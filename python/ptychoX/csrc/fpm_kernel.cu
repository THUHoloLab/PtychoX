#include <cuda_runtime.h>
#include <cufft.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>

#include <mutex>
#include <unordered_map>

namespace {

using complex64 = c10::complex<float>;

inline void cufft_check(cufftResult result, const char* message) {
  TORCH_CHECK(result == CUFFT_SUCCESS, message, " cuFFT error code=", static_cast<int>(result));
}

struct CufftPlanKey {
  int device = 0;
  int batch = 0;
  int height = 0;
  int width = 0;

  bool operator==(const CufftPlanKey& other) const {
    return device == other.device && batch == other.batch && height == other.height && width == other.width;
  }
};

struct CufftPlanKeyHash {
  size_t operator()(const CufftPlanKey& key) const {
    size_t h = static_cast<size_t>(key.device);
    h = h * 1315423911u + static_cast<size_t>(key.batch);
    h = h * 1315423911u + static_cast<size_t>(key.height);
    h = h * 1315423911u + static_cast<size_t>(key.width);
    return h;
  }
};

std::mutex g_plan_cache_mutex;
std::unordered_map<CufftPlanKey, cufftHandle, CufftPlanKeyHash> g_plan_cache;

cufftHandle get_cached_cufft_plan(int64_t batch, int64_t height, int64_t width) {
  int device = 0;
  cudaGetDevice(&device);

  CufftPlanKey key{
      device,
      static_cast<int>(batch),
      static_cast<int>(height),
      static_cast<int>(width),
  };

  std::lock_guard<std::mutex> lock(g_plan_cache_mutex);
  auto it = g_plan_cache.find(key);
  if (it != g_plan_cache.end()) {
    return it->second;
  }

  cufftHandle plan;
  int n[2] = {static_cast<int>(height), static_cast<int>(width)};
  const int istride = 1;
  const int ostride = 1;
  const int idist = static_cast<int>(height * width);
  const int odist = static_cast<int>(height * width);
  cufft_check(
      cufftPlanMany(
          &plan,
          2,
          n,
          nullptr,
          istride,
          idist,
          nullptr,
          ostride,
          odist,
          CUFFT_C2C,
          static_cast<int>(batch)),
      "cufftPlanMany failed.");

  g_plan_cache.emplace(key, plan);
  return plan;
}

void clear_cached_cufft_plans_impl() {
  std::lock_guard<std::mutex> lock(g_plan_cache_mutex);
  for (auto& item : g_plan_cache) {
    cufftDestroy(item.second);
  }
  g_plan_cache.clear();
}

__device__ __forceinline__ complex64 cmul(complex64 a, complex64 b) {
  return complex64(a.real() * b.real() - a.imag() * b.imag(),
                   a.real() * b.imag() + a.imag() * b.real());
}

__device__ __forceinline__ complex64 cconj(complex64 a) {
  return complex64(a.real(), -a.imag());
}

__device__ __forceinline__ void atomic_add_complex(complex64* ptr, complex64 value) {
  float* raw = reinterpret_cast<float*>(ptr);
  atomicAdd(raw, value.real());
  atomicAdd(raw + 1, value.imag());
}

__global__ void fpm_extract_shift_kernel(
    const complex64* __restrict__ object_fft,
    const complex64* __restrict__ pupil,
    const int64_t* __restrict__ led_indices_xy,
    int64_t object_h,
    int64_t object_w,
    int64_t low_h,
    int64_t low_w,
    int64_t batch,
    complex64* __restrict__ sub_spectrum,
    complex64* __restrict__ low_field_shifted) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * low_h * low_w;
  if (linear >= total) {
    return;
  }

  const int64_t local = linear % (low_h * low_w);
  const int64_t view = linear / (low_h * low_w);
  const int64_t y = local / low_w;
  const int64_t x = local % low_w;
  const int64_t led_x = led_indices_xy[2 * view + 0];
  const int64_t led_y = led_indices_xy[2 * view + 1];

  const int64_t src_y = (y + led_y + object_h / 2) % object_h;
  const int64_t src_x = (x + led_x + object_w / 2) % object_w;
  const complex64 sub = object_fft[src_y * object_w + src_x];
  const complex64 p = pupil[y * low_w + x];
  const complex64 field = cmul(sub, p);

  sub_spectrum[linear] = sub;
  const int64_t dst_y = (y + low_h / 2) % low_h;
  const int64_t dst_x = (x + low_w / 2) % low_w;
  low_field_shifted[view * low_h * low_w + dst_y * low_w + dst_x] = field;
}

__global__ void fpm_normalize_amp_kernel(
    complex64* __restrict__ image_field,
    int64_t total,
    float ratio,
    float* __restrict__ amplitude) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  complex64 z = image_field[linear];
  z = complex64(z.real() * ratio, z.imag() * ratio);
  image_field[linear] = z;
  amplitude[linear] = sqrtf(z.real() * z.real() + z.imag() * z.imag());
}

__global__ void fpm_phase_kernel(
    const float* __restrict__ grad_output,
    const complex64* __restrict__ image_field,
    int64_t total,
    complex64* __restrict__ grad_image) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const complex64 z = image_field[linear];
  const float mag = sqrtf(z.real() * z.real() + z.imag() * z.imag());
  const float scale = grad_output[linear] / fmaxf(mag, 1.1920928955078125e-7f);
  grad_image[linear] = complex64(z.real() * scale, z.imag() * scale);
}

__global__ void fpm_scatter_spectrum_kernel(
    const complex64* __restrict__ grad_low_shifted,
    const complex64* __restrict__ sub_spectrum,
    const complex64* __restrict__ pupil,
    const int64_t* __restrict__ led_indices_xy,
    int64_t object_h,
    int64_t object_w,
    int64_t low_h,
    int64_t low_w,
    int64_t batch,
    float low_ratio,
    complex64* __restrict__ grad_object_fft,
    complex64* __restrict__ grad_pupil) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * low_h * low_w;
  if (linear >= total) {
    return;
  }

  const int64_t local = linear % (low_h * low_w);
  const int64_t view = linear / (low_h * low_w);
  const int64_t y = local / low_w;
  const int64_t x = local % low_w;
  const int64_t shifted_y = (y + low_h / 2) % low_h;
  const int64_t shifted_x = (x + low_w / 2) % low_w;
  const complex64 g0 = grad_low_shifted[view * low_h * low_w + shifted_y * low_w + shifted_x];
  const complex64 g = complex64(g0.real() * low_ratio, g0.imag() * low_ratio);

  const complex64 sub = sub_spectrum[linear];
  const complex64 p = pupil[y * low_w + x];
  atomic_add_complex(&grad_pupil[y * low_w + x], cmul(g, cconj(sub)));

  const int64_t led_x = led_indices_xy[2 * view + 0];
  const int64_t led_y = led_indices_xy[2 * view + 1];
  const int64_t dst_y = (y + led_y + object_h / 2) % object_h;
  const int64_t dst_x = (x + led_x + object_w / 2) % object_w;
  atomic_add_complex(&grad_object_fft[dst_y * object_w + dst_x], cmul(g, cconj(p)));
}

}  // namespace

void run_cufft_c2c(torch::Tensor tensor, int64_t batch, int64_t height, int64_t width, int direction) {
  cufftHandle plan = get_cached_cufft_plan(batch, height, width);
  cufft_check(cufftSetStream(plan, c10::cuda::getCurrentCUDAStream()), "cufftSetStream failed.");
  cufft_check(
      cufftExecC2C(
          plan,
          reinterpret_cast<cufftComplex*>(tensor.data_ptr<c10::complex<float>>()),
          reinterpret_cast<cufftComplex*>(tensor.data_ptr<c10::complex<float>>()),
          direction),
      "cufftExecC2C failed.");
}

void ptychox_fpm_clear_plan_cache_cuda() {
  clear_cached_cufft_plans_impl();
}

std::vector<torch::Tensor> ptychox_fpm_forward_cuda(
    torch::Tensor object,
    torch::Tensor pupil,
    torch::Tensor led_indices_xy) {
  const auto batch = led_indices_xy.size(0);
  const auto object_h = object.size(0);
  const auto object_w = object.size(1);
  const auto low_h = pupil.size(0);
  const auto low_w = pupil.size(1);

  auto object_fft = object.clone();
  run_cufft_c2c(object_fft, 1, object_h, object_w, CUFFT_FORWARD);

  auto sub_spectrum = torch::empty({batch, low_h, low_w}, object.options());
  auto image_field = torch::empty_like(sub_spectrum);
  const int threads = 256;
  const int64_t total = batch * low_h * low_w;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  fpm_extract_shift_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(object_fft.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(pupil.data_ptr<c10::complex<float>>()),
      led_indices_xy.data_ptr<int64_t>(),
      object_h,
      object_w,
      low_h,
      low_w,
      batch,
      reinterpret_cast<complex64*>(sub_spectrum.data_ptr<c10::complex<float>>()),
      reinterpret_cast<complex64*>(image_field.data_ptr<c10::complex<float>>()));

  run_cufft_c2c(image_field, batch, low_h, low_w, CUFFT_INVERSE);

  auto amplitude = torch::empty({batch, low_h, low_w}, object.options().dtype(torch::kFloat));
  fpm_normalize_amp_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<complex64*>(image_field.data_ptr<c10::complex<float>>()),
      total,
      1.0f / static_cast<float>(low_h * low_w),
      amplitude.data_ptr<float>());

  return {sub_spectrum, image_field, amplitude};
}

std::vector<torch::Tensor> ptychox_fpm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor sub_spectrum,
    torch::Tensor pupil,
    torch::Tensor image_field,
    torch::Tensor led_indices_xy,
    int64_t object_h,
    int64_t object_w) {
  const auto batch = led_indices_xy.size(0);
  const auto low_h = pupil.size(0);
  const auto low_w = pupil.size(1);
  const int64_t total = batch * low_h * low_w;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  auto grad_low_shifted = torch::empty_like(image_field);
  fpm_phase_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      grad_output.data_ptr<float>(),
      reinterpret_cast<const complex64*>(image_field.data_ptr<c10::complex<float>>()),
      total,
      reinterpret_cast<complex64*>(grad_low_shifted.data_ptr<c10::complex<float>>()));

  run_cufft_c2c(grad_low_shifted, batch, low_h, low_w, CUFFT_FORWARD);

  auto grad_object_fft = torch::zeros({object_h, object_w}, pupil.options());
  auto grad_pupil = torch::zeros_like(pupil);
  fpm_scatter_spectrum_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(grad_low_shifted.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(sub_spectrum.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(pupil.data_ptr<c10::complex<float>>()),
      led_indices_xy.data_ptr<int64_t>(),
      object_h,
      object_w,
      low_h,
      low_w,
      batch,
      1.0f / static_cast<float>(low_h * low_w),
      reinterpret_cast<complex64*>(grad_object_fft.data_ptr<c10::complex<float>>()),
      reinterpret_cast<complex64*>(grad_pupil.data_ptr<c10::complex<float>>()));

  run_cufft_c2c(grad_object_fft, 1, object_h, object_w, CUFFT_INVERSE);
  return {grad_object_fft, grad_pupil};
}
