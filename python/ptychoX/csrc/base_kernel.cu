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

__global__ void crop_mul_kernel(
    const complex64* __restrict__ sample,
    const complex64* __restrict__ probe,
    const int64_t* __restrict__ positions_xy,
    int64_t sample_w,
    int64_t probe_h,
    int64_t probe_w,
    int64_t batch,
    complex64* __restrict__ patch_stack,
    complex64* __restrict__ exit_wave) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * probe_h * probe_w;
  if (linear >= total) {
    return;
  }

  const int64_t local = linear % (probe_h * probe_w);
  const int64_t view = linear / (probe_h * probe_w);
  const int64_t py = local / probe_w;
  const int64_t px = local % probe_w;

  // PyTorch row-major convention: flat index = y * width + x.
  const int64_t x0 = positions_xy[2 * view + 0];
  const int64_t y0 = positions_xy[2 * view + 1];
  const complex64 w = sample[(y0 + py) * sample_w + (x0 + px)];
  const complex64 p = probe[py * probe_w + px];

  patch_stack[linear] = w;
  exit_wave[linear] = cmul(w, p);
}

__global__ void scatter_backward_kernel(
    const complex64* __restrict__ grad_exit,
    const complex64* __restrict__ patch_stack,
    const complex64* __restrict__ probe,
    const int64_t* __restrict__ positions_xy,
    int64_t sample_w,
    int64_t probe_h,
    int64_t probe_w,
    int64_t batch,
    complex64* __restrict__ grad_sample,
    complex64* __restrict__ grad_probe) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * probe_h * probe_w;
  if (linear >= total) {
    return;
  }

  const int64_t local = linear % (probe_h * probe_w);
  const int64_t view = linear / (probe_h * probe_w);
  const int64_t py = local / probe_w;
  const int64_t px = local % probe_w;
  const int64_t x0 = positions_xy[2 * view + 0];
  const int64_t y0 = positions_xy[2 * view + 1];

  const complex64 x = grad_exit[linear];
  const complex64 p = probe[py * probe_w + px];
  const complex64 w = patch_stack[linear];

  atomic_add_complex(&grad_sample[(y0 + py) * sample_w + (x0 + px)], cmul(x, cconj(p)));
  atomic_add_complex(&grad_probe[py * probe_w + px], cmul(x, cconj(w)));
}

__global__ void fftshift_amplitude_kernel(
    const complex64* __restrict__ fft_unshifted,
    int64_t height,
    int64_t width,
    int64_t batch,
    complex64* __restrict__ fft_shifted,
    float* __restrict__ amplitude) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * height * width;
  if (linear >= total) {
    return;
  }

  const int64_t local = linear % (height * width);
  const int64_t view = linear / (height * width);
  const int64_t y = local / width;
  const int64_t x = local % width;
  const int64_t src_y = (y + height / 2) % height;
  const int64_t src_x = (x + width / 2) % width;
  const int64_t src = view * height * width + src_y * width + src_x;
  const complex64 value = fft_unshifted[src];
  fft_shifted[linear] = value;
  amplitude[linear] = sqrtf(value.real() * value.real() + value.imag() * value.imag());
}

__global__ void phase_ifftshift_kernel(
    const float* __restrict__ grad_output,
    const complex64* __restrict__ far_field_shifted,
    int64_t height,
    int64_t width,
    int64_t batch,
    complex64* __restrict__ grad_unshifted) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * height * width;
  if (linear >= total) {
    return;
  }

  const int64_t local = linear % (height * width);
  const int64_t view = linear / (height * width);
  const int64_t y = local / width;
  const int64_t x = local % width;
  const int64_t shifted_y = (y + (height + 1) / 2) % height;
  const int64_t shifted_x = (x + (width + 1) / 2) % width;
  const int64_t shifted = view * height * width + shifted_y * width + shifted_x;

  const complex64 f = far_field_shifted[shifted];
  const float mag = sqrtf(f.real() * f.real() + f.imag() * f.imag());
  const float scale = grad_output[shifted] / fmaxf(mag, 1.1920928955078125e-7f);
  grad_unshifted[linear] = complex64(f.real() * scale, f.imag() * scale);
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

void ptychox_base_clear_plan_cache_cuda() {
  clear_cached_cufft_plans_impl();
}

std::vector<torch::Tensor> ptychox_base_crop_mul_cuda(
    torch::Tensor sample,
    torch::Tensor probe,
    torch::Tensor positions_xy) {
  const auto batch = positions_xy.size(0);
  const auto probe_h = probe.size(0);
  const auto probe_w = probe.size(1);
  const auto sample_w = sample.size(1);

  auto patch_stack = torch::empty({batch, probe_h, probe_w}, sample.options());
  auto exit_wave = torch::empty_like(patch_stack);

  const int threads = 256;
  const int64_t total = batch * probe_h * probe_w;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  crop_mul_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(sample.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(probe.data_ptr<c10::complex<float>>()),
      positions_xy.data_ptr<int64_t>(),
      sample_w,
      probe_h,
      probe_w,
      batch,
      reinterpret_cast<complex64*>(patch_stack.data_ptr<c10::complex<float>>()),
      reinterpret_cast<complex64*>(exit_wave.data_ptr<c10::complex<float>>()));

  return {patch_stack, exit_wave};
}

std::vector<torch::Tensor> ptychox_base_scatter_cuda(
    torch::Tensor grad_exit,
    torch::Tensor patch_stack,
    torch::Tensor probe,
    torch::Tensor positions_xy,
    int64_t sample_h,
    int64_t sample_w) {
  const auto batch = positions_xy.size(0);
  const auto probe_h = probe.size(0);
  const auto probe_w = probe.size(1);

  auto grad_sample = torch::zeros({sample_h, sample_w}, probe.options());
  auto grad_probe = torch::zeros_like(probe);

  const int threads = 256;
  const int64_t total = batch * probe_h * probe_w;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  scatter_backward_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(grad_exit.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(patch_stack.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(probe.data_ptr<c10::complex<float>>()),
      positions_xy.data_ptr<int64_t>(),
      sample_w,
      probe_h,
      probe_w,
      batch,
      reinterpret_cast<complex64*>(grad_sample.data_ptr<c10::complex<float>>()),
      reinterpret_cast<complex64*>(grad_probe.data_ptr<c10::complex<float>>()));

  return {grad_sample, grad_probe};
}

std::vector<torch::Tensor> ptychox_base_forward_cuda(
    torch::Tensor sample,
    torch::Tensor probe,
    torch::Tensor positions_xy) {
  auto crop = ptychox_base_crop_mul_cuda(sample, probe, positions_xy);
  auto patch_stack = crop[0];
  auto fft_unshifted = crop[1];

  const auto batch = positions_xy.size(0);
  const auto probe_h = probe.size(0);
  const auto probe_w = probe.size(1);
  run_cufft_c2c(fft_unshifted, batch, probe_h, probe_w, CUFFT_FORWARD);

  auto far_field_shifted = torch::empty_like(fft_unshifted);
  auto amplitude = torch::empty({batch, probe_h, probe_w}, sample.options().dtype(torch::kFloat));
  const int threads = 256;
  const int64_t total = batch * probe_h * probe_w;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  fftshift_amplitude_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(fft_unshifted.data_ptr<c10::complex<float>>()),
      probe_h,
      probe_w,
      batch,
      reinterpret_cast<complex64*>(far_field_shifted.data_ptr<c10::complex<float>>()),
      amplitude.data_ptr<float>());

  return {patch_stack, far_field_shifted, amplitude};
}

std::vector<torch::Tensor> ptychox_base_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor patch_stack,
    torch::Tensor probe,
    torch::Tensor far_field_shifted,
    torch::Tensor positions_xy,
    int64_t sample_h,
    int64_t sample_w) {
  const auto batch = positions_xy.size(0);
  const auto probe_h = probe.size(0);
  const auto probe_w = probe.size(1);
  auto grad_unshifted = torch::empty_like(far_field_shifted);

  const int threads = 256;
  const int64_t total = batch * probe_h * probe_w;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  phase_ifftshift_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      grad_output.data_ptr<float>(),
      reinterpret_cast<const complex64*>(far_field_shifted.data_ptr<c10::complex<float>>()),
      probe_h,
      probe_w,
      batch,
      reinterpret_cast<complex64*>(grad_unshifted.data_ptr<c10::complex<float>>()));

  run_cufft_c2c(grad_unshifted, batch, probe_h, probe_w, CUFFT_INVERSE);
  return ptychox_base_scatter_cuda(grad_unshifted, patch_stack, probe, positions_xy, sample_h, sample_w);
}
