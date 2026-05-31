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

struct PlanKey {
  int device = 0;
  int batch = 0;
  int height = 0;
  int width = 0;

  bool operator==(const PlanKey& other) const {
    return device == other.device && batch == other.batch && height == other.height && width == other.width;
  }
};

struct PlanKeyHash {
  size_t operator()(const PlanKey& key) const {
    size_t h = static_cast<size_t>(key.device);
    h = h * 1315423911u + static_cast<size_t>(key.batch);
    h = h * 1315423911u + static_cast<size_t>(key.height);
    h = h * 1315423911u + static_cast<size_t>(key.width);
    return h;
  }
};

std::mutex g_plan_mutex;
std::unordered_map<PlanKey, cufftHandle, PlanKeyHash> g_plan_cache;

cufftHandle get_plan(int64_t batch, int64_t height, int64_t width) {
  int device = 0;
  cudaGetDevice(&device);
  PlanKey key{device, static_cast<int>(batch), static_cast<int>(height), static_cast<int>(width)};
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  auto it = g_plan_cache.find(key);
  if (it != g_plan_cache.end()) {
    return it->second;
  }

  cufftHandle plan;
  int n[2] = {static_cast<int>(height), static_cast<int>(width)};
  cufft_check(
      cufftPlanMany(
          &plan,
          2, n, nullptr, 1,
          static_cast<int>(height * width),
          nullptr, 1,
          static_cast<int>(height * width),
          CUFFT_C2C,
          static_cast<int>(batch)),
      "cufftPlanMany failed.");
  g_plan_cache.emplace(key, plan);
  return plan;
}

void run_cufft(torch::Tensor tensor, int64_t batch, int64_t height, int64_t width, int direction) {
  cufftHandle plan = get_plan(batch, height, width);
  cufft_check(cufftSetStream(plan, c10::cuda::getCurrentCUDAStream()), "cufftSetStream failed.");
  cufft_check(
      cufftExecC2C(
          plan,
          reinterpret_cast<cufftComplex*>(tensor.data_ptr<c10::complex<float>>()),
          reinterpret_cast<cufftComplex*>(tensor.data_ptr<c10::complex<float>>()),
          direction),
      "cufftExecC2C failed.");
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

__global__ void coded_shift_prop_kernel(
    const complex64* __restrict__ object_spectrum,
    const complex64* __restrict__ transfer_1,
    const float* __restrict__ shifts_xy,
    int64_t height,
    int64_t width,
    int64_t batch,
    complex64* __restrict__ x_forward_freq) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * height * width;
  if (linear >= total) {
    return;
  }
  const int64_t local = linear % (height * width);
  const int64_t view = linear / (height * width);
  const int64_t y = local / width;
  const int64_t x = local % width;
  const float fx = static_cast<float>(x - ((2 * x >= width) ? width : 0));
  const float fy = static_cast<float>(y - ((2 * y >= height) ? height : 0));
  const float dx = shifts_xy[2 * view + 0];
  const float dy = shifts_xy[2 * view + 1];
  const float arg = 6.283185307179586f * (fx * dx + fy * dy);
  const complex64 phase(cosf(arg), sinf(arg));
  x_forward_freq[linear] = cmul(cmul(object_spectrum[local], transfer_1[local]), phase);
}

__global__ void coded_mul_kernel(
    const complex64* __restrict__ x_forward,
    const complex64* __restrict__ coded_surface,
    int64_t total,
    int64_t page_size,
    float ratio,
    complex64* __restrict__ out) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t local = linear % page_size;
  complex64 x = x_forward[linear];
  x = complex64(x.real() * ratio, x.imag() * ratio);
  out[linear] = cmul(x, coded_surface[local]);
}

__global__ void coded_mul_transfer_kernel(
    complex64* __restrict__ x,
    const complex64* __restrict__ transfer,
    int64_t total,
    int64_t page_size) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t local = linear % page_size;
  x[linear] = cmul(x[linear], transfer[local]);
}

__global__ void coded_mul_transfer_conj_kernel(
    complex64* __restrict__ x,
    const complex64* __restrict__ transfer,
    int64_t total,
    int64_t page_size) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t local = linear % page_size;
  x[linear] = cmul(x[linear], cconj(transfer[local]));
}

__global__ void coded_downsample_kernel(
    const complex64* __restrict__ sensor_field,
    int downsample,
    int64_t low_h,
    int64_t low_w,
    int64_t batch,
    float ratio,
    float* __restrict__ amplitude) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * low_h * low_w;
  if (linear >= total) {
    return;
  }
  const int64_t local = linear % (low_h * low_w);
  const int64_t view = linear / (low_h * low_w);
  const int64_t y = local / low_w;
  const int64_t x = local % low_w;
  const int64_t high_w = low_w * downsample;
  const int64_t page_offset = view * low_h * low_w * downsample * downsample;
  float sum = 0.0f;
  for (int yy = 0; yy < downsample; ++yy) {
    for (int xx = 0; xx < downsample; ++xx) {
      const int64_t idx = page_offset + (y * downsample + yy) * high_w + (x * downsample + xx);
      const complex64 v = sensor_field[idx];
      const float re = v.real() * ratio;
      const float im = v.imag() * ratio;
      sum += re * re + im * im;
    }
  }
  amplitude[linear] = sqrtf(sum / static_cast<float>(downsample * downsample));
}

__global__ void coded_downsample_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ amplitude,
    const complex64* __restrict__ sensor_field,
    int downsample,
    int64_t low_h,
    int64_t low_w,
    int64_t batch,
    float ratio,
    complex64* __restrict__ grad_sensor) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * low_h * low_w * downsample * downsample;
  if (linear >= total) {
    return;
  }
  const int64_t high_h = low_h * downsample;
  const int64_t high_w = low_w * downsample;
  const int64_t page_size = high_h * high_w;
  const int64_t view = linear / page_size;
  const int64_t local = linear % page_size;
  const int64_t y = local / high_w;
  const int64_t x = local % high_w;
  const int64_t ly = y / downsample;
  const int64_t lx = x / downsample;
  const int64_t low_idx = view * low_h * low_w + ly * low_w + lx;
  const float scale = grad_output[low_idx] /
      (fmaxf(amplitude[low_idx], 1.0e-7f) * static_cast<float>(downsample * downsample));
  const complex64 v = sensor_field[linear];
  grad_sensor[linear] = complex64(v.real() * ratio * scale, v.imag() * ratio * scale);
}

__global__ void coded_coded_grad_kernel(
    const complex64* __restrict__ grad_after_code,
    const complex64* __restrict__ x_forward,
    const complex64* __restrict__ coded_surface,
    int64_t total,
    int64_t page_size,
    complex64* __restrict__ grad_x_forward,
    complex64* __restrict__ grad_coded) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t local = linear % page_size;
  const complex64 g = grad_after_code[linear];
  const complex64 xf = x_forward[linear];
  const complex64 c = coded_surface[local];
  grad_x_forward[linear] = cmul(g, cconj(c));
  atomic_add_complex(&grad_coded[local], cmul(g, cconj(xf)));
}

__global__ void coded_object_grad_kernel(
    const complex64* __restrict__ grad_x_forward_freq,
    const complex64* __restrict__ transfer_1,
    const float* __restrict__ shifts_xy,
    int64_t height,
    int64_t width,
    int64_t batch,
    float ratio,
    complex64* __restrict__ grad_object_spectrum) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * height * width;
  if (linear >= total) {
    return;
  }
  const int64_t local = linear % (height * width);
  const int64_t view = linear / (height * width);
  const int64_t y = local / width;
  const int64_t x = local % width;
  const float fx = static_cast<float>(x - ((2 * x >= width) ? width : 0));
  const float fy = static_cast<float>(y - ((2 * y >= height) ? height : 0));
  const float dx = shifts_xy[2 * view + 0];
  const float dy = shifts_xy[2 * view + 1];
  const float arg = 6.283185307179586f * (fx * dx + fy * dy);
  const complex64 phase(cosf(arg), sinf(arg));
  complex64 g = grad_x_forward_freq[linear];
  g = complex64(g.real() * ratio, g.imag() * ratio);
  atomic_add_complex(&grad_object_spectrum[local], cmul(cmul(g, cconj(transfer_1[local])), cconj(phase)));
}

}  // namespace

std::vector<torch::Tensor> ptychox_coded_forward_cuda(
    torch::Tensor object,
    torch::Tensor coded_surface,
    torch::Tensor transfer_1,
    torch::Tensor transfer_2,
    torch::Tensor shifts_xy,
    int64_t downsample) {
  const auto batch = shifts_xy.size(0);
  const auto height = object.size(0);
  const auto width = object.size(1);
  const int64_t page_size = height * width;
  const int64_t total = batch * page_size;
  const int threads = 256;
  const int blocks_total = static_cast<int>((total + threads - 1) / threads);

  auto object_spectrum = object.clone();
  run_cufft(object_spectrum, 1, height, width, CUFFT_FORWARD);

  auto x_forward = torch::empty({batch, height, width}, object.options());
  coded_shift_prop_kernel<<<blocks_total, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(object_spectrum.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(transfer_1.data_ptr<c10::complex<float>>()),
      shifts_xy.data_ptr<float>(),
      height,
      width,
      batch,
      reinterpret_cast<complex64*>(x_forward.data_ptr<c10::complex<float>>()));
  run_cufft(x_forward, batch, height, width, CUFFT_INVERSE);

  auto after_code = torch::empty_like(x_forward);
  coded_mul_kernel<<<blocks_total, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(x_forward.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(coded_surface.data_ptr<c10::complex<float>>()),
      total,
      page_size,
      1.0f / static_cast<float>(page_size),
      reinterpret_cast<complex64*>(after_code.data_ptr<c10::complex<float>>()));

  auto sensor_field = after_code.clone();
  run_cufft(sensor_field, batch, height, width, CUFFT_FORWARD);
  coded_mul_transfer_kernel<<<blocks_total, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<complex64*>(sensor_field.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(transfer_2.data_ptr<c10::complex<float>>()),
      total,
      page_size);
  run_cufft(sensor_field, batch, height, width, CUFFT_INVERSE);

  const auto low_h = height / downsample;
  const auto low_w = width / downsample;
  auto amplitude = torch::empty({batch, low_h, low_w}, object.options().dtype(torch::kFloat));
  const int64_t low_total = batch * low_h * low_w;
  const int blocks_low = static_cast<int>((low_total + threads - 1) / threads);
  coded_downsample_kernel<<<blocks_low, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(sensor_field.data_ptr<c10::complex<float>>()),
      static_cast<int>(downsample),
      low_h,
      low_w,
      batch,
      1.0f / static_cast<float>(page_size),
      amplitude.data_ptr<float>());

  return {object_spectrum, x_forward, sensor_field, amplitude};
}

std::vector<torch::Tensor> ptychox_coded_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor object_spectrum,
    torch::Tensor x_forward,
    torch::Tensor sensor_field,
    torch::Tensor coded_surface,
    torch::Tensor transfer_1,
    torch::Tensor transfer_2,
    torch::Tensor shifts_xy,
    int64_t downsample) {
  const auto batch = shifts_xy.size(0);
  const auto height = coded_surface.size(0);
  const auto width = coded_surface.size(1);
  const auto low_h = height / downsample;
  const auto low_w = width / downsample;
  const int64_t page_size = height * width;
  const int64_t total = batch * page_size;
  const int threads = 256;
  const int blocks_total = static_cast<int>((total + threads - 1) / threads);

  auto amplitude = torch::empty({batch, low_h, low_w}, grad_output.options());
  const int64_t low_total = batch * low_h * low_w;
  const int blocks_low = static_cast<int>((low_total + threads - 1) / threads);
  coded_downsample_kernel<<<blocks_low, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(sensor_field.data_ptr<c10::complex<float>>()),
      static_cast<int>(downsample),
      low_h,
      low_w,
      batch,
      1.0f / static_cast<float>(page_size),
      amplitude.data_ptr<float>());

  auto grad_sensor = torch::empty_like(sensor_field);
  coded_downsample_backward_kernel<<<blocks_total, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      grad_output.data_ptr<float>(),
      amplitude.data_ptr<float>(),
      reinterpret_cast<const complex64*>(sensor_field.data_ptr<c10::complex<float>>()),
      static_cast<int>(downsample),
      low_h,
      low_w,
      batch,
      1.0f / static_cast<float>(page_size),
      reinterpret_cast<complex64*>(grad_sensor.data_ptr<c10::complex<float>>()));

  run_cufft(grad_sensor, batch, height, width, CUFFT_FORWARD);
  coded_mul_transfer_conj_kernel<<<blocks_total, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<complex64*>(grad_sensor.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(transfer_2.data_ptr<c10::complex<float>>()),
      total,
      page_size);
  run_cufft(grad_sensor, batch, height, width, CUFFT_INVERSE);

  auto grad_x_forward = torch::empty_like(x_forward);
  auto grad_coded = torch::zeros_like(coded_surface);
  coded_coded_grad_kernel<<<blocks_total, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(grad_sensor.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(x_forward.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(coded_surface.data_ptr<c10::complex<float>>()),
      total,
      page_size,
      reinterpret_cast<complex64*>(grad_x_forward.data_ptr<c10::complex<float>>()),
      reinterpret_cast<complex64*>(grad_coded.data_ptr<c10::complex<float>>()));

  run_cufft(grad_x_forward, batch, height, width, CUFFT_FORWARD);
  auto grad_object_spectrum = torch::zeros({height, width}, coded_surface.options());
  coded_object_grad_kernel<<<blocks_total, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const complex64*>(grad_x_forward.data_ptr<c10::complex<float>>()),
      reinterpret_cast<const complex64*>(transfer_1.data_ptr<c10::complex<float>>()),
      shifts_xy.data_ptr<float>(),
      height,
      width,
      batch,
      1.0f / static_cast<float>(page_size),
      reinterpret_cast<complex64*>(grad_object_spectrum.data_ptr<c10::complex<float>>()));

  run_cufft(grad_object_spectrum, 1, height, width, CUFFT_INVERSE);
  return {grad_object_spectrum, grad_coded};
}

void ptychox_coded_clear_plan_cache_cuda() {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  for (auto& item : g_plan_cache) {
    cufftDestroy(item.second);
  }
  g_plan_cache.clear();
}
