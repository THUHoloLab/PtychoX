#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> ptychox_coded_forward_cuda(
    torch::Tensor object,
    torch::Tensor coded_surface,
    torch::Tensor transfer_1,
    torch::Tensor transfer_2,
    torch::Tensor shifts_xy,
    int64_t downsample);

std::vector<torch::Tensor> ptychox_coded_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor object_spectrum,
    torch::Tensor x_forward,
    torch::Tensor sensor_field,
    torch::Tensor coded_surface,
    torch::Tensor transfer_1,
    torch::Tensor transfer_2,
    torch::Tensor shifts_xy,
    int64_t downsample);

void ptychox_coded_clear_plan_cache_cuda();

std::vector<torch::Tensor> forward(
    torch::Tensor object,
    torch::Tensor coded_surface,
    torch::Tensor transfer_1,
    torch::Tensor transfer_2,
    torch::Tensor shifts_xy,
    int64_t downsample) {
  TORCH_CHECK(object.is_cuda(), "object must be a CUDA tensor");
  TORCH_CHECK(coded_surface.is_cuda(), "coded_surface must be a CUDA tensor");
  TORCH_CHECK(transfer_1.is_cuda(), "transfer_1 must be a CUDA tensor");
  TORCH_CHECK(transfer_2.is_cuda(), "transfer_2 must be a CUDA tensor");
  TORCH_CHECK(shifts_xy.is_cuda(), "shifts_xy must be a CUDA tensor");
  TORCH_CHECK(object.scalar_type() == torch::kComplexFloat, "object must be complex64");
  TORCH_CHECK(coded_surface.scalar_type() == torch::kComplexFloat, "coded_surface must be complex64");
  TORCH_CHECK(transfer_1.scalar_type() == torch::kComplexFloat, "transfer_1 must be complex64");
  TORCH_CHECK(transfer_2.scalar_type() == torch::kComplexFloat, "transfer_2 must be complex64");
  TORCH_CHECK(shifts_xy.scalar_type() == torch::kFloat, "shifts_xy must be float32");
  return ptychox_coded_forward_cuda(
      object.contiguous(),
      coded_surface.contiguous(),
      transfer_1.contiguous(),
      transfer_2.contiguous(),
      shifts_xy.contiguous(),
      downsample);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_output,
    torch::Tensor object_spectrum,
    torch::Tensor x_forward,
    torch::Tensor sensor_field,
    torch::Tensor coded_surface,
    torch::Tensor transfer_1,
    torch::Tensor transfer_2,
    torch::Tensor shifts_xy,
    int64_t downsample) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(object_spectrum.is_cuda(), "object_spectrum must be a CUDA tensor");
  TORCH_CHECK(x_forward.is_cuda(), "x_forward must be a CUDA tensor");
  TORCH_CHECK(sensor_field.is_cuda(), "sensor_field must be a CUDA tensor");
  TORCH_CHECK(coded_surface.is_cuda(), "coded_surface must be a CUDA tensor");
  TORCH_CHECK(transfer_1.is_cuda(), "transfer_1 must be a CUDA tensor");
  TORCH_CHECK(transfer_2.is_cuda(), "transfer_2 must be a CUDA tensor");
  TORCH_CHECK(shifts_xy.is_cuda(), "shifts_xy must be a CUDA tensor");
  TORCH_CHECK(grad_output.scalar_type() == torch::kFloat, "grad_output must be float32");
  TORCH_CHECK(object_spectrum.scalar_type() == torch::kComplexFloat, "object_spectrum must be complex64");
  TORCH_CHECK(x_forward.scalar_type() == torch::kComplexFloat, "x_forward must be complex64");
  TORCH_CHECK(sensor_field.scalar_type() == torch::kComplexFloat, "sensor_field must be complex64");
  TORCH_CHECK(coded_surface.scalar_type() == torch::kComplexFloat, "coded_surface must be complex64");
  TORCH_CHECK(transfer_1.scalar_type() == torch::kComplexFloat, "transfer_1 must be complex64");
  TORCH_CHECK(transfer_2.scalar_type() == torch::kComplexFloat, "transfer_2 must be complex64");
  TORCH_CHECK(shifts_xy.scalar_type() == torch::kFloat, "shifts_xy must be float32");
  return ptychox_coded_backward_cuda(
      grad_output.contiguous(),
      object_spectrum.contiguous(),
      x_forward.contiguous(),
      sensor_field.contiguous(),
      coded_surface.contiguous(),
      transfer_1.contiguous(),
      transfer_2.contiguous(),
      shifts_xy.contiguous(),
      downsample);
}

PYBIND11_MODULE(_C, m) {
  m.def("forward", &forward, "PtychoX coded forward with cached cuFFT (CUDA)");
  m.def("backward", &backward, "PtychoX coded backward with cached cuFFT (CUDA)");
  m.def("clear_plan_cache", &ptychox_coded_clear_plan_cache_cuda, "Destroy cached cuFFT plans");
}

