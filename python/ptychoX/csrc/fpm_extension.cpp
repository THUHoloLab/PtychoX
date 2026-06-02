#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> ptychox_fpm_forward_cuda(
    torch::Tensor object,
    torch::Tensor pupil,
    torch::Tensor led_indices_xy);

std::vector<torch::Tensor> ptychox_fpm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor sub_spectrum,
    torch::Tensor pupil,
    torch::Tensor image_field,
    torch::Tensor led_indices_xy,
    int64_t object_h,
    int64_t object_w);

void ptychox_fpm_clear_plan_cache_cuda();

std::vector<torch::Tensor> forward(
    torch::Tensor object,
    torch::Tensor pupil,
    torch::Tensor led_indices_xy) {
  TORCH_CHECK(object.is_cuda(), "object must be a CUDA tensor");
  TORCH_CHECK(pupil.is_cuda(), "pupil must be a CUDA tensor");
  TORCH_CHECK(led_indices_xy.is_cuda(), "led_indices_xy must be a CUDA tensor");
  TORCH_CHECK(object.scalar_type() == torch::kComplexFloat, "object must be complex64");
  TORCH_CHECK(pupil.scalar_type() == torch::kComplexFloat, "pupil must be complex64");
  TORCH_CHECK(led_indices_xy.scalar_type() == torch::kLong, "led_indices_xy must be int64");
  return ptychox_fpm_forward_cuda(object.contiguous(), pupil.contiguous(), led_indices_xy.contiguous());
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_output,
    torch::Tensor sub_spectrum,
    torch::Tensor pupil,
    torch::Tensor image_field,
    torch::Tensor led_indices_xy,
    int64_t object_h,
    int64_t object_w) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(sub_spectrum.is_cuda(), "sub_spectrum must be a CUDA tensor");
  TORCH_CHECK(pupil.is_cuda(), "pupil must be a CUDA tensor");
  TORCH_CHECK(image_field.is_cuda(), "image_field must be a CUDA tensor");
  TORCH_CHECK(led_indices_xy.is_cuda(), "led_indices_xy must be a CUDA tensor");
  TORCH_CHECK(grad_output.scalar_type() == torch::kFloat, "grad_output must be float32");
  TORCH_CHECK(sub_spectrum.scalar_type() == torch::kComplexFloat, "sub_spectrum must be complex64");
  TORCH_CHECK(pupil.scalar_type() == torch::kComplexFloat, "pupil must be complex64");
  TORCH_CHECK(image_field.scalar_type() == torch::kComplexFloat, "image_field must be complex64");
  TORCH_CHECK(led_indices_xy.scalar_type() == torch::kLong, "led_indices_xy must be int64");
  return ptychox_fpm_backward_cuda(
      grad_output.contiguous(),
      sub_spectrum.contiguous(),
      pupil.contiguous(),
      image_field.contiguous(),
      led_indices_xy.contiguous(),
      object_h,
      object_w);
}

PYBIND11_MODULE(_C, m) {
  m.def("forward", &forward, "PtychoX FPM forward with cached cuFFT (CUDA)");
  m.def("backward", &backward, "PtychoX FPM backward with cached cuFFT (CUDA)");
  m.def("clear_plan_cache", &ptychox_fpm_clear_plan_cache_cuda, "Destroy cached cuFFT plans");
}
