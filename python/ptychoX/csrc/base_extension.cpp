#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> ptychox_base_forward_cuda(
    torch::Tensor sample,
    torch::Tensor probe,
    torch::Tensor positions_xy);

std::vector<torch::Tensor> ptychox_base_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor patch_stack,
    torch::Tensor probe,
    torch::Tensor far_field_shifted,
    torch::Tensor positions_xy,
    int64_t sample_h,
    int64_t sample_w);

void ptychox_base_clear_plan_cache_cuda();

std::vector<torch::Tensor> forward(
    torch::Tensor sample,
    torch::Tensor probe,
    torch::Tensor positions_xy) {
  TORCH_CHECK(sample.is_cuda(), "sample must be a CUDA tensor");
  TORCH_CHECK(probe.is_cuda(), "probe must be a CUDA tensor");
  TORCH_CHECK(positions_xy.is_cuda(), "positions_xy must be a CUDA tensor");
  TORCH_CHECK(sample.scalar_type() == torch::kComplexFloat, "sample must be complex64");
  TORCH_CHECK(probe.scalar_type() == torch::kComplexFloat, "probe must be complex64");
  TORCH_CHECK(positions_xy.scalar_type() == torch::kLong, "positions_xy must be int64");
  return ptychox_base_forward_cuda(sample.contiguous(), probe.contiguous(), positions_xy.contiguous());
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_output,
    torch::Tensor patch_stack,
    torch::Tensor probe,
    torch::Tensor far_field_shifted,
    torch::Tensor positions_xy,
    int64_t sample_h,
    int64_t sample_w) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(patch_stack.is_cuda(), "patch_stack must be a CUDA tensor");
  TORCH_CHECK(probe.is_cuda(), "probe must be a CUDA tensor");
  TORCH_CHECK(far_field_shifted.is_cuda(), "far_field_shifted must be a CUDA tensor");
  TORCH_CHECK(positions_xy.is_cuda(), "positions_xy must be a CUDA tensor");
  TORCH_CHECK(grad_output.scalar_type() == torch::kFloat, "grad_output must be float32");
  TORCH_CHECK(patch_stack.scalar_type() == torch::kComplexFloat, "patch_stack must be complex64");
  TORCH_CHECK(probe.scalar_type() == torch::kComplexFloat, "probe must be complex64");
  TORCH_CHECK(far_field_shifted.scalar_type() == torch::kComplexFloat, "far_field_shifted must be complex64");
  TORCH_CHECK(positions_xy.scalar_type() == torch::kLong, "positions_xy must be int64");
  return ptychox_base_backward_cuda(
      grad_output.contiguous(),
      patch_stack.contiguous(),
      probe.contiguous(),
      far_field_shifted.contiguous(),
      positions_xy.contiguous(),
      sample_h,
      sample_w);
}

PYBIND11_MODULE(_C, m) {
  m.def("forward", &forward, "PtychoX base forward with cached cuFFT (CUDA)");
  m.def("backward", &backward, "PtychoX base backward with cached cuFFT (CUDA)");
  m.def("clear_plan_cache", &ptychox_base_clear_plan_cache_cuda, "Destroy cached cuFFT plans");
}
