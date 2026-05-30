"""Classical ptychography autograd operator."""

from __future__ import annotations

from pathlib import Path

import torch

from ptychoX._utils import can_use_cuda_ext, check_complex_scan_inputs, load_cuda_extension

_CUDA_EXT = None


def ptychoX_base(
    sample: torch.Tensor,
    probe: torch.Tensor,
    positions_xy: torch.Tensor,
    use_cuda_kernels: bool = False,
) -> torch.Tensor:
    """Return `abs(fftshift(fft2(crop(sample) * probe)))`.

    Args:
        sample: complex object tensor, shape `[sample_h, sample_w]`.
        probe: complex probe tensor, shape `[probe_h, probe_w]`.
        positions_xy: zero-based `(x, y)` scan positions, shape `[batch, 2]`.
        use_cuda_kernels: use the compiled cuFFT extension when available.
    """
    return PtychoXBaseFunction.apply(sample, probe, positions_xy, use_cuda_kernels)


ptychography_base = ptychoX_base


def clear_cuda_plan_cache() -> None:
    cuda_ext = _CUDA_EXT
    if cuda_ext is not None and hasattr(cuda_ext, "clear_plan_cache"):
        cuda_ext.clear_plan_cache()


class PtychoXBaseFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        sample: torch.Tensor,
        probe: torch.Tensor,
        positions_xy: torch.Tensor,
        use_cuda_kernels: bool = False,
    ) -> torch.Tensor:
        check_complex_scan_inputs(sample, probe, positions_xy)
        positions_xy = positions_xy.to(device=sample.device, dtype=torch.long).contiguous()
        sample = sample.contiguous()
        probe = probe.contiguous()

        probe_h, probe_w = probe.shape
        cuda_ext = _load_cuda_ext() if can_use_cuda_ext(sample, probe, use_cuda_kernels) else None
        if cuda_ext is not None:
            patch_stack, far_field, amplitude = cuda_ext.forward(sample, probe, positions_xy)
            used_cuda_ext = True
        else:
            patches = []
            for x, y in positions_xy.tolist():
                patches.append(sample[y : y + probe_h, x : x + probe_w])
            patch_stack = torch.stack(patches, dim=0)
            exit_wave = patch_stack * probe.unsqueeze(0)
            far_field = torch.fft.fftshift(torch.fft.fft2(exit_wave), dim=(-2, -1))
            amplitude = far_field.abs()
            used_cuda_ext = False

        ctx.save_for_backward(patch_stack, probe, far_field, positions_xy)
        ctx.sample_shape = tuple(sample.shape)
        ctx.used_cuda_ext = used_cuda_ext
        return amplitude

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        patch_stack, probe, far_field, positions_xy = ctx.saved_tensors
        sample_h, sample_w = ctx.sample_shape
        _, probe_h, probe_w = patch_stack.shape

        grad_output = grad_output.to(dtype=far_field.real.dtype).contiguous()
        if ctx.used_cuda_ext:
            cuda_ext = _load_cuda_ext()
            grad_sample, grad_probe = cuda_ext.backward(
                grad_output,
                patch_stack,
                probe,
                far_field,
                positions_xy,
                sample_h,
                sample_w,
            )
        else:
            phase = far_field / far_field.abs().clamp_min(torch.finfo(far_field.real.dtype).eps)
            grad_far = grad_output * phase
            grad_exit = torch.fft.ifft2(torch.fft.ifftshift(grad_far, dim=(-2, -1))) * (probe_h * probe_w)
            grad_sample = torch.zeros((sample_h, sample_w), dtype=probe.dtype, device=probe.device)
            grad_probe = torch.zeros_like(probe)

            probe_conj = probe.conj()
            for view in range(patch_stack.shape[0]):
                x = int(positions_xy[view, 0])
                y = int(positions_xy[view, 1])
                this_grad = grad_exit[view]
                grad_sample[y : y + probe_h, x : x + probe_w] += this_grad * probe_conj
                grad_probe += this_grad * patch_stack[view].conj()

        return grad_sample, grad_probe, None, None


def _load_cuda_ext():
    global _CUDA_EXT
    if _CUDA_EXT is not None:
        return _CUDA_EXT

    root = Path(__file__).resolve().parents[1]
    _CUDA_EXT = load_cuda_extension(
        module_name="ptychoX.base._C",
        build_name="ptychoX_base_C",
        sources=[
            root / "csrc" / "base" / "extension.cpp",
            root / "csrc" / "base" / "kernel.cu",
        ],
    )
    return _CUDA_EXT
