"""Coded ptychography differentiable operator."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ptychoX._utils import can_use_cuda_ext, load_cuda_extension

_CUDA_EXT = None


def clear_cuda_plan_cache() -> None:
    cuda_ext = _CUDA_EXT
    if cuda_ext is not None and hasattr(cuda_ext, "clear_plan_cache"):
        cuda_ext.clear_plan_cache()


def ptychoX_coded(
    object_field: torch.Tensor,
    coded_surface: torch.Tensor,
    transfer_1: torch.Tensor,
    transfer_2: torch.Tensor,
    shifts_xy: torch.Tensor,
    downsample: int,
    use_cuda_kernels: bool = False,
) -> torch.Tensor:
    """Forward model for coded ptychography.

    Args:
        object_field: high-resolution complex object, `[H, W]`.
        coded_surface: high-resolution complex coded surface, `[H, W]`.
        transfer_1: propagation transfer function before the coded surface.
        transfer_2: propagation transfer function after the coded surface.
        shifts_xy: normalized subpixel shifts, `[batch, 2]`, `(dx, dy)`.
        downsample: integer detector downsample factor.

    Returns:
        RMS downsampled amplitudes, shape `[batch, H/downsample, W/downsample]`.
    """
    _check_inputs(object_field, coded_surface, transfer_1, transfer_2, shifts_xy, downsample)
    if not can_use_cuda_ext(object_field, coded_surface, use_cuda_kernels):
        return _coded_forward_torch(
            object_field,
            coded_surface,
            transfer_1,
            transfer_2,
            shifts_xy,
            downsample,
        )[3]
    return PtychoXCodedFunction.apply(
        object_field,
        coded_surface,
        transfer_1,
        transfer_2,
        shifts_xy,
        downsample,
        use_cuda_kernels,
    )


coded_ptychography = ptychoX_coded


class PtychoXCodedFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        object_field: torch.Tensor,
        coded_surface: torch.Tensor,
        transfer_1: torch.Tensor,
        transfer_2: torch.Tensor,
        shifts_xy: torch.Tensor,
        downsample: int,
        use_cuda_kernels: bool = False,
    ) -> torch.Tensor:
        _check_inputs(object_field, coded_surface, transfer_1, transfer_2, shifts_xy, downsample)
        object_field = object_field.contiguous()
        coded_surface = coded_surface.contiguous()
        transfer_1 = transfer_1.contiguous()
        transfer_2 = transfer_2.contiguous()
        shifts_xy = shifts_xy.to(device=object_field.device, dtype=object_field.real.dtype).contiguous()
        cuda_ext = _load_cuda_ext() if can_use_cuda_ext(object_field, coded_surface, use_cuda_kernels) else None
        if cuda_ext is not None and object_field.dtype == torch.complex64:
            object_spectrum, x_forward, sensor_field, amplitude = cuda_ext.forward(
                object_field,
                coded_surface,
                transfer_1,
                transfer_2,
                shifts_xy,
                downsample,
            )
            used_cuda_ext = True
        else:
            object_spectrum, x_forward, sensor_field, amplitude = _coded_forward_torch(
                object_field,
                coded_surface,
                transfer_1,
                transfer_2,
                shifts_xy,
                downsample,
            )
            used_cuda_ext = False

        ctx.save_for_backward(object_spectrum, x_forward, sensor_field, coded_surface, transfer_1, transfer_2, shifts_xy)
        ctx.downsample = downsample
        ctx.used_cuda_ext = used_cuda_ext
        return amplitude

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        object_spectrum, x_forward, sensor_field, coded_surface, transfer_1, transfer_2, shifts_xy = ctx.saved_tensors
        downsample = ctx.downsample
        grad_output = grad_output.to(dtype=sensor_field.real.dtype).contiguous()
        if ctx.used_cuda_ext:
            cuda_ext = _load_cuda_ext()
            grad_object, grad_coded = cuda_ext.backward(
                grad_output,
                object_spectrum,
                x_forward,
                sensor_field,
                coded_surface,
                transfer_1,
                transfer_2,
                shifts_xy,
                downsample,
            )
        else:
            raise RuntimeError("Internal error: torch fallback should not call custom coded backward.")
        return grad_object, grad_coded, None, None, None, None, None


def _coded_forward_torch(
    object_field: torch.Tensor,
    coded_surface: torch.Tensor,
    transfer_1: torch.Tensor,
    transfer_2: torch.Tensor,
    shifts_xy: torch.Tensor,
    downsample: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shifts_xy = shifts_xy.to(device=object_field.device, dtype=object_field.real.dtype)
    height, width = object_field.shape

    phase = shift_phase_ramp(height, width, shifts_xy, object_field.device, object_field.real.dtype)
    object_spectrum = torch.fft.fft2(object_field)
    x_forward = torch.fft.ifft2(object_spectrum.unsqueeze(0) * transfer_1.unsqueeze(0) * phase)
    after_code = x_forward * coded_surface.unsqueeze(0)
    sensor_field = torch.fft.ifft2(torch.fft.fft2(after_code) * transfer_2.unsqueeze(0))
    intensity = sensor_field.abs().square().unsqueeze(1)
    pooled = F.avg_pool2d(intensity, kernel_size=downsample, stride=downsample)
    return object_spectrum, x_forward, sensor_field, torch.sqrt(pooled[:, 0].clamp_min(0.0))


def shift_phase_ramp(
    height: int,
    width: int,
    shifts_xy: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, d=1.0, device=device).to(dtype) * height
    fx = torch.fft.fftfreq(width, d=1.0, device=device).to(dtype) * width
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    phase_arg = 2.0 * math.pi * (
        shifts_xy[:, 0, None, None] * xx.unsqueeze(0)
        + shifts_xy[:, 1, None, None] * yy.unsqueeze(0)
    )
    return torch.exp(1j * phase_arg)


def _check_inputs(
    object_field: torch.Tensor,
    coded_surface: torch.Tensor,
    transfer_1: torch.Tensor,
    transfer_2: torch.Tensor,
    shifts_xy: torch.Tensor,
    downsample: int,
) -> None:
    if object_field.ndim != 2:
        raise ValueError("object_field must be a 2D complex tensor.")
    for name, tensor in (
        ("coded_surface", coded_surface),
        ("transfer_1", transfer_1),
        ("transfer_2", transfer_2),
    ):
        if tensor.shape != object_field.shape:
            raise ValueError(f"{name} must have the same shape as object_field.")
        if tensor.device != object_field.device:
            raise ValueError(f"{name} must live on the same device as object_field.")
    if not object_field.is_complex() or not coded_surface.is_complex():
        raise TypeError("object_field and coded_surface must be complex tensors.")
    if not transfer_1.is_complex() or not transfer_2.is_complex():
        raise TypeError("transfer_1 and transfer_2 must be complex tensors.")
    if shifts_xy.ndim != 2 or shifts_xy.shape[-1] != 2:
        raise ValueError("shifts_xy must have shape [batch, 2].")
    if object_field.shape[0] % downsample != 0 or object_field.shape[1] % downsample != 0:
        raise ValueError("object_field shape must be divisible by downsample.")


def _load_cuda_ext():
    global _CUDA_EXT
    if _CUDA_EXT is not None:
        return _CUDA_EXT

    root = Path(__file__).resolve().parents[1]
    _CUDA_EXT = load_cuda_extension(
        module_name="ptychoX.coded._C",
        build_name="ptychoX_coded_C",
        sources=[
            root / "csrc" / "coded_extension.cpp",
            root / "csrc" / "coded_kernel.cu",
        ],
    )
    return _CUDA_EXT
