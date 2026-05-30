"""Fourier ptychography autograd operator."""

from __future__ import annotations

from pathlib import Path

import torch

from ptychoX._utils import can_use_cuda_ext, check_complex_scan_inputs, load_cuda_extension

_CUDA_EXT = None


def ptychoX_fourier(
    object_field: torch.Tensor,
    pupil: torch.Tensor,
    led_indices_xy: torch.Tensor,
    use_cuda_kernels: bool = False,
) -> torch.Tensor:
    """Fourier ptychography amplitude operator.

    The PyTorch convention is row-major and zero-based:
    `led_indices_xy[:, 0]` is x and `led_indices_xy[:, 1]` is y.
    """
    return PtychoXFPMFunction.apply(object_field, pupil, led_indices_xy, use_cuda_kernels)


fourier_ptychography = ptychoX_fourier


def clear_cuda_plan_cache() -> None:
    cuda_ext = _CUDA_EXT
    if cuda_ext is not None and hasattr(cuda_ext, "clear_plan_cache"):
        cuda_ext.clear_plan_cache()


class PtychoXFPMFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        object_field: torch.Tensor,
        pupil: torch.Tensor,
        led_indices_xy: torch.Tensor,
        use_cuda_kernels: bool = False,
    ) -> torch.Tensor:
        check_complex_scan_inputs(object_field, pupil, led_indices_xy, "object_field", "pupil")
        led_indices_xy = led_indices_xy.to(device=object_field.device, dtype=torch.long).contiguous()
        object_field = object_field.contiguous()
        pupil = pupil.contiguous()

        cuda_ext = _load_cuda_ext() if can_use_cuda_ext(object_field, pupil, use_cuda_kernels) else None
        if cuda_ext is not None:
            sub_spectrum, image_field, amplitude = cuda_ext.forward(object_field, pupil, led_indices_xy)
            used_cuda_ext = True
        else:
            object_fft = torch.fft.fft2(object_field)
            sub_spectrum = _gather_subspectrum(object_fft, pupil.shape, led_indices_xy)
            image_field = torch.fft.ifft2(torch.fft.fftshift(sub_spectrum * pupil.unsqueeze(0), dim=(-2, -1)))
            amplitude = image_field.abs()
            used_cuda_ext = False

        ctx.save_for_backward(sub_spectrum, pupil, image_field, led_indices_xy)
        ctx.object_shape = tuple(object_field.shape)
        ctx.used_cuda_ext = used_cuda_ext
        return amplitude

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        sub_spectrum, pupil, image_field, led_indices_xy = ctx.saved_tensors
        object_h, object_w = ctx.object_shape
        grad_output = grad_output.to(dtype=image_field.real.dtype).contiguous()

        if ctx.used_cuda_ext:
            cuda_ext = _load_cuda_ext()
            grad_object, grad_pupil = cuda_ext.backward(
                grad_output,
                sub_spectrum,
                pupil,
                image_field,
                led_indices_xy,
                object_h,
                object_w,
            )
        else:
            low_h, low_w = pupil.shape
            phase = image_field / image_field.abs().clamp_min(torch.finfo(image_field.real.dtype).eps)
            grad_image = grad_output * phase
            grad_low_shifted = torch.fft.fft2(grad_image) / (low_h * low_w)
            grad_low = torch.fft.ifftshift(grad_low_shifted, dim=(-2, -1))

            grad_pupil = torch.sum(grad_low * sub_spectrum.conj(), dim=0)
            flat_indices = _fpm_flat_indices((object_h, object_w), (low_h, low_w), led_indices_xy)
            grad_values = (grad_low * pupil.conj().unsqueeze(0)).reshape(-1)
            grad_object_fft_flat = torch.zeros(object_h * object_w, dtype=pupil.dtype, device=pupil.device)
            grad_object_fft_flat.index_add_(0, flat_indices.reshape(-1), grad_values)
            grad_object_fft = grad_object_fft_flat.reshape(object_h, object_w)
            grad_object = torch.fft.ifft2(grad_object_fft) * (object_h * object_w)

        return grad_object, grad_pupil, None, None


def _gather_subspectrum(
    object_fft: torch.Tensor,
    low_shape: torch.Size | tuple[int, int],
    led_indices_xy: torch.Tensor,
) -> torch.Tensor:
    object_h, object_w = object_fft.shape
    low_h, low_w = int(low_shape[0]), int(low_shape[1])
    flat_indices = _fpm_flat_indices((object_h, object_w), (low_h, low_w), led_indices_xy)
    return object_fft.reshape(-1).gather(0, flat_indices.reshape(-1)).reshape(led_indices_xy.shape[0], low_h, low_w)


def _fpm_flat_indices(
    object_shape: tuple[int, int],
    low_shape: tuple[int, int],
    led_indices_xy: torch.Tensor,
) -> torch.Tensor:
    object_h, object_w = object_shape
    low_h, low_w = low_shape
    rows = torch.arange(low_h, device=led_indices_xy.device, dtype=torch.long)
    cols = torch.arange(low_w, device=led_indices_xy.device, dtype=torch.long)
    yy = (led_indices_xy[:, 1, None, None] + rows[None, :, None] + object_h // 2) % object_h
    xx = (led_indices_xy[:, 0, None, None] + cols[None, None, :] + object_w // 2) % object_w
    return yy * object_w + xx


def _load_cuda_ext():
    global _CUDA_EXT
    if _CUDA_EXT is not None:
        return _CUDA_EXT

    root = Path(__file__).resolve().parents[1]
    _CUDA_EXT = load_cuda_extension(
        module_name="ptychoX.fpm._C",
        build_name="ptychoX_fpm_C",
        sources=[
            root / "csrc" / "fpm" / "extension.cpp",
            root / "csrc" / "fpm" / "kernel.cu",
        ],
    )
    return _CUDA_EXT
