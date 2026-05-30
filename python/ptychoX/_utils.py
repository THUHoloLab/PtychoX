from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

DEFAULT_CUDA_ARCH_LIST = "8.6"


def detect_cuda_arch_list() -> str:
    if torch.cuda.is_available():
        archs = {
            f"{major}.{minor}"
            for index in range(torch.cuda.device_count())
            for major, minor in [torch.cuda.get_device_capability(index)]
        }
        if archs:
            return ";".join(sorted(archs))
    return DEFAULT_CUDA_ARCH_LIST


def check_complex_scan_inputs(
    first: torch.Tensor,
    second: torch.Tensor,
    positions_xy: torch.Tensor,
    first_name: str = "sample",
    second_name: str = "probe",
) -> None:
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(f"{first_name} and {second_name} must be 2D complex tensors.")
    if not first.is_complex() or not second.is_complex():
        raise TypeError(f"{first_name} and {second_name} must be complex tensors.")
    if first.device != second.device:
        raise ValueError(f"{first_name} and {second_name} must live on the same device.")
    if first.dtype != second.dtype:
        raise ValueError(f"{first_name} and {second_name} must have the same dtype.")
    if positions_xy.ndim != 2 or positions_xy.shape[-1] != 2:
        raise ValueError("positions_xy must have shape [batch, 2].")
    if positions_xy.device.type != "cpu" and positions_xy.device != first.device:
        raise ValueError(f"positions_xy must be on CPU or on the same device as {first_name}.")


def can_use_cuda_ext(first: torch.Tensor, second: torch.Tensor, requested: bool) -> bool:
    return requested and first.is_cuda and first.dtype == torch.complex64 and second.dtype == torch.complex64


def configure_matching_cuda_home() -> None:
    torch_cuda = torch.version.cuda
    if not torch_cuda:
        return

    default_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    candidate = default_root / f"v{torch_cuda}"
    if candidate.exists():
        os.environ["CUDA_HOME"] = str(candidate)
        os.environ["CUDA_PATH"] = str(candidate)
        os.environ["PATH"] = str(candidate / "bin") + os.pathsep + os.environ.get("PATH", "")


def configure_cuda_arch_list() -> None:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", DEFAULT_CUDA_ARCH_LIST)


def load_cuda_extension(
    module_name: str,
    build_name: str,
    sources: list[Path],
):
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    configure_matching_cuda_home()
    configure_cuda_arch_list()
    from torch.utils import cpp_extension
    from torch.utils.cpp_extension import load

    if "CUDA_HOME" in os.environ:
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]

    project_root = Path(__file__).resolve().parents[1]
    build_dir = project_root / ".torch_extensions" / build_name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=module_name,
        sources=[str(source) for source in sources],
        build_directory=str(build_dir),
        extra_cuda_cflags=["--use_fast_math"],
        extra_ldflags=["cufft.lib"] if sys.platform == "win32" else ["-lcufft"],
        verbose=True,
    )


def matlab_positions_to_torch(positions_xy_1based: torch.Tensor) -> torch.Tensor:
    """Convert MATLAB one-based `(x, y)` scan positions to PyTorch zero-based."""
    return positions_xy_1based.to(torch.long) - 1
