from pathlib import Path
import os

from setuptools import setup
import torch


ROOT = Path(__file__).resolve().parent
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


def configure_cuda_arch_list() -> None:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", detect_cuda_arch_list())


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


configure_matching_cuda_home()
configure_cuda_arch_list()

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def cuda_extension(name: str, binding: str) -> CUDAExtension:
    return CUDAExtension(
        name=name,
        sources=[
            str(ROOT / "ptychoX" / "csrc" / binding / "extension.cpp"),
            str(ROOT / "ptychoX" / "csrc" / binding / "kernel.cu"),
        ],
        libraries=["cufft"],
        extra_compile_args={
            "cxx": [],
            "nvcc": ["--use_fast_math"],
        },
    )


setup(
    name="ptychoX",
    package_dir={"": str(ROOT)},
    packages=["ptychoX", "ptychoX.base", "ptychoX.fpm", "ptychoX.coded"],
    package_data={"ptychoX": ["csrc/*/*.cpp", "csrc/*/*.cu"]},
    ext_modules=[
        cuda_extension("ptychoX.base._C", "base"),
        cuda_extension("ptychoX.fpm._C", "fpm"),
        cuda_extension("ptychoX.coded._C", "coded"),
    ],
    cmdclass={"build_ext": BuildExtension},
)
