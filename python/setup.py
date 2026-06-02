from pathlib import Path
import os

# ===============================
# Force MSVC toolset version
# ===============================
FORCED_MSVC_ROOT = Path(
    r"C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.44.35207"
)

FORCED_WINSDK_VERSION = "10.0.26100.0"
FORCED_WINSDK_ROOT = Path(r"C:\Program Files (x86)\Windows Kits\10")


def prepend_env_path(name: str, value: Path) -> None:
    value_str = str(value)
    old = os.environ.get(name, "")
    parts = [p for p in old.split(os.pathsep) if p]
    parts = [p for p in parts if value_str.lower() != p.lower()]
    os.environ[name] = value_str + (os.pathsep + os.pathsep.join(parts) if parts else "")


def remove_env_contains(name: str, keywords: list[str]) -> None:
    old = os.environ.get(name, "")
    if not old:
        return
    parts = [p for p in old.split(os.pathsep) if p]
    filtered = []
    for p in parts:
        pl = p.lower()
        if any(k.lower() in pl for k in keywords):
            continue
        filtered.append(p)
    os.environ[name] = os.pathsep.join(filtered)


def force_msvc_1444() -> None:
    # 避免 PyTorch / setuptools 从用户 Python、LibTorch 或 VS18 新工具链里混入路径
    bad_keywords = [
        r"MSVC\14.51.36231",
        r"AppData\Local\Programs\Python\Python313",
        r"AppData\Roaming\Python\Python313",
        r"C:\Windows\LibTorch",
    ]

    for var in ["PATH", "INCLUDE", "LIB", "LIBPATH"]:
        remove_env_contains(var, bad_keywords)

    # 指定 cl.exe
    prepend_env_path("PATH", FORCED_MSVC_ROOT / "bin" / "Hostx64" / "x64")

    # 指定 MSVC include/lib
    os.environ["INCLUDE"] = os.pathsep.join([
        str(FORCED_MSVC_ROOT / "include"),
        str(FORCED_WINSDK_ROOT / "Include" / FORCED_WINSDK_VERSION / "ucrt"),
        str(FORCED_WINSDK_ROOT / "Include" / FORCED_WINSDK_VERSION / "um"),
        str(FORCED_WINSDK_ROOT / "Include" / FORCED_WINSDK_VERSION / "shared"),
    ])

    os.environ["LIB"] = os.pathsep.join([
        str(FORCED_MSVC_ROOT / "lib" / "x64"),
        str(FORCED_WINSDK_ROOT / "Lib" / FORCED_WINSDK_VERSION / "ucrt" / "x64"),
        str(FORCED_WINSDK_ROOT / "Lib" / FORCED_WINSDK_VERSION / "um" / "x64"),
    ])

    # 让 distutils 尽量使用当前环境，而不是重新自动探测 VS
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"

    os.environ["VCToolsInstallDir"] = str(FORCED_MSVC_ROOT) + "\\"
    os.environ["VCINSTALLDIR"] = str(FORCED_MSVC_ROOT.parent.parent) + "\\"
    os.environ["PYTHONNOUSERSITE"] = "1"


force_msvc_1444()

prepend_env_path(
    "PATH",
    Path(r"C:\Users\Administrator\AppData\Roaming\Python\Python313\Scripts")
)

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
            str(ROOT / "ptychoX" / "csrc" / f"{binding}_extension.cpp"),
            str(ROOT / "ptychoX" / "csrc" / f"{binding}_kernel.cu"),
        ],
        libraries=["cufft"],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-std=c++17",
                "--use_fast_math",
                "-allow-unsupported-compiler",
            ],
        },
    )


setup(
    name="ptychoX",
    package_dir={"": str(ROOT)},
    packages=["ptychoX", "ptychoX.base", "ptychoX.fpm", "ptychoX.coded"],
    package_data={"ptychoX": ["csrc/*.cpp", "csrc/*.cu"]},
    ext_modules=[
        cuda_extension("ptychoX.base._C", "base"),
        cuda_extension("ptychoX.fpm._C", "fpm"),
        cuda_extension("ptychoX.coded._C", "coded"),
    ],
    cmdclass={"build_ext": BuildExtension},
)
