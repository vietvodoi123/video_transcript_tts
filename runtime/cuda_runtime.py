import os
import sys
from pathlib import Path


_DLL_DIRECTORIES = []


def setup_cuda_runtime():
    """
    Configure NVIDIA CUDA runtime DLLs for Windows.

    The NVIDIA runtime is installed inside the Python virtual environment.
    """

    if os.name != "nt":
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_dir = site_packages / "nvidia"

    directories = [
        nvidia_dir / "cublas" / "bin",
        nvidia_dir / "cudnn" / "bin",
        nvidia_dir / "cuda_nvrtc" / "bin",
    ]

    valid_directories = []

    for directory in directories:
        if directory.exists():
            valid_directories.append(directory)

    if not valid_directories:
        return

    # ---------------------------------------------------------
    # 1. Add DLL directories for Python's Windows DLL loader
    # ---------------------------------------------------------

    for directory in valid_directories:
        handle = os.add_dll_directory(str(directory))
        _DLL_DIRECTORIES.append(handle)

        print(f"CUDA DLL directory added: {directory}")

    # ---------------------------------------------------------
    # 2. Add them to PATH
    #
    # CTranslate2 loads CUDA libraries from native C++ code.
    # On Windows, PATH is important for this lookup.
    # ---------------------------------------------------------

    current_path = os.environ.get("PATH", "")

    paths_to_add = [
        str(directory)
        for directory in valid_directories
        if str(directory) not in current_path
    ]

    if paths_to_add:
        os.environ["PATH"] = (
            os.pathsep.join(paths_to_add)
            + os.pathsep
            + current_path
        )

        print("CUDA DLL directories added to PATH.")