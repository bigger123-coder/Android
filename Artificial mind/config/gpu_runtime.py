import os
from pathlib import Path


_NVIDIA_ROOT = Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages" / "nvidia"
_DLL_DIRECTORIES = (
    _NVIDIA_ROOT / "cu13" / "bin" / "x86_64",
    _NVIDIA_ROOT / "cu13" / "bin",
    _NVIDIA_ROOT / "cublas" / "bin",
    _NVIDIA_ROOT / "cudnn" / "bin",
    _NVIDIA_ROOT / "nvjitlink" / "bin",
)


_dll_handles = []
_existing_path = os.environ.get("PATH", "")
_path_directories = []
for _directory in _DLL_DIRECTORIES:
    if _directory.is_dir():
        _path_directories.append(str(_directory))
        _dll_handles.append(os.add_dll_directory(str(_directory)))

if _path_directories:
    os.environ["PATH"] = os.pathsep.join(_path_directories + [_existing_path])
