import ctypes
import os
import platform
from pathlib import Path

# Check for environment variable PPAFM_RECOMPILE to determine whether
# we should recompile the C++ extensions.
if "PPAFM_RECOMPILE" in os.environ and os.environ["PPAFM_RECOMPILE"] != "":
    _recompile = True
else:
    _recompile = False

# Shared libraries are called .dll on Windows and .so on Linux by convention
system = platform.system()
if system == "Windows":
    _lib_ext = "_lib.dll"
else:
    _lib_ext = "_lib.so"

_vars_path = None

PACKAGE_PATH = Path(__file__).resolve().parent
CPP_PATH = PACKAGE_PATH.joinpath("cpp")

print(" PACKAGE_PATH = ", PACKAGE_PATH)
print(" CPP_PATH     = ", CPP_PATH)

cpp_modules = {"PP": "ProbeParticle", "GU": "GridUtils", "fitting": "fitting", "fitSpline": "fitSpline"}
"""Dictionary of C++ extension modules. Keys are targets for make and values are module names."""


def get_cdll(module):
    """
    Get a handle to a C++ extension module.

    Arguments:
        module: str. Module to load. Should be one listed in :data:`cpp_modules`.

    Returns:
        cdll: ctypes.CDLL. Loaded module handle.
    """
    if module not in cpp_modules:
        raise ValueError(f"Unrecognized module `{module}`. Should be one of {list(cpp_modules.keys())}")
    module_path = CPP_PATH / (cpp_modules[module] + _lib_ext)
    if _recompile:
        _make(module)
    elif not module_path.exists():
        raise RuntimeError(
            f"Could not find compiled extension module in `{module_path}`. "
            "Either check pip installation or enable dynamic compilation by setting "
            "the environment variable PPAFM_RECOMPILE=1"
        )
    return ctypes.CDLL(str(module_path))  # Changing to str is required on Windows


def _make(module):
    current_directory = Path.cwd()
    os.chdir(CPP_PATH)
    if system == "Windows":
        _build_windows(module)
    else:
        os.system("make clean")
        os.system("make " + module)
    os.chdir(current_directory)


def _get_vars_path():
    global _vars_path
    if _vars_path is None:
        vs_path = Path("C:/") / "Program Files (x86)" / "Microsoft Visual Studio"
        if not vs_path.exists():
            raise RuntimeError("Could not detect Microsoft Visual Studio installation")
        vars_paths = list(vs_path.glob("**/VsDevCmd.bat"))
        if len(vars_paths) == 0:
            raise RuntimeError("Could not find VsDevCmd.bat")
        _vars_path = vars_paths[0]
    return _vars_path


def _build_windows(target):
    vars_path = _get_vars_path()
    if target == "all":
        targets = cpp_modules.values()
    else:
        targets = [cpp_modules[target]]
    for module in targets:
        cmd = f'"{vars_path}" /no_logo /arch=amd64 && cl.exe /openmp /O2 /D_USRDLL /D_WINDLL {module}.cpp /link /dll /out:{module}{_lib_ext}'
        print(cmd)
        os.system(cmd)
