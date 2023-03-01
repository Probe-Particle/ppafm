import os
import glob
import platform
from pathlib import Path

# Check for environment variable PPAFM_RECOMPILE to determine whether
# we should recompile the C++ extensions.
if 'PPAFM_RECOMPILE' in os.environ and os.environ['PPAFM_RECOMPILE'] != '':
    recompile = True
else:
    recompile = False

# Shared libraries are called .dll on Windows and .so on Linux by convention
system = platform.system()
if system == 'Windows':
    lib_ext = '_lib.dll'
else:
    lib_ext = '_lib.so'

cpp_modules = {
    'PP': 'ProbeParticle',
    'GU': 'GridUtils',
    'fitting_': 'fitting',
    'fitSpline_': 'fitSpline'
}

vars_path = None

PACKAGE_PATH = Path(__file__).resolve().parent
CPP_PATH     = PACKAGE_PATH / 'ppafm' / 'cpp'

print(" PACKAGE_PATH = ", PACKAGE_PATH)
print(" CPP_PATH     = ", CPP_PATH)

def make( what="" ):
    if recompile:
        current_directory = Path.cwd()
        os.chdir(CPP_PATH)
        if system == 'Windows':
            build_windows(what)
        else:
            os.system("make clean")
            os.system("make " + what)
        os.chdir(current_directory)

def get_vars_path():
    global vars_path
    if vars_path is None:
        vs_path = Path('C:/') / 'Program Files (x86)' / 'Microsoft Visual Studio'
        if not vs_path.exists():
            raise RuntimeError('Could not detect Microsoft Visual Studio installation')
        vars_paths = list(vs_path.glob('**/VsDevCmd.bat'))
        if len(vars_paths) == 0:
            raise RuntimeError('Could not find VsDevCmd.bat')
        vars_path = vars_paths[0]
    return vars_path

def build_windows(target):
    vars_path = get_vars_path()
    if target == 'all':
        targets = cpp_modules.values()
    else:
        targets = [cpp_modules[target]]
    for module in targets:
        cmd = f'"{vars_path}" /no_logo /arch=amd64 && cl.exe /O2 /D_USRDLL /D_WINDLL {module}.cpp /link /dll /out:{module}{lib_ext}'
        print(cmd)
        os.system(cmd)
