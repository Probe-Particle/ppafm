
import os
import glob
import platform
from setuptools import Extension, setup
from setuptools.command.build import build

cpp_modules = ['ProbeParticle', 'GridUtils', 'fitSpline', 'fitting']

def make():
    package_path = os.path.dirname(os.path.realpath(__file__))
    cpp_path = os.path.normpath(package_path + '/ppafm/cpp/')
    current_directory = os.getcwd()
    os.chdir (cpp_path)
    os.system("make clean")
    os.system("make all")
    os.chdir (current_directory)

def get_vars_path():
    if not os.path.exists('C:\Program Files (x86)\Microsoft Visual Studio'):
        raise RuntimeError('Could not detect Microsoft Visual Studio installation')
    vars_path = glob.glob('C:/Program Files (x86)\Microsoft Visual Studio/*/BuildTools/Common7/Tools/VsDevCmd.bat')
    if len(vars_path) == 0:
        raise RuntimeError('Could not find VsDevCmd.bat')
    return vars_path[0]

def build_windows():
    package_path = os.path.dirname(os.path.realpath(__file__))
    cpp_path = os.path.normpath(package_path + '/ppafm/cpp/')
    current_directory = os.getcwd()
    os.chdir(cpp_path)
    vars_path = get_vars_path()
    for module in cpp_modules:
        cmd = f'"{vars_path}" /no_logo /arch=amd64 && cl.exe /O2 /D_USRDLL /D_WINDLL {module}.cpp /link /dll /out:{module}_lib.dll'
        print(cmd)
        os.system(cmd)
    os.chdir (current_directory)

class Build(build):
    '''Custom build for setuptools to compile C shared libraries.'''
    def run(self):
        if platform.system() == 'Windows':
            build_windows()
        else:
            make()
        super().run()

setup(
    cmdclass={'build': Build},
    has_ext_modules=lambda: True
)
