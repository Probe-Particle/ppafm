import os
import platform
from pathlib import Path

from setuptools import setup
from setuptools.command.build import build


class Build(build):
    '''Custom build for setuptools to compile C++ shared libraries.'''

    cpp_modules = ['ProbeParticle', 'GridUtils', 'fitSpline', 'fitting']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpp_path = Path(__file__).resolve().parent / 'ppafm' / 'cpp'
        self.current_directory = Path.cwd()

    def run(self):
        if platform.system() == 'Windows':
            self.build_windows()
        else:
            self.make()
        super().run()

    def make(self):
        os.chdir(self.cpp_path)
        os.system("make clean")
        os.system("make all")
        os.chdir(self.current_directory)

    def build_windows(self):
        os.chdir(self.cpp_path)
        # vars_path = self.get_vars_path()
        for module in self.cpp_modules:
            # cmd = f'"{vars_path}" /no_logo /arch=amd64 && cl.exe /O2 /D_USRDLL /D_WINDLL {module}.cpp /link /dll /out:{module}_lib.dll'
            cmd = f'cl.exe /O2 /D_USRDLL /D_WINDLL {module}.cpp /link /dll /out:{module}_lib.dll'
            print(cmd)
            os.system(cmd)
        os.chdir(self.current_directory)

    def get_vars_path(self):
        vs_path = Path('C:/') / 'Program Files (x86)' / 'Microsoft Visual Studio'
        if not vs_path.exists():
            raise RuntimeError('Could not detect Microsoft Visual Studio installation')
        vars_path = list(vs_path.glob('**/VsDevCmd.bat'))
        if len(vars_path) == 0:
            raise RuntimeError('Could not find VsDevCmd.bat')
        return vars_path[0]

setup(
    cmdclass={'build': Build},
    has_ext_modules=lambda: True
)
