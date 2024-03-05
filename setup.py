import os
import platform
from pathlib import Path

from setuptools import setup
from setuptools.command.build import build


class Build(build):
    """Custom build for setuptools to compile C++ shared libraries."""

    cpp_modules = ["ProbeParticle", "GridUtils", "fitSpline", "fitting"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpp_path = Path(__file__).resolve().parent / "ppafm" / "cpp"
        self.current_directory = Path.cwd()

    def run(self):
        os.chdir(self.cpp_path)
        if platform.system() == "Windows":
            self.build_windows()
        else:
            self.make()
        os.chdir(self.current_directory)
        super().run()

    def make(self):
        os.system("make clean")
        os.system("make all")

    def build_windows(self):
        for module in self.cpp_modules:
            cmd = f"cl.exe /openmp /O2 /D_USRDLL /D_WINDLL {module}.cpp /link /dll /out:{module}_lib.dll"
            print(cmd)
            os.system(cmd)


setup(cmdclass={"build": Build}, has_ext_modules=lambda: True)
