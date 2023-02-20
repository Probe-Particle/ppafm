
import os
from setuptools import Extension, setup
from setuptools.command.build import build

def make( what="" ):
    package_path = os.path.dirname(os.path.realpath(__file__))
    cpp_path = os.path.normpath(package_path + '/ppafm/cpp/')
    current_directory = os.getcwd()
    os.chdir (cpp_path)
    os.system("make clean")
    os.system("make all")
    os.chdir (current_directory)

class Build(build):
    '''Custom build for setuptools to compile C shared libraries.'''
    def run(self):
        make()
        super().run()

setup(
    cmdclass={'build': Build},
    has_ext_modules=lambda: True
)
