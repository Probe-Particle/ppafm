
import os
from setuptools import Extension, setup
# from setuptools.command.build import build

# def make():
#     package_path = os.path.dirname(os.path.realpath(__file__))
#     cpp_path = os.path.normpath(package_path + '/ppafm/cpp/')
#     current_directory = os.getcwd()
#     os.chdir (cpp_path)
#     os.system("make clean")
#     os.system("make all")
#     os.chdir (current_directory)

# class Build(build):
#     '''Custom build for setuptools to compile C shared libraries.'''
#     def run(self):
#         make()
#         super().run()

# setup(
#     cmdclass={'build': Build}
# )

from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def get_ext_filename(self, ext_name):
        return ext_name.replace('.', '/') + '_lib.so'

cpp_modules = ['ProbeParticle', 'GridUtils', 'fitSpline', 'fitting']

setup(
    ext_modules=[
        Extension(
            name=f'ppafm.cpp.{module}',
            sources=[f'ppafm/cpp/{module}.cpp']
        )
        for module in cpp_modules
    ],
    cmdclass={'build_ext': BuildExt},
)
