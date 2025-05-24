# Description: Setup file
#
# Installation of package: python -m pip install .
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import sys
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

# -----------------------------------------------------------------------------
# Operating system dependent settings
# -----------------------------------------------------------------------------

path_lib_conda = os.environ["CONDA_PREFIX"] + "/lib/"
if sys.platform in ["linux", "linux2"]:
    print("Operating system: Linux")
    lib_end = ".so"
    compiler = "gcc"
    extra_compile_args_cython = ["-O3", "-ffast-math"]
    extra_compile_args_cpp = ["-O3"]
elif sys.platform in ["darwin"]:
    print("Operating system: Mac OS X")
    lib_end = ".dylib"
    compiler = "clang"
    extra_compile_args_cython = ["-O3", "-ffast-math",
                                 "-Wl,-rpath," + path_lib_conda,
                                 "-L" + path_lib_conda]
    extra_compile_args_cpp = ["-O3", "-std=c++11"]
elif sys.platform in ["win32"]:
    print("Operating system: Windows")
    print("Warning: Package not yet tested for Windows")
else:
    raise ValueError("Unsupported operating system")
libraries_cython = ["m", "pthread"]
include_dirs_cpp = [np.get_include()]
extra_objects_cpp = [path_lib_conda + i + lib_end for i in ["libembree3"]]

# -----------------------------------------------------------------------------
# Compile Cython/C++ code
# -----------------------------------------------------------------------------

os.environ["CC"] = compiler

ext_modules = [
    Extension("horayzon.transform",
              ["horayzon/transform.pyx"],
              libraries=libraries_cython,
              extra_compile_args=extra_compile_args_cython,
              include_dirs=[np.get_include()]),
    Extension("horayzon.direction",
              ["horayzon/direction.pyx"],
              libraries=libraries_cython,
              extra_compile_args=extra_compile_args_cython,
              include_dirs=[np.get_include()]),
    Extension("horayzon.topo_param",
              ["horayzon/topo_param.pyx"],
              libraries=libraries_cython,
              extra_compile_args=extra_compile_args_cython,
              include_dirs=[np.get_include()]),
    Extension("horayzon.horizon",
              sources=["horayzon/horizon.pyx", "horayzon/horizon_comp.cpp"],
              include_dirs=include_dirs_cpp,
              extra_objects=extra_objects_cpp,
              extra_compile_args=extra_compile_args_cpp,
              language="c++"),
    Extension("horayzon.shadow",
              sources=["horayzon/shadow.pyx", "horayzon/shadow_comp.cpp"],
              include_dirs=include_dirs_cpp,
              extra_objects=extra_objects_cpp,
              extra_compile_args=extra_compile_args_cpp,
              language="c++")
    ]

setup(name="horayzon",
      version="1.2",
      packages=["horayzon"],
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules)
