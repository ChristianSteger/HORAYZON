# Description: Setup file for 'manual' installation (this method works for
#              instance with 'pip')
#
# Installation of package: mv setup_manual.py setup.py
#                          python -m pip install .
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
# Manual settings
# -----------------------------------------------------------------------------

# Paths for Intel Embree and Threading Building Blocks (TBB)
path_include = ["/opt/local/include/"]
path_lib = ["/opt/local/lib/libembree3"]  # without file ending
# - depending on defined library paths and loaded modules, it might be
#   necessary to add paths to further libraries like 'libimf' and 'libtbb'
# - in case a library is not found during execution of HORAYZON, it has to be
#   defined before running Python/HORAYZON via 'LD_LIBRARY_PATH'.

# Compiler
compiler = "clang++"  # (like gcc, clang, clang++, default)
# "default": compiler is not redefined

# -----------------------------------------------------------------------------
# Operating system dependent settings
# -----------------------------------------------------------------------------

if sys.platform in ["linux", "linux2"]:
    print("Operating system: Linux")
    lib_end = ".so"
    extra_compile_args_cpp = ["-O3"]
elif sys.platform in ["darwin"]:
    print("Operating system: Mac OS X")
    lib_end = ".dylib"
    extra_compile_args_cpp = ["-O3", "-std=c++11"]
elif sys.platform in ["win32"]:
    print("Operating system: Windows")
    print("Warning: Package not yet tested for Windows")
else:
    raise ValueError("Unsupported operating system")
extra_compile_args_cython = ["-O3", "-ffast-math"]
libraries_cython = ["m", "pthread"]
include_dirs_cpp = [np.get_include()] + path_include
extra_objects_cpp = [i + lib_end for i in path_lib]

# -----------------------------------------------------------------------------
# Compile Cython/C++ code
# -----------------------------------------------------------------------------

if compiler != "default":
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
