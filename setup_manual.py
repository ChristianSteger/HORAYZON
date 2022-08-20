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
from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np

# -----------------------------------------------------------------------------
# Manual settings
# -----------------------------------------------------------------------------

# Paths for Intel Embree, Threading Building Blocks (TBB) and NetCDF-4 C++
path_include = ["/opt/local/include/"]
path_lib = ["/opt/local/lib/libembree3",
            "/opt/local/lib/libnetcdf",
            "/opt/local/lib/libnetcdf_c++4"]  # without file ending

# Compiler
compiler = "clang++"  # (like gcc, clang, clang++)

# NetCDF 3/4 library (legacy; no adaptation required here)
lib_netcdf = ["libnetcdf", "libnetcdf_c++4"]  # NetCDF4
# lib_netcdf = ["libnetcdf_c++"]  # NetCDF3

# -----------------------------------------------------------------------------
# Operating system dependent settings
# -----------------------------------------------------------------------------

if sys.platform in ["linux", "linux2"]:
    print("Operating system: Linux")
    lib_end = ".so"
elif sys.platform in ["darwin"]:
    print("Operating system: Mac OS X")
    lib_end = ".dylib"
elif sys.platform in ["win32"]:
    print("Operating system: Windows")
    print("Warning: Package not yet tested for Windows")
else:
    raise ValueError("Unsupported operating system")
extra_compile_args_cython = ["-O3", "-ffast-math", "-fopenmp"]
libraries_cython = ["m", "pthread"]
include_dirs_cpp = [np.get_include()] + path_include
extra_objects_cpp = [i + lib_end for i in path_lib]

# -----------------------------------------------------------------------------
# Compile Cython/C++ code
# -----------------------------------------------------------------------------

os.environ["CC"] = compiler

ext_modules = [
    Extension("horayzon.transform",
              ["horayzon/transform.pyx"],
              libraries=libraries_cython,
              extra_compile_args=extra_compile_args_cython,
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension("horayzon.direction",
              ["horayzon/direction.pyx"],
              libraries=libraries_cython,
              extra_compile_args=extra_compile_args_cython,
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension("horayzon.topo_param",
              ["horayzon/topo_param.pyx"],
              libraries=libraries_cython,
              extra_compile_args=extra_compile_args_cython,
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension("horayzon.horizon",
              sources=["horayzon/horizon.pyx", "horayzon/horizon_comp.cpp"],
              include_dirs=include_dirs_cpp,
              extra_objects=extra_objects_cpp,
              extra_compile_args=["-O3"],
              language="c++"),
    Extension("horayzon.shadow",
              sources=["horayzon/shadow.pyx", "horayzon/shadow_comp.cpp"],
              include_dirs=include_dirs_cpp,
              extra_objects=extra_objects_cpp,
              extra_compile_args=["-O3"],
              language="c++")
    ]

setup(name="horayzon",
      version="1.2",
      packages=["horayzon"],
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules)
