# Description: Setup file
#
# Installation of package: python -m pip install .
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

# Path to manually installed NetCDF4 C++ library (Mac OS X; compiled with gcc)
path_lib_netcdf = "/opt/local/lib/"

# NetCDF 3/4 library
lib_netcdf = ["libnetcdf", "libnetcdf_c++4"]  # NetCDF4
# lib_netcdf = ["libnetcdf_c++"]  # NetCDF3

# -----------------------------------------------------------------------------
# Operating system dependent settings
# -----------------------------------------------------------------------------

path_lib_conda = os.environ["CONDA_PREFIX"] + "/lib/"
if sys.platform in ["linux", "linux2"]:
    print("Operating system: Linux")
    lib_end = ".so"
    libraries_cython = ["m", "pthread"]
    include_dirs_cpp = [np.get_include()]
    extra_objects_cpp = [path_lib_conda + "libembree3" + lib_end] \
        + [path_lib_conda + i + lib_end for i in lib_netcdf]
elif sys.platform in ["darwin"]:
    print("Operating system: Mac OS X")
    lib_end = ".dylib"
    libraries_cython = ["m", "iomp5", "pthread"]
    include_dirs_cpp = [np.get_include()]
    extra_objects_cpp = [path_lib_conda + "libembree3" + lib_end] \
        + [path_lib_netcdf + i + lib_end for i in lib_netcdf]
    for i in extra_objects_cpp:
        if not os.path.isfile(i):
            raise ValueError("Library " + i + " not found")
elif sys.platform in ["win32"]:
    print("Operating system: Windows")
    print("Warning: Package not yet tested for Windows")
else:
    raise ValueError("Unsupported operating system")

# -----------------------------------------------------------------------------
# Compile Cython/C++ code
# -----------------------------------------------------------------------------

os.environ["CC"] = "gcc"

ext_modules = [
    Extension("horayzon.transform",
              ["horayzon/transform.pyx"],
              libraries=libraries_cython,
              extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension("horayzon.direction",
              ["horayzon/direction.pyx"],
              libraries=libraries_cython,
              extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension("horayzon.topo_param",
              ["horayzon/topo_param.pyx"],
              libraries=libraries_cython,
              extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
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
