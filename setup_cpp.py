# Description: Setup file for c++ horizon algorithm
#
# Building: python setup_cpp.py build_ext --build-lib lib/

# Load modules
import os
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

os.environ["CC"] = "gcc"

setup(ext_modules=cythonize(Extension(
           "horizon",
           sources=["src/horizon.pyx", "src/horizon_comp.cpp"],
           include_dirs=[numpy.get_include(), "/opt/local/include"],
           extra_objects=["/opt/local/lib/libembree3.dylib",
                          "/opt/local/lib/libnetcdf.dylib",
                          "/opt/local/lib/libnetcdf_c++4.dylib"],  # NetCDF4
           # extra_objects=["/opt/local/lib/libembree3.dylib",
           #                "/opt/local/lib/libnetcdf_c++.dylib"],  # NetCDF3
           extra_compile_args=["-O3"],
           language="c++",
      )))
