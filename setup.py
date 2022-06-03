# Description: Setup file
#
# Building: python setup.py build_ext --build-lib horayzon/
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

lib_netcdf = ["/opt/local/lib/libnetcdf.dylib",
              "/opt/local/lib/libnetcdf_c++4.dylib"]  # NetCDF4
# lib_netcdf = ["/opt/local/lib/libnetcdf_c++.dylib"]  # NetCDF3

# -----------------------------------------------------------------------------

os.environ["CC"] = "gcc"

ext_modules = [
    Extension("transform",
              ["horayzon/transform.pyx"],
              libraries=["m", "iomp5", "pthread"],
              extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension("ecef",
              ["horayzon/ecef.pyx"],
              libraries=["m", "iomp5", "pthread"],
              extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension("topo_param",
              ["horayzon/topo_param.pyx"],
              libraries=["m", "iomp5", "pthread"],
              extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension(
               "horizon",
               sources=["horayzon/horizon.pyx", "horayzon/horizon_comp.cpp"],
               include_dirs=[np.get_include(), "/opt/local/include"],
               extra_objects=["/opt/local/lib/libembree3.dylib"] + lib_netcdf,
               extra_compile_args=["-O3"],
               language="c++",
          )
    ]

setup(name="test",
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules)
