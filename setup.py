# Description: Setup file
#
# Installation of package: python -m pip install .
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import sys
from pathlib import Path

import numpy as np
from Cython.Distutils import build_ext
from setuptools import Extension, setup

# -----------------------------------------------------------------------------
# Native dependency discovery
# -----------------------------------------------------------------------------


def _candidate_prefixes(*env_vars):
    prefixes = []
    for name in env_vars:
        value = os.environ.get(name)
        if value:
            prefixes.append(Path(value))
    for name in ("HORAYZON_NATIVE_DIR", "CONDA_PREFIX"):
        value = os.environ.get(name)
        if value:
            prefixes.append(Path(value))
    prefixes.extend(
        [
            Path(sys.prefix),
            Path("/opt/homebrew"),
            Path("/opt/local"),
            Path("/usr/local"),
            Path("/usr"),
        ]
    )
    seen = set()
    for prefix in prefixes:
        prefix = prefix.expanduser().resolve()
        if prefix not in seen:
            seen.add(prefix)
            yield prefix


def _find_native_dependency(name, env_vars, header, library_names):
    for prefix in _candidate_prefixes(*env_vars):
        include_dir = prefix / "include"
        lib_dirs = [prefix / "lib", prefix / "lib64", prefix / "bin"]
        if not (include_dir / header).exists():
            continue
        for lib_dir in lib_dirs:
            if any(
                (lib_dir / lib_name).exists() for lib_name in library_names
            ):
                return str(include_dir), str(lib_dir)
    env_hint = " or ".join(env_vars)
    raise RuntimeError(
        f"{name} was not found. Set {env_hint} to its installation prefix "
        f"(the directory containing include/ and lib/)."
    )


def _native_library_names(base_name):
    if sys.platform == "win32":
        return (f"{base_name}.lib", f"{base_name}.dll")
    if sys.platform == "darwin":
        return (f"lib{base_name}.dylib",)
    return (f"lib{base_name}.so",)


embree_include, embree_lib = _find_native_dependency(
    "Embree",
    ("HORAYZON_EMBREE_DIR", "EMBREE_DIR"),
    "embree4/rtcore.h",
    _native_library_names("embree4"),
)
tbb_include, tbb_lib = _find_native_dependency(
    "TBB",
    ("HORAYZON_TBB_DIR", "TBB_DIR"),
    "tbb/parallel_for.h",
    _native_library_names("tbb") + _native_library_names("tbb12"),
)

# -----------------------------------------------------------------------------
# Operating system dependent settings
# -----------------------------------------------------------------------------

if sys.platform in ["linux", "linux2"]:
    print("Operating system: Linux")
    compiler = "gcc"
    extra_compile_args_cython = ["-O3", "-ffast-math"]
    extra_compile_args_cpp = ["-O3"]
    libraries_cython = ["m", "pthread"]
    libraries_cpp = ["embree4", "tbb"]
    runtime_library_dirs = [embree_lib, tbb_lib]
elif sys.platform in ["darwin"]:
    print("Operating system: Mac OS X")
    compiler = "clang"
    extra_compile_args_cython = [
        "-O3",
        "-ffast-math",
        "-Wno-nan-infinity-disabled",
    ]
    extra_compile_args_cpp = ["-O3", "-std=c++11"]
    libraries_cython = ["m", "pthread"]
    libraries_cpp = ["embree4", "tbb"]
    runtime_library_dirs = [embree_lib, tbb_lib]
elif sys.platform in ["win32"]:
    print("Operating system: Windows")
    compiler = None
    extra_compile_args_cython = ["/O2"]
    extra_compile_args_cpp = ["/O2", "/std:c++14"]
    libraries_cython = []
    libraries_cpp = ["embree4", "tbb12"]
    runtime_library_dirs = []
else:
    raise ValueError("Unsupported operating system")

include_dirs_cpp = [np.get_include(), embree_include, tbb_include]
library_dirs_cpp = [embree_lib, tbb_lib]
numpy_define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

masking_requires = ["fiona", "scikit-image", "shapely"]
examples_requires = [
    "ipython",
    "matplotlib",
    "netCDF4",
    "pillow",
    "pyproj",
    "rasterio",
    "skyfield",
    "trimesh",
]

# -----------------------------------------------------------------------------
# Compile Cython/C++ code
# -----------------------------------------------------------------------------

if compiler is not None:
    os.environ["CC"] = compiler

ext_modules = [
    Extension(
        "horayzon.transform",
        ["horayzon/transform.pyx"],
        libraries=libraries_cython,
        extra_compile_args=extra_compile_args_cython,
        define_macros=numpy_define_macros,
        include_dirs=[np.get_include()],
    ),
    Extension(
        "horayzon.direction",
        ["horayzon/direction.pyx"],
        libraries=libraries_cython,
        extra_compile_args=extra_compile_args_cython,
        define_macros=numpy_define_macros,
        include_dirs=[np.get_include()],
    ),
    Extension(
        "horayzon.topo_param",
        ["horayzon/topo_param.pyx"],
        libraries=libraries_cython,
        extra_compile_args=extra_compile_args_cython,
        define_macros=numpy_define_macros,
        include_dirs=[np.get_include()],
    ),
    Extension(
        "horayzon.horizon",
        sources=["horayzon/horizon.pyx", "horayzon/horizon_comp.cpp"],
        include_dirs=include_dirs_cpp,
        library_dirs=library_dirs_cpp,
        libraries=libraries_cpp,
        runtime_library_dirs=runtime_library_dirs,
        extra_compile_args=extra_compile_args_cpp,
        define_macros=numpy_define_macros,
        language="c++",
    ),
    Extension(
        "horayzon.shadow",
        sources=["horayzon/shadow.pyx", "horayzon/shadow_comp.cpp"],
        include_dirs=include_dirs_cpp,
        library_dirs=library_dirs_cpp,
        libraries=libraries_cpp,
        runtime_library_dirs=runtime_library_dirs,
        extra_compile_args=extra_compile_args_cpp,
        define_macros=numpy_define_macros,
        language="c++",
    ),
]

setup(
    name="horayzon",
    version="1.2",
    description=(
        "Efficient terrain horizon, sky view factor, and shadow computation"
    ),
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Christian R. Steger",
    maintainer="Christian R. Steger",
    url="https://github.com/ChristianSteger/HORAYZON",
    project_urls={
        "Source": "https://github.com/ChristianSteger/HORAYZON",
        "Issues": "https://github.com/ChristianSteger/HORAYZON/issues",
        "Publication": "https://doi.org/10.5194/gmd-15-6817-2022",
        "Zenodo": "https://doi.org/10.5281/zenodo.7013764",
    },
    license="MIT",
    license_files=["LICENSE", "horayzon/licenses/*.txt"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "terrain",
        "horizon",
        "shadow",
        "sky-view-factor",
        "ray-tracing",
        "embree",
    ],
    packages=["horayzon"],
    package_data={"horayzon": ["licenses/*.txt"]},
    python_requires=">=3.10",
    install_requires=[
        "geographiclib",
        "numpy",
        "pytest",
        "requests",
        "scikit-image",
        "scipy",
        "shapely",
        "tqdm",
        "xarray",
    ],
    extras_require={
        "all": examples_requires + masking_requires,
        "examples": examples_requires,
        "masking": masking_requires,
        "test": ["pytest"],
    },
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    zip_safe=False,
)
