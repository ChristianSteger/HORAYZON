# Description: Setup file for Python functions
#
# Building: python setup_python.py
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import importlib
import py_compile

###############################################################################
# Check availability of external packages
###############################################################################

# List of packages
packages = ("numpy", "scipy", "geographiclib", "pyproj", "osgeo",
            "shapely", "fiona", "pygeos", "skimage")

# Check if available
for i in packages:
    if importlib.util.find_spec(i) is None:
        raise ImportError("Package " + i + " not installed")
print("All required Python packages are available.")

###############################################################################
# Compile functions
###############################################################################

py_compile.compile("src/auxiliary.py", cfile="lib/auxiliary.pyc")
py_compile.compile("src/geoid.py", cfile="lib/geoid.pyc")
py_compile.compile("src/load_dem.py", cfile="lib/load_dem.pyc")
py_compile.compile("src/ocean_masking.py", cfile="lib/ocean_masking.pyc")
print("Python function compiled.")
