# HORAYZON

# General Information
Package to compute terrain parameters horizon, sky view factor and slope angle/aspect from high-resolution elevation data. Horizon computation is based on the high-performance ray-tracing library Embree. The package is written in Python, Cython and C++.

# Dependencies

## Python packages

Source code:
- numpy
- cython
- scipy
- geographiclib
- pyproj
- gdal
- shapely
- fiona
- pygeos
- scikit-image

Application:
- xarray
- matplotlib
- netcdf4
- rasterio

Optional for remote terrain simplification:
- trimesh

## Further dependencies
- Intel&copy; Embree with Intel Threading Building Blocks (TBB) and GLFW. Source code and compilation instructions can be found here: https://github.com/embree/embree
- NetCDF4 C++. Source code and compilation instructions can be found here: https://github.com/Unidata/netcdf-cxx4
- hmm. Optional &ndash; only required if remote terrain simplification is needed in case of elevation data with very high (<5 m) resolution. Source code and compilation instructions can be found here: https://github.com/fogleman/hmm

# Installation
The source code can be compiled in the following way: First, the Cython functions have to be compiled by calling `python setup_cython.py build_ext --build-lib lib/` in the main directory. Subsequently, the correct paths to the NetCDF4-C++ and the Embree library have to be set in the file *setup_cpp.py*. The C++ code can then be compiled by calling `python setup_cpp.py build_ext --build-lib lib/`. Finally, the Python code can be compiled with `python setup_python.py`. All libraries are placed in the directory *lib* in the main folder.

# Required data

## Digital elevation model data

Digital elevation model data is available from various sources, e.g.:
- [NASADEM](https://search.earthdata.nasa.gov/)
- [SRTM](https://srtm.csi.cgiar.org)
- [MERIT](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/)
- [USGS 1/3rd arc-second DEM](https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5)
- [swissALTI3D](https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html)

## Auxiliary data

Auxiliary data, like geoid undulation data (EGM96 and GEOID12A) and coastline polygons (GSHHG) are available here:
- [EGM96](https://earth-info.nga.mil)
- [GEOID12A](https://geodesy.noaa.gov/GEOID/GEOID12A/GEOID12A_AK.shtml)
- [GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/)

# Usage

# Example output

# Reference
Link to Geoscientific Model Development [manuscript](https://www.geoscientific-model-development.net)

# Support 
In case of issues or questions, please contact Christian Steger (christian.steger@env.ethz.ch).