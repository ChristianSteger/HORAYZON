# HORAYZON

# General Information
Package to efficiently compute terrain parameters **horizon**, **sky view factor** and slope angle/aspect from high-resolution elevation data. Horizon computation is based on the high-performance ray-tracing library Intel&copy; Embree. Calculations are parallelised with OpenMP (Cython code) or Threading Building Blocks (C++ code). A description of the algorithm is published in the journal of Geoscientific Model Development (Steger et al., 2022).

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
- cartopy
- cmcrameri
- netcdf4
- rasterio

Optional for remote terrain simplification:
- trimesh

## Further dependencies
- Intel&copy; Embree with Intel Threading Building Blocks (TBB) and GLFW. Source code and compilation instructions can be found here: https://github.com/embree/embree
- NetCDF4 C++ library. Source code and compilation instructions can be found here: https://github.com/Unidata/netcdf-cxx4
- Heightmap meshing utility (hmm). Optional &ndash; only required if remote terrain simplification should be applied in case of elevation data with very high (<5 m) resolution. Source code and compilation instructions can be found here: https://github.com/fogleman/hmm

# Installation
The source code can be compiled in the following way: First, the Cython functions have to be compiled by calling `python setup_cython.py build_ext --build-lib lib/` in the main directory. Subsequently, the correct paths to the NetCDF4-C++ and the Embree library have to be set in the file **setup_cpp.py**. The C++ code can then be compiled by calling `python setup_cpp.py build_ext --build-lib lib/`. Finally, the Python code can be compiled with `python setup_python.py`. All libraries are placed in the directory **lib** in the main folder.

# Input data

## Digital elevation model (DEM) data

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

## Examples

The usage of the packages is best illustrated by means of five examples. To successfully run them, the paths to the input data and the folder **lib**, which are defined at the beginning of the example files, must be adapted. Three terrain horizon functions are available, which cover different application cases. The function **horizon_gridded()** allows to computed gridded terrain horizon from DEM data and its application is illustrated in the following three examples:
- **examples/gridded_NASADEM_Alps.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for a ~30x30 km region in the European Alps. Output from this script is shown below.
- **examples/gridded_NASADEM_HIMI.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for a ~100x100 km region centred at Heard Island and McDonald Islands. DEM grid cells, which are at least 20 km apart from land, are masked to speed-up the horizon computation.
- **examples/gridded_SwissALTI3D_Alps.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from swissALTI3D (~2 m) for an a 3x3 km region in the European Alps. The outer DEM domain is simplified and represented by a triangulated irregular network (TIN) to reduce the large memory footprint of the DEM data.
The additional functions **horizon_gridcells()** and **horizon_locations()** are useful in case terrain horizon is only needed for a subset of locations within a geographical region. These two functions additional allow to output the distance to the horizon. The former function allows to compute horizon at the grid cell's centre, while the latter can be used to compute terrain horizon for arbitrary spatial locations.
- **examples/gridcells_NASADEM_Himalayas.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for four grid cells in the Himalayas (Mount Everest region). The horizon's elevation and distance is visualised in a plot.
- **examples/locations_NASADEM_Switzerland.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for 10 arbitrary locations in Switzerland. The horizon's elevation and distance is visualised in a plot.

![Alt text](https://github.com/ChristianSteger/Images/blob/master/Topo_slope_SVF.png?raw=true "Output from examples/NASADEM_Alps.py")

## Sky view factor and related parameters
The term sky view factor (SVF) is defined ambiguously in literature. In Zakšek et al. (2011), it referes to the solid angle of the (celestial) hemisphere. We call this parameter *visible sky fraction* and its computation is performed with the function **functions_cy.visskyfrac()**. In applications related to radiation, the SVF is typically defined as the fraction of sky radiation received at a certain location in case of isotropic sky radiation (see e.g. Helbig et al., 2009). This parameter is called *sky view factor* in our application and its computation is performed with the function **functions_cy.skyviewfactor()**. Additionally, the positive topographic openness (Yokoyama et al., 2002) can be computed with the function **functions_cy.topoopen()**. 

# References
- Steger, C. R., Steger, B. and Schär, C (2022): HORAYZON v1.0: An efficient and flexible ray-tracing algorithm to compute horizon and sky view factor, Geoscientific Model Development, submitted
- Helbig, N., Löwe, H. and Lehning, M. (2009): Radiosity Approach for the Shortwave Surface Radiation Balance in Complex Terrain, Journal of the Atmospheric Sciences, 66(9), 2900-2912, https://doi.org/10.1175/2009JAS2940.1
- Yokoyama, R., Shirasawa, M. and Pike, R. J. (2002): Visualizing Topography by Openness: A New Application of Image Processing to Digital Elevation Models, Photogrammetric Engineering and Remote Sensing, 68, 257-265.
- Zakšek, K., Oštir, K. and Kokalj, Ž. (2011): Sky-View Factor as a Relief Visualization Technique, Remote Sensing, 3(2):398-415, https://doi.org/10.3390/rs3020398

# Support 
In case of issues or questions, please contact Christian Steger (christian.steger@env.ethz.ch).