# HORAYZON

Package to efficiently compute terrain parameters (like **horizon**, **sky view factor**, **topographic openness**, slope angle/aspect) from high-resolution digital elevation model data. The package also allows to compute **shadow maps** (and **correction factors for downwelling direct shortwave radiation**) for specific sun positions. Horizon computation is based on the high-performance ray-tracing library Intel&copy; Embree. Calculations are parallelised with OpenMP (Cython code) or Threading Building Blocks (C++ code).

When you use HORAYZON, please cite:

**Steger, C. R., Steger, B. and Schär, C (2022): HORAYZON v1.0: An efficient and flexible ray-tracing algorithm to compute horizon and sky view factor, Geosci. Model Dev., https://doi.org/10.5194/gmd-2022-58**

# Package dependencies

HORAYZON depends on various external libraries and packages. The essential ones are listed below under **Core dependencies**. Further listed dependencies are only needed to run the examples. The examples **horizon/gridded_curved_DEM_masked.py**, **horizon/gridded_planar_DEM_2m.py** and **shadow/gridded_curved_DEM_NASADEM.py** require more complex dependencies, which are listed under **Specific dependencies for examples**. It is recommended to handle dependencies via [Conda](https://docs.conda.io/en/latest/#), which covers all dependencies except **hmm**. A new Conda environment for HORAYZON can be created according to the below examples.

**Core dependencies**
- [Intel Embree](https://www.embree.org) and [Threading Building Blocks (TBB)](https://github.com/oneapi-src/oneTBB)
- [NetCDF-4 C++ library](https://github.com/Unidata/netcdf-cxx4)
- Python packages: *cython numpy scipy geographiclib tqdm requests xarray*
```bash
conda create -n horayzon_core -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray
```

**Base dependencies of examples**
- Python packages: *netcdf4 matplotlib pillow skyfield pyproj ipython*
```bash
conda create -n horayzon_base -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray netcdf4 matplotlib pillow skyfield pyproj ipython
```

**Specific dependencies for examples (masking and high-resolution DEM examples; GDAL dependency)**
- Python packages: *shapely fiona pygeos scikit-image rasterio trimesh*
- [heightmap meshing utility (hmm)](https://github.com/fogleman/hmm)
```bash
conda create -n horayzon_all -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray netcdf4 matplotlib pillow skyfield pyproj ipython shapely fiona pygeos scikit-image rasterio trimesh
```
Installation instruction for **hmm** can be found [here](https://github.com/fogleman/hmm). **hmm**'s dependency **glm** can be installed via a package manager (e.g. APT, Homebrew, MacPorts) or via manual building from [source code](https://glm.g-truc.net/0.9.9/index.html).

# Installation

HORAYZON has been tested on **Python 3.10** under **Linux** and **Mac OS X**. Installation can be accomplished as follows:

## Linux
Placeholder

## MacOS
Placeholder

# Usage

The usage of the packages is best illustrated by means of examples, which can either be run in a Python IDE (like PyCharm or Spyder) or in the terminal. To run an examples, the path *path_out* must be adapted to a location that provides enough disk space. All input data (DEM or auxiliary data; see section below) required for the example is downloaded automatically.

## Examples: Terrain horizon and other terrain parameters

. To successfully run them, the paths to the input data and the folder **lib**, which are defined at the beginning of the example files, must be adapted. Three terrain horizon functions are available, which cover different application cases. The function **horizon_gridded()** allows to computed gridded terrain horizon from DEM data and its application is illustrated in the following three examples:
- **examples/gridded_NASADEM_Alps.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for a ~30x30 km region in the European Alps. Output from this script is shown below.
- **examples/gridded_NASADEM_HIMI.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for a ~100x100 km region centred at Heard Island and McDonald Islands. DEM grid cells, which are at least 20 km apart from land, are masked to speed-up the horizon computation.
- **examples/gridded_SwissALTI3D_Alps.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from swissALTI3D (~2 m) for an a 3x3 km region in the European Alps. The outer DEM domain is simplified and represented by a triangulated irregular network (TIN) to reduce the large memory footprint of the DEM data.

The additional functions **horizon_gridcells()** and **horizon_locations()** are useful in case terrain horizon is only needed for a subset of locations within a geographical region. These two functions additional allow to output the distance to the horizon. The former function allows to compute horizon only for the grid cells' centre, while the latter can be used to compute terrain horizon for arbitrary spatial locations.
- **examples/gridcells_NASADEM_Himalayas.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for four grid cells in the Himalayas (Mount Everest region). The horizon's elevation angle and distance is visualised in a plot.
- **examples/locations_NASADEM_Switzerland.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for 10 arbitrary locations in Switzerland. The horizon's elevation angle and distance is visualised in a plot.

Sky view factor and related parameters
The term sky view factor (SVF) is defined ambiguously in literature. In Zakšek et al. (2011), it referes to the solid angle of the (celestial) hemisphere. We call this parameter *visible sky fraction* and its computation is performed with the function **functions_cy.visskyfrac()**. In applications related to radiation, the SVF is typically defined as the fraction of sky radiation received at a certain location in case of isotropic sky radiation (see e.g. Helbig et al., 2009). This parameter is called *sky view factor* in our application and its computation is performed with the function **functions_cy.skyviewfactor()**. Additionally, the positive topographic openness (Yokoyama et al., 2002) can be computed with the function **functions_cy.topoopen()**. 


![Alt text](https://github.com/ChristianSteger/Images/blob/master/Topo_slope_SVF.png?raw=true "Output from examples/NASADEM_Alps.py")

## Examples: Shadow map and correction factor for downward direct shortwave radiation

# Digital elevation model and auxiliary data

Digital elevation model (DEM) data is available from various sources, e.g.:
- [NASADEM](https://search.earthdata.nasa.gov/)
- [SRTM](https://srtm.csi.cgiar.org)
- [MERIT](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/)
- [USGS 1/3rd arc-second DEM](https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5)
- [swissALTI3D](https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html)

Auxiliary data, like geoid undulation data (EGM96 and GEOID12A), coastline polygons (GSHHG) or glacier outlines (GAMDAM) are for instance available here:
- [EGM96](https://earth-info.nga.mil)
- [GEOID12A](https://geodesy.noaa.gov/GEOID/GEOID12A/GEOID12A_AK.shtml)
- [GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/)
- [GAMDAM](https://doi.pangaea.de/10.1594/PANGAEA.891423)

# References
- Helbig, N., Löwe, H. and Lehning, M. (2009): Radiosity Approach for the Shortwave Surface Radiation Balance in Complex Terrain, Journal of the Atmospheric Sciences, 66(9), 2900-2912, https://doi.org/10.1175/2009JAS2940.1
- Yokoyama, R., Shirasawa, M. and Pike, R. J. (2002): Visualizing Topography by Openness: A New Application of Image Processing to Digital Elevation Models, Photogrammetric Engineering and Remote Sensing, 68, 257-265.
- Zakšek, K., Oštir, K. and Kokalj, Ž. (2011): Sky-View Factor as a Relief Visualization Technique, Remote Sensing, 3(2):398-415, https://doi.org/10.3390/rs3020398
- Müller, M. D., and Scherer, D. (2005): A Grid- and Subgrid-Scale Radiation Parameterization of Topographic Effects for Mesoscale Weather Forecast Models, Monthly Weather Review, 133(6), 1431-1442, https://journals.ametsoc.org/view/journals/mwre/133/6/mwr2927.1.xml

# Support 
In case of issues or questions, please contact Christian Steger (christian.steger@env.ethz.ch).