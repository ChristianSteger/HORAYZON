# HORAYZON

Package to efficiently compute terrain parameters (like **horizon**, **sky view factor**, **topographic openness**, slope angle/aspect) from high-resolution digital elevation model (DEM) data. The package also allows to compute **shadow maps** (and **correction factors for downwelling direct shortwave radiation**) for specific sun positions. Horizon computation is based on the high-performance ray-tracing library Intel&copy; Embree. Calculations are parallelised with OpenMP (Cython code) or Threading Building Blocks (C++ code).

When you use HORAYZON, please cite:

**Steger, C. R., Steger, B. and Schär, C (2022): HORAYZON v1.0: An efficient and flexible ray-tracing algorithm to compute horizon and sky view factor, Geosci. Model Dev., https://doi.org/10.5194/gmd-2022-58**

# Package dependencies

HORAYZON depends on multiple external libraries and packages. The essential ones are listed below under **Core dependencies**. Further listed dependencies are only needed to run the examples. The examples **horizon/gridded_curved_DEM_masked.py**, **horizon/gridded_planar_DEM_2m.py** and **shadow/gridded_curved_DEM_NASADEM.py** require more complex dependencies, which are listed under **Specific dependencies for examples**. It is recommended to handle dependencies via [Conda](https://docs.conda.io/en/latest/#), which covers all dependencies except **hmm**. A new Conda environment can be created according to the below examples.

**Core dependencies**
- [Intel Embree](https://www.embree.org) and [Threading Building Blocks (TBB)](https://github.com/oneapi-src/oneTBB)
- [NetCDF-4 C++ library](https://github.com/Unidata/netcdf-cxx4)
- Python packages: Cython, NumPy, SciPy, GeographicLib, tqdm, requests, xarray
```bash
conda create -n horayzon_core -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray
```

**Base dependencies of examples**
- Python packages: netCDF4, Matplotlib, Pillow, Skyfield, pyproj, IPython
```bash
conda create -n horayzon_base -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray netcdf4 matplotlib pillow skyfield pyproj ipython
```

**Specific dependencies for examples (masking and high-resolution DEM examples; GDAL dependency)**
- Python packages: Shapely, fiona, PyGEOS, scikit-image, Rasterio, Trimesh
- [heightmap meshing utility (hmm)](https://github.com/fogleman/hmm)
```bash
conda create -n horayzon_all -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray netcdf4 matplotlib pillow skyfield pyproj ipython shapely fiona pygeos scikit-image rasterio trimesh
```
Installation instruction for **hmm** can be found [here](https://github.com/fogleman/hmm). **hmm**'s dependency **glm** can be installed via a package manager (e.g. APT, MacPorts, Homebrew) or via manual building from [source code](https://glm.g-truc.net/0.9.9/index.html).

# Installation

HORAYZON has been tested on **Python 3.10** under **Linux** and **Mac OS X**. Installation requires the [GNU Compiler Collection (GCC)](https://gcc.gnu.org) and can be accomplished as follows:

## Linux

Create an appropriate Conda environment (see examples above) and activate this environment. The HORAYZON package can then be installed with:
```bash
git clone https://github.com/ChristianSteger/HORAYZON.git
cd HORAYZON  
python -m pip install .
```

## Mac OS X
Create an appropriate Conda environment (see examples above) and activate this environment. Download the HORAYZON package and change to the main directory:
```bash
git clone https://github.com/ChristianSteger/HORAYZON.git
cd HORAYZON  
```
Currently, the default method of installing HORAYZON under Mac OS X fails because the NetCDF-4 C++ library provided by Conda was build with the C++ Standard Library **libc++**, which causes an incompatibility when the source code is compiled with GCC. It is therefore necessary to build and install the NetCDF-4 C++ library with GCC as well. This is for instance possible via MacPorts or Homebrew. The appropriate command for MacPorts is:

```bash
sudo port install netcdf-cxx4 +gcc10 
```
Subsequently, the path to the newly installed NetCDF-4 C++ library has to be set in HORAYZON's setup file (**setup.py**) and the package can be installed with:
```bash 
python -m pip install .
```

# Usage

The usage of the packages is best illustrated by means of examples, which can either be run in a Python IDE (like PyCharm or Spyder) or in the terminal. To run an examples, the path **path_out** must be adapted to a location that provides enough disk space. All input data (DEM or auxiliary data; see section below) required for running the examples is downloaded automatically.

## Examples: Terrain parameters (slope, horizon and sky view factor)

Two terrain horizon functions are available, **horizon_gridded()** and **horizon_locations()**. The former function allows to compute horizon for gridded input while the latter allows to compute horizon for arbitrary selected locations. The second function can optionally also output the distance to the horizon. The following five examples are provided:
- **examples/gridded_curved_DEM.py**: Compute topographic parameters (slope angle/aspect, horizon and sky view factor) from SRTM (geodetic coordinates, ~90 m resolution) for a ~50x50 km example region in the European Alps. Earth's surface curvature is considered. Plot output of this script is shown below.
![Alt text](https://github.com/ChristianSteger/Images/blob/master/Topo_slope_SVF_new.png?raw=true "Output from examples/gridded_curved_DEM.py")
- **examples/gridded_planar_DEM.py**: Compute topographic parameters (slope angle/aspect, horizon and sky view factor) from swisstopo DHM25 (map projection, 25 m resolution) for a ~25x40 km example region in Switzerland. Earth's surface curvature is neglected.
- **examples/locations_curved_DEM.py**: Compute topographic parameters (slope angle/aspect, horizon, distance to horizon and sky view factor) from SRTM (geodetic coordinates, ~90 m resolution) for 11 locations in Switzerland. Earth's surface curvature is considered. Plot output of this script for one location is shown below.
![Alt text](https://github.com/ChristianSteger/Images/blob/master/Wengen.png?raw=true "Output from examples/locations_curved_DEM.py")
- **examples/gridded_curved_DEM_masked.py**:
- **examples/gridded_planar_DEM_2m.py**:



gridded_curved_DEM_masked.py  DEM data with latitude/longitude coordinates (e.g. SRTM, NASADEM, MERIT) and a rectangular gridded domain (mask ocean grid cells based on distance to coast)
gridded_planar_DEM_2m.py	  DEM data with x/y coordinates (swissALTI3D); terrain simplification of outer domain
- **examples/gridded_NASADEM_HIMI.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from NASADEM (~30 m) for a ~100x100 km region centred at Heard Island and McDonald Islands. DEM grid cells, which are at least 20 km apart from land, are masked to speed-up the horizon computation.
- **examples/gridded_SwissALTI3D_Alps.py**: Compute topographic parameters (slope angle and aspect, horizon and sky view factor) from swissALTI3D (~2 m) for an a 3x3 km region in the European Alps. The outer DEM domain is simplified and represented by a triangulated irregular network (TIN) to reduce the large memory footprint of the DEM data.

Sky view factor and related parameters
The term sky view factor (SVF) is defined ambiguously in literature. In Zakšek et al. (2011), it referes to the solid angle of the (celestial) hemisphere. We call this parameter *visible sky fraction* and its computation is performed with the function **functions_cy.visskyfrac()**. In applications related to radiation, the SVF is typically defined as the fraction of sky radiation received at a certain location in case of isotropic sky radiation (see e.g. Helbig et al., 2009). This parameter is called *sky view factor* in our application and its computation is performed with the function **functions_cy.skyviewfactor()**. Additionally, the positive topographic openness (Yokoyama et al., 2002) can be computed with the function **functions_cy.topoopen()**. 


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