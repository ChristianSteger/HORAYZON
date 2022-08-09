# HORAYZON

Package to efficiently compute terrain parameters (like **horizon**, **sky view factor**, **topographic openness**, slope angle/aspect) from high-resolution digital elevation model (DEM) data. The package also allows to compute **shadow maps** and **correction factors for downwelling direct shortwave radiation** for specific sun positions. Horizon computation is based on the high-performance ray-tracing library Intel&copy; Embree. Calculations are parallelised with OpenMP (Cython code) or Threading Building Blocks (C++ code).

When you use HORAYZON, please cite:

**Steger, C. R., Steger, B. and Schär, C (2022): HORAYZON v1.2: An efficient and flexible ray-tracing algorithm to compute horizon and sky view factor, Geosci. Model Dev., https://doi.org/10.5194/gmd-2022-58**

# Package dependencies

HORAYZON depends on multiple external libraries and packages. The essential ones are listed below under **Core dependencies**. Further dependencies are needed to run the examples (**Base dependencies for examples**). The examples **horizon/gridded_curved_DEM_masked.py**, **horizon/gridded_planar_DEM_2m.py** and **shadow/gridded_curved_DEM_NASADEM.py** require more complex dependencies, which are listed under **All dependencies for examples**. It is recommended to install dependencies via [Conda](https://docs.conda.io/en/latest/#), which covers all dependencies except **hmm**.

**Core dependencies**
- [Intel Embree](https://www.embree.org) and [Threading Building Blocks (TBB)](https://github.com/oneapi-src/oneTBB)
- [NetCDF-4 C++ library](https://github.com/Unidata/netcdf-cxx4)
- Python packages: Cython, NumPy, SciPy, GeographicLib, tqdm, requests, xarray

**Base dependencies for examples**
- Python packages: netCDF4, Matplotlib, Pillow, Skyfield, pyproj, IPython

**All dependencies for examples (masking and high-resolution DEM examples; GDAL dependency)**
- Python packages: Shapely, fiona, PyGEOS, scikit-image, Rasterio, Trimesh
- [heightmap meshing utility (hmm)](https://github.com/fogleman/hmm)

# Installation

HORAYZON has been tested with **Python 3.10** (Linux) and **Python 3.9** (Mac OS X). Installation can be accomplished as follows:

## Linux

Installation requires the [GNU Compiler Collection (GCC)](https://gcc.gnu.org). Create an appropriate Conda environment

**Core dependencies**
```bash
conda create -n horayzon_core -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray
```

**Base dependencies for examples**
```bash
conda create -n horayzon_base -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray netcdf4 matplotlib pillow skyfield pyproj ipython
```

**All dependencies for examples (masking and high-resolution DEM examples; GDAL dependency)**
```bash
conda create -n horayzon_all -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray netcdf4 matplotlib pillow skyfield pyproj ipython shapely fiona pygeos scikit-image rasterio trimesh
```

and **activate this environment**. The HORAYZON package can then be installed with:
```bash
git clone --branch installation https://github.com/ChristianSteger/HORAYZON.git
cd HORAYZON  
python -m pip install .
```

## Mac OS X

HORAYZON is compiled with **Clang** under Mac OS X. As the Apple-provided **Clang** does not support OpenMP, an alternative **Clang** with OpenMP support has to be installed. This can be done via Conda. Create an appropriate Conda environment

**Core dependencies**
```bash
conda create -n horayzon_core -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray c-compiler openmp python=3.9
```

**Base dependencies for examples**
```bash
conda create -n horayzon_base -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray netcdf4 matplotlib pillow skyfield pyproj ipython c-compiler openmp python=3.9
```

**All dependencies for examples (masking and high-resolution DEM examples; GDAL dependency)**
```bash
conda create -n horayzon_all -c conda-forge embree3 tbb-devel netcdf-cxx4 cython numpy scipy geographiclib tqdm requests xarray netcdf4 matplotlib pillow skyfield pyproj ipython shapely fiona pygeos scikit-image rasterio trimesh c-compiler openmp python=3.9
```

and **activate this environment**. The HORAYZON package can then be installed with:
```bash
git clone --branch installation https://github.com/ChristianSteger/HORAYZON.git
cd HORAYZON
python -m pip install .
```

## Optional installation of hmm
**hmm** depends on **glm**, which can also be installed via Conda
```bash
conda install -c conda-forge glm
```
Alternatively, **glm** can also be built manually from [source](https://glm.g-truc.net/0.9.9/index.html). **hmm** can then be downloaded with
```bash
git clone https://github.com/fogleman/hmm.git
cd hmm
```
The following two lines in **hmm**'s Makefile might have to be adapted to (the include directory in the first line is valid in case **glm**  was installed with Conda):
```bash
COMPILE_FLAGS = -std=c++11 -flto -O3 -Wall -Wextra -Wno-sign-compare -march=native -lGL -lglut -lGLEW -I<path to directory 'include' of conda environment>
INSTALL_PREFIX = <binary install path>
```
Finally, **hmm** can be installed with
```bash
make
make install
```

# Usage

The usage of the packages is best illustrated by means of examples, which can either be run in a Python IDE (like PyCharm or Spyder) or in the terminal.
To run the examples, the path **path_out** must be adapted to a location that provides enough disk space. For the example **horizon/gridded_planar_DEM_2m.py**, the path to the hmm executable (**hmm_ex**) has to be additionally adapted.
All input data [DEM or auxiliary data](# Digital-elevation-model-and-auxiliary-data) required for running the examples is downloaded automatically.

## Examples: Terrain parameters (slope, horizon and sky view factor)

Two terrain horizon functions are available, **horizon_gridded()** and **horizon_locations()**. The former function allows to compute horizon for gridded input while the latter allows to compute horizon for arbitrary selected locations. The second function can optionally output the distance to the horizon. The following five examples are provided:
- **horizon/gridded_curved_DEM.py**: Compute topographic parameters from SRTM (geodetic coordinates, ~90 m resolution) for a ~50x50 km example region in the European Alps. Earth's surface curvature is considered. Plot output of this script is shown below.
![Alt text](https://github.com/ChristianSteger/Images/blob/master/Topo_slope_SVF_new.png?raw=true "Output from horizon/gridded_curved_DEM.py")
- **horizon/gridded_planar_DEM.py**: Compute topographic parameters from swisstopo DHM25 (map projection, 25 m resolution) for a ~25x40 km example region in Switzerland. Earth's surface curvature is neglected.
- **horizon/locations_curved_DEM.py**: Compute topographic parameters (additionally distance to horizon) from SRTM (geodetic coordinates, ~90 m resolution) for 11 locations in Switzerland. Earth's surface curvature is considered. Plot output of this script for one location is shown below.
![Alt text](https://github.com/ChristianSteger/Images/blob/master/Wengen.png?raw=true "Output from horizon/locations_curved_DEM.py")
- **horizon/gridded_curved_DEM_masked.py**: Compute topographic parameters from SRTM (geodetic coordinates, ~90 m resolution) for South Georgia in the South Atlantic Ocean. Earth's surface curvature is considered. DEM grid cells, which are at least 20 km apart from land, are masked to speed-up horizon computation.
- **horizon/gridded_planar_DEM_2m.py**:  Compute gridded topographic parameters from swissALTI3D (map projection, 2 m resolution) for a 3x3 km example region in Switzerland. Earth's surface curvature is neglected. The outer DEM domain is simplified and represented by a triangulated irregular network (TIN) to reduce the large memory footprint of the DEM data.


**A remark on sky view factor and related parameters**<br/>
The term sky view factor (SVF) is defined ambiguously in literature. In Zakšek et al. (2011), it refers to the solid angle of the (celestial) hemisphere. We call this parameter *visible sky fraction* and its computation is performed with the function **topo_param.visible_sky_fraction()**. In applications related to radiation, the SVF is typically defined as the fraction of sky radiation received at a certain location in case of isotropic sky radiation (see e.g. Helbig et al., 2009). This parameter is called *sky view factor* in our application and its computation is performed with the function **topo_param.sky_view_factor()**. Additionally, the positive topographic openness (Yokoyama et al., 2002) can be computed with the function **topo_param.topographic_openness()**. 

## Examples: Shadow map and shortwave correction factor

The module **shadow** allows to compute shadow maps and correction factors for downwelling direct shortwave radiation for arbitrary terrains and sun positions. 
This module was not part of the initial HORAYZON release and is thus not described in the [reference publication](https://doi.org/10.5194/gmd-2022-58). A more detailed description is therefore provided here.
To compute gridded shadow maps or shortwave correction factors, a class **shadow.Terrain** must first be created and initialised.
In this step, the gridded terrain input is first converted to a triangle mesh and these triangles are then stored in a bounding volume hierarchy (BVH) to perform ray casting efficiently.
During initialisation, and optional mask can be provided to ignore certain grid cells and a flag to consider [atmospheric refraction](#link_atmos_refrac) can be enabled.
The two methods **Terrain.shadow()** and **Terrain.sw_dir_cor()** can then be called for various sun positions.
The output of the method **Terrain.shadow()** is encoded as follows: 0: illuminated, 1: self-shaded, 2: terrain-shaded, 3: not considered (respectively masked).
The correction factors for downwelling direct shortwave radiation is computed with the method **Terrain.sw_dir_cor()** according to Müller and Scherer (2005).
This factor can be applied to radiation output from a regional climate or general circulation model, in which radiation is only simulated along the vertical dimension and all grid cells are assumed to have a horizontal surface. 
The correction factor accounts for all terrain-induced modifications in radiation, like self/terrain-shading, changes in angles between the sun and the surface's normal vector and the geometric surface enlargement of grid cells due to sloping surfaces.
According to Equation (2) in Müller and Scherer (2005), the correction factor $f_{cor}$ is computed as

$f_{cor} = \left( \dfrac{1.0}{\vec{h} \times \vec{s}} \right) \left( \dfrac{1.0}{\vec{h} \times \vec{t}} \right) \ {mask}_{shadow} \ \left( \vec{t} \times \vec{s} \right)$

where $\vec{h}$ is the normal of the horizontal surface, $\vec{t}$ the normal of the tilted surface, $\vec{s}$ the sun position vector and ${mask}_{shadow}$ the terrain-shading mask (0: shadow, 1: illuminated). All above vectors represent unit vectors.

- **shadow/gridded_curved_DEM_SRTM.py**: Compute shadow map and shortwave correction factor from SRTM (geodetic coordinates, ~90 m resolution) for South Georgia in the South Atlantic Ocean for a day in southern-hemisphere winter. Earth's surface curvature and atmospheric refraction are considered. Plot output of this script is shown below.
![Alt text](https://github.com/ChristianSteger/Images/blob/master/Elevation_sw_dir_cor.png?raw=true "Output from shadow/gridded_curved_DEM_SRTM.py")
- **shadow/gridded_curved_DEM_REMA.py**: Compute shortwave correction factor from REMA (map projection, ~100 m resolution) for an example region in Antarctica for a day in southern-hemisphere summer. Earth's surface curvature and atmospheric refraction are considered and ocean grid cells are ignored (masked).
- **shadow/gridded_planar_DEM_artificial.py**: Compute shortwave correction factor from artificial topography (hemispherical mountain in the centre). The illumination source (sun) rotates once around the centre.
- **shadow/gridded_curved_DEM_NASADEM.py** Compute shortwave correction factor from NASADEM (geodetic coordinates, ~30 m resolution) for an example region in the Karakoram for a day in northern-hemisphere winter. Earth's surface curvature is considered and atmospheric refraction ignored. All non-glacier grid cells are masked to speed-up computation.

<a name="link_atmos_refrac"> **Atmospheric refraction**<br/> </a>
Close to the unobstructed terrestrial horizon, the position of the sun is significantly influenced by [atmospheric refraction](https://en.wikipedia.org/wiki/Atmospheric_refraction).
The solar elevation angle of the true position is thereby lower than the apparent position.
We included an option (**refrac_cor=True**) to account for this effect by applying the formula of Sæmundsson (1986). This formula is also presented in Meeus (1998) and reads

$r = \frac{1}{60} \left(1.02 \cdot \cot \left(h_{t} + \frac{10.3}{h_{t} + 5.11}\right) + 0.0019279 \right) \cdot \left(\frac{p}{101} \frac{283}{273 + T}\right)$

with r representing atmospheric refraction (degree), h<sub>t</sub> the sun's true elevation angle (degree), p atmospheric pressure (kPa) and T temperature (° C). Note that the function argument of *cot* must be provided in **radian**.
Atmospheric refraction increases with increasing air pressure and decreasing temperature and is only significant for very low solar elevation angles, as illustrated in the below figure.
The dotted lines represent the raw output according to the above equation. We keep refraction correction constant for elevation angles smaller than -1.0°.
![Alt text](https://github.com/ChristianSteger/Images/blob/master/Atmos_refrac_Saemundsson.png?raw=true "Atmospheric refraction accoring to Sæmundsson (1986)")
At sea level, we assume a temperature of T<sub>0</sub> = 10° C and an atmospheric pressure of p<sub>0</sub> = 101.0 kPa. These quantities are extrapolated to higher elevations with a constant linear temperature
lapse rate and the hydrostatic assumption according to the following two equations

$T(z) = T_{0} - \Gamma \cdot z$<br/>
$p(z) = p_{0} \cdot \left(\frac{T_{0} - \Gamma \cdot z}{T_{0}}\right)^{\frac{g}{R_{d} \cdot \Gamma}}$

with g representing the acceleration due to gravity (9.81 m <sup>-2</sup>), R<sub>d</sub> the gas constant for dry air (287.0 J K<sup>-1</sup> kg<sup>-1</sup>) and $\Gamma$ the lapse rate (0.0065° C m<sup>-1</sup>).
These assumptions yield a temperature of -9.5° C and an atmospheric pressure of ~70 kPa for an elevation of 3000 m a.s.l. According to the above figure, changes in atmospheric pressure dominate the influence on atmospheric refraction,
which results in less significant refraction effects for elevated areas like mountains.

# Digital elevation model and auxiliary data

Digital elevation model (DEM) data is available from various sources, e.g.:
- [SRTM](https://srtm.csi.cgiar.org)
- [DHM25](https://www.swisstopo.admin.ch/en/geodata/height/dhm25.html)
- [swissALTI3D](https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html)
- [NASADEM](https://search.earthdata.nasa.gov/)
- [MERIT](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/)
- [USGS 1/3rd arc-second DEM](https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5)

Auxiliary data, like geoid undulation data (EGM96 and GEOID12A), coastline polygons (GSHHG) or glacier outlines (GAMDAM) are available from here:
- [EGM96](https://earth-info.nga.mil)
- [GEOID12A](https://geodesy.noaa.gov/GEOID/GEOID12A/GEOID12A_AK.shtml)
- [GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/)
- [GAMDAM](https://doi.pangaea.de/10.1594/PANGAEA.891423)

# References
- Helbig, N., Löwe, H. and Lehning, M. (2009): Radiosity Approach for the Shortwave Surface Radiation Balance in Complex Terrain, Journal of the Atmospheric Sciences, 66(9), 2900-2912, https://doi.org/10.1175/2009JAS2940.1
- Meeus (1998). Astronomical algorithms (Second edition). Richmond, Va.: Willmann-Bell. pp. 105–108. ISBN 0943396611.
- Müller, M. D., and Scherer, D. (2005): A Grid- and Subgrid-Scale Radiation Parameterization of Topographic Effects for Mesoscale Weather Forecast Models, Monthly Weather Review, 133(6), 1431-1442, https://journals.ametsoc.org/view/journals/mwre/133/6/mwr2927.1.xml
- Sæmundsson (1986). Astronomical Refraction. Sky and Telescope. 72: 70.
- Yokoyama, R., Shirasawa, M. and Pike, R. J. (2002): Visualizing Topography by Openness: A New Application of Image Processing to Digital Elevation Models, Photogrammetric Engineering and Remote Sensing, 68, 257-265.
- Zakšek, K., Oštir, K. and Kokalj, Ž. (2011): Sky-View Factor as a Relief Visualization Technique, Remote Sensing, 3(2):398-415, https://doi.org/10.3390/rs3020398

# Support, feedback and collaboration
In case of issues or questions, contact Christian R. Steger (christian.steger@env.ethz.ch). Please report any bugs you find in HORAYZON. You are welcome to fork this repository to modify the source code - we are open to consider *pull requests* for future HORAYZON versions/releases.