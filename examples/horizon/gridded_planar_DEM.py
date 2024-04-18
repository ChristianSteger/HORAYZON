# Description: Compute gridded topographic parameters (slope angle and aspect,
#              horizon and sky view factor) from swisstopo DHM25 data (25 m)
#              for an example region in Switzerland. Ignore Earth's surface
#              curvature.
#
# Source of applied DEM data:
#   https://www.swisstopo.admin.ch/en/geodata/height/dhm25.html
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import xarray as xr
import zipfile
import horayzon as hray

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
domain = {"x_min": 668000, "x_max": 707000,
          "y_min": 172000, "y_max": 200000}  # domain boundaries [metre]
dist_search = 20.0  # search distance for horizon [kilometre]
azim_num = 180  # number of azimuth sampling directions [-]

# Paths and file names
dem_file_url = "https://cms.geo.admin.ch/ogd/topography/" \
               + "DHM25_MM_ASCII_GRID.zip"
path_out = "/Users/csteger/Desktop/Output/"
file_hori = "hori_DHM25_Switzerland.nc"
file_topo_par = "topo_par_DHM25_Switzerland.nc"

# -----------------------------------------------------------------------------
# Compute and save topographic parameters
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise ValueError("Output directory does not exist")
path_out += "horizon/gridded_DHM25_Switzerland/"
if not os.path.isdir(path_out):
    os.makedirs(path_out)

# Download and unzip DHM25 data
print("Download DHM25 data:")
hray.download.file(dem_file_url, path_out)
with zipfile.ZipFile(path_out + "DHM25_MM_ASCII_GRID.zip", "r") as zip_ref:
    zip_ref.extractall(path_out)
os.remove(path_out + "DHM25_MM_ASCII_GRID.zip")

# Load required DEM data (including outer boundary zone)
domain_outer = hray.domain.planar_grid(domain, dist_search)
file_dem = path_out + "ASCII_GRID_1part/dhm25_grid_raster.asc"
x, y, elevation = hray.load_dem.dhm25(file_dem, domain_outer, engine="numpy")
# -> ESRI ASCII GRID file can also be read with GDAL if available (-> faster)

# Compute indices of inner domain
slice_in = (slice(np.where(y >= domain["y_max"])[0][-1],
                  np.where(y <= domain["y_min"])[0][0] + 1),
            slice(np.where(x <= domain["x_min"])[0][-1],
                  np.where(x >= domain["x_max"])[0][0] + 1))
offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
print("Inner domain size: " + str(elevation[slice_in].shape))

# Create directional unit vectors for inner domain
dem_dim_0, dem_dim_1 = elevation.shape
vec_norm = np.zeros((dem_dim_0 - (2 * offset_0),
                     dem_dim_1 - (2 * offset_1), 3), dtype=np.float32)
vec_norm[:, :, 2] = 1.0
vec_north = np.zeros((dem_dim_0 - (2 * offset_0),
                      dem_dim_1 - (2 * offset_1), 3), dtype=np.float32)
vec_north[:, :, 1] = 1.0

# Merge vertex coordinates and pad geometry buffer
vert_grid = hray.auxiliary.rearrange_pad_buffer(*np.meshgrid(x, y), elevation)

# Compute horizon
hori, azim = hray.horizon.horizon_gridded(
    vert_grid, dem_dim_0, dem_dim_1,
    vec_norm, vec_north, offset_0, offset_1,
    dist_search=dist_search, azim_num=azim_num)

# Save horizon to NetCDF file (in Ncview-compatible format)
ds = xr.Dataset(
    coords=dict(
        azim=(["azim"], azim, {"units": "radian"}),
        y=(["y"], y[slice_in[0]], {"units": "m"}),
        x=(["x"], x[slice_in[1]], {"units": "m"}),
    ),
    data_vars=dict(
        horizon=(["azim", "y", "x"], np.moveaxis(hori, 2, 0),
                   {"units": "radian"})
    )
)
encoding = {i: {"_FillValue": None} for i in ("azim", "y", "x")}
ds.to_netcdf(path_out + file_hori, encoding=encoding)

# Compute slope
x_2d, y_2d = np.meshgrid(x, y)
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))
vec_tilt = hray.topo_param.slope_plane_meth(x_2d[slice_in_a1],
                                            y_2d[slice_in_a1],
                                            elevation[slice_in_a1])[1:-1, 1:-1]
del x_2d, y_2d

# Compute Sky View Factor
svf = hray.topo_param.sky_view_factor(azim, hori, vec_tilt)

# Compute slope angle and aspect
slope = np.arccos(vec_tilt[:, :, 2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(vec_tilt[:, :, 1],
                                  vec_tilt[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

# Save topographic parameters to NetCDF file
ds = xr.Dataset(
    coords=dict(
        y=(["y"], y[slice_in[0]], {"units": "m"}),
        x=(["x"], x[slice_in[1]], {"units": "m"}),
    ),
    data_vars=dict(
        elevation=(["y", "x"], elevation[slice_in],
                   {"long_name": "elevation", "units": "m"}),
        slope=(["y", "x"], slope,
               {"long_name": "slope angle", "units": "radian"}),
        aspect=(["y", "x"], aspect,
                {"long_name": "slope aspect (clockwise from North)",
                 "units": "radian"}),
        svf=(["y", "x"], svf,
             {"long_name": "sky view factor", "units": "-"}),
    )
)
encoding = {i: {"_FillValue": None} for i in
            ("y", "x", "elevation", "slope", "aspect")}
ds.to_netcdf(path_out + file_topo_par, encoding=encoding)
