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
import sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import zipfile
from osgeo import gdal
sys.path.append("/Users/csteger/Downloads/HORAYZON/")  # ------------ temporary
from horayzon import auxiliary, domain, horizon, load_dem, topo_param, download  # temporary
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
path_out += "gridded_DHM25_Switzerland/"
if not os.path.isdir(path_out):
    os.mkdir(path_out)

# Download and unzip DHM25 data
print("Download DHM25 data:")
hray.download.file(dem_file_url, path_out)
with zipfile.ZipFile(path_out + "DHM25_MM_ASCII_GRID.zip", "r") as zip_ref:
    zip_ref.extractall(path_out)
os.remove(path_out + "DHM25_MM_ASCII_GRID.zip")

# Load required DEM data (including outer boundary zone)
domain_outer = hray.domain.planar_grid(domain, dist_search)
file_dem = path_out + "ASCII_GRID_1part/dhm25_grid_raster.asc"
x, y, elevation = hray.load_dem.dhm25(file_dem, domain_outer, engine="gdal")
# -> ESRI ASCII GRID file can also be read with NumPy in case GDAL is not
#    available!

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
hray.horizon.horizon_gridded(vert_grid, dem_dim_0, dem_dim_1,
                             vec_norm, vec_north, offset_0, offset_1,
                             file_out=path_out + file_hori,
                             x_axis_val=x[slice_in[1]].astype(np.float32),
                             y_axis_val=y[slice_in[0]].astype(np.float32),
                             x_axis_name="x", y_axis_name="y",
                             units="metre", dist_search=dist_search,
                             azim_num=azim_num)

# Load horizon data
ds = xr.open_dataset(path_out + file_hori)
hori = ds["horizon"].values
azim = ds["azim"].values
ds.close()
# del vec_north, vec_norm

# Swap coordinate axes (-> make viewable with ncview)
ds_ncview = ds.transpose("azim", "y", "x")
ds_ncview.to_netcdf(path_out + file_hori[:-3] + "_ncview.nc")

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
ncfile = Dataset(filename=path_out + file_topo_par, mode="w")
ncfile.createDimension(dimname="y", size=svf.shape[0])
ncfile.createDimension(dimname="x", size=svf.shape[1])
nc_lat = ncfile.createVariable(varname="y", datatype="f",
                               dimensions="y")
nc_lat[:] = y[slice_in[0]]
nc_lat.units = "metre"
nc_lon = ncfile.createVariable(varname="x", datatype="f",
                               dimensions="x")
nc_lon[:] = x[slice_in[1]]
nc_lon.units = "metre"
nc_data = ncfile.createVariable(varname="elevation", datatype="f",
                                dimensions=("y", "x"))
nc_data[:] = elevation[slice_in]
nc_data.long_name = "elevation"
nc_data.units = "m"
nc_data = ncfile.createVariable(varname="slope", datatype="f",
                                dimensions=("y", "x"))
nc_data[:] = slope
nc_data.long_name = "slope angle"
nc_data.units = "rad"
nc_data = ncfile.createVariable(varname="aspect", datatype="f",
                                dimensions=("y", "x"))
nc_data[:] = aspect
nc_data.long_name = "slope aspect (clockwise from North)"
nc_data.units = "rad"
nc_data = ncfile.createVariable(varname="svf", datatype="f",
                                dimensions=("y", "x"))
nc_data[:] = svf
nc_data.long_name = "sky view factor"
nc_data.units = "-"
ncfile.close()
