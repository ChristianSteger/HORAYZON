# Description: Compute topographic parameters (slope angle and aspect, horizon
#              and Sky View Factor) from NASADEM (~30 m) for the example region
#              'Heard Island and McDonald Islands' and mask ocean grid cells
#              that have a certain minimal distance to land
#
# Required input data:
#   - NASADEM: https://search.earthdata.nasa.gov/
#     -> NASADEM Merged DEM Global 1 arc second nc V001
#   - EGM96:
#     https://earth-info.nga.mil/php/download.php?file=egm-96interpolation
#   - GSHHG:
#     http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import Affine

mpl.style.use("classic")

# Paths to folders
path_DEM = "/Users/csteger/Desktop/HIMI/"
path_EGM96 = "/Users/csteger/Desktop/EGM96/"
path_GSHHG = "/Users/csteger/Desktop/gshhg-shp-2.3.7/GSHHS_shp/f/"
path_temp = "/Users/csteger/Desktop/temp/"
path_out = "/Users/csteger/Desktop/output/"

# Load required functions
sys.path.append("/Users/csteger/Desktop/lib/")
from horizon import horizon
import functions_cy
from auxiliary import pad_geometry_buffer
from load_dem import dem_domain_loc
from geoid import geoid_undulation
from ocean_masking import get_GSHHS_coastlines, coastline_contours
from ocean_masking import coastline_distance, coastline_buffer

###############################################################################
# Compute and save topographic parameters
###############################################################################

# Settings
loc = (-53.11, 73.55)  # centre location (latitude, longitude) [degree]
width_in = 100.0  # width of considered domain [kilometre]
dist_search = 20.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
azim_num = 60  # number of azimuth sectors [-]
hori_acc = 1.5  # accuracy of horizon computation [degree]
dem_res = 1.0 / 3600.0  # resolution of DEM [degree]
files_dem = ((None, "NASADEM_NC_s53e073.nc", None),
             ("NASADEM_NC_s54e072.nc", "NASADEM_NC_s54e073.nc", None))
file_hori = path_out + "hori_NASADEM_HIMI.nc"
file_topo_par = path_out + "topo_par_NASADEM_HIMI.nc"

# Load DEM data (2 x 3 tiles)
dem = np.zeros((3600 * 2, 3600 * 3), dtype=np.float32)
for i in range(2):
    for j in range(3):
        if files_dem[i][j] is not None:
            slic = (slice(i * 3600, (i + 1) * 3600),
                    slice(j * 3600, (j + 1) * 3600))
            dem[slic] = xr.open_dataset(path_DEM + files_dem[i][j]) \
                .isel(lon=slice(0, 3600), lat=slice(0, 3600))["NASADEM_HGT"] \
                .values
lat = np.linspace(-52.0, -54.0, dem.shape[0] + 1, dtype=np.float64)[:-1]
lon = np.linspace(72.0, 75.0, dem.shape[1] + 1, dtype=np.float64)[:-1]

# Crop DEM data
dom = dem_domain_loc(loc, width_in, dist_search, ellps)
sd = (slice(np.argmin(np.abs(dom["tot"]["lat_max"] - lat)),
            np.argmin(np.abs(dom["tot"]["lat_min"] - lat))),
      slice(np.argmin(np.abs(dom["tot"]["lon_min"] - lon)),
            np.argmin(np.abs(dom["tot"]["lon_max"] - lon))))
dem = dem[sd]
lat = lat[sd[0]]
lon = lon[sd[1]]
mask_land_dem = (dem != 0.0)
print("Total domain size: " + str(dem.shape))

# Compute ellipsoidal heights
undul = geoid_undulation(lon, lat, geoid="EGM96", path=path_EGM96)
dem += undul  # ellipsoidal height [m]

# Compute indices of inner domain
sd_in = (slice(np.argmin(np.abs(dom["in"]["lat_max"] - lat)),
               np.argmin(np.abs(dom["in"]["lat_min"] - lat))),
         slice(np.argmin(np.abs(dom["in"]["lon_min"] - lon)),
               np.argmin(np.abs(dom["in"]["lon_max"] - lon))))
offset_0 = sd_in[0].start
offset_1 = sd_in[1].start
print("Inner domain size: " + str(dem[sd_in].shape))

# Compute ECEF coordinates
x_ecef, y_ecef, z_ecef = functions_cy.lonlat2ecef_gc1d(lon, lat, dem,
                                                       ellps=ellps)
dem_dim_0, dem_dim_1 = dem.shape

# ENU origin of coordinates
ind_0, ind_1 = int(len(lat) / 2), int(len(lon) / 2)
lon_or, lat_or = lon[ind_1], lat[ind_0]
x_ecef_or = x_ecef[ind_0, ind_1]
y_ecef_or = y_ecef[ind_0, ind_1]
z_ecef_or = z_ecef[ind_0, ind_1]

# Compute ENU coordinates
x_enu, y_enu, z_enu = functions_cy.ecef2enu(x_ecef, y_ecef, z_ecef,
                                            x_ecef_or, y_ecef_or, z_ecef_or,
                                            lon_or, lat_or)

# Compute unit vectors (in ENU coordinates)
lon_in, lat_in = np.meshgrid(lon[sd_in[1]], lat[sd_in[0]])
vec_norm_ecef = functions_cy.surf_norm(lon_in, lat_in)
del lon_in, lat_in
vec_north_ecef = functions_cy.north_dir(x_ecef[sd_in], y_ecef[sd_in],
                                        z_ecef[sd_in], vec_norm_ecef,
                                        ellps=ellps)
vec_norm_enu = functions_cy.ecef2enu_vec(vec_norm_ecef, lon_or, lat_or)
vec_north_enu = functions_cy.ecef2enu_vec(vec_north_ecef, lon_or, lat_or)
del vec_norm_ecef, vec_north_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = np.hstack((x_enu.reshape(x_enu.size, 1),
                       y_enu.reshape(x_enu.size, 1),
                       z_enu.reshape(x_enu.size, 1))).ravel()
vert_grid = pad_geometry_buffer(vert_grid)

# -----------------------------------------------------------------------------
# Compute mask for ocean grid cells
# -----------------------------------------------------------------------------

# Important note: compute coastlines for total domain -> land grid cells in the
# outer domain can influence terrain horizon of inner domain

# Get and rasterise GSHHS data
poly_coastlines = get_GSHHS_coastlines(dom["tot"], path_GSHHG, path_temp)
coastline_GSHHS = unary_union([i for i in poly_coastlines])
d_lon = np.diff(lon).mean()
d_lat = np.diff(lat).mean()
transform = Affine(d_lon, 0.0, lon[0] - d_lon / 2.0,
                   0.0, d_lat, lat[0] - d_lat / 2.0)
mask_land_GSHHS = rasterize(coastline_GSHHS, (len(lat), len(lon)),
                            transform=transform).astype(bool)

# Compute common land-sea-mask, contours and transform coordinates
mask_land = (mask_land_dem | mask_land_GSHHS)
contours_latlon = coastline_contours(lon, lat, mask_land.astype(np.uint8))
print("Number of vertices (DEM): " + str(sum([i.shape[0]
                                              for i in contours_latlon])))
pts_latlon = np.vstack(([i for i in contours_latlon]))
h = np.zeros((pts_latlon.shape[0], 1), dtype=np.float32)
coords = functions_cy.lonlat2ecef(pts_latlon[:, 0:1], pts_latlon[:, 1:2], h,
                                  ellps=ellps)
pts_ecef = np.hstack((coords[0], coords[1], coords[2]))

# Compute coastline buffer (only for inner domain)
block_size = 5 * 2 + 1
mask_buffer = coastline_buffer(x_ecef[sd_in], y_ecef[sd_in], z_ecef[sd_in],
                               mask_land[sd_in], pts_ecef, lat[sd_in[0]],
                               (dist_search * 1000.0), dem_res, ellps,
                               block_size)
frac_ma = ((mask_buffer == 1).sum() / mask_buffer.size * 100.0)
print("Fraction of masked grid cells %.2f" % frac_ma + " %")

# Mask with all types (-1: outside buffer, 0: buffer, 1: land)
mask_type = mask_land[sd_in].astype(np.int32)
mask_type[mask_buffer] = -1

# Test plot
plt.figure()
plt.pcolormesh(x_enu[sd_in], y_enu[sd_in], mask_type, shading="auto")
plt.colorbar()

# Binary mask
mask = np.ones(vec_norm_enu.shape[:2], dtype=np.uint8)
mask[(mask_type == -1)] = 0  # mask area outside of buffer
# mask[(mask_type < 1)] = 0  # mask ocean entirely

# -----------------------------------------------------------------------------

# Compute horizon
horizon(vert_grid, dem_dim_0, dem_dim_1,
        vec_norm_enu, vec_north_enu,
        offset_0, offset_1,
        file_out=file_hori,
        x_axis_val=lon[sd_in[1]].astype(np.float32),
        y_axis_val=lat[sd_in[0]].astype(np.float32),
        x_axis_name="lon", y_axis_name="lat", units="degree",
        hori_buffer_size_max=4.5,
        mask=mask,
        dist_search=dist_search, azim_num=azim_num, hori_acc=hori_acc)

# Load horizon data
ds = xr.open_dataset(file_hori)
hori = ds["horizon"].values
azim = ds["azim"].values
ds.close()

# Rotation matrix (global ENU -> local ENU)
rot_mat = np.empty((vec_north_enu.shape[0] + 2, vec_north_enu.shape[1] + 2,
                    3, 3), dtype=np.float32)
rot_mat.fill(np.nan)
rot_mat[1:-1, 1:-1, 0, :] = np.cross(vec_north_enu, vec_norm_enu, axisa=2,
                                     axisb=2)  # vector pointing towards east
rot_mat[1:-1, 1:-1, 1, :] = vec_north_enu
rot_mat[1:-1, 1:-1, 2, :] = vec_norm_enu
del vec_north_enu, vec_norm_enu

# Compute slope
sd_in_a1 = (slice(sd_in[0].start - 1, sd_in[0].stop + 1),
            slice(sd_in[1].start - 1, sd_in[1].stop + 1))
vec_tilt = functions_cy.slope_plane_meth(x_enu[sd_in_a1], y_enu[sd_in_a1],
                                         z_enu[sd_in_a1], rot_mat)[1:-1, 1:-1]
del rot_mat
del x_enu, y_enu, z_enu

# Compute Sky View Factor
svf = functions_cy.skyviewfactor(azim, hori, vec_tilt)

# Compute slope angle and aspect
slope = np.arccos(vec_tilt[:, :, 2])
aspect = np.pi / 2.0 - np.arctan2(vec_tilt[:, :, 1],
                                  vec_tilt[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

# Save topographic parameters to NetCDF file
ncfile = Dataset(filename=file_topo_par, mode="w")
ncfile.createDimension(dimname="lat", size=svf.shape[0])
ncfile.createDimension(dimname="lon", size=svf.shape[1])
nc_lat = ncfile.createVariable(varname="lat", datatype="f",
                               dimensions="lat")
nc_lat[:] = lat[sd_in[0]]
nc_lat.units = "degree"
nc_lon = ncfile.createVariable(varname="lon", datatype="f",
                               dimensions="lon")
nc_lon[:] = lon[sd_in[1]]
nc_lon.units = "degree"
nc_data = ncfile.createVariable(varname="elevation", datatype="f",
                                dimensions=("lat", "lon"))
nc_data[:] = dem[sd_in]
nc_data.long_name = "ellipsoidal height"
nc_data.units = "m"
nc_data = ncfile.createVariable(varname="slope", datatype="f",
                                dimensions=("lat", "lon"))
nc_data[:] = slope
nc_data.long_name = "slope angle"
nc_data.units = "rad"
nc_data = ncfile.createVariable(varname="aspect", datatype="f",
                                dimensions=("lat", "lon"))
nc_data[:] = aspect
nc_data.long_name = "slope aspect (clockwise from North)"
nc_data.units = "rad"
nc_data = ncfile.createVariable(varname="svf", datatype="f",
                                dimensions=("lat", "lon"))
nc_data[:] = svf
nc_data.long_name = "sky view factor"
nc_data.units = "-"
ncfile.close()
