# Description: Compute topographic parameters (slope angle and aspect,
#              horizon and sky view factor) from NASADEM (~30 m) for a
#              selection of grid cells in the Himalayas (Mount Everest)
#
# Required input data:
#   - NASADEM: https://search.earthdata.nasa.gov/
#     -> NASADEM Merged DEM Global 1 arc second nc V001
#   - EGM96:
#     https://earth-info.nga.mil/php/download.php?file=egm-96interpolation
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("classic")

# Paths to folders
path_DEM = "/Users/csteger/Desktop/Himalayas/"
path_EGM96 = "/Users/csteger/Desktop/EGM96/"
path_out = "/Users/csteger/Desktop/output/"

# Load required functions
sys.path.append("/Users/csteger/Desktop/lib/")
from horizon import horizon_gridcells
import functions_cy
from auxiliary import pad_geometry_buffer
from load_dem import dem_domain_loc
from geoid import geoid_undulation


###############################################################################
# Functions
###############################################################################

# Preprocess function for NASADEM tiles
def preprocess(ds):
    """Remove double grid cell row/column at margins """
    return ds.isel(lon=slice(0, 3600), lat=slice(0, 3600))


###############################################################################
# Compute and save topographic parameters
###############################################################################

# Settings
loc = (27.988056, 86.925278)  # centre location (latitude, longitude) [degree]
width_in = 30.0  # width of considered domain [kilometre]
dist_search = 50.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
azim_num = 720  # number of azimuth sectors [-]
files_dem = ("NASADEM_NC_n27e086.nc", "NASADEM_NC_n27e087.nc",
             "NASADEM_NC_n28e086.nc", "NASADEM_NC_n28e087.nc")
file_hori = path_out + "hori_NASADEM_Himalayas.nc"
file_topo_par = path_out + "topo_par_NASADEM_Himalayas.nc"

# Load DEM data
dom = dem_domain_loc(loc, width_in, dist_search, ellps)
ds = xr.open_mfdataset([path_DEM + i for i in files_dem],
                       preprocess=preprocess)
ds = ds.sel(lon=slice(dom["tot"]["lon_min"], dom["tot"]["lon_max"]),
            lat=slice(dom["tot"]["lat_max"], dom["tot"]["lat_min"]))
dem = ds["NASADEM_HGT"].values
lon = ds["lon"].values
lat = ds["lat"].values
ds.close()
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

# Merge vertex coordinates and pad geometry buffer
vert_grid = np.hstack((x_enu.reshape(x_enu.size, 1),
                       y_enu.reshape(x_enu.size, 1),
                       z_enu.reshape(x_enu.size, 1))).ravel()
vert_grid = pad_geometry_buffer(vert_grid)

# Compute unit vectors (in ENU coordinates)
lon_in, lat_in = np.meshgrid(lon[sd_in[1]], lat[sd_in[0]])
vec_norm_ecef = functions_cy.surf_norm(lon_in, lat_in)
vec_north_ecef = functions_cy.north_dir(x_ecef[sd_in], y_ecef[sd_in],
                                        z_ecef[sd_in], vec_norm_ecef,
                                        ellps=ellps)
del x_ecef, y_ecef, z_ecef
vec_norm_enu = functions_cy.ecef2enu_vec(vec_norm_ecef, lon_or, lat_or)
vec_north_enu = functions_cy.ecef2enu_vec(vec_north_ecef, lon_or, lat_or)
del vec_norm_ecef, vec_north_ecef

# Select individual grid cells (indices for inner domain)
loc_sel = {"Mount_Everest": (27.988611, 86.925000),
           "South_Col":     (27.973486, 86.930190),
           "Camp_I":        (27.987227, 86.876561),
           "Base_Camp":     (28.002148, 86.852473)}
indices = np.empty([len(loc_sel), 2], dtype=np.int32)
for ind, i in enumerate(loc_sel.keys()):
    indices[ind, 0] = np.argmin(np.abs(loc_sel[i][0] - lat[sd_in[0]]))
    indices[ind, 1] = np.argmin(np.abs(loc_sel[i][1] - lon[sd_in[1]]))

vec_norm_enu_gc = vec_norm_enu[indices[:, 0], indices[:, 1], :]
vec_north_enu_gc = vec_north_enu[indices[:, 0], indices[:, 1], :]

# Compute horizon
horizon_gridcells(vert_grid, dem_dim_0, dem_dim_1,
                  indices, vec_norm_enu_gc, vec_north_enu_gc,
                  offset_0, offset_1,
                  dist_search=dist_search, azim_num=azim_num,
                  file_out=file_hori,
                  ray_algorithm="binary_search", geom_type="quad",
                  ray_org_elev=2.0,
                  hori_dist_out=True)

# Load horizon data
ds = xr.open_dataset(file_hori)
hori = ds["horizon"].values
azim = ds["azim"].values
hori_dist = ds["horizon_distance"].values / 1000.0  # [km]
ds.close()

# Plot horizon and distance to horizon for specific location
ind = 2
plt.figure(figsize=(12, 5))
ax_l = plt.axes()
plt.plot(np.rad2deg(azim), np.rad2deg(hori[ind, :]),
         color="black", lw=1.5)
plt.axis([0.0, 360.0, -10.0, 50.0])
plt.xlabel("Azimuth angle (measured clockwise from North) [deg]")
plt.ylabel("Horizon elevation angle [deg]")
ax_r = ax_l.twinx()
plt.plot(np.rad2deg(azim), hori_dist[ind, :],
         color="blue", lw=1.5)
plt.ylabel("Distance to horizon line [km]", color="blue")
ax_r.tick_params(axis="y", colors="blue")
plt.title(list(loc_sel.keys())[ind].replace("_", " "))
