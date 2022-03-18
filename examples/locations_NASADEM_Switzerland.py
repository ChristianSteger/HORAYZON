# Description: Compute topographic parameters (slope angle and aspect,
#              horizon and sky view factor) from NASADEM (~30 m) for
#              arbitrary locations in Switzerland
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
from scipy import interpolate

mpl.style.use("classic")

# Paths to folders
path_DEM = "/Users/csteger/Desktop/European_Alps/"
path_EGM96 = "/Users/csteger/Desktop/EGM96/"
path_out = "/Users/csteger/Desktop/output/"

# Load required functions
sys.path.append("/Users/csteger/Desktop/lib/")
from horizon import horizon_locations
import functions_cy
from auxiliary import pad_geometry_buffer
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
dist_search = 50.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
azim_num = 1440  # number of azimuth sectors [-]
hori_acc = 0.1  # [degree]
files_dem = ["NASADEM_NC_n" + str(j) + "e" + str(i).zfill(3) + ".nc"
             for j in range(45, 48) for i in range(5, 11)]
file_hori = path_out + "hori_NASADEM_Switzerland.nc"
file_topo_par = path_out + "topo_par_NASADEM_Switzerland.nc"

# Load DEM data
ds = xr.open_mfdataset([path_DEM + i for i in files_dem],
                       preprocess=preprocess)
dem = ds["NASADEM_HGT"].values
lon = ds["lon"].values
lat = ds["lat"].values
ds.close()
print("Total domain size: " + str(dem.shape))

# Compute ellipsoidal heights
undul = geoid_undulation(lon, lat, geoid="EGM96", path=path_EGM96)
dem += undul  # ellipsoidal height [m]

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
del x_ecef, y_ecef, z_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = np.hstack((x_enu.reshape(x_enu.size, 1),
                       y_enu.reshape(x_enu.size, 1),
                       z_enu.reshape(x_enu.size, 1))).ravel()
vert_grid = pad_geometry_buffer(vert_grid)
del x_enu, y_enu, z_enu

# -----------------------------------------------------------------------------
# Individual locations
# -----------------------------------------------------------------------------

# Locations
loc_sel = {"Eiger_Nordwand": (46.58210, 8.00038),
           "Wengen":         (46.60691, 7.92347),
           "Muerren":        (46.55944, 7.89222),
           "Goeschenen":     (46.66777, 8.58639),
           "Vals":           (46.61666, 9.18334),
           "Biasca":         (46.35991, 8.97105),
           "Zinal":          (46.13556, 7.62583),
           "Blatten":        (46.42221, 7.82083),
           "Leukerbad":      (46.38333, 7.63333),
           "Zuerich":        (47.37174, 8.54226)}

# Geodetic coordinates
lon_loc = np.array([loc_sel[i][1] for i in loc_sel.keys()],
                   dtype=np.float64)[:, np.newaxis]
lat_loc = np.array([loc_sel[i][0] for i in loc_sel.keys()],
                   dtype=np.float64)[:, np.newaxis]
z_loc = np.zeros((len(loc_sel), 1), dtype=np.float32)

# Compute ECEF coordinates
x_ecef_loc, y_ecef_loc, z_ecef_loc \
    = functions_cy.lonlat2ecef(lon_loc, lat_loc, z_loc, ellps=ellps)

# Compute ENU coordinates
x_enu_loc, y_enu_loc, z_enu_loc \
    = functions_cy.ecef2enu(x_ecef_loc, y_ecef_loc, z_ecef_loc,
                            x_ecef_or, y_ecef_or, z_ecef_or,
                            lon_or, lat_or)
coords = np.hstack((x_enu_loc, y_enu_loc, z_enu_loc))
del x_enu_loc, y_enu_loc, z_enu_loc

# Compute unit vectors (in ENU coordinates)
vec_norm_ecef = functions_cy.surf_norm(lon_loc, lat_loc)
vec_north_ecef = functions_cy.north_dir(x_ecef_loc, y_ecef_loc,
                                        z_ecef_loc, vec_norm_ecef,
                                        ellps=ellps)
del x_ecef_loc, y_ecef_loc, z_ecef_loc
vec_norm_enu = functions_cy.ecef2enu_vec(vec_norm_ecef, lon_or, lat_or) \
    .squeeze()
vec_north_enu = functions_cy.ecef2enu_vec(vec_north_ecef, lon_or, lat_or) \
    .squeeze()
del vec_norm_ecef, vec_north_ecef

# -----------------------------------------------------------------------------

# Compute horizon
horizon_locations(vert_grid, dem_dim_0, dem_dim_1,
                  coords, vec_norm_enu, vec_north_enu,
                  dist_search=dist_search, azim_num=azim_num,
                  hori_acc=hori_acc,
                  x_axis_val=lon_loc.ravel().astype(np.float32),
                  y_axis_val=lat_loc.ravel().astype(np.float32),
                  x_axis_name="lon", y_axis_name="lat", units="degree",
                  file_out=file_hori,
                  ray_algorithm="binary_search", geom_type="grid",
                  ray_org_elev=np.array([2.0], dtype=np.float32),
                  hori_dist_out=True)

# Load horizon data
ds = xr.open_dataset(file_hori)
azim = ds["azim"].values
hori = ds["horizon"].values
hori_dist = ds["horizon_distance"].values / 1000.0  # [km]
ds.close()
del vert_grid

# -----------------------------------------------------------------------------
# Compute slope and sky view factor for locations
# -----------------------------------------------------------------------------

topo_param = {}
for i in list(loc_sel.keys()):

    # 5 x 5 grid cell domain
    ind_0 = np.argmin(np.abs(loc_sel[i][0] - lat))
    ind_1 = np.argmin(np.abs(loc_sel[i][1] - lon))
    slic = (slice(ind_0 - 2, ind_0 + 3), slice(ind_1 - 2, ind_1 + 3))

    # Compute ECEF coordinates
    lon_5x5, lat_5x5 = np.meshgrid(lon[slic[1]], lat[slic[0]])
    x_ecef, y_ecef, z_ecef = functions_cy.lonlat2ecef(lon_5x5, lat_5x5,
                                                      dem[slic], ellps=ellps)

    # Compute ENU coordinates
    x_enu, y_enu, z_enu = functions_cy.ecef2enu(x_ecef, y_ecef, z_ecef,
                                                x_ecef_or, y_ecef_or,
                                                z_ecef_or,
                                                lon_or, lat_or)

    # Compute unit vectors (in ENU coordinates)
    vec_norm_ecef = functions_cy.surf_norm(lon_5x5, lat_5x5)
    vec_north_ecef = functions_cy.north_dir(x_ecef, y_ecef, z_ecef,
                                            vec_norm_ecef, ellps=ellps)
    del x_ecef, y_ecef, z_ecef
    vec_norm_enu = functions_cy.ecef2enu_vec(vec_norm_ecef, lon_or, lat_or)
    vec_north_enu = functions_cy.ecef2enu_vec(vec_north_ecef, lon_or, lat_or)
    del vec_norm_ecef, vec_north_ecef

    # Rotation matrix (global ENU -> local ENU)
    rot_mat = np.empty((vec_north_enu.shape[0], vec_north_enu.shape[1],
                        3, 3), dtype=np.float32)
    rot_mat[:, :, 0, :] = np.cross(vec_north_enu, vec_norm_enu, axisa=2,
                                   axisb=2)
    rot_mat[:, :, 1, :] = vec_north_enu
    rot_mat[:, :, 2, :] = vec_norm_enu
    del vec_north_enu, vec_norm_enu

    # Compute slope
    vec_tilt = functions_cy.slope_plane_meth(x_enu, y_enu, z_enu,
                                             rot_mat)[1: -1, 1: -1, :]

    # Bilinear interpolation of slope at location
    vec_tilt_ip = np.empty((1, 1, 3), dtype=np.float32)
    for j in range(3):
        f = interpolate.interp2d(lon_5x5[1:-1, 1:-1], lat_5x5[1:-1, 1:-1],
                                 vec_tilt[:, :, j], bounds_error=True)
        vec_tilt_ip[0, 0, j] = f(loc_sel[i][1], loc_sel[i][0])
    vec_tilt_ip /= np.sqrt(np.sum(vec_tilt_ip ** 2))  # unit vector

    # Compute slope angle and aspect
    slope = np.arccos(vec_tilt_ip[0, 0, 2])
    aspect = np.pi / 2.0 - np.arctan2(vec_tilt_ip[0, 0, 1],
                                      vec_tilt_ip[0, 0, 0])
    if aspect < 0.0:
        aspect += np.pi * 2.0  # [0.0, 2.0 * np.pi]

    # Compute Sky View Factor
    hori_rs = hori[list(loc_sel).index(i), :][np.newaxis, np.newaxis, :]
    svf = functions_cy.skyviewfactor(azim, hori_rs, vec_tilt_ip)[0, 0]

    topo_param[i] = "Slope angle: %.2f" % np.rad2deg(slope) + " deg, " \
                    + "slope aspect: %.2f" % np.rad2deg(aspect) + " deg, " \
                    + "sky view factor: %.2f" % svf

# -----------------------------------------------------------------------------

# Plot horizon and distance to horizon for locations
for i in list(loc_sel.keys()):
    plt.figure(figsize=(12, 6))
    ax_l = plt.axes()
    ind = list(loc_sel.keys()).index(i)
    plt.plot(np.rad2deg(azim), np.rad2deg(hori[ind, :]), color="black", lw=2.5)
    plt.xlabel("Azimuth angle (measured clockwise from North) [deg]")
    plt.ylabel("Horizon elevation angle [deg]")
    plt.xlim([-5.0, 365.0])
    ax_r = ax_l.twinx()
    plt.plot(np.rad2deg(azim), hori_dist[ind, :], color="blue", lw=1.5)
    plt.ylabel("Distance to horizon line [km]", color="blue")
    ax_r.tick_params(axis="y", colors="blue")
    plt.title(i.replace("_", " "), fontsize=12, fontweight="bold",
              loc="left")
    plt.title(topo_param[i], fontsize=12, loc="right")
