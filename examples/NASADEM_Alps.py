# Description: Compute topographic parameters (slope angle and aspect, horizon
#              and Sky View Factor) from NASADEM (~30 m) for an example region
#              in the European Alps
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
import os
import sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cmcrameri import cm

mpl.style.use("classic")

# Paths to folders
path_DEM = "/Users/csteger/Desktop/European_Alps/"
path_EGM96 = "/Users/csteger/Desktop/EGM96/"
path_out = "/Users/csteger/Desktop/output/"

# Load required functions
sys.path.append("/Users/csteger/Desktop/lib/")
from horizon import horizon
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
loc = (46.9, 9.0)  # centre location (latitude, longitude) [degree]
width_in = 30.0  # width of considered domain [kilometre]
dist_search = 50.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
azim_num = 180  # number of azimuth sectors [-]
files_dem = ("NASADEM_NC_n46e008.nc", "NASADEM_NC_n46e009.nc",
             "NASADEM_NC_n47e008.nc", "NASADEM_NC_n47e009.nc")
file_hori = path_out + "hori_NASADEM_Alps.nc"
file_topo_par = path_out + "topo_par_NASADEM_Alps.nc"

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

# Compute unit vectors (in ENU coordinates)
lon_in, lat_in = np.meshgrid(lon[sd_in[1]], lat[sd_in[0]])
vec_norm_ecef = functions_cy.surf_norm(lon_in, lat_in)
del lon_in, lat_in
vec_north_ecef = functions_cy.north_dir(x_ecef[sd_in], y_ecef[sd_in],
                                        z_ecef[sd_in], vec_norm_ecef,
                                        ellps=ellps)
del x_ecef, y_ecef, z_ecef
vec_norm_enu = functions_cy.ecef2enu_vec(vec_norm_ecef, lon_or, lat_or)
vec_north_enu = functions_cy.ecef2enu_vec(vec_north_ecef, lon_or, lat_or)
del vec_norm_ecef, vec_north_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = np.hstack((x_enu.reshape(x_enu.size, 1),
                       y_enu.reshape(x_enu.size, 1),
                       z_enu.reshape(x_enu.size, 1))).ravel()
vert_grid = pad_geometry_buffer(vert_grid)

# Compute horizon
horizon(vert_grid, dem_dim_0, dem_dim_1,
        vec_norm_enu, vec_north_enu,
        offset_0, offset_1,
        file_out=file_hori,
        x_axis_val=lon[sd_in[1]].astype(np.float32),
        y_axis_val=lat[sd_in[0]].astype(np.float32),
        x_axis_name="lon", y_axis_name="lat", units="degree",
        dist_search=dist_search, azim_num=azim_num)

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

###############################################################################
# Plot topographic parameters
###############################################################################

# Plot settings
data_plot = {"elevation": dem[sd_in],
             "slope": np.rad2deg(slope),
             "aspect": np.rad2deg(aspect),
             "svf": svf}
cmaps = {"elevation": cm.batlowW, "slope": cm.lajolla,
         "aspect": cm.romaO, "svf": cm.davos}
pos = {"elevation": [0, 0], "slope": [0, 1],
       "aspect":    [3, 0], "svf":   [3, 1]}
titles = {"elevation": "Elevation [m]", "slope": "Slope [degree]",
          "aspect": "Aspect (clockwise from North) [degree]",
          "svf": "Sky View Factor [-]"}

# Plot
geo_crs = ccrs.PlateCarree()
fig = plt.figure(figsize=(14.0, 16.0))
gs = gridspec.GridSpec(5, 2, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       hspace=0.05, wspace=0.05,
                       height_ratios=[1, 0.04, 0.07, 1, 0.04])
for i in list(data_plot.keys()):
    # -------------------------------------------------------------------------
    if i != "aspect":
        levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
            .tick_values(np.percentile(data_plot[i], 5.0),
                         np.percentile(data_plot[i], 95.0))
        cmap = cmaps[i]
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
        ticks = levels
    else:
        levels = np.arange(0.0, 380.0, 20.0)
        cmap = cmaps[i]
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N)
        ticks = np.arange(20.0, 360.0, 40.0)
    ax = plt.subplot(gs[pos[i][0], pos[i][1]], projection=geo_crs)
    plt.pcolormesh(lon[sd_in[1]], lat[sd_in[0]], data_plot[i], cmap=cmap,
                   norm=norm, shading="auto")
    ax.set_aspect("auto")
    if i == "elevation":
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = True
        gl.left_labels = True
        gl.bottom_labels = False
        gl.right_labels = False
        t = plt.text(0.17, 0.95, titles[i], fontsize=13, fontweight="bold",
                     horizontalalignment="center", verticalalignment="center",
                     transform=ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none"))
    else:
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i], fontsize=13, fontweight="bold")
    plt.axis([lon[sd_in[1]].min(), lon[sd_in[1]].max(),
              lat[sd_in[0]].min(), lat[sd_in[0]].max()])
    # -------------------------------------------------------------------------
    ax = plt.subplot(gs[pos[i][0] + 1, pos[i][1]])
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                   orientation="horizontal")
    # -------------------------------------------------------------------------
fig.savefig(path_out + "Topo_slope_SVF.png", dpi=300, bbox_inches="tight")
plt.close(fig)
