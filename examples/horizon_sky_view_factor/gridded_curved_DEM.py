# Description: Compute gridded topographic parameters (slope angle and aspect,
#              horizon and sky view factor) from SRTM data (~90 m) for an
#              example region in the European Alps. Consider Earth's surface
#              curvature.
#
# Source of applied DEM data: https://srtm.csi.cgiar.org
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import sys  # ------------------------------------------------------- temporary
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cmcrameri import cm
import zipfile
sys.path.append("/Users/csteger/Downloads/HORAYZON/")  # ------------ temporary
from horayzon import auxiliary, direction, domain, geoid, horizon, load_dem, topo_param, transform  # temporary
import horayzon as hray

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
domain = {"lon_min": 7.70, "lon_max": 8.30,
          "lat_min": 46.3, "lat_max": 46.75}  # domain boundaries [degree]
dist_search = 50.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
azim_num = 360  # number of azimuth sampling directions [-]

# Paths and file names
dem_file_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/"\
               + "srtm_5x5/TIFF/srtm_38_03.zip"
path_out = "/Users/csteger/Desktop/Output/"
file_hori = "hori_SRTM_Alps.nc"
file_topo_par = "topo_par_SRTM_Alps.nc"

# -----------------------------------------------------------------------------
# Compute and save topographic parameters
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "gridded_SRTM_Alps/"
if not os.path.isdir(path_out):
    os.mkdir(path_out)

# Download and unzip SRTM tile (5 x 5 degree)
print("Download SRTM tile (5 x 5 degree):")
hray.download.file(dem_file_url, path_out)
with zipfile.ZipFile(path_out + "srtm_38_03.zip", "r") as zip_ref:
    zip_ref.extractall(path_out + "srtm_38_03")
os.remove(path_out + "srtm_38_03.zip")

# Load required DEM data (including outer boundary zone)
domain_outer = hray.domain.curved_grid(domain, dist_search, ellps)
file_dem = path_out + "srtm_38_03/srtm_38_03.tif"
lon, lat, elevation = hray.load_dem.srtm(file_dem, domain_outer, engine="gdal")
# -> GeoTIFF can also be read with Pillow in case GDAL is not available!

# Compute ellipsoidal heights
elevation += hray.geoid.undulation(lon, lat, geoid="EGM96")  # [m]

# Compute indices of inner domain
slice_in = (slice(np.where(lat >= domain["lat_max"])[0][-1],
                  np.where(lat <= domain["lat_min"])[0][0] + 1),
            slice(np.where(lon <= domain["lon_min"])[0][-1],
                  np.where(lon >= domain["lon_max"])[0][0] + 1))
offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
print("Inner domain size: " + str(elevation[slice_in].shape))

# Compute ECEF coordinates
x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef_1d(lon, lat, elevation,
                                                       ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape

# Compute ENU coordinates
trans = hray.transform.TransformerEcef2enu(lon, lat, x_ecef, y_ecef, z_ecef)
x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef, trans)

# Compute unit vectors (up and north) in ENU coordinates for inner domain
vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(lon[slice_in[1]],
                                                      lat[slice_in[0]]))
vec_north_ecef = hray.direction.north_dir(x_ecef[slice_in], y_ecef[slice_in],
                                          z_ecef[slice_in], vec_norm_ecef,
                                          ellps=ellps)
del x_ecef, y_ecef, z_ecef
vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans)
vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans)
del vec_norm_ecef, vec_north_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

# Compute horizon
hray.horizon.horizon_gridded(vert_grid, dem_dim_0, dem_dim_1,
                             vec_norm_enu, vec_north_enu,
                             offset_0, offset_1,
                             file_out=path_out + file_hori,
                             x_axis_val=lon[slice_in[1]].astype(np.float32),
                             y_axis_val=lat[slice_in[0]].astype(np.float32),
                             x_axis_name="lon", y_axis_name="lat",
                             units="degree", dist_search=dist_search,
                             azim_num=azim_num)

# Load horizon data
ds = xr.open_dataset(path_out + file_hori)
hori = ds["horizon"].values
azim = ds["azim"].values
ds.close()

# Swap coordinate axes (-> make viewable with ncview)
ds_ncview = ds.transpose("azim", "lat", "lon")
ds_ncview.to_netcdf(path_out + file_hori[:-3] + "_ncview.nc")

# Compute rotation matrix (global ENU -> local ENU)
rot_mat = hray.transform.rotation_matrix(vec_north_enu, vec_norm_enu)
del vec_north_enu, vec_norm_enu

# Compute slope
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))
vec_tilt = hray.topo_param.slope_plane_meth(x_enu[slice_in_a1],
                                            y_enu[slice_in_a1],
                                            z_enu[slice_in_a1],
                                            rot_mat)[1:-1, 1:-1]
del rot_mat
del x_enu, y_enu, z_enu

# Compute Sky View Factor
svf = hray.topo_param.sky_view_factor(azim, hori, vec_tilt)

# Compute slope angle and aspect
slope = np.arccos(vec_tilt[:, :, 2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(vec_tilt[:, :, 1],
                                  vec_tilt[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

# Save topographic parameters to NetCDF file
ncfile = Dataset(filename=path_out + file_topo_par, mode="w")
ncfile.createDimension(dimname="lat", size=svf.shape[0])
ncfile.createDimension(dimname="lon", size=svf.shape[1])
nc_lat = ncfile.createVariable(varname="lat", datatype="f",
                               dimensions="lat")
nc_lat[:] = lat[slice_in[0]]
nc_lat.units = "degree"
nc_lon = ncfile.createVariable(varname="lon", datatype="f",
                               dimensions="lon")
nc_lon[:] = lon[slice_in[1]]
nc_lon.units = "degree"
nc_data = ncfile.createVariable(varname="elevation", datatype="f",
                                dimensions=("lat", "lon"))
nc_data[:] = elevation[slice_in]
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

# -----------------------------------------------------------------------------
# Plot topographic parameters
# -----------------------------------------------------------------------------

# Plot settings
data_plot = {"elevation": elevation[slice_in],
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
    plt.pcolormesh(lon[slice_in[1]], lat[slice_in[0]], data_plot[i], cmap=cmap,
                   norm=norm, shading="auto")
    ax.set_aspect("auto")
    if i == "elevation":
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = True
        gl.left_labels = True
        gl.bottom_labels = False
        gl.right_labels = False
        t = plt.text(0.17, 0.95, titles[i], fontsize=12, fontweight="bold",
                     horizontalalignment="center", verticalalignment="center",
                     transform=ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
    else:
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i], fontsize=12, fontweight="bold")
    plt.axis([lon[slice_in[1]].min(), lon[slice_in[1]].max(),
              lat[slice_in[0]].min(), lat[slice_in[0]].max()])
    # -------------------------------------------------------------------------
    ax = plt.subplot(gs[pos[i][0] + 1, pos[i][1]])
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                   orientation="horizontal")
    # -------------------------------------------------------------------------
fig.savefig(path_out + "Topo_slope_SVF.png", dpi=300, bbox_inches="tight")
plt.close(fig)
