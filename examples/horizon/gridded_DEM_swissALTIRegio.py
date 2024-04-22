# Description: Compute gridded topographic parameters (slope angle and aspect,
#              horizon and sky view factor) from swissALTIRegio (10 m) for an
#              example region in Switzerland. Consider Earth's surface
#              curvature. Note: entire swissALTIRegio data set (12.25 GB) is 
#              downloaded, which may required siginficant time.
#
# Source of applied DEM data: 
#   https://www.swisstopo.admin.ch/en/height-model-swissaltiregio
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import time
import horayzon as hray

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
domain = {"x_min": 2_633_000 + 1000, "x_max": 2_658_000 - 1000,
          "y_min": 1_103_000 + 1000, "y_max": 1_122_000 - 1000}
# domain boundaries [metre]
dist_search = 80.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
azim_num = 180  # 360  # number of azimuth sampling directions [-]
# -> Note: above configuration works on a machine with 16 GB RAM

# Paths and file names
dem_file_url = "https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/" \
               + "swissaltiregio/swissaltiregio_2056_5728.tif"
path_out = "/Users/csteger/Desktop/Output/"
file_hori = "hori_swissALTIRegio_Switzerland.nc"
file_topo_par = "topo_par_swissALTIRegio_Switzerland.nc"

# -----------------------------------------------------------------------------
# Compute and save topographic parameters
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "horizon/gridded_swissALTIRegio_Switzerland/"
if not os.path.isdir(path_out):
    os.makedirs(path_out)

# Download swissALTIRegio data
print("Download swissALTIRegio data:")
hray.download.file(dem_file_url, path_out)

# Load required DEM data (including outer boundary zone)
domain_outer = hray.domain.planar_grid(domain, dist_search)
file_dem = path_out + "swissaltiregio_2056_5728.tif"
x, y, elevation_ch = hray.load_dem.swissaltiregio(file_dem, domain_outer)

# Compute indices of inner domain
slice_in = (slice(np.where(y >= domain["y_max"])[0][-1],
                  np.where(y <= domain["y_min"])[0][0] + 1),
            slice(np.where(x <= domain["x_min"])[0][-1],
                  np.where(x >= domain["x_max"])[0][0] + 1))
offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
print("Inner domain size: " + str(elevation_ch[slice_in].shape))

# Compute geodetic coordinates
lon, lat, elevation = hray.transform.swiss2wgs(*np.meshgrid(x, y),
                                               elevation_ch)
del elevation_ch

# Compute ECEF coordinates
x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(lon, lat, elevation,
                                                    ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape
lon_in = lon[slice_in].copy()
lat_in = lat[slice_in].copy()
time.sleep(0.5)
del lon, lat

# Compute ENU coordinates
trans_ecef2enu = hray.transform.TransformerEcef2enu(
    lon_or=lon_in.mean(), lat_or=lat_in.mean(), ellps=ellps)
x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

# Compute unit vectors (up and north) in ENU coordinates for inner domain
vec_norm_ecef = hray.direction.surf_norm(lon_in, lat_in)
vec_north_ecef = hray.direction.north_dir(x_ecef[slice_in], y_ecef[slice_in],
                                          z_ecef[slice_in], vec_norm_ecef,
                                          ellps=ellps)
time.sleep(0.5)
del x_ecef, y_ecef, z_ecef
vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
del vec_norm_ecef, vec_north_ecef

# Compute rotation matrix (global ENU -> local ENU)
rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                           vec_norm_enu)

# Compute slope
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))
vec_tilt = hray.topo_param.slope_plane_meth(x_enu[slice_in_a1],
                                            y_enu[slice_in_a1],
                                            z_enu[slice_in_a1],
                                            rot_mat=rot_mat_glob2loc,
                                            output_rot=True)[1:-1, 1:-1]
del rot_mat_glob2loc

# Compute slope angle and aspect
slope = np.arccos(vec_tilt[:, :, 2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(vec_tilt[:, :, 1],
                                  vec_tilt[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

# Merge vertex coordinates and pad geometry buffer
vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
time.sleep(0.5)
del x_enu, y_enu, z_enu

# Compute horizon
hori, azim = hray.horizon.horizon_gridded(
    vert_grid, dem_dim_0, dem_dim_1,
    vec_norm_enu, vec_north_enu,
    offset_0, offset_1,
    dist_search=dist_search,
    azim_num=azim_num)
del vec_north_enu, vec_norm_enu

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

# Compute Sky View Factor
svf = hray.topo_param.sky_view_factor(azim, hori, vec_tilt)

# Save topographic parameters to NetCDF file
ds = xr.Dataset(
    coords=dict(
        y=(["y"], y[slice_in[0]], {"units": "m"}),
        x=(["x"], x[slice_in[1]], {"units": "m"}),
    ),
    data_vars=dict(
        elevation=(["y", "x"], elevation[slice_in],
                   {"long_name": "ellipsoidal height", "units": "m"}),
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

# -----------------------------------------------------------------------------
# Plot topographic parameters
# -----------------------------------------------------------------------------

# Plot settings
data_plot = {"elevation": elevation[slice_in],
             "slope": np.rad2deg(slope),
             "aspect": np.rad2deg(aspect),
             "svf": svf}
cmaps = {"elevation": plt.get_cmap("gist_earth"),
         "slope": plt.get_cmap("YlOrBr"),
         "aspect": plt.get_cmap("twilight"),
         "svf": plt.get_cmap("YlGnBu_r")}
pos = {"elevation": [0, 0], "slope": [0, 1],
       "aspect":    [3, 0], "svf":   [3, 1]}
titles = {"elevation": "Elevation [m]", "slope": "Slope [degree]",
          "aspect": "Aspect (clockwise from North) [degree]",
          "svf": "Sky View Factor [-]"}

# Plot
fig = plt.figure(figsize=(20.0, 15.0))
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
    ax = plt.subplot(gs[pos[i][0], pos[i][1]])
    plt.pcolormesh(x[slice_in[1]], y[slice_in[0]], data_plot[i], cmap=cmap,
                   norm=norm, shading="auto")
    ax.set_aspect("auto")
    plt.xticks([])
    plt.yticks([])
    plt.title(titles[i], fontsize=12, fontweight="bold")
    plt.axis((x[slice_in[1]].min(), x[slice_in[1]].max(),
              y[slice_in[0]].min(), y[slice_in[0]].max()))
    # -------------------------------------------------------------------------
    ax = plt.subplot(gs[pos[i][0] + 1, pos[i][1]])
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                   orientation="horizontal")
    # -------------------------------------------------------------------------
fig.savefig(path_out + "Topo_slope_SVF.png", dpi=300, bbox_inches="tight")
plt.close(fig)
