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
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import zipfile
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
dem_file_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/" \
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
path_out += "horizon/gridded_SRTM_Alps/"
if not os.path.isdir(path_out):
    os.makedirs(path_out)

# Download and unzip SRTM tile (5 x 5 degree)
print("Download SRTM tile (5 x 5 degree):")
hray.download.file(dem_file_url, path_out)
with zipfile.ZipFile(path_out + "srtm_38_03.zip", "r") as zip_ref:
    zip_ref.extractall(path_out + "srtm_38_03")
os.remove(path_out + "srtm_38_03.zip")

# Load required DEM data (including outer boundary zone)
domain_outer = hray.domain.curved_grid(domain, dist_search, ellps)
file_dem = path_out + "srtm_38_03/srtm_38_03.tif"
lon, lat, elevation = hray.load_dem.srtm(file_dem, domain_outer,
                                         engine="pillow")
# -> GeoTIFF can also be read with GDAL if available (-> faster)

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
x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(lon, lat),
                                                    elevation, ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape

# Compute ENU coordinates
trans_ecef2enu = hray.transform.TransformerEcef2enu(
    lon_or=lon[int(len(lon) / 2)], lat_or=lat[int(len(lat) / 2)], ellps=ellps)
x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

# Compute unit vectors (up and north) in ENU coordinates for inner domain
vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(lon[slice_in[1]],
                                                      lat[slice_in[0]]))
vec_north_ecef = hray.direction.north_dir(x_ecef[slice_in], y_ecef[slice_in],
                                          z_ecef[slice_in], vec_norm_ecef,
                                          ellps=ellps)
del x_ecef, y_ecef, z_ecef
vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
del vec_norm_ecef, vec_north_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

# Compute horizon
hori, azim = hray.horizon.horizon_gridded(
    vert_grid, dem_dim_0, dem_dim_1,
    vec_norm_enu, vec_north_enu,
    offset_0, offset_1,
    dist_search=dist_search,
    azim_num=azim_num)

# Save horizon to NetCDF file (in Ncview-compatible format)
ds = xr.Dataset({
    "horizon": xr.DataArray(
        data=np.moveaxis(hori, 2, 0),
        dims=["azim", "lat", "lon"],
        coords={
            "azim": azim,
            "lat": lat[slice_in[0]],
            "lon": lon[slice_in[1]]},
        attrs={
            "units": "radian"
        })
    })
ds["azim"].attrs["units"] = "radian"
ds["lat"].attrs["units"] = "degree"
ds["lon"].attrs["units"] = "degree"
encoding = {i: {"_FillValue": None} for i in ("azim", "lat", "lon")}
ds.to_netcdf(path_out + file_hori, encoding=encoding)

# Compute rotation matrix (global ENU -> local ENU)
rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                           vec_norm_enu)
del vec_north_enu, vec_norm_enu

# Compute slope
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))
vec_tilt = hray.topo_param.slope_plane_meth(x_enu[slice_in_a1],
                                            y_enu[slice_in_a1],
                                            z_enu[slice_in_a1],
                                            rot_mat=rot_mat_glob2loc,
                                            output_rot=True)[1:-1, 1:-1]
del rot_mat_glob2loc
del x_enu, y_enu, z_enu

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
        lat=(["lat"], lat[slice_in[0]]),
        lon=(["lon"], lon[slice_in[1]]),
    ),
    data_vars=dict(
        elevation=(["lat", "lon"], elevation[slice_in],
                   {"long_name": "ellipsoidal height",
                    "units": "m"}),
        slope=(["lat", "lon"], slope,
               {"long_name": "slope angle",
                "units": "radian"}),
        aspect=(["lat", "lon"], aspect,
                {"long_name": "slope aspect (clockwise from North)",
                "units": "radian"}),
        svf=(["lat", "lon"], svf,
             {"long_name": "sky view factor",
              "units": "-"}),
    )
)
ds["lat"].attrs["units"] = "degree"
ds["lon"].attrs["units"] = "degree"
encoding = {i: {"_FillValue": None} for i in
            ("lat", "lon", "elevation", "slope", "aspect")}
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
    ax = plt.subplot(gs[pos[i][0], pos[i][1]])
    plt.pcolormesh(lon[slice_in[1]], lat[slice_in[0]], data_plot[i], cmap=cmap,
                   norm=norm, shading="auto")
    ax.set_aspect("auto")
    if i == "elevation":
        ax.xaxis.tick_top()
        plt.grid(True, ls=":")
        x_ticks = np.arange(7.7, 8.3, 0.1)
        plt.xticks(x_ticks, ["%.1f" % i + r"$^{\circ}$E" for i in x_ticks])
        y_ticks = np.arange(46.3, 46.8, 0.1)
        plt.yticks(y_ticks, ["%.1f" % i + r"$^{\circ}$N" for i in y_ticks])
        t = plt.text(0.17, 0.95, titles[i], fontsize=12, fontweight="bold",
                     horizontalalignment="center", verticalalignment="center",
                     transform=ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
    else:
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i], fontsize=12, fontweight="bold")
    plt.axis((lon[slice_in[1]].min(), lon[slice_in[1]].max(),
              lat[slice_in[0]].min(), lat[slice_in[0]].max()))
    # -------------------------------------------------------------------------
    ax = plt.subplot(gs[pos[i][0] + 1, pos[i][1]])
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                   orientation="horizontal")
    # -------------------------------------------------------------------------
fig.savefig(path_out + "Topo_slope_SVF.png", dpi=300, bbox_inches="tight")
plt.close(fig)
