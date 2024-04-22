# Description: Compute gridded shadow map and correction factor for downward
#              direct shortwave radiation from SRTM data (~90 m) for South
#              Georgia in the South Atlantic Ocean. Consider Earth's surface
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
from netCDF4 import Dataset, date2num
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import zipfile
from skyfield.api import load, wgs84
import time
import datetime as dt
import horayzon as hray

mpl.style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
domain = {"lon_min": -38.45, "lon_max": -35.65,
          "lat_min": -55.10, "lat_max": -53.90}
# domain boundaries [degree]
dist_search = 75.0  # search distance for terrain shading [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# Paths and file names
dem_file_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/" \
               + "srtm_30x30/TIFF/S60W060.zip"
path_out = "/Users/csteger/Desktop/Output/"
file_shadow = "shadow_SRTM_South_Georgia.nc"
file_sw_dir_cor = "sw_dir_cor_SRTM_South_Georgia.nc"

# -----------------------------------------------------------------------------
# Prepare data and initialise Terrain class
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "shadow/gridded_SRTM_South_Georgia/"
if not os.path.isdir(path_out):
    os.makedirs(path_out)

# Download and unzip SRTM tile (30 x 30 degree)
print("Download SRTM tile (30 x 30 degree):")
hray.download.file(dem_file_url, path_out)
with zipfile.ZipFile(path_out + "S60W060.zip", "r") as zip_ref:
    zip_ref.extractall(path_out + "S60W060")
os.remove(path_out + "S60W060.zip")

# Load required DEM data (including outer boundary zone)
domain_outer = hray.domain.curved_grid(domain, dist_search, ellps)
file_dem = path_out + "S60W060/cut_s60w060.tif"
lon, lat, elevation = hray.load_dem.srtm(file_dem, domain_outer,
                                         engine="pillow")
# -> GeoTIFF can also be read with GDAL if available (-> faster)
mask_ocean = (elevation == -32768.0)
elevation[mask_ocean] = 0.0

# Compute indices of inner domain
slice_in = (slice(np.where(lat >= domain["lat_max"])[0][-1],
                  np.where(lat <= domain["lat_min"])[0][0] + 1),
            slice(np.where(lon <= domain["lon_min"])[0][-1],
                  np.where(lon >= domain["lon_max"])[0][0] + 1))
offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
print("Inner domain size: " + str(elevation[slice_in].shape))
elevation_ortho = np.ascontiguousarray(elevation[slice_in])
# orthometric height (-> height above mean sea level)

# Compute ellipsoidal heights
elevation += hray.geoid.undulation(lon, lat, geoid="EGM96")  # [m]

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

# Compute rotation matrix (global ENU -> local ENU)
rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                           vec_norm_enu)
del vec_north_enu

# Compute slope (in global ENU coordinates!)
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))
vec_tilt_enu \
    = np.ascontiguousarray(hray.topo_param.slope_plane_meth(
        x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
        rot_mat=rot_mat_glob2loc, output_rot=False)[1:-1, 1:-1])

# Compute surface enlargement factor
surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
# surf_enl_fac[:] = 1.0
print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
      + ", %.3f" % surf_enl_fac.max())

# Initialise terrain
mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
terrain = hray.shadow.Terrain()
dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                   offset_0, offset_1, vec_tilt_enu, vec_norm_enu,
                   surf_enl_fac, mask=mask, elevation=elevation_ortho,
                   refrac_cor=True)

# Load Skyfield data
load.directory = path_out
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_or = earth + wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)
# -> position lies on the surface of the ellipsoid by default

# Create time axis
time_dt_beg = dt.datetime(2022, 6, 21, 10, 30, tzinfo=dt.timezone.utc)
time_dt_end = dt.datetime(2022, 6, 21, 18, 45, tzinfo=dt.timezone.utc)
dt_step = dt.timedelta(hours=0.125)
num_ts = int((time_dt_end - time_dt_beg) / dt_step)
ta = [time_dt_beg + dt_step * i for i in range(num_ts)]

# -----------------------------------------------------------------------------
# Compute shadow map
# -----------------------------------------------------------------------------

# Loop through time steps and save data to NetCDF file
ncfile = Dataset(filename=path_out + file_shadow, mode="w")
ncfile.createDimension(dimname="time", size=None)
ncfile.createDimension(dimname="lat", size=dim_in_0)
ncfile.createDimension(dimname="lon", size=dim_in_1)
nc_time = ncfile.createVariable(varname="time", datatype="f",
                                dimensions="time")
nc_time.units = "hours since 2015-01-01 00:00:00"
nc_time.calendar = "gregorian"
nc_lat = ncfile.createVariable(varname="lat", datatype="f",
                               dimensions="lat")
nc_lat[:] = lat[slice_in[0]]
nc_lat.units = "degree"
nc_lon = ncfile.createVariable(varname="lon", datatype="f",
                               dimensions="lon")
nc_lon[:] = lon[slice_in[1]]
nc_lon.units = "degree"
nc_data = ncfile.createVariable(varname="shadow", datatype="u2",
                                dimensions=("time", "lat", "lon"))
nc_data.long_name = "0: illuminated, 1: self-shaded, 2: terrain-shaded, " \
                    + "3: not considered"
nc_data.units = "-"
ncfile.close()
comp_time_shadow = []
shadow_buffer = np.zeros(vec_tilt_enu.shape[:2], dtype=np.uint8)
for i in range(len(ta)):

    t_beg = time.time()

    ts = load.timescale()
    t = ts.from_datetime(ta[i])
    astrometric = loc_or.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    x = d.m * np.cos(alt.radians) * np.sin(az.radians)
    y = d.m * np.cos(alt.radians) * np.cos(az.radians)
    z = d.m * np.sin(alt.radians)
    sun_position = np.array([x, y, z], dtype=np.float32)

    terrain.shadow(sun_position, shadow_buffer)

    comp_time_shadow.append((time.time() - t_beg))

    ncfile = Dataset(filename=path_out + file_shadow, mode="a")
    nc_time = ncfile.variables["time"]
    nc_time[i] = date2num(ta[i], units=nc_time.units,
                          calendar=nc_time.calendar)
    nc_data = ncfile.variables["shadow"]
    nc_data[i, :, :] = shadow_buffer
    ncfile.close()

# -----------------------------------------------------------------------------
# Compute correction factor for direct downward shortwave radiation
# -----------------------------------------------------------------------------

# Loop through time steps and save data to NetCDF file
ncfile = Dataset(filename=path_out + file_sw_dir_cor, mode="w")
ncfile.createDimension(dimname="time", size=None)
ncfile.createDimension(dimname="lat", size=dim_in_0)
ncfile.createDimension(dimname="lon", size=dim_in_1)
nc_time = ncfile.createVariable(varname="time", datatype="f",
                                dimensions="time")
nc_time.units = "hours since 2015-01-01 00:00:00"
nc_time.calendar = "gregorian"
nc_lat = ncfile.createVariable(varname="lat", datatype="f",
                               dimensions="lat")
nc_lat[:] = lat[slice_in[0]]
nc_lat.units = "degree"
nc_lon = ncfile.createVariable(varname="lon", datatype="f",
                               dimensions="lon")
nc_lon[:] = lon[slice_in[1]]
nc_lon.units = "degree"
nc_data = ncfile.createVariable(varname="sw_dir_cor", datatype="f",
                                dimensions=("time", "lat", "lon"))
nc_data.long_name = "correction factor for direct downward shortwave radiation"
nc_data.units = "-"
ncfile.close()
comp_time_sw_dir_cor = []
sw_dir_cor_buffer = np.zeros(vec_tilt_enu.shape[:2], dtype=np.float32)
for i in range(len(ta)):

    t_beg = time.time()

    ts = load.timescale()
    t = ts.from_datetime(ta[i])
    astrometric = loc_or.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    x = d.m * np.cos(alt.radians) * np.sin(az.radians)
    y = d.m * np.cos(alt.radians) * np.cos(az.radians)
    z = d.m * np.sin(alt.radians)
    sun_position = np.array([x, y, z], dtype=np.float32)

    terrain.sw_dir_cor(sun_position, sw_dir_cor_buffer)

    comp_time_sw_dir_cor.append((time.time() - t_beg))

    ncfile = Dataset(filename=path_out + file_sw_dir_cor, mode="a")
    nc_time = ncfile.variables["time"]
    nc_time[i] = date2num(ta[i], units=nc_time.units,
                          calendar=nc_time.calendar)
    nc_data = ncfile.variables["sw_dir_cor"]
    nc_data[i, :, :] = sw_dir_cor_buffer
    ncfile.close()

# -----------------------------------------------------------------------------
# Analyse performance and spatial mean of correction factor
# -----------------------------------------------------------------------------

# Performance plot
fig = plt.figure(figsize=(10, 6))
plt.plot(ta, comp_time_shadow, lw=1.5, color="blue",
         label="Shadow (mean: %.2f" % np.array(comp_time_shadow).mean() + ")")
plt.plot(ta, comp_time_sw_dir_cor, lw=1.5, color="red",
         label="SW_dir_cor (mean: %.2f"
               % np.array(comp_time_sw_dir_cor).mean() + ")")
plt.ylabel("Computing time [seconds]")
plt.legend(loc="upper center", frameon=False, fontsize=11)
plt.title("Terrain size (" + str(dim_in_0) + " x " + str(dim_in_1) + ")",
          fontweight="bold", fontsize=12)
fig.savefig(path_out + "Performance.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Check spatial mean of correction factor
ds = xr.open_dataset(path_out + file_sw_dir_cor)
sw_dir_cor = ds["sw_dir_cor"].values
ds.close()

fig = plt.figure(figsize=(10, 6))
for i in (0.0, 1.0):
    plt.hlines(i, ta[0], ta[-1], lw=1.5, ls="--", color="black")
plt.plot(ta, sw_dir_cor.mean(axis=(1, 2)), lw=1.5, color="blue")
plt.ylim([-0.1, 1.1])
plt.ylabel("Spatial mean of correction factor [-]")
fig.savefig(path_out + "SW_dir_cor_spatial_mean.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

del sw_dir_cor

# -----------------------------------------------------------------------------
# Plot elevation and shortwave correction factor fo specific time step
# -----------------------------------------------------------------------------

# Load data
ind = 10  # select time step
ds = xr.open_dataset(path_out + file_sw_dir_cor)
sw_dir_cor = ds["sw_dir_cor"][ind, :, :].values
ds.close()

# Plot
ax_lim = (lon[slice_in[1]].min(), lon[slice_in[1]].max(),
          lat[slice_in[0]].min(), lat[slice_in[0]].max())
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       hspace=0.05, wspace=0.05, width_ratios=[1.0, 0.027])
ax = plt.subplot(gs[0, 0])
ax.set_facecolor(plt.get_cmap("terrain")(0.15)[:3] + (0.25,))
levels = np.arange(0.0, 2600.0, 200.0)
cmap = colors.LinearSegmentedColormap.from_list(
    "terrain", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 100)))
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
data_plot = np.ma.masked_where(mask_ocean[slice_in], elevation_ortho)
plt.pcolormesh(lon[slice_in[1]], lat[slice_in[0]], data_plot,
               cmap=cmap, norm=norm)
x_ticks = np.arange(-38.0, -35.5, 0.5)
plt.xticks(x_ticks, ["" for i in x_ticks])
y_ticks = np.arange(-55.0, -53.9, 0.2)
plt.yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
plt.axis(ax_lim)
ax = plt.subplot(gs[0, 1])
mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
plt.ylabel("Elevation [m a.s.l.]", labelpad=10.0)
ax = plt.subplot(gs[1, 0])
levels = np.arange(0.0, 5.25, 0.25)
ticks = np.arange(0.0, 5.5, 0.5)
cmap = plt.get_cmap("viridis")
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
plt.pcolormesh(lon[slice_in[1]], lat[slice_in[0]], sw_dir_cor,
               cmap=cmap, norm=norm)
plt.xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
plt.yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
plt.axis(ax_lim)
txt = ta[ind].strftime("%Y-%m-%d %H:%M:%S") + " UTC"
t = plt.text(0.835, 0.935, txt, fontsize=11, fontweight="bold",
             horizontalalignment="center", verticalalignment="center",
             transform=ax.transAxes)
t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
ts = load.timescale()
astrometric = loc_or.at(ts.from_datetime(ta[ind])).observe(sun)
alt, az, d = astrometric.apparent().altaz()
txt = "Mean solar elevation angle: %.1f" % alt.degrees + "$^{\circ}$"
t = plt.text(0.21, 0.06, txt, fontsize=11, fontweight="bold",
             horizontalalignment="center", verticalalignment="center",
             transform=ax.transAxes)
t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
ax = plt.subplot(gs[1, 1])
mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=ticks,
                          orientation="vertical")
plt.ylabel("${\downarrow}SW_{dir}$ correction factor [-]", labelpad=10.0)
fig.savefig(path_out + "Elevation_sw_dir_cor.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
