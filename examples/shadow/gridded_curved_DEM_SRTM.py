# Description: Compute gridded shadow map and correction factor for downward
#              direct shortwave radiation from SRTM data (~90 m) for an
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
from netCDF4 import Dataset, date2num
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cmcrameri import cm
import zipfile
from skyfield.api import load, wgs84
import time
import datetime as dt
sys.path.append("/Users/csteger/Downloads/HORAYZON/")  # ------------ temporary
from horayzon import auxiliary, direction, domain, geoid, horizon, load_dem, topo_param, transform, shadow  # temporary
import horayzon as hray

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
add = 0.5
domain = {"lon_min": 7.65 - add, "lon_max": 8.40 + add,
          "lat_min": 46.3 - add, "lat_max": 46.8 + add}
# domain boundaries [degree]
dist_search = 75.0  # search distance for terrain shading [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# Paths and file names
dem_file_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/" \
               + "srtm_5x5/TIFF/srtm_38_03.zip"
path_out = "/Users/csteger/Desktop/Output/"
file_shadow = "shadow_SRTM_Alps.nc"
file_sw_dir_cor = "sw_dir_cor_SRTM_Alps.nc"

# -----------------------------------------------------------------------------
# Prepare data and initialise Terrain class
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "shadow/gridded_SRTM_Alps/"
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
lon, lat, elevation = hray.load_dem.srtm(file_dem, domain_outer, engine="gdal")
# -> GeoTIFF can also be read with Pillow in case GDAL is not available!

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

# Compute slope
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
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_or = earth + wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)

# Create time axis
time_dt_beg = dt.datetime(2022, 1, 6, 0, 0, tzinfo=dt.timezone.utc)
time_dt_end = dt.datetime(2022, 1, 7, 0, 0, tzinfo=dt.timezone.utc)
dt_step = dt.timedelta(hours=0.125)  # 0.25
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

# Performance plot
fig = plt.figure(figsize=(10, 6))
plt.plot(ta, comp_time_shadow, lw=1.5, color="blue",
         label="Shadow (mean: %.2f" % np.array(comp_time_shadow).mean() + ")")
plt.plot(ta, comp_time_sw_dir_cor, lw=1.5, color="red",
         label="SW_dir_cor (mean: %.2f"
               % np.array(comp_time_sw_dir_cor).mean() + ")")
plt.ylabel("Computing time [seconds]")
plt.legend(loc="upper right", frameon=False, fontsize=11)
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
