# Description: Compute gridded shadow mask and correction factor for downward
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
from netCDF4 import Dataset
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
domain = {"lon_min": 7.65, "lon_max": 8.40,
          "lat_min": 46.3, "lat_max": 46.8}  # domain boundaries [degree]

add = 0.6
domain = {"lon_min": 7.65 - add, "lon_max": 8.40 + add,
          "lat_min": 46.3 - add, "lat_max": 46.8 + add}  # domain boundaries [degree]

# add = 0.5
# domain = {"lon_min": 8.005278 - add, "lon_max": 8.005278 + add,
#           "lat_min": 46.5775 - add, "lat_max": 46.5775 + add}
dist_search = 50.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# Paths and file names
dem_file_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/"\
               + "srtm_5x5/TIFF/srtm_38_03.zip"
path_out = "/Users/csteger/Desktop/Output/"
file_shadow = "shadow_SRTM_Alps.nc"

# -----------------------------------------------------------------------------
# Compute and save topographic parameters
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "gridded_SRTM_Alps_shadow/"
if not os.path.isdir(path_out):
    os.mkdir(path_out)

# # Download and unzip SRTM tile (5 x 5 degree)
# print("Download SRTM tile (5 x 5 degree):")
# hray.download.file(dem_file_url, path_out)
# with zipfile.ZipFile(path_out + "srtm_38_03.zip", "r") as zip_ref:
#     zip_ref.extractall(path_out + "srtm_38_03")
# os.remove(path_out + "srtm_38_03.zip")

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

# Compute rotation matrix (global ENU -> local ENU)
rot_mat = hray.transform.rotation_matrix(vec_north_enu, vec_norm_enu)
# del vec_north_enu, vec_norm_enu

# Compute slope
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))
vec_tilt = hray.topo_param.slope_plane_meth(x_enu[slice_in_a1],
                                            y_enu[slice_in_a1],
                                            z_enu[slice_in_a1],
                                            rot_mat)[1:-1, 1:-1]




# Compute horizon
terrain = hray.shadow.Terrain(1, 1, 5, 5)
dim_in_0, dim_in_1 = vec_tilt.shape[0], vec_tilt.shape[1]

# Compute surface enlargement factor
surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt).sum(axis=2)
# surf_enl_fac[:] = 1.0

# Test plot
plt.figure()
plt.pcolormesh(lon[slice_in[1]], lat[slice_in[0]], surf_enl_fac,
               vmin=0.0, vmax=2.0)
plt.colorbar()


vec_tilt = np.ascontiguousarray(vec_tilt)
print(vert_grid.flags["C_CONTIGUOUS"])
print(vec_tilt.flags["C_CONTIGUOUS"])
print(vec_norm_enu.flags["C_CONTIGUOUS"])
print(surf_enl_fac.flags["C_CONTIGUOUS"])
# -> all passed arrays must be C-contiguous!
# -> passed vectors must be unit vectors!


terrain.initialise(vert_grid, dem_dim_0, dem_dim_1, "grid",
                   offset_0, offset_1, vec_tilt, vec_norm_enu,
                   dim_in_0, dim_in_1, surf_enl_fac)

shadow = np.zeros(vec_tilt.shape[:2], dtype=np.float32)


# Load data/planets
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_or = earth + wgs84.latlon(trans.lat_or, trans.lon_or)

time_dt_beg = dt.datetime(2022, 1, 6, 0, 0, tzinfo=dt.timezone.utc)
time_dt_end = dt.datetime(2022, 1, 7, 0, 0, tzinfo=dt.timezone.utc)
dt_step = dt.timedelta(hours=0.25)
num_ts = int((time_dt_end - time_dt_beg) / dt_step)
ta = [time_dt_beg + dt_step * i for i in range(num_ts)]


ncfile = Dataset(filename=path_out + file_shadow, mode="w")
ncfile.createDimension(dimname="time", size=None)
ncfile.createDimension(dimname="lat", size=dim_in_0)
ncfile.createDimension(dimname="lon", size=dim_in_1)
nc_time = ncfile.createVariable(varname="time", datatype="f",
                                dimensions="time")
nc_lat = ncfile.createVariable(varname="lat", datatype="f",
                               dimensions="lat")
nc_lat[:] = lat[slice_in[0]]
nc_lat.units = "degree"
nc_lon = ncfile.createVariable(varname="lon", datatype="f",
                               dimensions="lon")
nc_lon[:] = lon[slice_in[1]]
nc_lon.units = "degree"
nc_data = ncfile.createVariable(varname="shadow", datatype="f",
                                dimensions=("time", "lat", "lon"))
ncfile.close()

comp_time = []
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

    # terrain.shadow(sun_position, shadow)
    terrain.sw_dir_cor(sun_position, shadow)

    comp_time.append((time.time() - t_beg))

    ncfile = Dataset(filename=path_out + file_shadow, mode="a")
    nc_time = ncfile.variables["time"]
    nc_time[i] = i
    nc_data = ncfile.variables["shadow"]
    nc_data[i, :, :] = shadow
    # nc_data[i, :, :] = (shadow > 0).astype(int)
    ncfile.close()


# Performance plot
plt.figure()
plt.plot(ta, comp_time)
print(sum(comp_time))
print(np.array(comp_time).mean())


ds = xr.open_dataset(path_out + "shadow_SRTM_Alps_surf_enhanc.nc")
sw_dir_cor_enl = ds["shadow"].values
ds.close()

ds = xr.open_dataset(path_out + file_shadow)
sw_dir_cor = ds["shadow"].values
ds.close()

plt.figure()
plt.plot(sw_dir_cor_enl.mean(axis = (1, 2)), color="blue",
                             label="with surf. enl.")
plt.plot(sw_dir_cor.mean(axis = (1, 2)), color="red",
                         label="without surf. enl.")
plt.hlines(1.0, 0.0, 92.0, lw=1.5, color="black")
plt.legend(frameon=False)
