# Description: Compute gridded shadow correction factor for downward direct
#              shortwave radiation from REMA data (~100 m) for an example
#              region in Antarctica (mask ocean). Consider Earth's surface
#              curvature.
#
# Source of applied DEM data: https://www.pgc.umn.edu/data/rema/
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import sys  # ------------------------------------------------------- temporary
import numpy as np
from netCDF4 import Dataset, date2num
from skyfield.api import load, wgs84
import time
import datetime as dt
from pyproj import CRS, Transformer
sys.path.append("/Users/csteger/Downloads/HORAYZON/")  # ------------ temporary
from horayzon import auxiliary, direction, domain, geoid, horizon, load_dem, topo_param, transform, shadow  # temporary
import horayzon as hray

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
domain = {"x_min": -2530000.0, "x_max": -2300000.0,
          "y_min": 1450000.0, "y_max": 1700000.0}
# domain boundaries [degree]
dist_search = 65.0  # search distance for terrain shading [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# Paths and file names
dem_file_url = "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/" \
               + "v1.1/100m/REMA_100m_peninsula_dem_filled.tif"
path_out = "/Users/csteger/Desktop/Output/"
file_sw_dir_cor = "sw_dir_cor_REMA_Antarctica.nc"

# -----------------------------------------------------------------------------
# Prepare data and initialise Terrain class
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "shadow/gridded_REMA_Antarctica/"
if not os.path.isdir(path_out):
    os.makedirs(path_out)

# Download REMA tile for Antarctic Peninsula
print("Download REMA tile for Antarctic Peninsula:")
hray.download.file(dem_file_url, path_out)

# Load required DEM data (including outer boundary zone)
domain_outer = hray.domain.planar_grid(domain, dist_search)
file_dem = path_out + "REMA_100m_peninsula_dem_filled.tif"
x, y, elevation = hray.load_dem.rema(file_dem, domain_outer, engine="gdal")
# -> GeoTIFF can also be read with Pillow in case GDAL is not available!
# -> elevation is referenced to WGS84 ellipsoid

# Set ocean grid cells to 0.0 m
mask_ocean = (elevation == -9999.0)
elevation[mask_ocean] = 0.0

# Compute indices of inner domain
slice_in = (slice(np.where(y >= domain["y_max"])[0][-1],
                  np.where(y <= domain["y_min"])[0][0] + 1),
            slice(np.where(x <= domain["x_min"])[0][-1],
                  np.where(x >= domain["x_max"])[0][0] + 1))
offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
print("Inner domain size: " + str(elevation[slice_in].shape))
elevation_in = np.ascontiguousarray(elevation[slice_in])
# -> assume that orthometric height ~ ellipsoidal height

# Compute geodetic coordinates
crs_proj = CRS.from_epsg(3031)
crs_wgs84 = CRS.from_epsg(4326)
transformer = Transformer.from_crs(crs_proj, crs_wgs84, always_xy=True)
lon, lat = transformer.transform(*np.meshgrid(x, y))

# Compute ECEF coordinates
x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(lon, lat, elevation,
                                                    ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape

# Compute ENU coordinates
ind_or = (int(lon.shape[0] / 2), int(lon.shape[1] / 2))
trans_ecef2enu = hray.transform.TransformerEcef2enu(
    lon_or=lon[ind_or], lat_or=lat[ind_or], ellps=ellps)
x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

# Compute unit vectors (up and north) in ENU coordinates for inner domain
vec_norm_ecef = hray.direction.surf_norm(lon[slice_in], lat[slice_in])
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
vec_tilt_enu = \
    np.ascontiguousarray(hray.topo_param.slope_vector_meth(
        x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
        rot_mat=rot_mat_glob2loc, output_rot=False)[1:-1, 1:-1])

# Compute surface enlargement factor
surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
      + ", %.3f" % surf_enl_fac.max())

# Initialise terrain
mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
mask[mask_ocean[slice_in]] = 0  # mask ocean grid cells
terrain = hray.shadow.Terrain()
dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                   offset_0, offset_1, vec_tilt_enu, vec_norm_enu,
                   surf_enl_fac, mask=mask, elevation=elevation_in,
                   refrac_cor=True)

# Load Skyfield data
load.directory = path_out
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_or = earth + wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)
# -> position lies on the surface of the ellipsoid by default

# Create time axis
time_dt_beg = dt.datetime(2022, 1, 6, 0, 0, tzinfo=dt.timezone.utc)
time_dt_end = dt.datetime(2022, 1, 7, 0, 0, tzinfo=dt.timezone.utc)
dt_step = dt.timedelta(hours=0.125)
num_ts = int((time_dt_end - time_dt_beg) / dt_step)
ta = [time_dt_beg + dt_step * i for i in range(num_ts)]

# -----------------------------------------------------------------------------
# Compute correction factor for direct downward shortwave radiation
# -----------------------------------------------------------------------------

# Loop through time steps and save data to NetCDF file
ncfile = Dataset(filename=path_out + file_sw_dir_cor, mode="w")
ncfile.createDimension(dimname="time", size=None)
ncfile.createDimension(dimname="y", size=dim_in_0)
ncfile.createDimension(dimname="x", size=dim_in_1)
nc_time = ncfile.createVariable(varname="time", datatype="f",
                                dimensions="time")
nc_time.units = "hours since 2015-01-01 00:00:00"
nc_time.calendar = "gregorian"
nc_y = ncfile.createVariable(varname="y", datatype="f", dimensions="y")
nc_y[:] = y[slice_in[0]]
nc_y.units = "metre"
nc_x = ncfile.createVariable(varname="x", datatype="f", dimensions="x")
nc_x[:] = x[slice_in[1]]
nc_x.units = "metre"
nc_data = ncfile.createVariable(varname="sw_dir_cor", datatype="f",
                                dimensions=("time", "y", "x"))
nc_data.long_name = "correction factor for direct downward shortwave radiation"
nc_data.units = "-"
ncfile.close()
comp_time_shadow = []
sw_dir_cor = np.zeros(vec_tilt_enu.shape[:2], dtype=np.float32)
for i in range(len(ta)):

    t_beg = time.time()

    ts = load.timescale()
    t = ts.from_datetime(ta[i])
    astrometric = loc_or.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    x_sun = d.m * np.cos(alt.radians) * np.sin(az.radians)
    y_sun = d.m * np.cos(alt.radians) * np.cos(az.radians)
    z_sun = d.m * np.sin(alt.radians)
    sun_position = np.array([x_sun, y_sun, z_sun], dtype=np.float32)

    terrain.sw_dir_cor(sun_position, sw_dir_cor)

    comp_time_shadow.append((time.time() - t_beg))

    ncfile = Dataset(filename=path_out + file_sw_dir_cor, mode="a")
    nc_time = ncfile.variables["time"]
    nc_time[i] = date2num(ta[i], units=nc_time.units,
                          calendar=nc_time.calendar)
    nc_data = ncfile.variables["sw_dir_cor"]
    nc_data[i, :, :] = sw_dir_cor
    ncfile.close()

time_tot = np.array(comp_time_shadow).sum()
print("Elapsed time (total / per time step): " + "%.2f" % time_tot
      + " , %.2f" % (time_tot / len(ta)) + " s")

# -----------------------------------------------------------------------------
# Append further fields to NetCDF file
# -----------------------------------------------------------------------------

# Compute slope (in local ENU coordinates!)
vec_tilt_enu_loc = \
    np.ascontiguousarray(hray.topo_param.slope_vector_meth(
        x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
        rot_mat=rot_mat_glob2loc, output_rot=True)[1:-1, 1:-1])

# Compute slope angle and aspect (in local ENU coordinates)
slope = np.arccos(vec_tilt_enu_loc[:, :, 2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(vec_tilt_enu_loc[:, :, 1],
                                  vec_tilt_enu_loc[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

# Append further fields to NetCDF file
fields = {"elevation": {"array": elevation_in, "datatype": "f",
                        "long_name": "ellipsoidal height", "units": "m"},
          "slope": {"array": np.rad2deg(slope), "datatype": "f",
                    "long_name": "slope", "units": "degree"},
          "aspect": {"array": np.rad2deg(aspect), "datatype": "f",
                     "long_name": "aspect (measured clockwise from North)",
                     "units": "degree"},
          "surf_enl_fac": {"array": surf_enl_fac, "datatype": "f",
                           "long_name": "surface enlargement factor",
                           "units": "-"}}
ncfile = Dataset(filename=path_out + file_sw_dir_cor, mode="a")
for i in fields:
    nc_data = ncfile.createVariable(varname=i, datatype=fields[i]["datatype"],
                                    dimensions=("y", "x"))
    nc_data[:] = fields[i]["array"]
    nc_data.long_name = fields[i]["long_name"]
    nc_data.units = fields[i]["units"]
ncfile.close()
