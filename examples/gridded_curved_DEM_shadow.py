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
from horayzon import auxiliary, direction, domain, geoid, horizon, load_dem, topo_param, transform, shadow  # temporary
import horayzon as hray

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
domain = {"lon_min": 7.70, "lon_max": 8.30,
          "lat_min": 46.3, "lat_max": 46.75}  # domain boundaries [degree]

domain = {"lon_min": 7.70 - 0.5, "lon_max": 8.30 + 0.5,
          "lat_min": 46.3 - 0.5, "lat_max": 46.75 + 0.5}  # domain boundaries [degree]

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

terrain.initialise(vert_grid, dem_dim_0, dem_dim_1, "grid",
                   offset_0, offset_1, vec_tilt, dim_in_0, dim_in_1)

shadow = np.zeros(vec_tilt.shape[:2], dtype=np.float32)

sun_position = np.array([0.0, 0.0, 8000.0], dtype=np.float32)

terrain.shootray(sun_position, shadow)
print(shadow[np.isfinite(shadow)].shape)

# Test plot
plt.figure()
plt.pcolormesh(lon[slice_in[1]], lat[slice_in[0]], shadow)
plt.colorbar()