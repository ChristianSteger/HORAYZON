# Description: Compute gridded topographic parameters (slope angle and aspect,
#              horizon and sky view factor) from SRTM data (~90 m) for South
#              Georgia in the South Atlantic Ocean and mask ocean grid cells
#              that have a certain minimal distance to land. Consider Earth's
#              surface curvature.
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
import zipfile
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import Affine
import horayzon as hray
import horayzon.ocean_masking as ocean_masking

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
domain = {"lon_min": -38.65, "lon_max": -35.4,
          "lat_min": -55.2, "lat_max": -53.75}  # domain boundaries [degree]
dist_search = 20.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
azim_num = 120  # number of azimuth sectors [-]
hori_acc = 0.5  # accuracy of horizon computation [degree]
dem_res = 3.0 / 3600.0  # resolution of DEM [degree]

# Paths and file names
dem_file_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/" \
               + "srtm_30x30/TIFF/S60W060.zip"
path_out = "/Users/csteger/Desktop/Output/"
file_hori = "hori_SRTM_South_Georgia.nc"
file_topo_par = "topo_par_SRTM_South_Georgia.nc"

# -----------------------------------------------------------------------------
# Compute and save topographic parameters
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "horizon/gridded_SRTM_South_Georgia/"
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
lon, lat, elevation = hray.load_dem.srtm(file_dem, domain_outer, engine="gdal")
mask_land_dem = (elevation != -32768.0)

# Set ocean grid cells to 0.0 m
elevation[~mask_land_dem] = 0.0

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
vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
del vec_norm_ecef, vec_north_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

# -----------------------------------------------------------------------------
# Compute mask for ocean grid cells
# -----------------------------------------------------------------------------

# Important note:
# compute coastlines for total domain -> land grid cells in the outer domain
# can influence terrain horizon of inner domain

# Get and rasterise GSHHS data
poly_coastlines = ocean_masking.get_gshhs_coastlines(domain_outer)
coastline_GSHHS = unary_union([i for i in poly_coastlines])
d_lon = np.diff(lon).mean()
d_lat = np.diff(lat).mean()
transform = Affine(d_lon, 0.0, lon[0] - d_lon / 2.0,
                   0.0, d_lat, lat[0] - d_lat / 2.0)
mask_land_gshhs = rasterize(coastline_GSHHS.geoms, (len(lat), len(lon)),
                            transform=transform).astype(bool)

# Compute common land-sea-mask, contours and transform coordinates
mask_land = (mask_land_dem | mask_land_gshhs)
contours_latlon = ocean_masking.coastline_contours(lon, lat, mask_land
                                                   .astype(np.uint8))
print("Number of vertices (DEM): " + str(sum([i.shape[0]
                                              for i in contours_latlon])))
pts_latlon = np.vstack(([i for i in contours_latlon]))
h = np.zeros((pts_latlon.shape[0], 1), dtype=np.float32)
coords = hray.transform.lonlat2ecef(pts_latlon[:, 0:1], pts_latlon[:, 1:2], h,
                                    ellps=ellps)
pts_ecef = np.hstack((coords[0], coords[1], coords[2]))

# Compute coastline buffer (only for inner domain)
block_size = 5 * 2 + 1
mask_buffer = ocean_masking.coastline_buffer(
    x_ecef[slice_in], y_ecef[slice_in], z_ecef[slice_in], mask_land[slice_in],
    pts_ecef, lat[slice_in[0]], (dist_search * 1000.0), dem_res, ellps,
    block_size)
frac_ma = ((mask_buffer == 1).sum() / mask_buffer.size * 100.0)
print("Fraction of masked grid cells %.2f" % frac_ma + " %")

# Mask with all types (-1: outside buffer, 0: buffer, 1: land)
mask_type = mask_land[slice_in].astype(np.int32)
mask_type[mask_buffer] = -1

# Plot grid cell types
cmap = mpl.colors.ListedColormap(["gray", "deepskyblue", "sienna"])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
fig = plt.figure(figsize=(11, 6))
plt.pcolormesh(lon[slice_in[1]], lat[slice_in[0]], mask_type, shading="auto",
               cmap=cmap, norm=norm)
plt.axis((lon[slice_in[1]].min(), lon[slice_in[1]].max(),
          lat[slice_in[0]].min(), lat[slice_in[0]].max()))
plt.xlabel("Longitude [degree]")
plt.ylabel("Latitude [degree]")
cbar = plt.colorbar()
cbar.ax.set_yticks([-1.0, 0.0, 1.0])
cbar.ax.set_yticklabels(["outside buffer", "buffer", "land"])
cbar.ax.tick_params(rotation=90)
cbar.ax.yaxis.set_tick_params(pad=10)
fig.savefig(path_out + "Grid_cell_types.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Binary mask
mask = np.ones(vec_norm_enu.shape[:2], dtype=np.uint8)
mask[(mask_type == -1)] = 0  # mask area outside of buffer
# mask[(mask_type < 1)] = 0  # mask ocean entirely

# -----------------------------------------------------------------------------

# Compute horizon
hori, azim = hray.horizon.horizon_gridded(
    vert_grid, dem_dim_0, dem_dim_1,
    vec_norm_enu, vec_north_enu,
    offset_0, offset_1,
    mask=mask, dist_search=dist_search,
    azim_num=azim_num, hori_acc=hori_acc)

# Save horizon to NetCDF file (in Ncview-compatible format)
ds = xr.Dataset(
    coords=dict(
        azim=(["azim"], azim, {"units": "radian"}),
        lat=(["lat"], lat[slice_in[0]], {"units": "degree"}),
        lon=(["lon"], lon[slice_in[1]], {"units": "degree"}),
    ),
    data_vars=dict(
        horizon=(["azim", "lat", "lon"], np.moveaxis(hori, 2, 0),
                   {"units": "radian"})
    )
)
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
        lat=(["lat"], lat[slice_in[0]], {"units": "degree"}),
        lon=(["lon"], lon[slice_in[1]], {"units": "degree"}),
    ),
    data_vars=dict(
        elevation=(["lat", "lon"], elevation[slice_in],
                   {"long_name": "ellipsoidal height", "units": "m"}),
        slope=(["lat", "lon"], slope,
               {"long_name": "slope angle", "units": "radian"}),
        aspect=(["lat", "lon"], aspect,
                {"long_name": "slope aspect (clockwise from North)",
                "units": "radian"}),
        svf=(["lat", "lon"], svf,
             {"long_name": "sky view factor", "units": "-"}),
    )
)
encoding = {i: {"_FillValue": None} for i in
            ("lat", "lon", "elevation", "slope", "aspect")}
ds.to_netcdf(path_out + file_topo_par, encoding=encoding)
