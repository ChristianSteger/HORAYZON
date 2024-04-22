# Description: Compute gridded correction factor for downward direct shortwave
#              radiation from artifical topography.
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import horayzon as hray

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
dom_width_h = np.array([10000, 20000, 10000], dtype=np.float32)  # [m]
dx, dy = 100, 100  # [m]

# Paths and file names
path_out = "/Users/csteger/Desktop/Output/"
file_sw_dir_cor = "sw_dir_cor_artificial.nc"

# -----------------------------------------------------------------------------
# Prepare data and initialise Terrain class
# -----------------------------------------------------------------------------

if ((not np.all(dom_width_h % dx == 0.0))
        or (not np.all(dom_width_h % dy == 0.0))):
    raise ValueError("Domain widths must be exact multiples of grid spacing")

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "shadow/gridded_artificial/"
if not os.path.isdir(path_out):
    os.makedirs(path_out)

# Generate artifical topography
x = np.linspace(-(dom_width_h.sum() - dx / 2.0),
                (dom_width_h.sum() - dx / 2.0),
                int(dom_width_h.sum() / dx) * 2, dtype=np.float32)
y = np.linspace(-(dom_width_h.sum() - dy / 2.0),
                (dom_width_h.sum() - dy / 2.0),
                int(dom_width_h.sum() / dy) * 2, dtype=np.float32)
x, y = np.meshgrid(x, y)
slice_in = (slice(int(dom_width_h[-1] / dx), -int(dom_width_h[-1] / dx)),
            slice(int(dom_width_h[-1] / dy), -int(dom_width_h[-1] / dy)))
elevation = np.zeros(x.shape, dtype=np.float32)
slice_mod = (slice(int(dom_width_h[1:].sum() / dx),
                   -int(dom_width_h[1:].sum() / dx)),
             slice(int(dom_width_h[1:].sum() / dy),
                   -int(dom_width_h[1:].sum() / dy)))
rad_sqrt = (dom_width_h[0] * 0.95) ** 2
elevation[slice_mod] = np.sqrt(rad_sqrt - x[slice_mod] ** 2
                               - y[slice_mod] ** 2)
elevation[np.isnan(elevation)] = 0.0

print("Inner domain size: " + str(elevation[slice_in].shape))
elevation_in = np.ascontiguousarray(elevation[slice_in])

# Define unit vectors (up)
vec_norm = np.zeros(elevation[slice_in].shape + (3,), dtype=np.float32)
vec_norm[:, :, 2] = 1.0

# Merge vertex coordinates and pad geometry buffer
vert_grid = hray.auxiliary.rearrange_pad_buffer(x, y, elevation)

# Compute slope (in global ENU coordinates!)
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))
vec_tilt = \
    np.ascontiguousarray(hray.topo_param.slope_plane_meth(
        x[slice_in_a1], y[slice_in_a1], elevation[slice_in_a1],
        output_rot=False)[1:-1, 1:-1])

# Compute slope angle and aspect
slope = np.arccos(vec_tilt[:, :, 2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(vec_tilt[:, :, 1],
                                  vec_tilt[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

# Compute surface enlargement factor
surf_enl_fac = 1.0 / (vec_norm * vec_tilt).sum(axis=2)
# surf_enl_fac[:] = 1.0
print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
      + ", %.3f" % surf_enl_fac.max())

# Initialise terrain
offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
dem_dim_0, dem_dim_1 = elevation.shape
mask = np.ones(vec_tilt.shape[:2], dtype=np.uint8)
terrain = hray.shadow.Terrain()
dim_in_0, dim_in_1 = vec_tilt.shape[0], vec_tilt.shape[1]
terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                   offset_0, offset_1, vec_tilt, vec_norm,
                   surf_enl_fac, mask=mask, elevation=elevation_in,
                   refrac_cor=False, ang_max=89.99)

# Position of illumination point (sun)
azim = np.deg2rad(np.linspace(0.0, 360.0, 181))
elev = np.deg2rad(30.00)  # minimum: 20.0
dist_sun = 10000000.0  # distance to sun from coordinate origin [m]
# -> if "downward" projection of sun is within inner domain, erroneous terrain-
#    shadow can be generated for grid cells (-> ray shot from cell
#    towards sun intercepts with terrain "behind" the sun...)

# -----------------------------------------------------------------------------
# Compute correction factor for direct downward shortwave radiation
# -----------------------------------------------------------------------------

# Ensure that sun position is outside of terrain
mask_mod = (elevation[slice_mod] != 0.0)
dist_min = np.sqrt(x[slice_mod][mask_mod] ** 2 + y[slice_mod][mask_mod] ** 2
                   + elevation[slice_mod][mask_mod] ** 2).max() * 1.01
if dist_sun < dist_min:
    raise ValueError("Distance to sun is too small")

# Loop through different azimuth angles and save data to NetCDF file
ncfile = Dataset(filename=path_out + file_sw_dir_cor, mode="w")
ncfile.createDimension(dimname="azim", size=None)
ncfile.createDimension(dimname="y", size=dim_in_0)
ncfile.createDimension(dimname="x", size=dim_in_1)
nc_azim = ncfile.createVariable(varname="azim", datatype="f",
                                dimensions="azim")
nc_azim.units = "degree"
nc_y = ncfile.createVariable(varname="y", datatype="f", dimensions="y")
nc_y[:] = y[slice_in[0], 0]
nc_y.units = "metre"
nc_x = ncfile.createVariable(varname="x", datatype="f", dimensions="x")
nc_x[:] = x[0, slice_in[1]]
nc_x.units = "metre"
nc_data = ncfile.createVariable(varname="sw_dir_cor", datatype="f",
                                dimensions=("azim", "y", "x"))
nc_data.long_name = "correction factor for direct downward shortwave radiation"
nc_data.units = "-"
ncfile.close()
sw_dir_cor = np.zeros(vec_tilt.shape[:2], dtype=np.float32)
for i in range(len(azim)):

    x_sun = dist_sun * np.cos(elev) * np.sin(azim[i])
    y_sun = dist_sun * np.cos(elev) * np.cos(azim[i])
    z_sun = dist_sun * np.sin(elev)
    sun_position = np.array([x_sun, y_sun, z_sun], dtype=np.float32)
    terrain.sw_dir_cor(sun_position, sw_dir_cor)

    ncfile = Dataset(filename=path_out + file_sw_dir_cor, mode="a")
    nc_azim = ncfile.variables["azim"]
    nc_azim[i] = np.rad2deg(azim[i])
    nc_data = ncfile.variables["sw_dir_cor"]
    nc_data[i, :, :] = sw_dir_cor
    ncfile.close()

# -----------------------------------------------------------------------------
# Append further fields to NetCDF file
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Check spatially averaged correction factor
# -----------------------------------------------------------------------------

# Check spatial mean of correction factor
ds = xr.open_dataset(path_out + file_sw_dir_cor)
azim = ds["azim"].values
sw_dir_cor = ds["sw_dir_cor"].values.mean(axis=(1, 2))
ds.close()

# Plot
fig = plt.figure()
plt.plot(azim, sw_dir_cor)
plt.axis((azim[0] - 5.0, azim[-1] + 5.0, 0.85, 1.05))
plt.xlabel("Azimuth angle (measured clockwise from North) [degree]")
plt.ylabel("Spatial mean of correction factor [-]")
plt.title("Average: %.3f" % sw_dir_cor.mean(), fontsize=12)
fig.savefig(path_out + "SW_dir_cor_spatial_mean.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
