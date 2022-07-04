# Description: Compute gridded topographic parameters (slope angle and aspect,
#              horizon and sky view factor) from swissALTI3D data (2 m) for
#              an example region in Switzerland. Ignore Earth's surface
#              curvature and simplify the outer DEM domain.
#
# Source of applied DEM data:
#   https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import sys
import numpy as np
import xarray as xr
from skimage.io import imsave
import subprocess
import time
import trimesh
import glob
from netCDF4 import Dataset
sys.path.append("/Users/csteger/Downloads/HORAYZON/")  # ------------ temporary
from horayzon import auxiliary, domain, horizon, load_dem, topo_param, download  # temporary
import horayzon as hray

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Domain size and computation settings
domain = {"x_min": 2711700, "x_max": 2714700,
          "y_min": 1184400, "y_max": 1187400}  # domain boundaries [metre]
# -> 3 x 3 km domain centred around Toedi, Glarus
# -> Swiss LV95 coordinates -> required domain can e.g. be determined with:
#    https://map.geo.admin.ch/
dist_search = 30.0  # search distance for horizon [kilometre]
azim_num = 360  # number of azimuth sectors [-]
hori_acc = np.array([0.15, 0.1], dtype=np.float32)
# horizon accuracy due to algorithm and terrain simplification [degree]
dem_res = 2.0  # resolution of DEM [degree]
domain_out_frac = np.array([0.25, 0.75], dtype=np.float32)
# partitioning of outer domain: not simplified / simplified [-]

# Path to heightmap meshing utility (hmm) executable
hmm_ex = "/Applications/hmm/hmm-master/hmm"

# Paths and file names
dem_file_url = "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/" \
               + "swissalti3d_yyyy_eeee-nnnn/" \
               + "swissalti3d_yyyy_eeee-nnnn_2_2056_5728.tif"
path_out = "/Users/csteger/Desktop/Output/"
file_hori = "hori_swissALTI3D_Switzerland.nc"
file_topo_par = "topo_par_swissALTI3D_Switzerland.nc"

# -----------------------------------------------------------------------------
# Check settings and specified folders/executables
# -----------------------------------------------------------------------------

# Check settings
if domain_out_frac.sum() != 1.0:
    raise ValueError("Array 'domain_out_frac' must sum up to exactly 1.0")

# Check if output directory exists
if not os.path.isdir(path_out):
    raise ValueError("Output directory does not exist")
path_out += "gridded_swissALTI3D_Switzerland/"
if not os.path.isdir(path_out):
    os.mkdir(path_out)

# Check if heightmap meshing utility (hmm) executable exists
if not os.path.isfile(hmm_ex):
    raise ValueError("heightmap meshing utility (hmm) not installed or "
                     + "path to executable erroneous")

# -----------------------------------------------------------------------------
# Download swissALTI3D tiles
# -----------------------------------------------------------------------------

# Download data
path_tiles = path_out + "tiles_dem/"
if not os.path.isdir(path_tiles):
    os.mkdir(path_tiles)
add = dist_search * 1000.0
tiles_east = list(range(int(np.floor((domain["x_min"] - add) / 1000)),
                        int(np.ceil((domain["x_max"] + add) / 1000))))
tiles_north = list(range(int(np.floor((domain["y_min"] - add) / 1000)),
                         int(np.ceil((domain["y_max"] + add) / 1000))))[::-1]
files_url = [dem_file_url.replace("eeee", str(j)).replace("nnnn", str(i))
             for i in tiles_north for j in tiles_east]
for i in (2019, 2020, 2021):
    if len(files_url) > 0:
        print((" Try to download files for year " + str(i) + " ")
              .center(60, "-"))
        files_url_y = [j.replace("yyyy", str(i)) for j in files_url]
        downloaded = hray.download.files(files_url_y, path_tiles,
                                         mode="parallel", block_size=200,
                                         file_num=10)
        files_url = [j for j, k in zip(files_url, downloaded) if not k]
if len(files_url) != 0:
    raise ValueError("Not all required tiles were found/downloaded")

# -----------------------------------------------------------------------------
# Load and prepare DEM
# -----------------------------------------------------------------------------

# Load DEM data
domain_outer = hray.domain.planar_grid(domain, dist_search)
east, north, elevation = hray.load_dem.swissalti3d(path_tiles, domain_outer,
                                                   engine="gdal")
north, elevation = north[::-1], np.flipud(elevation)
# -> arrange both coordinate axis (east and north) in increasing order

# Slices for subdomains (-> equal dimensions in east and north direction)
bd_ind = (0,
          int((domain_out_frac[1] * dist_search * 1000.0) / dem_res),
          int((dist_search * 1000.0) / dem_res),
          len(north) - int((dist_search * 1000.0) / dem_res),
          len(north) - int((domain_out_frac[1] * dist_search * 1000.0)
                           / dem_res),
          len(north))
print(np.diff(bd_ind) * dem_res)
slic_quad = (slice(bd_ind[1], bd_ind[4]), slice(bd_ind[1], bd_ind[4]))
slic_hori = (slice(bd_ind[2], bd_ind[3]), slice(bd_ind[2], bd_ind[3]))
print("Size of quad domain: " + str(elevation[slic_quad].shape)
      + ", vertices: %.2f" % (elevation[slic_quad].nbytes / (10 ** 9) * 3)
      + " GB")
print("Size of full domain: " + str(elevation.shape) + ", vertices: %.2f"
      % (elevation.nbytes / (10 ** 9) * 3) + " GB")

# -----------------------------------------------------------------------------
# Compute TIN (Triangular Irregular Network) from gridded data
# -----------------------------------------------------------------------------

# Slices for outer 4 domains
slic_out = ((slice(bd_ind[0], bd_ind[1] + 1), slice(bd_ind[0], bd_ind[4])),
            (slice(bd_ind[0], bd_ind[4]), slice(bd_ind[4] - 1, bd_ind[5])),
            (slice(bd_ind[4] - 1, bd_ind[5]), slice(bd_ind[1], bd_ind[5])),
            (slice(bd_ind[1], bd_ind[5]), slice(bd_ind[0], bd_ind[1] + 1)))
# -> create "overlap" at domain boundaries

# Loop through outer domains and save as PNG
res_cp = dem_res
scal_fac = 16.0  # (1.0 / 16.0 -> 0.0625)
res_cp *= scal_fac
for i in range(4):

    # Copy data (-> keep original data unmodified)
    elevation_cp = elevation[slic_out[i]].copy()

    # Scale data
    elevation_cp *= scal_fac

    # Further process DEM data
    print("Range (min, max) of scaled DEM data: %.1f" % elevation_cp.min()
          + ", %.1f" % elevation_cp.max() + " m")
    # -> allowed range in np.uint16: [0, 65535]
    # np.array([0, 65535], dtype=np.uint16)
    if (elevation_cp.min() < 1.0) or (elevation_cp.max() > 65534.0):
        raise ValueError("Scaled DEM range too large -> issue for uint16 "
                         + "conversion")
    elevation_cp = elevation_cp.astype(np.uint16)
    elevation_cp = np.flipud(elevation_cp)  # flip

    # Save DEM as PNG-file (for triangulation)
    imsave(path_out + "out_" + str(i) + ".png",
           elevation_cp.astype(np.uint16),
           check_contrast=False, optimize=False, compress_level=0)
    time.sleep(1.0)
    del elevation_cp

# Z Scale
z_scale = "%.2f " % (65535.0 / res_cp)
print("Z Scale: " + z_scale)

# Compute relative error
dist = domain_out_frac[0] * dist_search * 1000.0  # [m]
err_ang = hori_acc[1]  # [deg]
err_max = np.tan(np.deg2rad(err_ang) / 2.0) * 2.0 * dist
print("Maximal vertical error: %.1f" % err_max + " m")
e = "%.6f" % ((err_max * scal_fac) / 65535.0)
print("e: " + e)

# Compute TIN from gridded data (parallel for 4 domains)
t_beg = time.time()
commands = [hmm_ex + " " + path_out + "out_" + str(i) + ".png" + " "
            + path_out + "out_" + str(i) + ".stl" + " -z " + z_scale
            + " -e " + e for i in range(4)]
procs = [subprocess.Popen(i, shell=True) for i in commands]
for p in procs:
    p.wait()
print("Elapsed time: %.2f" % (time.time() - t_beg) + " s")

# -----------------------------------------------------------------------------
# Combine 4 outer simplified domains and add skirt
# -----------------------------------------------------------------------------

# # Check minimal and maximal elevation in domains
# for i in range(4):
#     print((" Domain " + str(i) + " ").center(50, "-"))
#     print("Min: %.2f" % elevation[slic_out[i]].min()
#           + " m, max: %.2f" % elevation[slic_out[i]].max() + " m")
#     mesh_data = trimesh.load(path_out + "out_" + str(i) + ".stl")
#     vertices = mesh_data.vertices.view(np.ndarray) \
#         * res_cp * (1.0 / scal_fac)  # [m]
#     print("Min: %.2f" % vertices[:, 2].min()
#           + " m, max: %.2f" % vertices[:, 2].max() + " m")
#     time.sleep(1.0)
#     del mesh_data, vertices

# Delete outer part of DEM
elevation_quad = elevation[slic_quad].copy()
elevation_hori = elevation[slic_hori].copy()
elevation_size = ((elevation.nbytes * 3) / (10 ** 6))  # MB
del elevation

# Merge four outer domains (and add skirt)
add_skirt = True
t_beg = time.time()
mesh_data_all = []
skirt_val = (bd_ind[1] * dem_res, 0.0, 0.0, bd_ind[1] * dem_res)  # rel. coord.
for i in range(4):
    mesh_data = trimesh.load(path_out + "out_" + str(i) + ".stl")
    mesh_data.vertices *= res_cp * (1.0 / scal_fac)  # [m]
    # -> x and y coordinates are relative to lower left corner (0, 0)
    # -------------------------------------------------------------------------
    if add_skirt:
        x = np.array(mesh_data.vertices[:, 0])
        y = np.array(mesh_data.vertices[:, 1])
        z = np.array(mesh_data.vertices[:, 2])
        if i in (0, 2):  # y-coordinate constant
            ind = np.where(np.abs(y - skirt_val[i]) < (dem_res / 2.0))[0]
            ind_sort = np.argsort(x[ind])
            skirt_x = x[ind][ind_sort]
            skirt_y = np.repeat(skirt_val[i], len(skirt_x))
        else:  # x-coordinate constant
            ind = np.where(np.abs(x - skirt_val[i]) < (dem_res / 2.0))[0]
            ind_sort = np.argsort(y[ind])
            skirt_y = y[ind][ind_sort]
            skirt_x = np.repeat(skirt_val[i], len(skirt_y))
        skirt_z = z[ind][ind_sort]
        vert_num = len(skirt_x) * 2
        vert_skirt = np.empty((vert_num, 3), dtype=np.float32)
        vert_skirt[:, 0] = np.repeat(skirt_x, 2)
        vert_skirt[:, 1] = np.repeat(skirt_y, 2)
        vert_skirt[:, 2] = np.repeat(skirt_z, 2)
        vert_skirt[:, 2][1::2] -= (3.0 * err_max)
        faces_num = (len(skirt_x) - 1) * 2
        faces_skirt = np.empty((faces_num, 3), dtype=np.int32)
        faces_skirt[:, 0] = np.arange(faces_num, dtype=np.int32)
        faces_skirt[:, 1] = np.arange(faces_num, dtype=np.int32) + 1
        faces_skirt[:, 2] = np.arange(faces_num, dtype=np.int32) + 2
        # -> winding order of triangles not identical
        mesh_data_skirt = trimesh.Trimesh(vertices=vert_skirt,
                                          faces=faces_skirt)
        # ---------------------------------------------------------------------
        mesh_data = trimesh.util.concatenate([mesh_data, mesh_data_skirt])
        # identical vertices are not merged
    # -------------------------------------------------------------------------
    mesh_data.vertices[:, 0] += east[slic_out[i][1]][0]  # [m]
    mesh_data.vertices[:, 1] += north[slic_out[i][0]][0]  # [m]
    mesh_data_all.append(mesh_data)
mesh_data_comb = trimesh.util.concatenate(mesh_data_all)
out = mesh_data_comb.export(file_obj=path_out + "out_comb.stl")
print("Elapsed time: %.2f" % (time.time() - t_beg) + " s")
time.sleep(1.0)
del mesh_data, mesh_data_comb, mesh_data_all, mesh_data_skirt

# -----------------------------------------------------------------------------
# Prepare data for horizon computation
# -----------------------------------------------------------------------------

# Compute offset
offset_0 = bd_ind[2] - bd_ind[1]
offset_1 = bd_ind[2] - bd_ind[1]
dem_dim_0 = elevation_quad.shape[0]
dem_dim_1 = elevation_quad.shape[1]

# Create directional unit vectors for inner domain
vec_norm = np.zeros((dem_dim_0 - (2 * offset_0),
                     dem_dim_1 - (2 * offset_1), 3), dtype=np.float32)
vec_norm[:, :, 2] = 1.0
vec_north = np.zeros((dem_dim_0 - (2 * offset_0),
                      dem_dim_1 - (2 * offset_1), 3), dtype=np.float32)
vec_north[:, :, 1] = 1.0

# Merge vertex coordinates and pad geometry buffer
vert_grid = hray.auxiliary.rearrange_pad_buffer(
    *np.meshgrid(east[slic_quad[1]], north[slic_quad[0]]),
    elevation_quad)

# Load triangles
mesh_data = trimesh.load(path_out + "out_comb.stl")

# Store data in Embree input format
vert_simp = mesh_data.vertices.view(np.ndarray).astype(np.float32).ravel()
num_vert_simp = mesh_data.vertices.shape[0]
vert_simp = hray.auxiliary.pad_buffer(vert_simp)
tri_ind_simp = mesh_data.faces.view(np.ndarray).astype(np.int32).ravel()
num_tri_simp = mesh_data.faces.shape[0]
tri_ind_simp = hray.auxiliary.pad_buffer(tri_ind_simp)

# Compare memory requirements
print("Memory requirements:")
print("Total DEM: %.2f" % elevation_size + " MB")
print("Quad DEM: %.2f" % ((elevation_quad.nbytes * 3) / (10 ** 6)) + " MB")
print("Triangle DEM: %.2f" % ((vert_simp.nbytes + tri_ind_simp.nbytes)
                              / (10 ** 6)) + " MB")

# -----------------------------------------------------------------------------
# Compute and save topographic parameters
# -----------------------------------------------------------------------------

# Compute horizon
hray.horizon.horizon_gridded(vert_grid, dem_dim_0, dem_dim_1,
                             vec_norm, vec_north, offset_0, offset_1,
                             file_out=path_out + file_hori,
                             dist_search=dist_search,
                             azim_num=azim_num, hori_acc=hori_acc[0],
                             ray_algorithm="guess_constant", geom_type="grid",
                             vert_simp=vert_simp, num_vert_simp=num_vert_simp,
                             tri_ind_simp=tri_ind_simp,
                             num_tri_simp=num_tri_simp,
                             x_axis_val=east[slic_hori[1]].astype(np.float32),
                             y_axis_val=north[slic_hori[0]].astype(np.float32),
                             x_axis_name="east", y_axis_name="north",
                             units="m", hori_buffer_size_max=0.85)
time.sleep(1.0)
del vert_grid, vert_simp, tri_ind_simp

# Merge horizon slices
ds = xr.open_mfdataset(path_out + file_hori[:-3] + "_p??.nc")
ds.to_netcdf(path_out + file_hori, format="NETCDF4")
files_rm = glob.glob(path_out + file_hori[:-3] + "_p??.nc")
for i in files_rm:
    os.remove(i)

# Swap coordinate axes (-> make viewable with ncview)
ds_ncview = ds.transpose("azim", "north", "east")
ds_ncview.to_netcdf(path_out + file_hori[:-3] + "_ncview.nc")

# Compute slope
sd_in = (slice(offset_0, -offset_0), slice(offset_1, -offset_1))
sd_in_a1 = (slice(sd_in[0].start - 1, sd_in[0].stop + 1),
            slice(sd_in[1].start - 1, sd_in[1].stop + 1))
east_2d, north_2d = np.meshgrid(east[slic_quad[1]], north[slic_quad[0]])
vec_tilt = hray.topo_param.slope_plane_meth(
    east_2d[sd_in_a1], north_2d[sd_in_a1],
    elevation_quad[sd_in_a1])[1:-1, 1:-1]
del east_2d, north_2d

# Compute Sky View Factor
ds = xr.open_dataset(path_out + file_hori)
hori = ds["horizon"].values
azim = ds["azim"].values
ds.close()
svf = hray.topo_param.sky_view_factor(azim, hori, vec_tilt)
del hori

# Compute slope angle and aspect
slope = np.arccos(vec_tilt[:, :, 2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(vec_tilt[:, :, 1],
                                  vec_tilt[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

# Save topographic parameters to NetCDF file
ncfile = Dataset(filename=path_out + file_topo_par, mode="w")
ncfile.createDimension(dimname="north", size=svf.shape[0])
ncfile.createDimension(dimname="east", size=svf.shape[1])
nc_north = ncfile.createVariable(varname="north", datatype="f",
                                 dimensions="north")
nc_north[:] = north[slic_hori[0]]
nc_north.units = "m"
nc_east = ncfile.createVariable(varname="east", datatype="f",
                                dimensions="east")
nc_east[:] = east[slic_hori[1]]
nc_east.units = "m"
nc_data = ncfile.createVariable(varname="elevation", datatype="f",
                                dimensions=("north", "east"))
nc_data[:] = elevation_hori
nc_data.units = "m"
nc_data = ncfile.createVariable(varname="slope", datatype="f",
                                dimensions=("north", "east"))
nc_data[:] = slope
nc_data.long_name = "slope angle"
nc_data.units = "rad"
nc_data = ncfile.createVariable(varname="aspect", datatype="f",
                                dimensions=("north", "east"))
nc_data[:] = aspect
nc_data.long_name = "slope aspect (clockwise from North)"
nc_data.units = "rad"
nc_data = ncfile.createVariable(varname="svf", datatype="f",
                                dimensions=("north", "east"))
nc_data[:] = svf
nc_data.long_name = "sky view factor"
nc_data.units = "-"
ncfile.close()
