# Description: Compute topographic parameters (slope angle and aspect, horizon
#              and Sky View Factor) from swissALTI3D (~2 m) for an example
#              region in the European Alps and simplify the outer DEM domain
#
# Required input data:
#   - swissALTI3D: https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import numpy as np
import xarray as xr
from skimage.io import imsave
import subprocess
import time
import trimesh
import glob
from netCDF4 import Dataset

# Paths to folders
path_DEM = "/Users/csteger/Desktop/SwissALTI3D/"
path_temp = "/Users/csteger/Desktop/temp/"
path_out = "/Users/csteger/Desktop/output/"

# Load required functions
sys.path.append("/Users/csteger/Desktop/lib/")
from horizon import horizon
import functions_cy
from load_dem import load_swissalti3d
from auxiliary import pad_geometry_buffer

###############################################################################
# Settings
###############################################################################

# Miscellaneous
loc = (46.844219, 9.011392)  # centre location (latitude, longitude) [degree]
# -> Limmerensee, Glarus
dom_len = np.array([1.5, 7.0, 28.0], dtype=np.float32)
# inner domain, boundary domain (not simplified / simplified) [kilometre]
hori_acc = np.array([0.15, 0.1], dtype=np.float32)
# horizon accuracy due to algorithm and terrain simplification [degree]
dem_res = 2.0  # resolution of DEM [degree]
azim_num = 360  # number of azimuth sectors [-]
dist_search = dom_len[1:].sum()  # search distance for horizon [kilometre]

# Executables
hmm_ex = "/Applications/hmm/hmm-master/hmm"  # path to 'hmm' executable

# Files
file_dem = path_DEM + "swissalti3d_yyyy_eeee-nnnn_2_2056_5728.tif"
file_hori = path_out + "hori_swissALTI3D_Alps.nc"
file_topo_par = path_out + "topo_par_swissALTI3D_Alps.nc"

###############################################################################
# Load and prepare DEM
###############################################################################

# Load DEM data
east, north, dem = load_swissalti3d(loc, dom_len.sum() * 2.0, path_DEM)

# Slices for subdomains (-> equal dimensions in east and north direction)
bd_ind = (0,
          int(dom_len[2] * 1000 / dem_res),
          int(dom_len[1:].sum() * 1000 / dem_res),
          len(north) - int(dom_len[1:].sum() * 1000 / dem_res) - 1,
          len(north) - int(dom_len[2] * 1000 / dem_res) - 1,
          len(north) - 1)
print(np.diff(bd_ind) * dem_res)
slic_quad = (slice(bd_ind[1], bd_ind[4] + 1), slice(bd_ind[1], bd_ind[4] + 1))
slic_hori = (slice(bd_ind[2], bd_ind[3] + 1), slice(bd_ind[2], bd_ind[3] + 1))
print("Size of quad domain: " + str(dem[slic_quad].shape) + ", vertices: %.2f"
      % (dem[slic_quad].nbytes / (10 ** 9) * 3) + " GB")
print("Size of full domain: " + str(dem.shape) + ", vertices: %.2f"
      % (dem.nbytes / (10 ** 9) * 3) + " GB")

###############################################################################
# Compute TIN (Triangular Irregular Network) from gridded data
###############################################################################

# Slices for outer 4 domains
slic_out = ((slice(bd_ind[0], bd_ind[1] + 1), slice(bd_ind[0], bd_ind[4] + 1)),
            (slice(bd_ind[0], bd_ind[4] + 1), slice(bd_ind[4], bd_ind[5] + 1)),
            (slice(bd_ind[4], bd_ind[5] + 1), slice(bd_ind[1], bd_ind[5] + 1)),
            (slice(bd_ind[1], bd_ind[5] + 1), slice(bd_ind[0], bd_ind[1] + 1)))
# -> create "overlap" at domain boundaries

# Loop through outer domains and save as PNG
res_cp = dem_res
scal_fac = 16.0  # (1.0 / 16.0 -> 0.0625)
res_cp *= scal_fac
for i in range(4):

    # Copy data (-> keep original data unmodified)
    dem_cp = dem[slic_out[i]].copy()

    # Scale data
    dem_cp *= scal_fac

    # Further process DEM data
    print("Range (min, max) of (scaled) DEM data: %.1f" % dem_cp.min()
          + ", %.1f" % dem_cp.max() + " m")
    # -> allowed range in np.uint16: [0, 65535]
    # np.array([0, 65535], dtype=np.uint16)
    if (dem_cp.min() < 1.0) or (dem_cp.max() > 65534.0):
        raise ValueError("(Scaled) DEM range too large -> issue for uint16 "
                         + "conversion")
    dem_cp = dem_cp.astype(np.uint16)
    dem_cp = np.flipud(dem_cp)  # flip

    # Save DEM as PNG-file (for triangulation)
    imsave(path_temp + "out_" + str(i) + ".png", dem_cp.astype(np.uint16),
           check_contrast=False, optimize=False, compress_level=0)
    time.sleep(1.0)
    del dem_cp

# Z Scale
z_scale = "%.2f " % (65535.0 / res_cp)
print("Z Scale: " + z_scale)

# Compute relative error
dist = dom_len[1] * 1000.0  # [m]
err_ang = hori_acc[1]  # [deg]
err_max = np.tan(np.deg2rad(err_ang) / 2.0) * 2.0 * dist
print("Maximal vertical error: %.1f" % err_max + " m")
e = "%.6f" % ((err_max * scal_fac) / 65535.0)
print("e: " + e)

# Compute TIN from gridded data (parallel for 4 domains)
t_beg = time.time()
commands = [hmm_ex + " " + path_temp + "out_" + str(i) + ".png" + " "
            + path_temp + "out_" + str(i) + ".stl" + " -z " + z_scale
            + " -e " + e for i in range(4)]
procs = [subprocess.Popen(i, shell=True) for i in commands]
for p in procs:
    p.wait()
print("Elapsed time: %.2f" % (time.time() - t_beg) + " s")

###############################################################################
# Combine 4 outer simplified domains and add skirt
###############################################################################

# # Check minimal and maximal elevation in domains
# for i in range(4):
#     print((" Domain " + str(i) + " ").center(50, "-"))
#     print("Min: %.2f" % dem[slic_out[i]].min()
#           + " m, max: %.2f" % dem[slic_out[i]].max() + " m")
#     mesh_data = trimesh.load(path_temp + "out_" + str(i) + ".stl")
#     vertices = mesh_data.vertices.view(np.ndarray) \
#         * res_cp * (1.0 / scal_fac)  # [m]
#     print("Min: %.2f" % vertices[:, 2].min()
#           + " m, max: %.2f" % vertices[:, 2].max() + " m")
#     time.sleep(1.0)
#     del mesh_data, vertices

# Delete outer part of DEM
dem_quad = dem[slic_quad].copy()
dem_hori = dem[slic_hori].copy()
dem_size = ((dem.nbytes * 3) / (10 ** 6))  # MB
del dem

# -----------------------------------------------------------------------------

# Merge four outer domains (and add skirt)
add_skirt = True
t_beg = time.time()
mesh_data_all = []
skirt_val = (bd_ind[1] * dem_res, 0.0, 0.0, bd_ind[1] * dem_res)  # rel. coord.
for i in range(4):
    mesh_data = trimesh.load(path_temp + "out_" + str(i) + ".stl")
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
out = mesh_data_comb.export(file_obj=path_temp + "out_comb.stl")
print("Elapsed time: %.2f" % (time.time() - t_beg) + " s")
time.sleep(1.0)
del mesh_data, mesh_data_comb, mesh_data_all, mesh_data_skirt

###############################################################################
# Prepare data for horizon computation
###############################################################################

# Compute offset
offset_0 = bd_ind[2] - bd_ind[1]
offset_1 = bd_ind[2] - bd_ind[1]
dem_dim_0 = dem_quad.shape[0]
dem_dim_1 = dem_quad.shape[1]

# -----------------------------------------------------------------------------
# Transform coordinates from LV95 to global ENU coordinates (quad DEM)
# -----------------------------------------------------------------------------

print(" Transform LV95 to global ENU coordinates ".center(60, "-"))

# Compute geographic coordinates
east_2D, north_2D = np.meshgrid(east[slic_quad[1]], north[slic_quad[0]])
lon, lat, h_wgs = functions_cy.swiss2wgs(east_2D.astype(np.float64),
                                         north_2D.astype(np.float64),
                                         dem_quad)

# Compute geocentric/ECEF coordinates
ellps = "WGS84"
x_ecef, y_ecef, z_ecef = functions_cy.lonlat2ecef(lon, lat, h_wgs, ellps=ellps)
del h_wgs

# ENU origin of coordinates
ind_0, ind_1 = int(len(lat) / 2), int(len(lon) / 2)
lon_or, lat_or = lon[ind_0, ind_1], lat[ind_0, ind_1]
x_ecef_or = x_ecef[ind_0, ind_1]
y_ecef_or = y_ecef[ind_0, ind_1]
z_ecef_or = z_ecef[ind_0, ind_1]

# Compute topocentric/ENU coordinates
x_enu, y_enu, z_enu = functions_cy.ecef2enu(x_ecef, y_ecef, z_ecef,
                                            x_ecef_or, y_ecef_or,
                                            z_ecef_or,
                                            lon_or, lat_or)

# Compute unit vectors (in ENU coordinates)
sd_in = (slice(offset_0, -offset_0), slice(offset_1, -offset_1))
vec_norm_ecef = functions_cy.surf_norm(lon[sd_in], lat[sd_in])
del lon, lat
vec_north_ecef = functions_cy.north_dir(x_ecef[sd_in], y_ecef[sd_in],
                                        z_ecef[sd_in], vec_norm_ecef,
                                        ellps=ellps)
vec_norm = functions_cy.ecef2enu_vec(vec_norm_ecef, lon_or, lat_or)
vec_north = functions_cy.ecef2enu_vec(vec_north_ecef, lon_or, lat_or)
del x_ecef, y_ecef, z_ecef
del vec_norm_ecef, vec_north_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = np.hstack((x_enu.reshape(x_enu.size, 1),
                       y_enu.reshape(x_enu.size, 1),
                       z_enu.reshape(x_enu.size, 1))).ravel()
vert_grid = pad_geometry_buffer(vert_grid)

# -----------------------------------------------------------------------------
# Transform coordinates from LV95 to global ENU coordinates (triangle DEM)
# -----------------------------------------------------------------------------

# Load triangles
mesh_data = trimesh.load(path_temp + "out_comb.stl")

# Compute geographic coordinates
lon, lat, h_wgs = functions_cy.swiss2wgs(
    mesh_data.vertices.view(np.ndarray)[:, 0][:, np.newaxis],
    mesh_data.vertices.view(np.ndarray)[:, 1][:, np.newaxis],
    mesh_data.vertices.view(np.ndarray)[:, 2][:, np.newaxis]
    .astype(np.float32))

# Compute geocentric/ECEF coordinates
x_ecef, y_ecef, z_ecef = functions_cy.lonlat2ecef(lon, lat, h_wgs, ellps=ellps)
del lon, lat, h_wgs

# Compute topocentric/ENU coordinates
x_enu_tri, y_enu_tri, z_enu_tri \
    = functions_cy.ecef2enu(x_ecef, y_ecef, z_ecef,
                            x_ecef_or, y_ecef_or, z_ecef_or,
                            lon_or, lat_or)
del x_ecef, y_ecef, z_ecef

# Store data in Embree input format
vert_simp = np.hstack((x_enu_tri, y_enu_tri, z_enu_tri)).ravel()
num_vert_simp = mesh_data.vertices.shape[0]
vert_simp = pad_geometry_buffer(vert_simp)
tri_ind_simp = mesh_data.faces.view(np.ndarray).astype(np.int32).ravel()
num_tri_simp = mesh_data.faces.shape[0]
tri_ind_simp = pad_geometry_buffer(tri_ind_simp)
del x_enu_tri, y_enu_tri, z_enu_tri

# -----------------------------------------------------------------------------

# Compare memory requirements
print("Memory requirements:")
print("Total DEM: %.2f" % dem_size + " MB")
print("Quad DEM: %.2f" % ((dem_quad.nbytes * 3) / (10 ** 6)) + " MB")
print("Triangle DEM: %.2f" % ((vert_simp.nbytes + tri_ind_simp.nbytes)
                              / (10 ** 6)) + " MB")

###############################################################################
# Perform horizon computation with Embree
###############################################################################

# Compute horizon angles
horizon(vert_grid, dem_dim_0, dem_dim_1,
        vec_norm, vec_north,
        offset_0, offset_1,
        dist_search=dist_search, azim_num=azim_num, hori_acc=hori_acc[0],
        ray_algorithm="guess_constant", geom_type="grid",
        vert_simp=vert_simp, num_vert_simp=num_vert_simp,
        tri_ind_simp=tri_ind_simp, num_tri_simp=num_tri_simp,
        file_out=file_hori,
        x_axis_val=east[slic_hori[1]].astype(np.float32),
        y_axis_val=north[slic_hori[0]].astype(np.float32),
        x_axis_name="east", y_axis_name="north", units="m",
        hori_buffer_size_max=0.85)
time.sleep(1.0)
del vert_grid, vert_simp, tri_ind_simp

# Merge horizon slices
ds = xr.open_mfdataset(file_hori[:-3] + "_p??.nc", concat_dim="east")
ds = ds.assign_coords({"east": ("east", ds["east"].values.astype(np.float32),
                                ds["east"].attrs)})
ds.to_netcdf(file_hori, format="NETCDF4")
files_rm = glob.glob(file_hori[:-3] + "_p??.nc")
for i in files_rm:
    os.remove(i)

###############################################################################
# Compute slope and Sky View Factor and save to NetCDF file
###############################################################################

# Rotation matrix (global ENU -> local ENU)
rot_mat = np.empty((vec_north.shape[0] + 2, vec_north.shape[1] + 2,
                    3, 3), dtype=np.float32)
rot_mat.fill(np.nan)
rot_mat[1:-1, 1:-1, 0, :] = np.cross(vec_north, vec_norm, axisa=2,
                                     axisb=2)  # vector pointing towards east
rot_mat[1:-1, 1:-1, 1, :] = vec_north
rot_mat[1:-1, 1:-1, 2, :] = vec_norm

# Compute slope
sd_in_a1 = (slice(sd_in[0].start - 1, sd_in[0].stop + 1),
            slice(sd_in[1].start - 1, sd_in[1].stop + 1))
vec_tilt = functions_cy.slope_plane_meth(x_enu[sd_in_a1], y_enu[sd_in_a1],
                                         z_enu[sd_in_a1], rot_mat)[1:-1, 1:-1]

# Compute slope and aspect
print("Maximal z-value of vec_tilt_loc: %.8f" % np.max(vec_tilt[:, :, 2]))
slope = np.arccos(vec_tilt[:, :, 2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(vec_tilt[:, :, 1],
                                  vec_tilt[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

# Load horizon data
ds = xr.open_dataset(file_hori)
hori = ds["horizon"].values
azim = ds["azim"].values
ds.close()

# Compute Sky View Factor
svf = functions_cy.skyviewfactor(azim, hori, vec_tilt)

# Save topographic parameters to NetCDF file
ncfile = Dataset(filename=file_topo_par, mode="w")
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
nc_data[:] = dem_hori
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
