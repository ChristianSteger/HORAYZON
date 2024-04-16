# Description: Compute topographic parameters (slope angle and aspect,
#              horizon and sky view factor) from SRTM data (~90 m) for some
#              arbitrary locations in Switzerland. Consider Earth's surface
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
from scipy import interpolate
import zipfile
import horayzon as hray

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Locations and computation settings
loc_sel = {"Eiger_Nordwand":              (46.58210, 8.00038, 2.0),
           "Wengen":                      (46.60691, 7.92347, 2.0),
           "Muerren":                     (46.55944, 7.89222, 2.0),
           "Goeschenen":                  (46.66777, 8.58639, 2.0),
           "Zinal":                       (46.13556, 7.62583, 2.0),
           "Blatten":                     (46.42221, 7.82083, 2.0),
           "Leukerbad":                   (46.38333, 7.63333, 2.0),
           "Zuerich":                     (47.37174, 8.54226, 5.0),
           "Balm_bei_Guensberg":          (47.25203, 7.55674, 2.0),
           "Gredetschtal_(east-facing)":  (46.35742, 7.92887, 1.0),
           "Gredetschtal_(west-facing)":  (46.35823, 7.93863, 1.0)}
# (latitude [degree], longitude [degree], elevation above surface [m])
dist_search = 100.0  # search distance for horizon [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
azim_num = 1440  # number of azimuth sampling directions [-]
hori_acc = 0.1  # [degree]

# Paths and file names
dem_file_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/" \
               + "srtm_5x5/TIFF/srtm_38_03.zip"
path_out = "/Users/csteger/Desktop/Output/"
file_hori = "hori_SRTM_Switzerland.nc"

# -----------------------------------------------------------------------------
# Prepare digital elevation model data
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")
path_out += "horizon/locations_SRTM_Switzerland/"
if not os.path.isdir(path_out):
    os.makedirs(path_out)

# Download and unzip SRTM tile (5 x 5 degree)
print("Download SRTM tile (5 x 5 degree):")
hray.download.file(dem_file_url, path_out)
with zipfile.ZipFile(path_out + "srtm_38_03.zip", "r") as zip_ref:
    zip_ref.extractall(path_out + "srtm_38_03")
os.remove(path_out + "srtm_38_03.zip")

# Load required DEM data (including outer boundary zone)
lon_loc = np.array([loc_sel[i][1] for i in loc_sel.keys()], dtype=np.float64)
lat_loc = np.array([loc_sel[i][0] for i in loc_sel.keys()], dtype=np.float64)
domain = {"lon_min": lon_loc.min(), "lon_max": lon_loc.max(),
          "lat_min": lat_loc.min(), "lat_max": lat_loc.max()}
# domain boundaries [degree]
domain_outer = hray.domain.curved_grid(domain, dist_search, ellps)
file_dem = path_out + "srtm_38_03/srtm_38_03.tif"
lon, lat, elevation = hray.load_dem.srtm(file_dem, domain_outer,
                                         engine="pillow")
# -> GeoTIFF can also be read with GDAL if available (-> faster)

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
del x_ecef, y_ecef, z_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
del x_enu, y_enu, z_enu

# -----------------------------------------------------------------------------
# Prepare data for selected locations
# -----------------------------------------------------------------------------

# Compute ECEF coordinates
elev_0_loc = np.zeros(len(loc_sel), dtype=np.float32)
x_ecef_loc, y_ecef_loc, z_ecef_loc \
    = hray.transform.lonlat2ecef(lon_loc, lat_loc, elev_0_loc, ellps=ellps)

# Compute ENU coordinates
coords = np.array(hray.transform.ecef2enu(x_ecef_loc, y_ecef_loc, z_ecef_loc,
                              trans_ecef2enu)).transpose()

# Compute unit vectors (in ENU coordinates)
vec_norm_ecef = hray.direction.surf_norm(lon_loc, lat_loc)
vec_north_ecef = hray.direction.north_dir(x_ecef_loc, y_ecef_loc,
                                          z_ecef_loc, vec_norm_ecef,
                                          ellps=ellps)
del x_ecef_loc, y_ecef_loc, z_ecef_loc
vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
del vec_norm_ecef, vec_north_ecef

# -----------------------------------------------------------------------------
# Compute topographic parameters
# -----------------------------------------------------------------------------

# Compute horizon
ray_org_elev = np.array([loc_sel[i][2] for i in loc_sel.keys()],
                        dtype=np.float32)
hori, hori_dist, azim = hray.horizon.horizon_locations(
    vert_grid, dem_dim_0, dem_dim_1,
    coords, vec_norm_enu, vec_north_enu,
    dist_search=dist_search, azim_num=azim_num,
    hori_acc=hori_acc,
    ray_org_elev=ray_org_elev, hori_dist_out=True)
hori_dist /= 1000.0  # [km]

# Compute slope and sky view factor for locations
topo_param = {}
for i in list(loc_sel.keys()):

    # 5 x 5 grid cell domain
    ind_0 = np.argmin(np.abs(loc_sel[i][0] - lat))
    ind_1 = np.argmin(np.abs(loc_sel[i][1] - lon))
    slice_5x5 = (slice(ind_0 - 2, ind_0 + 3), slice(ind_1 - 2, ind_1 + 3))

    # Compute ECEF coordinates
    x_ecef, y_ecef, z_ecef \
        = hray.transform.lonlat2ecef(*np.meshgrid(lon[slice_5x5[1]],
                                                  lat[slice_5x5[0]]),
                                     elevation[slice_5x5], ellps=ellps)

    # Compute ENU coordinates
    x_enu, y_enu, z_enu \
        = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef, trans_ecef2enu)

    # Compute unit vectors (in ENU coordinates)
    slice_3x3 = (slice(slice_5x5[0].start + 1, slice_5x5[0].stop - 1),
                 slice(slice_5x5[1].start + 1, slice_5x5[1].stop - 1))
    vec_norm_ecef \
        = hray.direction.surf_norm(*np.meshgrid(lon[slice_3x3[1]],
                                                lat[slice_3x3[0]]))
    vec_north_ecef = hray.direction.north_dir(x_ecef[1:-1, 1:-1],
                                              y_ecef[1:-1, 1:-1],
                                              z_ecef[1:-1, 1:-1],
                                              vec_norm_ecef, ellps=ellps)
    del x_ecef, y_ecef, z_ecef
    vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef,
                                                  trans_ecef2enu)
    vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef,
                                                   trans_ecef2enu)
    del vec_norm_ecef, vec_north_ecef

    # Compute rotation matrix (global ENU -> local ENU)
    rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                               vec_norm_enu)
    del vec_north_enu, vec_norm_enu

    # Compute slope
    vec_tilt \
        = hray.topo_param.slope_plane_meth(x_enu, y_enu, z_enu,
                                           rot_mat=rot_mat_glob2loc,
                                           output_rot=True)[1:-1, 1:-1, :]

    # Bilinear interpolation of slope at location
    vec_tilt_ip = np.empty((1, 1, 3), dtype=np.float32)
    for j in range(3):
        f = interpolate.interp2d(*np.meshgrid(lon[slice_3x3[1]],
                                              lat[slice_3x3[0]]),
                                 vec_tilt[:, :, j], bounds_error=True)
        vec_tilt_ip[0, 0, j] = f(loc_sel[i][1], loc_sel[i][0])
    vec_tilt_ip /= np.sqrt(np.sum(vec_tilt_ip ** 2))  # unit vector

    # Compute slope angle and aspect
    slope = np.arccos(vec_tilt_ip[0, 0, 2])
    aspect = np.pi / 2.0 - np.arctan2(vec_tilt_ip[0, 0, 1],
                                      vec_tilt_ip[0, 0, 0])
    if aspect < 0.0:
        aspect += np.pi * 2.0  # [0.0, 2.0 * np.pi]

    # Compute Sky View Factor
    hori_loc = hori[list(loc_sel).index(i), :][np.newaxis, np.newaxis, :]
    svf = hray.topo_param.sky_view_factor(azim, hori_loc, vec_tilt_ip)[0, 0]

    topo_param[i] = {"slope_angle": np.rad2deg(slope),
                     "slope_aspect": np.rad2deg(aspect), "svf": svf}

# -----------------------------------------------------------------------------
# Plot topographic parameters
# -----------------------------------------------------------------------------

# Plot horizon and distance to horizon for locations
for i in list(loc_sel.keys()):
    fig = plt.figure(figsize=(14, 5))
    ax_l = plt.axes()
    ind = list(loc_sel.keys()).index(i)
    plt.plot(np.rad2deg(azim), np.rad2deg(hori[ind, :]), color="black", lw=1.5)
    plt.xlabel("Azimuth angle (measured clockwise from North) [$^{\circ}$]")
    plt.ylabel("Horizon elevation angle [$^{\circ}$]")
    plt.xlim([-5.0, 365.0])
    ax_r = ax_l.twinx()
    plt.fill_between(np.rad2deg(azim), 0.0, hori_dist[ind, :], color="blue",
                     alpha=0.25)
    plt.ylabel("Distance to horizon [km]", color="blue")
    ax_r.tick_params(axis="y", colors="blue")
    plt.title(i.replace("_", " "), fontsize=12, fontweight="bold",
              loc="left")
    title = "Slope angle: %.1f" % topo_param[i]["slope_angle"] + \
            "$^{\circ}$, slope aspect: %.1f" % topo_param[i]["slope_aspect"] \
            + "$^{\circ}$, SVF: %.2f" % topo_param[i]["svf"]
    plt.title(title, fontsize=12, loc="right")
    fig.savefig(path_out + i + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
