# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
from shapely.geometry import shape, box
import fiona
from scipy.spatial import cKDTree
import pygeos
import time
from skimage.measure import find_contours
import horayzon.transform as transform
from horayzon.auxiliary import get_path_aux_data
from horayzon.download import file as download_file
import zipfile
import shutil


# -----------------------------------------------------------------------------

def get_gshhs_coastlines(domain):
    """Get relevant GSHHS coastline data.

    Get relevant GSHHS coastline data for rectangular latitude/longitude
    domain.

    Parameters
    ----------
    domain : dict
        Dictionary with domain boundaries (lon_min, lon_max, lat_min, lat_max)
        [degree]

    Returns
    -------
    poly_coastlines : list
        Relevant coastline polygons as Shapely polygons"""

    # Check arguments
    keys_req = ("lon_min", "lon_max", "lat_min", "lat_max")
    if not set(keys_req).issubset(set(domain.keys())):
        raise ValueError("one or multiple key(s) are missing in 'domain'")
    if (domain["lon_min"] >= domain["lon_max"]) \
            or (domain["lat_min"] >= domain["lat_max"]):
        raise ValueError("invalid domain extent")

    # Download data
    path_aux_data = get_path_aux_data()
    if not os.path.isdir(path_aux_data + "GSHHG"):
        file_url = "http://www.soest.hawaii.edu/pwessel/gshhg/" \
                   + "gshhg-shp-2.3.7.zip"
        print("Download GSHHG data:")
        download_file(file_url, path_aux_data)
        file_zipped = path_aux_data + os.path.split(file_url)[-1]
        with zipfile.ZipFile(file_zipped, "r") as zip_ref:
            zip_ref.extractall(path_aux_data + "GSHHG")
        os.remove(file_zipped)

        # Remove superfluous data (larger files)
        shutil.rmtree(path_aux_data + "GSHHG/WDBII_shp/", ignore_errors=True)
        shutil.rmtree(path_aux_data + "GSHHG/GSHHS_shp/h/", ignore_errors=True)
        shutil.rmtree(path_aux_data + "GSHHG/GSHHS_shp/i/", ignore_errors=True)

    t_beg_func = time.time()

    # Compute and save bounding boxes of coastlines polygons
    file_bbc = path_aux_data + "GSHHG/Bounding_boxes_coastlines.npy"
    if not os.path.isfile(file_bbc):
        t_beg = time.time()
        ds = fiona.open(path_aux_data + "GSHHG/GSHHS_shp/f/GSHHS_f_L1.shp")
        bounds = np.empty((len(ds), 4), dtype=np.float32)
        for idx, var in enumerate(ds):
            bounds[idx, :] = shape(var["geometry"]).bounds
            # (lon_min, lat_min, lon_max, lat_max)
        ds.close()
        np.save(file_bbc, bounds)
        print("Bounding boxes for coastline polygons computed "
              + "(%.2f" % (time.time() - t_beg) + " s)")

    # Find relevant polygons for domain
    bounds = np.load(file_bbc)
    geoms = pygeos.box(bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3])
    tree = pygeos.STRtree(geoms)
    quer_rang = [domain["lon_min"], domain["lat_min"],
                 domain["lon_max"], domain["lat_max"]]
    ind = tree.query(pygeos.box(*quer_rang))

    # Load relevant polygons
    ds = fiona.open(path_aux_data + "GSHHG/GSHHS_shp/f/GSHHS_f_L1.shp")
    poly_all = [shape(ds[int(i)]["geometry"]) for i in ind]
    ds.close()
    print("Number of polygons: " + str(len(poly_all)))

    # Crop polygons (if necessary)
    quer_rang_s = box(*quer_rang)
    poly_coastlines = []
    for i in poly_all:
        if quer_rang_s.contains(i):
            poly_coastlines.append(i)
        elif quer_rang_s.intersects(i):
            poly_coastlines.append(quer_rang_s.intersection(i))

    print("Run time: %.2f" % (time.time() - t_beg_func) + " s")

    return poly_coastlines


# -----------------------------------------------------------------------------

def coastline_contours(lon, lat, mask_bin):
    """Compute coastline contours.

    Compute coastline contours from binary land-sea mask.

    Parameters
    ----------
    lon : ndarray of double
        Array (1-dimensional) with geographic longitude [degree]
    lat: ndarray of double
        Array (1-dimensional) with geographic latitude [degree]
    mask_bin: str
        Array (2-dimensional) with binary land-sea mask (0: water, 1: land)

    Returns
    -------
    contours_latlon : list
        List with contour lines in latitude/longitude coordinates [degree]"""

    # Check arguments
    if (lat.ndim != 1) or (lon.ndim != 1):
        raise ValueError("Input coordinates arrays must be 1-dimensional")
    if (mask_bin.shape[0] != len(lat)) or (mask_bin.shape[1] != len(lon)):
        raise ValueError("Input data has inconsistent dimension length(s)")
    if (mask_bin.dtype != "uint8") or (len(np.unique(mask_bin)) != 2) \
            or (not np.all(np.unique(mask_bin) == [0, 1])):
        raise ValueError("'mask_bin' must be of type 'uint8' and may "
                         + "only contain 0 and 1")

    t_beg_func = time.time()

    # Compute contour lines
    contours = find_contours(mask_bin, 0.5, fully_connected="high")

    # Get latitude/longitude coordinates of contours
    lon_ind = np.linspace(lon[0], lon[-1], len(lon) * 2 - 1)
    lat_ind = np.linspace(lat[0], lat[-1], len(lat) * 2 - 1)
    contours_latlon = []
    for i in contours:
        pts_latlon = np.empty(i.shape, dtype=np.float64)
        pts_latlon[:, 0] = lon_ind[(i[:, 1] * 2).astype(np.int32)]
        pts_latlon[:, 1] = lat_ind[(i[:, 0] * 2).astype(np.int32)]
        contours_latlon.append(pts_latlon)

    print("Run time: %.2f" % (time.time() - t_beg_func) + " s")

    return contours_latlon


# -----------------------------------------------------------------------------

def coastline_distance(x_ecef, y_ecef, z_ecef, mask_land, pts_ecef):
    """Compute minimal chord distance.

    Compute minimal chord distance between all water grid cells (centre)
    and the coastline.

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (two-dimensional) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (two-dimensional) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (two-dimensional) with ECEF z-coordinates [metre]
    mask_land: ndarray of bool
        Array (two-dimensional) with land mask
    pts_ecef: ndarray of double
        Array (two-dimensional) with ECEF coordinates of coastline vertices
        (number of vertices, x/y/z) [metre]

    Returns
    -------
    dist_chord : ndarray of double
        Array (2-dimensional) minimal chord distance between grid cells and
        coastline [metre]"""

    # Check arguments
    if x_ecef.shape != mask_land.shape:
        raise ValueError("Input data has inconsistent dimension length(s)")
    if mask_land.dtype != "bool":
        raise ValueError("'mask_land' must be a boolean mask")

    t_beg_func = time.time()

    # Build k-d tree
    tree = cKDTree(pts_ecef)

    # Query k-d tree
    pts_quer = np.vstack((x_ecef[~mask_land], y_ecef[~mask_land],
                          z_ecef[~mask_land])).transpose()
    dist_quer, idx = tree.query(pts_quer, k=1, workers=-1)

    # Save distances in two-dimensional array
    dist_chord = np.empty(x_ecef.shape, dtype=np.float64)
    dist_chord.fill(np.nan)
    dist_chord[~mask_land] = dist_quer  # [m]

    print("Run time: %.2f" % (time.time() - t_beg_func) + " s")

    return dist_chord


# -----------------------------------------------------------------------------

def coastline_buffer(x_ecef, y_ecef, z_ecef, mask_land, pts_ecef, lat,
                     dist_thr, dem_res, ellps, block_size=(5 * 2 + 1)):
    """Compute mask according to coastline buffer.

    Compute mask according to coastline buffer. Grid cells, whose minimal
    chord distance from the coastline is longer than 'dist_thr', are masked.

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (two-dimensional) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (two-dimensional) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (two-dimensional) with ECEF z-coordinates [metre]
    mask_land: ndarray of bool
        Array (two-dimensional) with land mask
    pts_ecef: ndarray of double
        Array (two-dimensional) with ECEF coordinates of coastline vertices
        (number of vertices, x/y/z) [metre]
    lat: ndarray of double
        Array (1-dimensional) with geographic latitude [degree]
    dist_thr: double
        Threshold for minimal distance from coastline [metre]
    dem_res: double
        Spatial resolution of digital elevation model [degree]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)
    block_size: int
        Block size of grid cells that are processed together.

    Returns
    -------
    mask_buffer : ndarray of bool
        Array (2-dimensional) with grid cells that are located outside
        of the coastline buffer [metre]"""

    # Check arguments
    if (x_ecef.shape != mask_land.shape) or (x_ecef.shape[0] != len(lat)):
        raise ValueError("Input data has inconsistent dimension length(s)")
    if mask_land.dtype != "bool":
        raise ValueError("'mask_land' must be a boolean mask")
    if ellps not in ("sphere", "WGS84", "GRS80"):
        raise ValueError("invalid value for 'ellps'")
    if block_size % 2 != 1:
        raise ValueError("Integer value for 'block_size' must be uneven")

    t_beg_func = time.time()

    # Compute maximal chord length for block (-> diagonal at equator)
    # lat_ini = 0.0  # equator
    lat_ini = np.maximum(np.abs(lat).min() - 1.0, 0.0)
    lon_max = np.array([[0.0,
                         0.0 + dem_res * int((block_size - 1) / 2)]],
                       dtype=np.float64).reshape(1, 2)
    lat_max = np.array([[lat_ini,
                         lat_ini + dem_res * int((block_size - 1) / 2)]],
                       dtype=np.float64).reshape(1, 2)
    h_max = np.zeros(lon_max.shape, dtype=np.float32)
    coord_ecef = transform.lonlat2ecef(lon_max, lat_max, h_max, ellps=ellps)
    chord_max = np.sqrt(np.diff(coord_ecef[0])[0][0] ** 2
                        + np.diff(coord_ecef[1])[0][0] ** 2
                        + np.diff(coord_ecef[2])[0][0] ** 2)
    if chord_max > dist_thr:
        raise ValueError("Maximal chord distance is larger than 'dist_thr'")

    # Build k-d tree
    tree = cKDTree(pts_ecef)

    # Query k-d tree
    slic = (slice(int((block_size - 1) / 2), None, block_size),
            slice(int((block_size - 1) / 2), None, block_size))
    t_beg = time.time()
    pts_quer = np.vstack((x_ecef[slic].ravel(), y_ecef[slic].ravel(),
                          z_ecef[slic].ravel())).transpose()
    dist_quer, idx = tree.query(pts_quer, k=1, workers=-1)
    print("Query k-d tree (%.2f" % (time.time() - t_beg) + " s)")

    # # Categorise blocks (old)
    # t_beg = time.time()
    # shp = x_ecef[slic].shape
    # dist_2d = dist_quer.reshape(shp)
    # mask_buffer_old = np.empty(x_ecef.shape, dtype=np.int32)
    # mask_buffer_old.fill(-1)
    # for i in range(shp[0]):
    #     for j in range(shp[1]):
    #         slic_sd = (slice(i * block_size, (i + 1) * block_size),
    #                    slice(j * block_size, (j + 1) * block_size))
    #         # -> can exceed limits of 'mask_buffer' -> exceeding is
    #         #    automatically handled
    #         if dist_2d[i, j] <= (dist_thr - chord_max):  # inside buffer
    #             mask_buffer_old[slic_sd] = 0
    #         elif dist_2d[i, j] > (dist_thr + chord_max):  # outside buffer
    #             mask_buffer_old[slic_sd] = 1
    # print("Categorise blocks (old) (%.2f" % (time.time() - t_beg) + " s)")

    # Categorise blocks
    # t_beg = time.time()
    shp = x_ecef[slic].shape
    dist_2d = dist_quer.reshape(shp)
    mask_buffer = np.empty(x_ecef.shape, dtype=np.int32)
    mask_buffer.fill(-1)
    mask = np.empty(dist_2d.shape, dtype=np.int32)
    mask.fill(-1)
    mask[dist_2d <= (dist_thr - chord_max)] = 0  # inside buffer
    mask[dist_2d > (dist_thr + chord_max)] = 1  # outside buffer
    slic_sd = (slice(0, shp[0] * block_size), slice(0, shp[1] * block_size))
    mask_buffer[slic_sd] = np.repeat(np.repeat(mask, block_size, axis=0),
                                     block_size, axis=1)[:x_ecef.shape[0],
                                                         :x_ecef.shape[1]]
    # print("Categorise blocks (new) (%.2f" % (time.time() - t_beg) + " s)")
    # print(np.all(mask_buffer == mask_buffer_old))

    # Categorise remaining grid cells
    mask_rem = (mask_buffer == -1)
    gc_frac = mask_rem.sum() / mask_buffer.size * 100.0
    print("Number of remaining grid cells: " + str(mask_rem.sum())
          + " (fraction: %.2f" % gc_frac + " %)")
    t_beg = time.time()
    pts_quer = np.vstack((x_ecef[mask_rem], y_ecef[mask_rem],
                          z_ecef[mask_rem])).transpose()
    dist_quer, idx = tree.query(pts_quer, k=1, workers=-1)
    print("Query k-d tree (%.2f" % (time.time() - t_beg) + " s)")
    mask_buffer[mask_rem] = (dist_quer > dist_thr).astype(np.int32)
    mask_buffer[mask_land] = 0

    print("Run time: %.2f" % (time.time() - t_beg_func) + " s")

    return mask_buffer.astype(bool)
