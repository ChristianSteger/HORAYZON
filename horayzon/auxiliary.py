# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
from tqdm import tqdm
import requests
from pyproj import CRS, Transformer
import glob
from osgeo import gdal
import horayzon


# -----------------------------------------------------------------------------

def get_path_aux_data():
    """Get path for auxiliary data.

        Get path for auxiliary data. Read from text file in 'HORAYZON' main
        directory if already defined, otherwise define by user.

    Returns
    -------
    path_aux_data: str
        Path of auxiliary data"""

    # Create text file with path to auxiliary data
    file_name = "path_aux_data.txt"
    path_horayzon = os.path.join(os.path.split(
        os.path.dirname(horayzon.__file__))[0], "")
    if not os.path.isfile(path_horayzon + "/" + file_name):
        valid_path = False
        print("Provide path for auxiliary data:")
        while not valid_path:
            path_aux_data = os.path.join(input(), "")
            if os.path.isdir(path_aux_data):
                valid_path = True
            else:
                print("Provided path is invalid - try again:")
        file = open(path_horayzon + "/" + file_name, "w")
        file.write(path_aux_data)
        file.close()
    else:
        file = open(path_horayzon + "/" + file_name, "r")
        path_aux_data = file.read()
        file.close()

    return path_aux_data


# -----------------------------------------------------------------------------

def download_file(file_url, file_local_path):
    """Download file from web.

    Download file from web and show progress with bar.

    Parameters
    ----------
    file_url : str
        URL of file to download
    file_local_path: str
        Local path for downloaded file"""

    # Check arguments
    if not os.path.isdir(os.path.split(file_local_path)[0]):
        raise ValueError("Local path does not exist")

    # Try to download file
    try:
        response = requests.get(file_url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 10
        # download seems to be faster with larger block size...
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB",
                            unit_scale=True)
        with open(file_local_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise ValueError("Inconsistency in file size")
    except Exception:
        print("URL of file does not exist")


# -----------------------------------------------------------------------------

def pad_geometry_buffer(buffer):
    """Padding of geometry buffer.

    Pads geometric buffer to make it conformal with 16-byte SSE load
    instructions.

    Parameters
    ----------
    buffer : ndarray
        Array (1-dimensional) with geometry buffer [arbitrary]

    Returns
    -------
    buffer : ndarray
        Array (1-dimensional) with padded geometry buffer [arbitrary]

    Notes
    -----
    This function ensures that vertex buffer size is divisible by 16 and hence
    conformal with 16-byte SSE load instructions (see Embree documentation;
    section 7.45 rtcSetSharedGeometryBuffer)."""

    # Check arguments
    if not isinstance(buffer, np.ndarray):
        raise ValueError("argument 'buffer' has invalid type")
    if buffer.ndim != 1:
        raise ValueError("argument 'buffer' must be one-dimensional")

    add_elem = 16
    if not (buffer.nbytes % 16) == 0:
        add_elem += ((16 - (buffer.nbytes % 16)) // buffer[0].nbytes)
    buffer = np.append(buffer, np.zeros(add_elem, dtype=buffer.dtype))

    return buffer


# -----------------------------------------------------------------------------

def load_swissalti3d(loc, width, path_tiles):
    """Compute Digital Elevation model (DEM) domain.

    Computes required domain of Digital Elevation model (DEM) from location
    and width of inner domain (swissALTI3D DEM).

    Parameters
    ----------
    loc : tuple
        Tuple with geodetic latitude/longitude of centre [degree]
    width : float
        Total x/y-width of domain [kilometre]
    path_tiles : str
        Path to swissALTI3D GeoTIFF tiles

    Returns
    -------
    east: ndarray
        Array (one-dimensional) with east-coordinate [metre]
    north: ndarray
        Array (one-dimensional) with north-coordinate [metre]
    dem: ndarray
        Array (two-dimensional) with DEM [metre]"""

    # Constant settings
    tiles_gc = 500  # number of grid cells per tile
    res_dem = 2.0  # horizontal resolution of DEM
    file_format = "swissalti3d_????_eeee-nnnn_2_2056_5728.tif"

    # Compute coordinates in swiss system (LV95)
    crs_4326 = CRS.from_epsg(4326)
    crs_2056 = CRS.from_epsg(2056)
    transformer = Transformer.from_crs(crs_4326, crs_2056, always_xy=True)
    east_cen, north_cen = transformer.transform(loc[1], loc[0])

    # Determine relevant tiles
    tiles_east = (np.array([int(east_cen - (width / 2.0) * 1000.0),
                            int(east_cen + (width / 2.0) * 1000.0)],
                           dtype=np.float32) / 1000.0).astype(np.int32)
    tiles_north = (np.array([int(north_cen - (width / 2.0) * 1000.0),
                             int(north_cen + (width / 2.0) * 1000.0)],
                            dtype=np.float32) / 1000.0).astype(np.int32)
    tiles_east = list(range(tiles_east[0], tiles_east[-1] + 1))
    tiles_north = list(range(tiles_north[0], tiles_north[-1] + 1))

    # Load DEM data
    file_dem = path_tiles + file_format
    dem_load = np.empty((len(tiles_north) * tiles_gc,
                         len(tiles_east) * tiles_gc),
                        dtype=np.float32)
    dem_load.fill(-9999.0)
    count = 0
    for i in range(len(tiles_north)):
        for j in range(len(tiles_east)):
            file = file_dem.replace("eeee", str(tiles_east[j])) \
                .replace("nnnn", str(tiles_north[i]))
            file = glob.glob(file)
            if len(file) == 0:
                print("Warning: no tile found for e" + str(tiles_east[j])
                      + "n" + str(tiles_north[i]))
            else:
                ds = gdal.Open(file[0])
                slic = (slice(i * tiles_gc, (i + 1) * tiles_gc),
                        slice(j * tiles_gc, (j + 1) * tiles_gc))
                dem_load[slic] = np.flipud(ds.GetRasterBand(1).ReadAsArray())
            count += 1
            if (count == 1) or (count % 100 == 0) \
                    or (count == (len(tiles_north) * len(tiles_east))):
                print("Tiles imported: " + str(count) + " of "
                      + str(len(tiles_north) * len(tiles_east)))

    # CH1903+ / LV95 coordinates
    east_load = np.linspace(tiles_east[0] * 1000.0 + res_dem / 2.0,
                            tiles_east[-1] * 1000.0
                            + tiles_gc * res_dem - res_dem / 2.0,
                            dem_load.shape[1], dtype=np.float32)
    north_load = np.linspace(tiles_north[0] * 1000.0 + res_dem / 2.0,
                             tiles_north[-1] * 1000.0
                             + tiles_gc * res_dem - res_dem / 2.0,
                             dem_load.shape[0], dtype=np.float32)

    # Crop DEM to relevant domain
    ind_east = np.argmin(np.abs(east_cen - east_load))
    ind_north = np.argmin(np.abs(north_cen - north_load))

    add = int(((width / 2.0) * 1000.0) / res_dem)
    slic = (slice(ind_north - add, ind_north + add + 1),
            slice(ind_east - add, ind_east + add + 1))
    dem, north, east = dem_load[slic], north_load[slic[0]], east_load[slic[1]]
    del dem_load, north_load, east_load
    if (dem.shape[0] != int((width / 2.0) * 1000) + 1
            or dem.shape[1] != int((width / 2.0) * 1000) + 1):
        raise ValueError("incorrect shape size of DEM")

    # Check for NaN-values
    if np.any(dem == -9999.0):
        print("Warning: Nan-values (-9999.0) detected")

    return east, north, dem
