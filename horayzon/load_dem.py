# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
from importlib import import_module
import xarray as xr
import glob


# -----------------------------------------------------------------------------

def srtm(file_dem, domain, engine="gdal"):
    """Load SRTM digital elevation model data.

    Load SRTM digital elevation model data from single GeoTIFF file.

    Parameters
    ----------
    file_dem : str
        Name of SRTM tile
    domain : dict
        Dictionary with domain boundaries (lon_min, lon_max, lat_min, lat_max)
        [degree]
    engine: str
        Backend for loading GeoTIFF file (either 'gdal' or 'pillow')

    Returns
    -------
    lon : ndarray
        Array (one-dimensional) with longitude [degree]
    lat : ndarray
        Array (one-dimensional) with latitude [degree]
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]

    Notes
    -----
    Data source: https://srtm.csi.cgiar.org"""

    # Check arguments
    if engine not in ("gdal", "pillow"):
        raise ValueError("Input for 'engine' must be either "
                         "'gdal' or 'pillow'")

    # Load digital elevation model data
    if engine == "gdal":
        print("Read GeoTIFF with GDAL")
        gdal = import_module("osgeo.gdal")
        ds = gdal.Open(file_dem)
        elevation = ds.GetRasterBand(1).ReadAsArray()  # 16-bit integer
        raster_size_x, raster_size_y = ds.RasterXSize, ds.RasterYSize
        lon_ulc, lat_ulc = ds.GetGeoTransform()[0], ds.GetGeoTransform()[3]
        d_lon, d_lat = ds.GetGeoTransform()[1], ds.GetGeoTransform()[5]
    else:
        print("Read GeoTIFF with Pillow")
        if (os.path.getsize(file_dem) / (1024 ** 2)) > 500.0:
            print("Warning: reading of large GeoTIFF file with Pillow is slow")
        Image = import_module("PIL.Image")
        Image.MAX_IMAGE_PIXELS = 1300000000
        img = Image.open(file_dem)
        elevation = np.array(img)  # 32-bit integer
        raster_size_x, raster_size_y = img.tag[256][0], img.tag[257][0]
        lon_ulc, lat_ulc = img.tag[33922][3], img.tag[33922][4]
        d_lon, d_lat = img.tag[33550][0], -img.tag[33550][1]
        # Warning: unclear where sign of n-s pixel resolution is stored!
    lon_edge = np.linspace(lon_ulc, lon_ulc + d_lon * raster_size_x,
                           raster_size_x + 1)
    lat_edge = np.linspace(lat_ulc, lat_ulc + d_lat * raster_size_y,
                           raster_size_y + 1)
    lon = lon_edge[:-1] + np.diff(lon_edge / 2.0)
    lat = lat_edge[:-1] + np.diff(lat_edge / 2.0)

    # Crop relevant domain
    if any([domain["lon_min"] < lon_edge.min(),
            domain["lon_max"] > lon_edge.max(),
            domain["lat_min"] < lat_edge.min(),
            domain["lat_max"] > lat_edge.max()]):
        raise ValueError("Provided tile does not cover domain")
    slice_lon = slice(np.where(lon_edge <= domain["lon_min"])[0][-1],
                      np.where(lon_edge >= domain["lon_max"])[0][0])
    slice_lat = slice(np.where(lat_edge >= domain["lat_max"])[0][-1],
                      np.where(lat_edge <= domain["lat_min"])[0][0])
    elevation = elevation[slice_lat, slice_lon].astype(np.float32)
    lon, lat = lon[slice_lon], lat[slice_lat]

    print_dem_info(elevation)

    return lon, lat, elevation


# -----------------------------------------------------------------------------

def nasadem(files_dem, domain):
    """Load NASADEM digital elevation model data.

    Load NASADEM digital elevation model data from (multiple) NetCDF file(s).

    Parameters
    ----------
    files_dem : str or list
        String with search pattern for NASADEM tiles or list with files
    domain : dict
        Dictionary with domain boundaries (lon_min, lon_max, lat_min, lat_max)
        [degree]

    Returns
    -------
    lon : ndarray
        Array (one-dimensional) with longitude [degree]
    lat : ndarray
        Array (one-dimensional) with latitude [degree]
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]

    Notes
    -----
    Data source: https://lpdaac.usgs.gov/tools/earthdata-search/

    To do
    -----
    Domain selection is not performed according to edge coordinates
    (-> inconsistent with other 'load_dem' functions)"""

    # Load digital elevation model data for relevant domain
    ds = xr.open_mfdataset(files_dem, preprocess=preprocess)
    if sum([domain["lon_min"] > ds["lon"].values.min(),
            domain["lon_max"] < ds["lon"].values.max(),
            domain["lat_min"] > ds["lat"].values.min(),
            domain["lat_max"] < ds["lat"].values.max()]) != 4:
        raise ValueError("Provided tile(s) does/do not cover domain")
    ds = ds.sel(lon=slice(domain["lon_min"], domain["lon_max"]),
                lat=slice(domain["lat_max"], domain["lat_min"]))
    elevation = ds["NASADEM_HGT"].values
    lon = ds["lon"].values
    lat = ds["lat"].values
    ds.close()

    print_dem_info(elevation)

    return lon, lat, elevation


def preprocess(ds):
    """Remove double grid cell row/column at margins """
    return ds.isel(lon=slice(0, 3600), lat=slice(0, 3600))


# -----------------------------------------------------------------------------

def dhm25(file_dem, domain, engine="gdal"):
    """Load DHM25 digital elevation model data.

    Load SRTM digital elevation model data from single ESRI ASCII GRID file.

    Parameters
    ----------
    file_dem : str
        Name of DHM25 tile
    domain : dict
        Dictionary with domain boundaries (x_min, x_max, y_min, y_max) [metre]
    engine: str
        Backend for loading ESRI ASCII GRID file (either 'gdal' or 'numpy')

    Returns
    -------
    x : ndarray
        Array (one-dimensional) with x-coordinate [metre]
    y : ndarray
        Array (one-dimensional) with y-coordinate [metre]
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]

    Notes
    -----
    Data source: https://www.swisstopo.admin.ch/en/geodata/height/dhm25.html"""

    # Check arguments
    if engine not in ("gdal", "numpy"):
        raise ValueError("Input for 'engine' must be either "
                         "'gdal' or 'numpy'")

    # Load digital elevation model data
    if engine == "gdal":
        print("Read ESRI ASCII GRID file with GDAL")
        gdal = import_module("osgeo.gdal")
        ds = gdal.Open(file_dem)
        elevation = ds.GetRasterBand(1).ReadAsArray()  # 32-bit float
        raster_size_x, raster_size_y = ds.RasterXSize, ds.RasterYSize
        x_ulc, y_ulc = ds.GetGeoTransform()[0], ds.GetGeoTransform()[3]
        d_x, d_y = ds.GetGeoTransform()[1], ds.GetGeoTransform()[5]
    else:
        print("Read ESRI ASCII GRID file with NumPy")
        if (os.path.getsize(file_dem) / (1024 ** 2)) > 500.0:
            print("Warning: reading of large ESRI ASCII GRID file with NumPy"
                  " is slow")
        elevation = np.loadtxt(file_dem, skiprows=6, dtype=np.float32)
        header = {}
        with open(file_dem) as file:
            for i in range(5):
                line = next(file).rstrip("\n").split()
                if line[0] in ("ncols", "nrows"):
                    header[line[0]] = int(line[1])
                else:
                    header[line[0]] = float(line[1])
        raster_size_x, raster_size_y = header["ncols"], header["nrows"]
        x_ulc = header["xllcorner"]
        y_ulc = header["yllcorner"] + header["nrows"] * header["cellsize"]
        d_x, d_y = header["cellsize"], -header["cellsize"]
        # Warning: unclear where sign of n-s pixel resolution is stored!

    x_edge = np.linspace(x_ulc, x_ulc + d_x * raster_size_x,
                         raster_size_x + 1, dtype=np.float32)
    y_edge = np.linspace(y_ulc, y_ulc + d_y * raster_size_y,
                         raster_size_y + 1, dtype=np.float32)
    x = x_edge[:-1] + np.diff(x_edge / 2.0)
    y = y_edge[:-1] + np.diff(y_edge / 2.0)

    # Crop relevant domain
    if any([domain["x_min"] < x_edge.min(),
            domain["x_max"] > x_edge.max(),
            domain["y_min"] < y_edge.min(),
            domain["y_max"] > y_edge.max()]):
        raise ValueError("Provided tile does not cover domain")
    slice_x = slice(np.where(x_edge <= domain["x_min"])[0][-1],
                    np.where(x_edge >= domain["x_max"])[0][0])
    slice_y = slice(np.where(y_edge >= domain["y_max"])[0][-1],
                    np.where(y_edge <= domain["y_min"])[0][0])
    elevation = elevation[slice_y, slice_x]
    x, y = x[slice_x], y[slice_y]

    # Set "no data values" to NaN
    elevation[elevation == -9999.0] = np.nan

    print_dem_info(elevation)

    return x, y, elevation


# -----------------------------------------------------------------------------

def swissalti3d(path_dem, domain, engine="gdal"):
    """Load swissALTI3D digital elevation model data.

    Load swissALTI3D digital elevation model data from multiple GeoTIFF files.

    Parameters
    ----------
    path_dem : str
        Path of swissALTI3D tiles
    domain : dict
        Dictionary with domain boundaries (x_min, x_max, y_min, y_max) [metre]
    engine: str
        Backend for loading GeoTIFF file (either 'gdal' or 'pillow')

    Returns
    -------
    x : ndarray
        Array (one-dimensional) with x-coordinate [metre]
    y : ndarray
        Array (one-dimensional) with y-coordinate [metre]
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]

    Notes
    -----
    Data source:
        https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html"""

    # Check arguments
    if engine not in ("gdal", "pillow"):
        raise ValueError("Input for 'engine' must be either "
                         "'gdal' or 'pillow'")

    # Constant settings
    tiles_gc = 500  # number of grid cells per tile
    file_format = "swissalti3d_????_eeee-nnnn_2_2056_5728.tif"

    # Determine relevant tiles
    tiles_east = list(range(int(np.floor(domain["x_min"] / 1000)),
                            int(np.ceil(domain["x_max"] / 1000))))
    tiles_north = list(range(int(np.floor(domain["y_min"] / 1000)),
                             int(np.ceil(domain["y_max"] / 1000))))[::-1]

    # Load required module
    if engine == "gdal":
        print("Read GeoTIFF with GDAL")
        gdal = import_module("osgeo.gdal")
    else:
        print("Read GeoTIFF with Pillow")
        Image = import_module("PIL.Image")

    # Load DEM data
    elevation = np.empty((len(tiles_north) * tiles_gc,
                          len(tiles_east) * tiles_gc), dtype=np.float32)
    elevation.fill(np.nan)
    count = 0
    num_tiles = len(tiles_north) * len(tiles_east)
    for i in tiles_north:
        for j in tiles_east:
            file = (path_dem + file_format).replace("eeee", str(j)) \
                .replace("nnnn", str(i))
            file = glob.glob(file)
            if len(file) == 0:
                print("Warning: no tile found for e" + str(j) + "n" + str(i))
            else:
                slic = (slice((tiles_north[0] - i) * tiles_gc,
                              (tiles_north[0] - i + 1) * tiles_gc),
                        slice((j - tiles_east[0]) * tiles_gc,
                              (j - tiles_east[0] + 1) * tiles_gc))
                if engine == "gdal":
                    ds = gdal.Open(file[0])
                    elevation[slic] = ds.GetRasterBand(1).ReadAsArray()
                else:
                    img = Image.open(file[0])
                    elevation[slic] = np.array(img)
            count += 1
            if (count == 1) or (count % 200 == 0) or (count == num_tiles):
                print("Tiles imported: " + str(count) + " of "
                      + str(num_tiles))

    # Generate LV95 coordinates
    d_x = 2.0  # resolution of DEM in x-direction [m]
    d_y = -2.0  # resolution of DEM in y-direction [m]
    x_edge = np.linspace(tiles_east[0] * 1000.0,
                         tiles_east[0] * 1000.0 + elevation.shape[1] * d_x,
                         elevation.shape[1] + 1, dtype=np.float32)
    y_edge = np.linspace((tiles_north[0] + 1) * 1000.0,
                         (tiles_north[0] + 1) * 1000.0
                         + elevation.shape[0] * d_y,
                         elevation.shape[0] + 1, dtype=np.float32)
    x = x_edge[:-1] + np.diff(x_edge / 2.0)
    y = y_edge[:-1] + np.diff(y_edge / 2.0)

    # Crop relevant domain
    slice_x = slice(np.where(x_edge <= domain["x_min"])[0][-1],
                    np.where(x_edge >= domain["x_max"])[0][0])
    slice_y = slice(np.where(y_edge >= domain["y_max"])[0][-1],
                    np.where(y_edge <= domain["y_min"])[0][0])
    x, y = x[slice_x], y[slice_y]
    elevation = elevation[slice_y, slice_x]

    print_dem_info(elevation)

    return x, y, elevation


# -----------------------------------------------------------------------------

def rema(file_dem, domain, engine="gdal"):
    """Load REMA digital elevation model data.

    Load REMA digital elevation model data from single GeoTIFF file.

    Parameters
    ----------
    file_dem : str
        Name of REMA tile
    domain : dict
        Dictionary with domain boundaries (x_min, x_max, y_min, y_max) [metre]
    engine: str
        Backend for loading GeoTIFF file (either 'gdal' or 'pillow')

    Returns
    -------
    x : ndarray
        Array (one-dimensional) with x-coordinate [metre]
    y : ndarray
        Array (one-dimensional) with y-coordinate [metre]
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]

    Notes
    -----
    Data source: https://www.pgc.umn.edu/data/rema/"""

    # Check arguments
    if engine not in ("gdal", "pillow"):
        raise ValueError("Input for 'engine' must be either "
                         "'gdal' or 'pillow'")

    # Load digital elevation model data
    if engine == "gdal":
        print("Read GeoTIFF with GDAL")
        gdal = import_module("osgeo.gdal")
        ds = gdal.Open(file_dem)
        elevation = ds.GetRasterBand(1).ReadAsArray()  # 32-bit float
        raster_size_x, raster_size_y = ds.RasterXSize, ds.RasterYSize
        x_ulc, y_ulc = ds.GetGeoTransform()[0], ds.GetGeoTransform()[3]
        d_x, d_y = ds.GetGeoTransform()[1], ds.GetGeoTransform()[5]
    else:
        print("Read GeoTIFF with Pillow")
        if (os.path.getsize(file_dem) / (1024 ** 2)) > 500.0:
            print("Warning: reading of large GeoTIFF file with Pillow is slow")
        Image = import_module("PIL.Image")
        Image.MAX_IMAGE_PIXELS = 1300000000
        img = Image.open(file_dem)
        elevation = np.array(img)  # 32-bit float
        raster_size_x, raster_size_y = img.tag[256][0], img.tag[257][0]
        x_ulc, y_ulc = img.tag[33922][3], img.tag[33922][4]
        d_x, d_y = img.tag[33550][0], -img.tag[33550][1]
        # Warning: unclear where sign of n-s pixel resolution is stored!
    x_edge = np.linspace(x_ulc, x_ulc + d_x * raster_size_x,
                         raster_size_x + 1)
    y_edge = np.linspace(y_ulc, y_ulc + d_y * raster_size_y,
                         raster_size_y + 1)
    x = x_edge[:-1] + np.diff(x_edge / 2.0)
    y = y_edge[:-1] + np.diff(y_edge / 2.0)

    # Crop relevant domain
    if any([domain["x_min"] < x_edge.min(),
            domain["x_max"] > x_edge.max(),
            domain["y_min"] < y_edge.min(),
            domain["y_max"] > y_edge.max()]):
        raise ValueError("Provided tile does not cover domain")
    slice_x = slice(np.where(x_edge <= domain["x_min"])[0][-1],
                    np.where(x_edge >= domain["x_max"])[0][0])
    slice_y = slice(np.where(y_edge >= domain["y_max"])[0][-1],
                    np.where(y_edge <= domain["y_min"])[0][0])
    elevation = elevation[slice_y, slice_x].astype(np.float32)
    x, y = x[slice_x], y[slice_y]

    print_dem_info(elevation)

    return x, y, elevation


# -----------------------------------------------------------------------------

def print_dem_info(elevation):
    """Print digital elevation model information.

    Parameters
    ----------
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]"""

    print("Size of loaded DEM domain: " + str(elevation.shape))
    txt = "Elevation range of DEM: %.1f" % np.nanmin(elevation) \
          + " - %.1f" % np.nanmax(elevation) + " m"
    if np.any(np.isnan(elevation)):
        txt = txt + " (Warning: NaN values are present)"
    print(txt)
