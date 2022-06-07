# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from osgeo import gdal
import xarray as xr
import glob


# -----------------------------------------------------------------------------

def srtm(file_dem, domain):
    """Load SRTM digital elevation model data.

    Load SRTM digital elevation model data from single GeoTIFF file.

    Parameters
    ----------
    file_dem : str
        Name of SRTM tile
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
    Data source: https://srtm.csi.cgiar.org"""

    # Load digital elevation model data
    ds = gdal.Open(file_dem)
    elevation = ds.GetRasterBand(1).ReadAsArray()  # 16-bit integer
    d_lon = ds.GetGeoTransform()[1]
    lon = np.linspace(ds.GetGeoTransform()[0] + (d_lon / 2.0),
                      ds.GetGeoTransform()[0] + d_lon * ds.RasterXSize
                      - (d_lon / 2.0), ds.RasterXSize)
    d_lat = ds.GetGeoTransform()[5]
    lat = np.linspace(ds.GetGeoTransform()[3] + (d_lat / 2.0),
                      ds.GetGeoTransform()[3] + d_lat * ds.RasterYSize
                      - (d_lat / 2.0), ds.RasterYSize)

    # Crop relevant domain
    if sum([domain["lon_min"] > lon.min(), domain["lon_max"] < lon.max(),
            domain["lat_min"] > lat.min(), domain["lat_max"] < lat.max()]) \
            != 4:
        raise ValueError("Provided tile does not cover domain")
    slice_lon = slice(np.where(lon <= domain["lon_min"])[0][-1],
                      np.where(lon >= domain["lon_max"])[0][0] + 1)
    slice_lat = slice(np.where(lat >= domain["lat_max"])[0][-1],
                      np.where(lat <= domain["lat_min"])[0][0] + 1)
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
    Data source: https://lpdaac.usgs.gov/tools/earthdata-search/"""

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

def dhm25(file_dem, domain):
    """Load DHM25 digital elevation model data.

    Load SRTM digital elevation model data from single ESRI ASCII GRID file.

    Parameters
    ----------
    file_dem : str
        Name of DHM25 tile
    domain : dict
        Dictionary with domain boundaries (x_min, x_max, y_min, y_max) [metre]

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

    # Load digital elevation model data
    ds = gdal.Open(file_dem)
    elevation = ds.GetRasterBand(1).ReadAsArray()  # 32-bit float
    d_x = ds.GetGeoTransform()[1]
    x = np.linspace(ds.GetGeoTransform()[0] + (d_x / 2.0),
                    ds.GetGeoTransform()[0] + d_x * ds.RasterXSize
                    - (d_x / 2.0), ds.RasterXSize)
    d_y = ds.GetGeoTransform()[5]
    y = np.linspace(ds.GetGeoTransform()[3] + (d_y / 2.0),
                    ds.GetGeoTransform()[3] + d_y * ds.RasterYSize
                    - (d_y / 2.0), ds.RasterYSize)

    # Crop relevant domain
    if sum([domain["x_min"] > x.min(), domain["x_max"] < x.max(),
            domain["y_min"] > y.min(), domain["y_max"] < y.max()]) \
            != 4:
        raise ValueError("Provided tile does not cover domain")
    slice_x = slice(np.where(x <= domain["x_min"])[0][-1],
                    np.where(x >= domain["x_max"])[0][0] + 1)
    slice_y = slice(np.where(y >= domain["y_max"])[0][-1],
                    np.where(y <= domain["y_min"])[0][0] + 1)
    elevation = elevation[slice_y, slice_x]
    x, y = x[slice_x], y[slice_y]

    # Set "no data values" to NaN
    elevation[elevation == -9999.0] = np.nan

    print_dem_info(elevation)

    return x, y, elevation


# -----------------------------------------------------------------------------

def swissalti3d(path_dem, domain):
    """Load swissALTI3D digital elevation model data.

    Load swissALTI3D digital elevation model data from multiple GeoTIFF files.

    Parameters
    ----------
    path_dem : str
        Path of swissALTI3D tiles
    domain : dict
        Dictionary with domain boundaries (x_min, x_max, y_min, y_max) [metre]

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

    # Constant settings
    tiles_gc = 500  # number of grid cells per tile
    file_format = "swissalti3d_????_eeee-nnnn_2_2056_5728.tif"

    # Determine relevant tiles
    tiles_east = list(range(int(np.floor(domain["x_min"] / 1000)),
                            int(np.floor(domain["x_max"] / 1000)) + 1))
    tiles_north = list(range(int(np.floor(domain["y_max"] / 1000)),
                             int(np.floor(domain["y_min"] / 1000)) - 1, -1))

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
                ds = gdal.Open(file[0])
                slic = (slice((tiles_north[0] - i) * tiles_gc,
                              (tiles_north[0] - i + 1) * tiles_gc),
                        slice((j - tiles_east[0]) * tiles_gc,
                              (j - tiles_east[0] + 1) * tiles_gc))
                elevation[slic] = ds.GetRasterBand(1).ReadAsArray()
            count += 1
            if (count == 1) or (count % 100 == 0) or (count == num_tiles):
                print("Tiles imported: " + str(count) + " of "
                      + str(num_tiles))

    # Generate LV95 coordinates
    dx = ds.GetGeoTransform()[1]  # resolution of DEM in x-direction [m]
    dy = ds.GetGeoTransform()[5]  # resolution of DEM in y-direction [m]
    x = np.linspace(tiles_east[0] * 1000.0 + (dx / 2.0),
                    tiles_east[0] * 1000.0 + (dx / 2.0)
                    + (elevation.shape[1] - 1) * dx,
                    elevation.shape[1], dtype=np.float32)
    y = np.linspace((tiles_north[0] + 1) * 1000.0 + (dy / 2.0),
                    (tiles_north[0] + 1) * 1000.0 + (dy / 2.0)
                    + (elevation.shape[0] - 1) * dy,
                    elevation.shape[0], dtype=np.float32)

    # Crop relevant domain
    if sum([domain["x_min"] > x.min(), domain["x_max"] < x.max(),
            domain["y_min"] > y.min(), domain["y_max"] < y.max()]) \
            != 4:
        raise ValueError("Provided tile does not cover domain")
    slice_x = slice(np.where(x <= domain["x_min"])[0][-1],
                    np.where(x >= domain["x_max"])[0][0] + 1)
    slice_y = slice(np.where(y >= domain["y_max"])[0][-1],
                    np.where(y <= domain["y_min"])[0][0] + 1)
    x, y = x[slice_x], y[slice_y]
    elevation = elevation[slice_y, slice_x]

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
