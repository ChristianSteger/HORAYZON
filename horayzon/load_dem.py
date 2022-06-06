# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from geographiclib.geodesic import Geodesic
from osgeo import gdal


# -----------------------------------------------------------------------------

def srtm(file_dem, domain):
    """Load SRTM digital elevation model data.

    Load SRTM digital elevation model data from single GeoTIFF file.

    Parameters
    ----------
    file_dem : str
        Path and file name of SRTM tile
    domain : dict
        List with domain boundaries [lon_min, lon_max, lat_min, lat_max]
        [degree]

    Returns
    -------
    lon : ndarray
        Array (one-dimensional) with longitude [degree]
    lat : ndarray
        Array (one-dimensional) with latitude [degree]
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]"""

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
        raise ValueError("SRTM tile does not entirely cover domain")
    slic_lon = slice(np.where(lon <= domain["lon_min"])[0][-1],
                     np.where(lon >= domain["lon_max"])[0][0] + 1)
    slic_lat = slice(np.where(lat >= domain["lat_max"])[0][-1],
                     np.where(lat <= domain["lat_min"])[0][0] + 1)
    elevation = elevation[slic_lat, slic_lon].astype(np.float32)
    lon, lat = lon[slic_lon], lat[slic_lat]

    print_dem_info(elevation)

    return lon, lat, elevation


# -----------------------------------------------------------------------------

def nasadem(files_dem, domain):
    """Load NASADEM digital elevation model data.

    Load NASADEM digital elevation model data from (multiple) NetCDF file(s).

    Parameters
    ----------
    files_dem : list
        List with path and file names of NASADEM tile(s)
    domain : dict
        List with domain boundaries [lon_min, lon_max, lat_min, lat_max]
        [degree]

    Returns
    -------
    lon : ndarray
        Array (one-dimensional) with longitude [degree]
    lat : ndarray
        Array (one-dimensional) with latitude [degree]
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]"""

    # Load digital elevation model data for relevant domain
    ds = xr.open_mfdataset(files_dem, preprocess=preprocess)
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

def dhm25(files_dem, domain):
    """Load DHM25 digital elevation model data.

    Load SRTM digital elevation model data from (multiple) NetCDF file(s).

    Parameters
    ----------
    files_dem : list
        List with path and file name(s) of NASADEM tile(s)
    domain : list
        List with domain boundaries [lon_min, lon_max, lat_min, lat_max]
        [degree]

    Returns
    -------
    lon : ndarray
        Array (one-dimensional) with longitude [degree]
    lat : ndarray
        Array (one-dimensional) with latitude [degree]
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]"""


# -----------------------------------------------------------------------------

def swissalti3d(loc, width, path_tiles):
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


# -----------------------------------------------------------------------------

def print_dem_info(elevation):
    """Print digital elevation model information.

    Parameters
    ----------
    elevation : ndarray
        Array (two-dimensional) with elevation [metre]"""

    print("Size of loaded DEM domain: " + str(elevation.shape))
    print("Elevation range of DEM: %.1f" % elevation.min()
          + " - %.1f" % elevation.max() + " m")
