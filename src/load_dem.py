# Load modules
import numpy as np
from geographiclib.geodesic import Geodesic
from pyproj import CRS, Transformer
import glob
from osgeo import gdal, osr


###############################################################################

def dem_domain_loc(loc, width_in, dist_s=50.0, ellps="sphere"):
    """Compute Digital Elevation model (DEM) domain.

    Computes required domain of Digital Elevation model (DEM) from location
    and width of inner domain (geodetic coordinates).

    Parameters
    ----------
    loc : tuple
        Tuple with geodetic latitude/longitude of centre [degree]
    width_in : float
        Total x/y-width of inner domain for which horizon is computed
        [kilometre]
    dist_s : float
        Search distance for horizon [kilometre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)

    Returns
    -------
    dom : dict
        Dictionary with boundaries of inner and total domain [degree]

    Notes
    -----
    Source:
    - Geoid parameters a and f: PROJ
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    # Check arguments
    if ellps not in ("sphere", "GRS80", "WGS84"):
        raise NotImplementedError("ellipsoid " + ellps + " is not supported")

    # Initialise geodesic
    if ellps == "sphere":
        a = 6370997.0  # earth radius [m]
        f = 0.0  # flattening [-]
    elif ellps == "GRS80":
        a = 6378137.0  # equatorial radius (semi-major axis) [m]
        f = (1.0 / 298.257222101)  # flattening [-]
    else:
        a = 6378137.0  # equatorial radius (semi-major axis) [m]
        f = (1.0 / 298.257223563)  # flattening [-]
    b = a * (1.0 - f)  # polar radius (semi-minor axis) [m]
    e_2 = 1.0 - (b ** 2 / a ** 2)  # squared num. eccentricity [-]
    geod = Geodesic(a, f)

    # Inner domain
    width_h = width_in / 2.0 * 1000.0  # [m]
    rad_sph = a / np.sqrt(1 - e_2 * np.sin(np.deg2rad(loc[0])) ** 2) \
        * np.cos(np.deg2rad(loc[0]))  # sphere radius [m]
    lon_add = (360.0 / (np.pi * 2 * rad_sph)) * width_h  # [deg]
    dom_in = {"lon_min": loc[1] - lon_add,
              "lon_max": loc[1] + lon_add,
              "lat_min": geod.Direct(loc[0], loc[1], 180.0, width_h)["lat2"],
              "lat_max": geod.Direct(loc[0], loc[1], 0.0, width_h)["lat2"]}

    # Total domain
    add_out = dist_s * 1000.0  # [m]
    lat_abs_max = max(abs(dom_in["lat_min"]), abs(dom_in["lat_max"]))
    rad_sph = a / np.sqrt(1 - e_2 * np.sin(np.deg2rad(lat_abs_max)) ** 2) \
        * np.cos(np.deg2rad(lat_abs_max))  # sphere radius [m]
    lon_add = (360.0 / (np.pi * 2 * rad_sph)) * add_out  # [deg]
    dom_tot = {"lon_min": dom_in["lon_min"] - lon_add,
               "lon_max": dom_in["lon_max"] + lon_add,
               "lat_min": geod.Direct(loc[0], loc[1], 180.0,
                                      width_h + add_out)["lat2"],
               "lat_max": geod.Direct(loc[0], loc[1], 0.0,
                                      width_h + add_out)["lat2"]}

    # Check if total domain is within valid range (lon: -/+180.0, lat: -/+90.0)
    if ((dom_tot["lon_min"] < -180.0) or (dom_tot["lon_max"] > 180.0)
            or (dom_tot["lat_min"] < -90.0) or (dom_tot["lat_max"] > 90.0)):
        raise ValueError("total domain exceeds valid range")

    return {"in": dom_in, "tot": dom_tot}


###############################################################################

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
