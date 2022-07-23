#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from math import prod
from libc.math cimport sin, cos, sqrt
from libc.math cimport M_PI
from cython.parallel import prange


# -----------------------------------------------------------------------------

def lonlat2ecef(lon, lat, h, ellps):
    """Coordinate transformation from lon/lat to ECEF.

    Transformation of geodetic longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates.

    Parameters
    ----------
    lon : ndarray of double
        Array (with arbitrary dimensions) with geographic longitude [degree]
    lat : ndarray of double
        Array (with arbitrary dimensions) with geographic latitude [degree]
    h : ndarray of float
        Array (with arbitrary dimensions) with elevation above ellipsoid
        [metre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)

    Returns
    -------
    x_ecef : ndarray of double
        Array (dimensions according to input) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (dimensions according to input) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (dimensions according to input) with ECEF z-coordinates [metre]
        """

    # Check arguments
    if (lon.shape != lat.shape) or (lat.shape != h.shape):
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")
    if ((lon.dtype != "float64") or (lat.dtype != "float64")
            or (h.dtype != "float32")):
        raise ValueError("Input array(s) has/have incorrect data type(s)")
    if ellps not in ("sphere", "GRS80", "WGS84"):
        raise ValueError("Unknown value for 'ellps'")

    # Wrapper for 1-dimensional function
    shp = lon.shape
    x_ecef, y_ecef, z_ecef = _lonlat2ecef_1d(lon.ravel(), lat.ravel(),
                                              h.ravel(), ellps)
    return x_ecef.reshape(shp), y_ecef.reshape(shp), z_ecef.reshape(shp)


def _lonlat2ecef_1d(double[:] lon, double[:] lat, float[:] h, ellps):
    """Coordinate transformation from lon/lat to ECEF (for 1-dimensional data).

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
    - Geoid parameters r, a and f: PROJ"""

    cdef int len_0 = lon.shape[0]
    cdef int i
    cdef double r, f, a, b, e_2, n
    cdef double[:] x_ecef = np.empty(len_0, dtype=np.float64)
    cdef double[:] y_ecef = np.empty(len_0, dtype=np.float64)
    cdef double[:] z_ecef = np.empty(len_0, dtype=np.float64)

    # Spherical coordinates
    if ellps == "sphere":
        r = 6370997.0  # earth radius [m]
        for i in prange(len_0, nogil=True, schedule="static"):
            x_ecef[i] = (r + h[i]) * cos(deg2rad(lat[i])) \
                * cos(deg2rad(lon[i]))
            y_ecef[i] = (r + h[i]) * cos(deg2rad(lat[i])) \
                * sin(deg2rad(lon[i]))
            z_ecef[i] = (r + h[i]) * sin(deg2rad(lat[i]))
        
    # Elliptic (geodetic) coordinates
    else:
        a = 6378137.0  # equatorial radius (semi-major axis) [m]
        if ellps == "GRS80":
            f = (1.0 / 298.257222101)  # flattening [-]
        else:  # WGS84
            f = (1.0 / 298.257223563)  # flattening [-]
        b = a * (1.0 - f)  # polar radius (semi-minor axis) [m]
        e_2 = 1.0 - (b ** 2 / a ** 2)  # squared num. eccentricity [-]
        for i in prange(len_0, nogil=True, schedule="static"):
            n = a / sqrt(1.0 - e_2 * sin(deg2rad(lat[i])) ** 2)
            x_ecef[i] = (n + h[i]) * cos(deg2rad(lat[i])) \
                * cos(deg2rad(lon[i]))
            y_ecef[i] = (n + h[i]) * cos(deg2rad(lat[i])) \
                * sin(deg2rad(lon[i]))
            z_ecef[i] = (b ** 2 / a ** 2 * n + h[i]) \
                * sin(deg2rad(lat[i]))

    return np.asarray(x_ecef), np.asarray(y_ecef), np.asarray(z_ecef)


# -----------------------------------------------------------------------------

def ecef2enu(x_ecef, y_ecef, z_ecef, trans_ecef2enu):
    """Coordinate transformation from ECEF to ENU.

    Transformation of earth-centered, earth-fixed (ECEF) to local tangent
    plane (ENU) coordinates.

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF z-coordinates [metre]
    trans_ecef2enu : class
        Instance of class `TransformerEcef2enu`

    Returns
    -------
    x_enu : ndarray of float
        Array (dimensions according to input) with ENU x-coordinates [metre]
    y_enu : ndarray of float
        Array (dimensions according to input) with ENU y-coordinates [metre]
    z_enu : ndarray of float
        Array (dimensions according to input) with ENU z-coordinates [metre]"""

    # Check arguments
    if (x_ecef.shape != y_ecef.shape) or (y_ecef.shape != z_ecef.shape):
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")
    if ((x_ecef.dtype != "float64") or (y_ecef.dtype != "float64")
            or (z_ecef.dtype != "float64")):
        raise ValueError("Input array(s) has/have incorrect data type(s)")

    # Wrapper for 1-dimensional function
    shp = x_ecef.shape
    x_enu, y_enu, z_enu = _ecef2enu_1d(x_ecef.ravel(), y_ecef.ravel(),
                                              z_ecef.ravel(), trans_ecef2enu)
    return x_enu.reshape(shp), y_enu.reshape(shp), z_enu.reshape(shp)


def _ecef2enu_1d(double[:] x_ecef, double[:] y_ecef, double[:] z_ecef,
                trans_ecef2enu):
    """Coordinate transformation from ECEF to ENU (for 1-dimensional data).

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    cdef int len_0 = x_ecef.shape[0]
    cdef int i
    cdef double sin_lon, cos_lon, sin_lat, cos_lat
    cdef float[:] x_enu = np.empty(len_0, dtype=np.float32)
    cdef float[:] y_enu = np.empty(len_0, dtype=np.float32)
    cdef float[:] z_enu = np.empty(len_0, dtype=np.float32)
    cdef double x_ecef_or = trans_ecef2enu.x_ecef_or
    cdef double y_ecef_or = trans_ecef2enu.y_ecef_or
    cdef double z_ecef_or = trans_ecef2enu.z_ecef_or
    cdef double lon_or = trans_ecef2enu.lon_or
    cdef double lat_or = trans_ecef2enu.lat_or

    # Trigonometric functions
    sin_lon = sin(deg2rad(lon_or))
    cos_lon = cos(deg2rad(lon_or))
    sin_lat = sin(deg2rad(lat_or))
    cos_lat = cos(deg2rad(lat_or))

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        x_enu[i] = (- sin_lon * (x_ecef[i] - x_ecef_or)
                    + cos_lon * (y_ecef[i] - y_ecef_or))
        y_enu[i] = (- sin_lat * cos_lon * (x_ecef[i] - x_ecef_or)
                    - sin_lat * sin_lon * (y_ecef[i] - y_ecef_or)
                    + cos_lat * (z_ecef[i] - z_ecef_or))
        z_enu[i] = (+ cos_lat * cos_lon * (x_ecef[i] - x_ecef_or)
                    + cos_lat * sin_lon * (y_ecef[i] - y_ecef_or)
                    + sin_lat * (z_ecef[i] - z_ecef_or))

    return np.asarray(x_enu), np.asarray(y_enu), np.asarray(z_enu)


# -----------------------------------------------------------------------------

def ecef2enu_vector(vec_ecef, trans_ecef2enu):
    """Coordinate transformation from ECEF to ENU.

    Transformation of earth-centered, earth-fixed (ECEF) to local tangent
    plane (ENU) coordinates (vectors).

    Parameters
    ----------
    vec_ecef : ndarray of float
        Array (at least two-dimensional; vector components must be stored in
        last dimension) with vectors in ECEF coordinates [metre]
    trans_ecef2enu : class
        Instance of class `TransformerEcef2enu`

    Returns
    -------
    vec_enu : ndarray of float
        Array (dimensions according to input; vector components are stored in
        last dimension) with vectors in ENU coordinates [metre]"""

    # Check arguments
    if (vec_ecef.ndim < 2) or (vec_ecef.shape[vec_ecef.ndim - 1] != 3):
        raise ValueError("Inccorect shape / number of dimensions of input "
                         + "array")
    if vec_ecef.dtype != "float32":
        raise ValueError("Input array has incorrect data type")

    # Wrapper for 1-dimensional function
    shp = vec_ecef.shape[:(vec_ecef.ndim - 1)]
    vec_enu = _ecef2enu_vector_1d(vec_ecef.reshape(prod(shp), 3),
                                 trans_ecef2enu)
    return vec_enu.reshape(shp + (3,))


def _ecef2enu_vector_1d(float[:, :] vec_ecef, trans_ecef2enu):
    """Coordinate transformation from ECEF to ENU (for 1-dimensional data).

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    cdef int len_0 = vec_ecef.shape[0]
    cdef int i
    cdef double sin_lon, cos_lon, sin_lat, cos_lat
    cdef float[:, :] vec_enu = np.empty((len_0, 3), dtype=np.float32)
    cdef double lon_or = trans_ecef2enu.lon_or
    cdef double lat_or = trans_ecef2enu.lat_or

    # Trigonometric functions
    sin_lon = sin(deg2rad(lon_or))
    cos_lon = cos(deg2rad(lon_or))
    sin_lat = sin(deg2rad(lat_or))
    cos_lat = cos(deg2rad(lat_or))

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        vec_enu[i, 0] = (- sin_lon * vec_ecef[i, 0] + cos_lon * vec_ecef[i, 1])
        vec_enu[i, 1] = (- sin_lat * cos_lon * vec_ecef[i, 0]
                         - sin_lat * sin_lon * vec_ecef[i, 1]
                         + cos_lat * vec_ecef[i, 2])
        vec_enu[i, 2] = (+ cos_lat * cos_lon * vec_ecef[i, 0]
                         + cos_lat * sin_lon * vec_ecef[i, 1]
                         + sin_lat * vec_ecef[i, 2])

    return np.asarray(vec_enu)


# -----------------------------------------------------------------------------

def wgs2swiss(lon, lat, h_wgs):
    """Coordinate transformation from lon/lat to LV95.

    Transformation of ellipsoidal WGS84 to Swiss projection coordinates (LV95).

    Parameters
    ----------
    lon : ndarray of double
        Array (with arbitrary dimensions) with geographic longitude [degree]
    lat : ndarray of double
        Array (with arbitrary dimensions) with geographic latitude [degree]
    h_wgs : ndarray of float
        Array (with arbitrary dimensions) with elevation above ellipsoid
        [metre]

    Returns
    -------
    e : ndarray of double
        Array (dimensions according to input) with coordinates in eastward
        direction [metre]
    n : ndarray of double
        Array (dimensions according to input) with coordinates in northward
        direction [metre]
    h_ch : ndarray of double
        Array (dimensions according to input) with height [metre]"""

    # Check arguments
    if (lon.shape != lat.shape) or (lat.shape != h_wgs.shape):
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")
    if ((lon.dtype != "float64") or (lat.dtype != "float64")
            or (h_wgs.dtype != "float32")):
        raise ValueError("Input array(s) has/have incorrect data type(s)")

    # Wrapper for 1-dimensional function
    shp = lon.shape
    e, n, h_ch = _wgs2swiss_1d(lon.ravel(), lat.ravel(), h_wgs.ravel())
    return e.reshape(shp), n.reshape(shp), h_ch.reshape(shp)


def _wgs2swiss_1d(double[:] lon, double[:] lat, float[:] h_wgs):
    """Coordinate transformation from lon/lat to LV95 (for 1-dimensional data).

    Sources
    -------
    - Document 'Approximate formulas for the transformation between Swiss
      projection coordinates and- WGS84'"""

    cdef int len_0 = lon.shape[0]
    cdef int i
    cdef double lon_pr, lat_pr
    cdef double[:] e = np.empty(len_0, dtype=np.float64)
    cdef double[:] n = np.empty(len_0, dtype=np.float64)
    cdef float[:] h_ch = np.empty(len_0, dtype=np.float32)

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):

        # Convert angles to arc-seconds and compute auxiliary values
        lon_pr = ((lon[i] * 3600.0) - 26782.5) / 10000.0
        lat_pr = ((lat[i] * 3600.0) - 169028.66) / 10000.0

        # Calculate projection coordinates in LV95
        e[i] = 2600072.37 \
               + 211455.93 * lon_pr \
               - 10938.51 * lon_pr * lat_pr \
               - 0.36 * lon_pr * lat_pr ** 2 \
               - 44.54 * lon_pr ** 3
        n[i] = 1200147.07 \
               + 308807.95 * lat_pr \
               + 3745.25 * lon_pr ** 2 \
               + 76.63 * lat_pr ** 2 \
               - 194.56 * lon_pr ** 2 * lat_pr \
               + 119.79 * lat_pr ** 3
        h_ch[i] = h_wgs[i] - 49.55 \
                  + 2.73 * lon_pr \
                  + 6.94 * lat_pr

    return np.asarray(e), np.asarray(n), np.asarray(h_ch)


# -----------------------------------------------------------------------------

def swiss2wgs(e, n, h_ch):
    """Coordinate transformation from LV95 to lon/lat.

    Transformation of swiss projection (LV95) to ellipsoidal WGS84 coordinates.

    Parameters
    -------
    e : ndarray of double
        Array (with arbitrary dimensions) with coordinates in eastward
        direction [metre]
    n : ndarray of double
        Array (with arbitrary dimensions) with coordinates in northward
        direction [metre]
    h_ch : ndarray of double
        Array (with arbitrary dimensions) with height [metre]

    Returns
    ----------
    lon : ndarray of double
        Array (dimensions according to input) with geographic longitude
        [degree]
    lat : ndarray of double
        Array (dimensions according to input) with geographic latitude [degree]
    h_wgs : ndarray of float
        Array (dimensions according to input) with elevation above ellipsoid
        [metre]"""

    # Check arguments
    if (e.shape != n.shape) or (n.shape != h_ch.shape):
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")
    if ((e.dtype != "float64") or (n.dtype != "float64")
            or (h_ch.dtype != "float32")):
        raise ValueError("Input array(s) has/have incorrect data type(s)")

    # Wrapper for 1-dimensional function
    shp = e.shape
    lon, lat, h_wgs = _swiss2wgs_1d(e.ravel(), n.ravel(), h_ch.ravel())
    return lon.reshape(shp), lat.reshape(shp), h_wgs.reshape(shp)


def _swiss2wgs_1d(double[:] e, double[:] n, float[:] h_ch):
    """Coordinate transformation from LV95 to lon/lat (for 1-dimensional data).

    Sources
    -------
    - Document 'Approximate formulas for the transformation between Swiss
      projection coordinates and- WGS84'"""

    cdef int len_0 = e.shape[0]
    cdef int i
    cdef double e_pr, n_pr
    cdef double[:] lon = np.empty(len_0, dtype=np.float64)
    cdef double[:] lat = np.empty(len_0, dtype=np.float64)
    cdef float[:] h_wgs = np.empty(len_0, dtype=np.float32)

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):

        # Convert projected coordinates in civilian system and convert
        # to 1000 km
        e_pr = (e[i] - 2600000.0) / 1000000.0
        n_pr = (n[i] - 1200000.0) / 1000000.0

        # Calculate longitude, latitude and elevation
        lon[i] = 2.6779094 \
                 + 4.728982 * e_pr \
                 + 0.791484 * e_pr * n_pr \
                 + 0.1306 * e_pr * n_pr ** 2 \
                 - 0.0436 * e_pr ** 3
        lat[i] = 16.9023892 \
                 + 3.238272 * n_pr \
                 - 0.270978 * e_pr ** 2 \
                 - 0.002528 * n_pr ** 2 \
                 - 0.0447 * e_pr ** 2 * n_pr \
                 - 0.0140 * n_pr ** 3
        h_wgs[i] = h_ch[i] + 49.55 \
                   - 12.60 * e_pr \
                   - 22.64 * n_pr

        # Convert longitude and latitude to degree
        lon[i] *= (100.0 / 36.)
        lat[i] *= (100.0 / 36.)

    return np.asarray(lon), np.asarray(lat), np.asarray(h_wgs)


# -----------------------------------------------------------------------------

class TransformerEcef2enu:
    """Class that stores attributes to transform from ECEF to ENU coordinates.

    Transformer class that stores attributes to convert between ECEF and ENU
    coordinates. The origin of the ENU coordinate system coincides with the
    surface of the sphere/ellipsoid.

    Parameters
    -------
    lon_or : double
        Longitude coordinate for origin of ENU coordinate system [degree]
    lat_or : double
        Latitude coordinate for origin of ENU coordinate system [degree]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)"""

    def __init__(self, lon_or, lat_or, ellps):
        if (lon_or < -180.0) or (lon_or > 180.0):
            raise ValueError("Value for 'lon_or' is outside of valid range")
        if (lat_or < -90.0) or (lat_or > 90.0):
            raise ValueError("Value for 'lat_or' is outside of valid range")
        self.lon_or = lon_or
        self.lat_or = lat_or

        if ellps == "sphere":
            r = 6370997.0  # earth radius [m]
            self.x_ecef_or = r * np.cos(np.deg2rad(self.lat_or)) \
                             * np.cos(np.deg2rad(self.lon_or))
            self.y_ecef_or = r * np.cos(np.deg2rad(self.lat_or)) \
                             * np.sin(np.deg2rad(self.lon_or))
            self.z_ecef_or = r * np.sin(np.deg2rad(self.lat_or))
        elif ellps in ("GRS80", "WGS84"):
            a = 6378137.0  # equatorial radius (semi-major axis) [m]
            if ellps == "GRS80":
                f = (1.0 / 298.257222101)  # flattening [-]
            else:  # WGS84
                f = (1.0 / 298.257223563)  # flattening [-]
            b = a * (1.0 - f)  # polar radius (semi-minor axis) [m]
            e_2 = 1.0 - (b ** 2 / a ** 2)  # squared num. eccentricity [-]
            n = a / np.sqrt(1.0 - e_2 * np.sin(np.deg2rad(self.lat_or)) ** 2)
            self.x_ecef_or = n * np.cos(np.deg2rad(self.lat_or)) \
                             * np.cos(np.deg2rad(self.lon_or))
            self.y_ecef_or = n * np.cos(np.deg2rad(self.lat_or)) \
                             * np.sin(np.deg2rad(self.lon_or))
            self.z_ecef_or = (b ** 2 / a ** 2 * n) \
                             * np.sin(np.deg2rad(self.lat_or))
        else:
            raise ValueError("Unknown value for 'ellps'")


# -----------------------------------------------------------------------------

def rotation_matrix_glob2loc(vec_north_enu, vec_norm_enu):
    """Matrices to rotate vectors from global to local ENU coordinates.

    Array with matrices to rotate vector from global to local ENU coordinates.

    Parameters
    -------
    vec_north_enu : ndarray of float
        Array (three-dimensional; vector components must be stored in last
        dimension) with north vector components in ENU coordinates [metre]
    vec_norm_enu : ndarray of float
        Array (three-dimensional; vector components must be stored in last
        dimension) with surface normal components in ENU coordinates
        [metre]

    Returns
    ----------
    lon : ndarray
        Array (dimensions according to input; rotation matrices stored in last
        two dimensions) with rotation matrices [metre]'"""

    # Check arguments
    if vec_north_enu.shape != vec_norm_enu.shape:
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")

    # Compute rotation matrix
    rot_mat_glob2loc = np.empty((vec_north_enu.shape[0] + 2,
                                 vec_north_enu.shape[1] + 2, 3, 3),
                                dtype=np.float32)
    rot_mat_glob2loc.fill(np.nan)
    rot_mat_glob2loc[1:-1, 1:-1, 0, :] = np.cross(vec_north_enu, vec_norm_enu,
                                                  axisa=2, axisb=2)
    # vector pointing towards east
    rot_mat_glob2loc[1:-1, 1:-1, 1, :] = vec_north_enu
    rot_mat_glob2loc[1:-1, 1:-1, 2, :] = vec_norm_enu

    return rot_mat_glob2loc


# -----------------------------------------------------------------------------
# Auxiliary function(s)
# -----------------------------------------------------------------------------

cdef inline double deg2rad(double ang_in) nogil:
    """Convert degree to radian"""

    cdef double ang_out
    
    ang_out = ang_in * (M_PI / 180.0)
       
    return ang_out
