#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from libc.math cimport sin, cos, sqrt
from libc.math cimport M_PI
from cython.parallel import prange


# -----------------------------------------------------------------------------

def lonlat2ecef(double[:, :] lon, double[:, :] lat, float[:, :] h,
                ellps="sphere"):
    """Coordinate transformation from lon/lat to ECEF.

    Transformation of geodetic longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates.

    Parameters
    ----------
    lon : ndarray of double
        Array (two-dimensional) with geographic longitude [degree]
    lat : ndarray of double
        Array (two-dimensional) with geographic latitude [degree]
    h : ndarray of float
        Array (two-dimensional) with elevation above ellipsoid [metre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)

    Returns
    -------
    x_ecef : ndarray of double
        Array (two-dimensional) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (two-dimensional) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (two-dimensional) with ECEF z-coordinates [metre]

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
    - Geoid parameters r, a and f: PROJ"""

    cdef int len_0 = lon.shape[0]
    cdef int len_1 = lon.shape[1]
    cdef int i, j
    cdef double r, f, a, b, e_2, n
    cdef double[:, :] x_ecef = np.empty((len_0, len_1), dtype=np.float64)
    cdef double[:, :] y_ecef = np.empty((len_0, len_1), dtype=np.float64)
    cdef double[:, :] z_ecef = np.empty((len_0, len_1), dtype=np.float64)

    # Spherical coordinates
    if ellps == "sphere":
        r = 6370997.0  # earth radius [m]
        for i in prange(len_0, nogil=True, schedule="static"):
            for j in range(len_1):
                x_ecef[i, j] = (r + h[i, j]) * cos(deg2rad(lat[i, j])) \
                    * cos(deg2rad(lon[i, j]))
                y_ecef[i, j] = (r + h[i, j]) * cos(deg2rad(lat[i, j])) \
                    * sin(deg2rad(lon[i, j]))
                z_ecef[i, j] = (r + h[i, j]) * sin(deg2rad(lat[i, j]))
        
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
            for j in range(len_1):
                n = a / sqrt(1.0 - e_2 * sin(deg2rad(lat[i, j])) ** 2)
                x_ecef[i, j] = (n + h[i, j]) * cos(deg2rad(lat[i, j])) \
                    * cos(deg2rad(lon[i, j]))
                y_ecef[i, j] = (n + h[i, j]) * cos(deg2rad(lat[i, j])) \
                    * sin(deg2rad(lon[i, j]))
                z_ecef[i, j] = (b ** 2 / a ** 2 * n + h[i, j]) \
                    * sin(deg2rad(lat[i, j]))
        
    return np.asarray(x_ecef), np.asarray(y_ecef), np.asarray(z_ecef)


# -----------------------------------------------------------------------------

def lonlat2ecef_coord1d(double[:] lon, double[:] lat, float[:,:] h,
                        ellps="sphere"):
    """Coordinate transformation from lon/lat to ECEF.

    Transformation of geodetic longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates (for one-dimensional input coordinates)

    Parameters
    ----------
    lon : ndarray of double
        Array (one-dimensional) with geographic longitude [degree]
    lat : ndarray of double
        Array (one-dimensional) with geographic latitude [degree]
    h : ndarray of float
        Array (two-dimensional) with elevation above ellipsoid [metre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)

    Returns
    -------
    x_ecef : ndarray of double
        Array (two-dimensional) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (two-dimensional) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (two-dimensional) with ECEF z-coordinates [metre]

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
    - Geoid parameters r, a and f: PROJ"""

    cdef int len_0 = lat.shape[0]
    cdef int len_1 = lon.shape[0]
    cdef int i, j
    cdef double r, f, a, b, e_2, n
    cdef double[:, :] x_ecef = np.empty((len_0, len_1), dtype=np.float64)
    cdef double[:, :] y_ecef = np.empty((len_0, len_1), dtype=np.float64)
    cdef double[:, :] z_ecef = np.empty((len_0, len_1), dtype=np.float64)

    # Spherical coordinates
    if ellps == "sphere":
        r = 6370997.0  # earth radius [m]
        for i in prange(len_0, nogil=True, schedule="static"):
            for j in range(len_1):
                x_ecef[i, j] = (r + h[i, j]) * cos(deg2rad(lat[i])) \
                    * cos(deg2rad(lon[j]))
                y_ecef[i, j] = (r + h[i, j]) * cos(deg2rad(lat[i])) \
                    * sin(deg2rad(lon[j]))
                z_ecef[i, j] = (r + h[i, j]) * sin(deg2rad(lat[i]))

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
            for j in range(len_1):
                n = a / sqrt(1.0 - e_2 * sin(deg2rad(lat[i])) ** 2)
                x_ecef[i, j] = (n + h[i, j]) * cos(deg2rad(lat[i])) \
                    * cos(deg2rad(lon[j]))
                y_ecef[i, j] = (n + h[i, j]) * cos(deg2rad(lat[i])) \
                    * sin(deg2rad(lon[j]))
                z_ecef[i, j] = (b ** 2 / a ** 2 * n + h[i, j]) \
                    * sin(deg2rad(lat[i]))

    return np.asarray(x_ecef), np.asarray(y_ecef), np.asarray(z_ecef)


# -----------------------------------------------------------------------------

def ecef2enu(double[:, :] x_ecef, double[:, :] y_ecef, double[:, :] z_ecef,
             double x_ecef_or, double y_ecef_or, double z_ecef_or,
             double lon_or, double lat_or):
    """Coordinate transformation from ECEF to ENU.

    Transformation of earth-centered, earth-fixed (ECEF) to local tangent
    plane (ENU) coordinates.

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (two-dimensional) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (two-dimensional) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (two-dimensional) with ECEF z-coordinates [metre]
    x_ecef_or : double
        ECEF x-coordinate of ENU origin [metre]
    y_ecef_or : double
        ECEF y-coordinate of ENU origin [metre]
    z_ecef_or : double
        ECEF z-coordinate of ENU origin [metre]
    lon_or : double
        Longitude of ENU origin [degree]
    lat_or : double
        Latitude of ENU origin [degree]

    Returns
    -------
    x_enu : ndarray of float
        Array (two-dimensional) with ENU x-coordinates [metre]
    y_enu : ndarray of float
        Array (two-dimensional) with ENU y-coordinates [metre]
    z_enu : ndarray of float
        Array (two-dimensional) with ENU z-coordinates [metre]

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    cdef int len_0 = x_ecef.shape[0]
    cdef int len_1 = x_ecef.shape[1]
    cdef int i, j
    cdef double sin_lon, cos_lon, sin_lat, cos_lat
    cdef float[:, :] x_enu = np.empty((len_0, len_1), dtype=np.float32)
    cdef float[:, :] y_enu = np.empty((len_0, len_1), dtype=np.float32)
    cdef float[:, :] z_enu = np.empty((len_0, len_1), dtype=np.float32)

    # Trigonometric functions
    sin_lon = sin(deg2rad(lon_or))
    cos_lon = cos(deg2rad(lon_or))
    sin_lat = sin(deg2rad(lat_or))
    cos_lat = cos(deg2rad(lat_or))

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        for j in range(len_1):
            
            x_enu[i, j] = (- sin_lon * (x_ecef[i, j] - x_ecef_or)
                           + cos_lon * (y_ecef[i, j] - y_ecef_or))
            y_enu[i, j] = (- sin_lat * cos_lon * (x_ecef[i, j] - x_ecef_or)
                           - sin_lat * sin_lon * (y_ecef[i, j] - y_ecef_or)
                           + cos_lat * (z_ecef[i, j] - z_ecef_or))
            z_enu[i, j] = (+ cos_lat * cos_lon * (x_ecef[i, j] - x_ecef_or)
                           + cos_lat * sin_lon * (y_ecef[i, j] - y_ecef_or)
                           + sin_lat * (z_ecef[i, j] - z_ecef_or))

    return np.asarray(x_enu), np.asarray(y_enu), np.asarray(z_enu)


# -----------------------------------------------------------------------------

def ecef2enu_vector(float[:, :, :] vec_ecef, double lon_or, double lat_or):
    """Coordinate transformation from ECEF to ENU.

    Transformation of earth-centered, earth-fixed (ECEF) to local tangent
    plane (ENU) coordinates (vectors).

    Parameters
    ----------
    vec_ecef : ndarray of float
        Array (three-dimensional) with vectors in ECEF coordinates
        (y, x, components) [metre]
    lon_or : double
        Longitude of ENU origin [degree]
    lat_or : double
        Latitude of ENU origin [degree]

    Returns
    -------
    vec_enu : ndarray of float
        Array (three-dimensional) with vectors in ENU coordinates
        (y, x, components) [metre]

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    cdef int len_0 = vec_ecef.shape[0]
    cdef int len_1 = vec_ecef.shape[1]
    cdef int i, j
    cdef double sin_lon, cos_lon, sin_lat, cos_lat
    cdef float[:, :, :] vec_enu = np.empty((len_0, len_1, 3), dtype=np.float32)

    # Trigonometric functions
    sin_lon = sin(deg2rad(lon_or))
    cos_lon = cos(deg2rad(lon_or))
    sin_lat = sin(deg2rad(lat_or))
    cos_lat = cos(deg2rad(lat_or))

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        for j in range(len_1):
            
            vec_enu[i, j, 0] = (- sin_lon * vec_ecef[i, j, 0]
                                + cos_lon * vec_ecef[i, j, 1])
            vec_enu[i, j, 1] = (- sin_lat * cos_lon * vec_ecef[i, j, 0]
                                - sin_lat * sin_lon * vec_ecef[i, j, 1]
                                + cos_lat * vec_ecef[i, j, 2])
            vec_enu[i, j, 2] = (+ cos_lat * cos_lon * vec_ecef[i, j, 0]
                                + cos_lat * sin_lon * vec_ecef[i, j, 1]
                                + sin_lat * vec_ecef[i, j, 2])

    return np.asarray(vec_enu)


# -----------------------------------------------------------------------------

def wgs2swiss(double[:, :] lon, double[:, :] lat, float[:, :] h_wgs):
    """Coordinate transformation from lon/lat to LV95.

    Transformation of ellipsoidal WGS84 to Swiss projection coordinates (LV95).

    Parameters
    ----------
    lon : ndarray of double
        Array (two-dimensional) with geographic longitude [degree]
    lat : ndarray of double
        Array (two-dimensional) with geographic latitude [degree]
    h_wgs : ndarray of float
        Array (two-dimensional) with elevation above ellipsoid [metre]

    Returns
    -------
    e : ndarray of double
        Array (two-dimensional) with coordinates in eastward direction [metre]
    n : ndarray of double
        Array (two-dimensional) with coordinates in northward direction [metre]
    h_ch : ndarray of double
        Array (two-dimensional) with height [metre]

    Sources
    -------
    - Document 'Approximate formulas for the transformation between Swiss
      projection coordinates and- WGS84'"""

    cdef int len_0 = lon.shape[0]
    cdef int len_1 = lon.shape[1]
    cdef int i, j
    cdef double lon_pr, lat_pr
    cdef double[:, :] e = np.empty((len_0, len_1), dtype=np.float64)
    cdef double[:, :] n = np.empty((len_0, len_1), dtype=np.float64)
    cdef float[:, :] h_ch = np.empty((len_0, len_1), dtype=np.float32)

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        for j in range(len_1):

            # Convert angles to arc-seconds and compute auxiliary values
            lon_pr = ((lon[i, j] * 3600.0) - 26782.5) / 10000.0
            lat_pr = ((lat[i, j] * 3600.0) - 169028.66) / 10000.0

            # Calculate projection coordinates in LV95
            e[i, j] = 2600072.37 \
                      + 211455.93 * lon_pr \
                      - 10938.51 * lon_pr * lat_pr \
                      - 0.36 * lon_pr * lat_pr ** 2 \
                      - 44.54 * lon_pr ** 3
            n[i, j] = 1200147.07 \
                      + 308807.95 * lat_pr \
                      + 3745.25 * lon_pr ** 2 \
                      + 76.63 * lat_pr ** 2 \
                      - 194.56 * lon_pr ** 2 * lat_pr \
                      + 119.79 * lat_pr ** 3
            h_ch[i, j] = h_wgs[i, j] - 49.55 \
                         + 2.73 * lon_pr \
                         + 6.94 * lat_pr

    return np.asarray(e), np.asarray(n), np.asarray(h_ch)


# -----------------------------------------------------------------------------

def swiss2wgs(double[:, :] e, double[:, :] n, float[:, :] h_ch):
    """Coordinate transformation from LV95 to lon/lat.

    Transformation of swiss projection (LV95) to ellipsoidal WGS84 coordinates.

    Parameters
    -------
    e : ndarray of double
        Array (two-dimensional) with coordinates in eastward direction [metre]
    n : ndarray of double
        Array (two-dimensional) with coordinates in northward direction [metre]
    h_ch : ndarray of double
        Array (two-dimensional) with height [metre]

    Returns
    ----------
    lon : ndarray of double
        Array (two-dimensional) with geographic longitude [degree]
    lat : ndarray of double
        Array (two-dimensional) with geographic latitude [degree]
    h_wgs : ndarray of float
        Array (two-dimensional) with elevation above ellipsoid [metre]

    Sources
    -------
    - Document 'Approximate formulas for the transformation between Swiss
      projection coordinates and- WGS84'"""

    cdef int len_0 = e.shape[0]
    cdef int len_1 = e.shape[1]
    cdef int i, j
    cdef double e_pr, n_pr
    cdef double[:, :] lon = np.empty((len_0, len_1), dtype=np.float64)
    cdef double[:, :] lat = np.empty((len_0, len_1), dtype=np.float64)
    cdef float[:, :] h_wgs = np.empty((len_0, len_1), dtype=np.float32)

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        for j in range(len_1):

            # Convert projected coordinates in civilian system and convert
            # to 1000 km
            e_pr = (e[i, j] - 2600000.0) / 1000000.0
            n_pr = (n[i, j] - 1200000.0) / 1000000.0

            # Calculate longitude, latitude and elevation
            lon[i, j] = 2.6779094 \
                        + 4.728982 * e_pr \
                        + 0.791484 * e_pr * n_pr \
                        + 0.1306 * e_pr * n_pr ** 2 \
                        - 0.0436 * e_pr ** 3
            lat[i, j] = 16.9023892 \
                        + 3.238272 * n_pr \
                        - 0.270978 * e_pr ** 2 \
                        - 0.002528 * n_pr ** 2 \
                        - 0.0447 * e_pr ** 2 * n_pr \
                        - 0.0140 * n_pr ** 3
            h_wgs[i, j] = h_ch[i, j] + 49.55 \
                          - 12.60 * e_pr \
                          - 22.64 * n_pr

            # Convert longitude and latitude to degree
            lon[i, j] *= (100.0 / 36.)
            lat[i, j] *= (100.0 / 36.)

    return np.asarray(lon), np.asarray(lat), np.asarray(h_wgs)


# -----------------------------------------------------------------------------
# Auxiliary function(s)
# -----------------------------------------------------------------------------

cdef inline double deg2rad(double ang_in) nogil:
    """Convert degree to radian"""

    cdef double ang_out
    
    ang_out = ang_in * (M_PI / 180.0)
       
    return ang_out
