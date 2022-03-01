#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from libc.math cimport sin, cos, sqrt, atan
from libc.math cimport M_PI
from libc.math cimport NAN
from libc.stdio cimport printf
from cython.parallel import prange
from scipy.linalg.cython_lapack cimport sgesv


###############################################################################
# Coordinate transformation
###############################################################################

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

    Notes
    -----
    Sources:
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


#------------------------------------------------------------------------------

def lonlat2ecef_gc1d(double[:] lon, double[:] lat, float[:,:] h,
                     ellps="sphere"):
    """Coordinate transformation from lon/lat to ECEF.

    Transformation of geodetic longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates.

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

    Notes
    -----
    Sources:
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


#------------------------------------------------------------------------------

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

    Notes
    -----
    Sources:
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


#------------------------------------------------------------------------------

def ecef2enu_vec(float[:, :, :] vec_ecef, double lon_or, double lat_or):
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

    Notes
    -----
    Sources:
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


#------------------------------------------------------------------------------

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

    Notes
    -----
    Sources:
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


#------------------------------------------------------------------------------

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

    Notes
    -----
    Sources:
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


###############################################################################
# Unit vectors
###############################################################################

def surf_norm(double[:, :] lon, double[:, :] lat):
    """Compute surface normal.

    Computation of surface normal unit vectors in earth-centered, earth-fixed
    (ECEF) coordinates.

    Parameters
    ----------
    lon : ndarray of double
        Array (two-dimensional) with geographic longitude [degree]
    lat : ndarray of double
        Array (two-dimensional) with geographic latitudes [degree]

    Returns
    -------
    vec_norm_ecef : ndarray of float
        Array (three-dimensional) with surface normal components in ECEF
        coordinates (y, x, components) [metre]

    Notes
    -----
    Source: https://en.wikipedia.org/wiki/N-vector"""

    cdef int len_0 = lon.shape[0]
    cdef int len_1 = lon.shape[1]
    cdef int i, j
    cdef double sin_lon, cos_lon, sin_lat, cos_lat
    cdef float[:, :, :] vec_norm_ecef = np.empty((len_0, len_1, 3),
                                                 dtype=np.float32)

    # Compute surface normals
    for i in prange(len_0, nogil=True, schedule="static"):
        for j in range(len_1):

            sin_lon = sin(deg2rad(lon[i, j]))
            cos_lon = cos(deg2rad(lon[i, j]))
            sin_lat = sin(deg2rad(lat[i, j]))
            cos_lat = cos(deg2rad(lat[i, j]))

            vec_norm_ecef[i, j, 0] = cos_lat * cos_lon
            vec_norm_ecef[i, j, 1] = cos_lat * sin_lon
            vec_norm_ecef[i, j, 2] = sin_lat

    return np.asarray(vec_norm_ecef)


# -----------------------------------------------------------------------------

def north_dir(double[:, :] x_ecef, double[:, :] y_ecef, double[:, :] z_ecef,
              float[:, :, :] vec_norm_ecef, ellps="sphere"):
    """Compute north vector.

    Computation of unit vectors in surface planes pointing towards North
    (in earth-centered, earth-fixed (ECEF) coordinates).

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (two-dimensional) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (two-dimensional) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (two-dimensional) with ECEF z-coordinates [metre]
    vec_norm_ecef : ndarray of float
        Array (three-dimensional) with surface normal components in ECEF
        coordinates (y, x, components) [metre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)

    Returns
    -------
    vec_north_ecef : ndarray of float
        Array (three-dimensional) with north vector components in ECEF
        coordinates (y, x, components) [metre]

    Notes
    -----
    Source: Geoid parameters r, a and f: PROJ"""

    cdef int len_0 = x_ecef.shape[0]
    cdef int len_1 = x_ecef.shape[1]
    cdef int i, j
    cdef double r, f, a, b
    cdef double np_x, np_y, np_z
    cdef double vec_nor_x, vec_nor_y, vec_nor_z
    cdef double dot_pr, vec_proj_x, vec_proj_y, vec_proj_z, norm
    cdef float[:,:,:] vec_north_ecef = np.empty((len_0, len_1, 3),
                                                dtype=np.float32)

    # Coordinates of North pole
    np_x = 0.0
    np_y = 0.0
    if ellps == "sphere":
        r = 6370997.0  # earth radius [m]
        np_z = r
    else:
        a = 6378137.0  # equatorial radius (semi-major axis) [m]
        if ellps == "GRS80":
            f = (1.0 / 298.257222101)  # flattening [-]
        else:  # WGS84
            f = (1.0 / 298.257223563)  # flattening [-]
        b = a * (1.0 - f)  # polar radius (semi-minor axis) [m]
        np_z = b

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        for j in range(len_1):

            # Vector to North Pole
            vec_nor_x = (np_x - x_ecef[i, j])
            vec_nor_y = (np_y - y_ecef[i, j])
            vec_nor_z = (np_z - z_ecef[i, j])

            # Project vector to North Pole on surface normal plane
            dot_pr = ((vec_nor_x * vec_norm_ecef[i, j, 0])
                      + (vec_nor_y * vec_norm_ecef[i, j, 1])
                      + (vec_nor_z * vec_norm_ecef[i, j, 2]))
            vec_proj_x = vec_nor_x - dot_pr * vec_norm_ecef[i, j, 0]
            vec_proj_y = vec_nor_y - dot_pr * vec_norm_ecef[i, j, 1]
            vec_proj_z = vec_nor_z - dot_pr * vec_norm_ecef[i, j, 2]

            # Normalise vector
            norm = sqrt(vec_proj_x ** 2 + vec_proj_y ** 2 + vec_proj_z ** 2)
            vec_north_ecef[i, j, 0] = vec_proj_x / norm
            vec_north_ecef[i, j, 1] = vec_proj_y / norm
            vec_north_ecef[i, j, 2] = vec_proj_z / norm

    return np.asarray(vec_north_ecef)


###############################################################################
# Slope computation
###############################################################################

def slope_plane_meth(float[:, :] x, float[:, :] y, float[:, :] z,
                     float[:, :, :, :] rot_mat=np.empty((0, 0, 3, 3),
                                                        dtype=np.float32)):
    """Slope computation.

    Compute surface slope of DEM from central and 8 neighbouring grid cells.

    Parameters
    ----------
    x : ndarray of float
        Array (two-dimensional) with x-coordinates [metre]
    y : ndarray of float
        Array (two-dimensional) with y-coordinates [metre]
    z : ndarray of float
        Array (two-dimensional) with z-coordinates [metre]
    rot_mat: ndarray of float, optional
        Array (four-dimensional) with rotation matrix (y, x, 3, 3) to transform
        coordinates to a local coordinate system in which the z-axis aligns
        with local up

    Returns
    -------
    vec_tilt : ndarray of float
        Array (three-dimensional) with titled surface normal components
        (y, x, components) [metre]

    Notes
    -----
    Plane-based method which computes the surface normal by fitting a plane
    to the central and 8 neighbouring grid cells. The optimal fit is computed
    by minimising the sum of the squared errors in the z-direction. The same
    method is used in ArcGIS: https://pro.arcgis.com/en/pro-app/tool-reference/
    spatial-analyst/how-slope-works.htm

    To do
    -----
    Parallelise function with OpenMP. Consider that various arrays
    (vec, mat, ...) must be thread-private."""

    cdef int len_0 = x.shape[0]
    cdef int len_1 = x.shape[1]
    cdef int i, j, k, l
    cdef float vec_x, vec_y, vec_z, vec_mag
    cdef int num, nrhs, lda, ldb, info
    cdef float x_l_sum, y_l_sum, z_l_sum
    cdef float x_l_x_l_sum, x_l_y_l_sum, x_l_z_l_sum, y_l_y_l_sum, y_l_z_l_sum
    cdef int count
    cdef float[:, :, :] vec_tilt = np.empty((len_0, len_1, 3),
                                            dtype=np.float32)
    cdef float[:] vec = np.empty(3, dtype=np.float32)
    cdef float[:] mat = np.zeros(9, dtype=np.float32)
    cdef int[:] ipiv = np.empty(3, dtype=np.int32)
    cdef float[:, :] coord = np.empty((9, 3), dtype=np.float32)

    # Settings for solving system of linear equations
    num = 3 # number of linear equations [-]
    nrhs = 1 # number of columns of matrix B [-]
    lda = 3 # leading dimension of array A [-]
    ldb = 3 # leading dimension of array B [-]

    # Initialise array
    vec_tilt[:] = NAN

    # Loop through grid cells
    if rot_mat.shape[0] == 0:  # perform no coordinate transformation

        for i in range(1, (len_0 - 1)):
            for j in range(1, (len_1 - 1)):

                # Compute normal vector of plane
                x_l_sum = 0.0
                y_l_sum = 0.0
                z_l_sum = 0.0
                x_l_x_l_sum = 0.0
                x_l_y_l_sum = 0.0
                x_l_z_l_sum = 0.0
                y_l_y_l_sum = 0.0
                y_l_z_l_sum = 0.0
                for k in range((i - 1), (i + 2)):
                    for l in range((j - 1), (j + 2)):
                        x_l_sum = x_l_sum + x[k, l]
                        y_l_sum = y_l_sum + y[k, l]
                        z_l_sum = z_l_sum + z[k, l]
                        x_l_x_l_sum = x_l_x_l_sum + (x[k, l] * x[k, l])
                        x_l_y_l_sum = x_l_y_l_sum + (x[k, l] * y[k, l])
                        x_l_z_l_sum = x_l_z_l_sum + (x[k, l] * z[k, l])
                        y_l_y_l_sum = y_l_y_l_sum + (y[k, l] * y[k, l])
                        y_l_z_l_sum = y_l_z_l_sum + (y[k, l] * z[k, l])
                # Fortran-contiguous
                mat[0] = x_l_x_l_sum
                mat[3] = x_l_y_l_sum
                mat[6] = x_l_sum
                mat[1] = x_l_y_l_sum
                mat[4] = y_l_y_l_sum
                mat[7] = y_l_sum
                mat[2] = x_l_sum
                mat[5] = y_l_sum
                mat[8] = 9.0
                vec[0] = x_l_z_l_sum
                vec[1] = y_l_z_l_sum
                vec[2] = z_l_sum
                sgesv(&num, &nrhs, &mat[0], &lda, &ipiv[0], &vec[0], &ldb,
                      &info)
                vec[2] = -1.0

                vec_x = vec[0]
                vec_y = vec[1]
                vec_z = vec[2]

                # Normalise vector
                vec_mag = sqrt(vec_x ** 2 + vec_y ** 2 + vec_z ** 2)
                vec_x = vec_x / vec_mag
                vec_y = vec_y / vec_mag
                vec_z = vec_z / vec_mag

                # Reverse orientation of plane's normal vector (if necessary)
                if vec_z < 0.0:
                    vec_x = vec_x * -1.0
                    vec_y = vec_y * -1.0
                    vec_z = vec_z * -1.0

                vec_tilt[i, j, 0] = vec_x
                vec_tilt[i, j, 1] = vec_y
                vec_tilt[i, j, 2] = vec_z

    else:
        printf("Perform local coordinate transformation\n")

        for i in range(1, (len_0 - 1)):
            for j in range(1, (len_1 - 1)):

                # Coordinate transformation (translation and rotation)
                count = 0
                for k in range((i - 1), (i + 2)):
                    for l in range((j - 1), (j + 2)):
                        coord[count, 0] = x[k, l] - x[i, j]
                        coord[count, 1] = y[k, l] - y[i, j]
                        coord[count, 2] = z[k, l] - z[i, j]
                        count = count + 1
                for k in range(9):
                    vec_x = rot_mat[i, j, 0, 0] * coord[k, 0] \
                            + rot_mat[i, j, 0, 1] * coord[k, 1] \
                            + rot_mat[i, j, 0, 2] * coord[k, 2]
                    vec_y = rot_mat[i, j, 1, 0] * coord[k, 0] \
                            + rot_mat[i, j, 1, 1] * coord[k, 1] \
                            + rot_mat[i, j, 1, 2] * coord[k, 2]
                    vec_z = rot_mat[i, j, 2, 0] * coord[k, 0] \
                            + rot_mat[i, j, 2, 1] * coord[k, 1] \
                            + rot_mat[i, j, 2, 2] * coord[k, 2]
                    coord[k, 0] = vec_x
                    coord[k, 1] = vec_y
                    coord[k, 2] = vec_z

                # Compute normal vector of plane
                x_l_sum = 0.0
                y_l_sum = 0.0
                z_l_sum = 0.0
                x_l_x_l_sum = 0.0
                x_l_y_l_sum = 0.0
                x_l_z_l_sum = 0.0
                y_l_y_l_sum = 0.0
                y_l_z_l_sum = 0.0
                for k in range(9):
                    x_l_sum = x_l_sum + coord[k, 0]
                    y_l_sum = y_l_sum + coord[k, 1]
                    z_l_sum = z_l_sum + coord[k, 2]
                    x_l_x_l_sum = x_l_x_l_sum + (coord[k, 0] * coord[k, 0])
                    x_l_y_l_sum = x_l_y_l_sum + (coord[k, 0] * coord[k, 1])
                    x_l_z_l_sum = x_l_z_l_sum + (coord[k, 0] * coord[k, 2])
                    y_l_y_l_sum = y_l_y_l_sum + (coord[k, 1] * coord[k, 1])
                    y_l_z_l_sum = y_l_z_l_sum + (coord[k, 1] * coord[k, 2])
                # Fortran-contiguous
                mat[0] = x_l_x_l_sum
                mat[3] = x_l_y_l_sum
                mat[6] = x_l_sum
                mat[1] = x_l_y_l_sum
                mat[4] = y_l_y_l_sum
                mat[7] = y_l_sum
                mat[2] = x_l_sum
                mat[5] = y_l_sum
                mat[8] = 9.0
                vec[0] = x_l_z_l_sum
                vec[1] = y_l_z_l_sum
                vec[2] = z_l_sum
                sgesv(&num, &nrhs, &mat[0], &lda, &ipiv[0], &vec[0], &ldb,
                      &info)
                vec[2] = -1.0

                vec_x = vec[0]
                vec_y = vec[1]
                vec_z = vec[2]

                # Normalise vector
                vec_mag = sqrt(vec_x ** 2 + vec_y ** 2 + vec_z ** 2)
                vec_x = vec_x / vec_mag
                vec_y = vec_y / vec_mag
                vec_z = vec_z / vec_mag

                # Reverse orientation of plane's normal vector (if necessary)
                if vec_z < 0.0:
                    vec_x = vec_x * -1.0
                    vec_y = vec_y * -1.0
                    vec_z = vec_z * -1.0

                vec_tilt[i, j, 0] = vec_x
                vec_tilt[i, j, 1] = vec_y
                vec_tilt[i, j, 2] = vec_z

    return np.asarray(vec_tilt)


# -----------------------------------------------------------------------------

def slope_vector_meth(float[:, :] x, float[:, :] y, float[:, :] z,
                      float[:, :, :, :] rot_mat=np.empty((0, 0, 3, 3),
                                                         dtype=np.float32)):
    """Slope computation.

    Compute surface slope of DEM from central and 4 neighbouring grid cells.

    Parameters
    ----------
    x : ndarray of float
        Array (two-dimensional) with x-coordinates [metre]
    y : ndarray of float
        Array (two-dimensional) with y-coordinates [metre]
    z : ndarray of float
        Array (two-dimensional) with z-coordinates [metre]
    rot_mat: ndarray of float, optional
        Array (four-dimensional) with rotation matrix (y, x, 3, 3) to transform
        coordinates to a local coordinate system in which the z-axis aligns
        with local up

    Returns
    -------
    vec_tilt : ndarray of float
        Array (three-dimensional) with titled surface normal components
        (y, x, components) [metre]

    Notes
    -----
    Vector-based method which averages the surface normals of the 4 adjacent
    triangles. Concept based on Corripio et al. (2003): Vectorial algebra
    algorithms for calculating terrain parameters from DEMs and solar
    radiation modelling in mountainous terrain."""

    cdef int len_0 = x.shape[0]
    cdef int len_1 = x.shape[1]
    cdef int i, j
    cdef float vec_x, vec_y, vec_z, vec_mag
    cdef float a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z, d_x, d_y, d_z
    cdef float[:, :, :] vec_tilt = np.empty((len_0, len_1, 3),
                                            dtype=np.float32)

    # Initialise array
    vec_tilt[:] = NAN

    # Loop through grid cells
    # for i in range(1, (len_0 - 1)):
    for i in prange(1, (len_0 - 1), nogil=True, schedule="static"):
        for j in range(1, (len_1 - 1)):

            # Compute normal vector of plane (average of 4 triangles)
            a_x = x[i, j - 1] - x[i, j]
            a_y = y[i, j - 1] - y[i, j]
            a_z = z[i, j - 1] - z[i, j]
            b_x = x[i + 1, j] - x[i, j]
            b_y = y[i + 1, j] - y[i, j]
            b_z = z[i + 1, j] - z[i, j]
            c_x = x[i, j + 1] - x[i, j]
            c_y = y[i, j + 1] - y[i, j]
            c_z = z[i, j + 1] - z[i, j]
            d_x = x[i - 1, j] - x[i, j]
            d_y = y[i - 1, j] - y[i, j]
            d_z = z[i - 1, j] - z[i, j]
            # ((a x b) + (b x c) + (c x d) + (d x a))) / 4.0
            vec_x = ((a_y * b_z - a_z * b_y)
                     + (b_y * c_z - b_z * c_y)
                     + (c_y * d_z - c_z * d_y)
                     + (d_y * a_z - d_z * a_y)) / 4.0
            vec_y = ((a_z * b_x - a_x * b_z)
                     + (b_z * c_x - b_x * c_z)
                     + (c_z * d_x - c_x * d_z)
                     + (d_z * a_x - d_x * a_z)) / 4.0
            vec_z = ((a_x * b_y - a_y * b_x)
                     + (b_x * c_y - b_y * c_x)
                     + (c_x * d_y - c_y * d_x)
                     + (d_x * a_y - d_y * a_x)) / 4.0

            # Normalise vector
            vec_mag = sqrt(vec_x ** 2 + vec_y ** 2 + vec_z ** 2)
            vec_x = vec_x / vec_mag
            vec_y = vec_y / vec_mag
            vec_z = vec_z / vec_mag

            # Reverse orientation of plane's normal vector (if necessary)
            if vec_z < 0.0:
                vec_x = vec_x * -1.0
                vec_y = vec_y * -1.0
                vec_z = vec_z * -1.0

            vec_tilt[i, j, 0] = vec_x
            vec_tilt[i, j, 1] = vec_y
            vec_tilt[i, j, 2] = vec_z

    # Perform local coordinate transformation (optional)
    if rot_mat.shape[0] != 0:
        printf("Perform local coordinate transformation\n")

        # for i in range(1, (len_0 - 1)):
        for i in prange(1, (len_0 - 1), nogil=True, schedule="static"):
            for j in range(1, (len_1 - 1)):
                vec_x = rot_mat[i, j, 0, 0] * vec_tilt[i, j, 0] \
                        + rot_mat[i, j, 0, 1] * vec_tilt[i, j, 1] \
                        + rot_mat[i, j, 0, 2] * vec_tilt[i, j, 2]
                vec_y = rot_mat[i, j, 1, 0] * vec_tilt[i, j, 0] \
                        + rot_mat[i, j, 1, 1] * vec_tilt[i, j, 1] \
                        + rot_mat[i, j, 1, 2] * vec_tilt[i, j, 2]
                vec_z = rot_mat[i, j, 2, 0] * vec_tilt[i, j, 0] \
                        + rot_mat[i, j, 2, 1] * vec_tilt[i, j, 1] \
                        + rot_mat[i, j, 2, 2] * vec_tilt[i, j, 2]
                vec_tilt[i, j, 0] = vec_x
                vec_tilt[i, j, 1] = vec_y
                vec_tilt[i, j, 2] = vec_z

    return np.asarray(vec_tilt)


###############################################################################
# Sky View Factor
###############################################################################

def skyviewfactor(float[:] azim, float[:, :, :] hori, float[:, :, :] vec_tilt):
    """Sky View Factor (SVF) computation.

    Computes Sky View Factor in local horizontal coordinate system.

    Parameters
    ----------
    azim : ndarray of float
        Array (one-dimensional) with azimuth [radian]
    hori : ndarray of float
        Array (three-dimensional) with horizon (azim, y, x) [radian]
    vec_tilt : ndarray of float
        Array (three-dimensional) with titled surface normal components
        (y, x, components) [metre]

    Returns
    -------
    svf : ndarray of float
        Array (two-dimensional) with Sky View Factor [-]"""

    cdef int len_0 = hori.shape[0]
    cdef int len_1 = hori.shape[1]
    cdef int len_2 = hori.shape[2]
    cdef int i, j, k
    cdef float azim_spac
    cdef float vec_east_enu_x, vec_east_enu_y, vec_east_enu_z
    cdef float vec_loc_cs_x, vec_loc_cs_y, vec_loc_cs_z
    cdef float agg, hori_plane, hori_zen, term
    cdef float[:, :] svf = np.empty((len_1, len_2), dtype=np.float32)
    cdef float[:] azim_sin = np.empty(len_0, dtype=np.float32)
    cdef float[:] azim_cos = np.empty(len_0, dtype=np.float32)

    # Precompute values of trigonometric functions
    for i in range(len_0):
        azim_sin[i] = sin(azim[i])
        azim_cos[i] = cos(azim[i])
    # -> these arrays can be shared between threads (read-only)

    # Compute Sky View Factor
    azim_spac = (azim[1] - azim[0])
    for i in prange(len_1, nogil=True, schedule="static"):
        for j in range(len_2):

            # Iterate over sky
            agg = 0.0
            for k in range(len_0):

                # Compute plane-sphere intersection
                hori_plane = atan(- azim_sin[k] * vec_tilt[i, j, 0]
                                  / vec_tilt[i, j, 2]
                                  - azim_cos[k] * vec_tilt[i, j, 1]
                                  / vec_tilt[i, j, 2])
                if hori[k, i, j] >= hori_plane:
                    hori_zen = (M_PI / 2.0) - hori[k, i, j]
                else:
                    hori_zen = (M_PI / 2.0) - hori_plane

                # Compute inner integral
                term = (hori_zen - sin(hori_zen) * cos(hori_zen))
                agg = agg + (vec_tilt[i, j, 0] * azim_sin[k] * term
                             + vec_tilt[i, j, 1] * azim_cos[k] * term
                             + vec_tilt[i, j, 2] * sin(hori_zen) ** 2)

            svf[i, j] = (1.0 / (2.0 * M_PI)) * agg * azim_spac

    return np.asarray(svf)

###############################################################################
# Auxiliary functions
###############################################################################

cdef inline double deg2rad(double ang_in) nogil:
    """Convert degree to radian"""

    cdef double ang_out
    
    ang_out = ang_in * (M_PI / 180.0)
       
    return ang_out
