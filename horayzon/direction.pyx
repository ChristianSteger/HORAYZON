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

def surf_norm(lon, lat):
    """Compute surface normal unit vectors.

    Computation of surface normal unit vectors in earth-centered, earth-fixed
    (ECEF) coordinates.

    Parameters
    ----------
    lon : ndarray of double
        Array (with arbitrary dimensions) with geographic longitude [degree]
    lat : ndarray of double
        Array (with arbitrary dimensions) with geographic latitudes [degree]

    Returns
    -------
    vec_norm_ecef : ndarray of float
        Array (dimensions according to input; vector components are stored in
        last dimension) with surface normal components in ECEF coordinates
        [metre]"""

    # Check arguments
    if lon.shape != lat.shape:
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")
    if ((lon.dtype != "float64") or (lat.dtype != "float64")):
        raise ValueError("Input array(s) has/have incorrect data type(s)")

    # Wrapper for 1-dimensional function
    shp = lon.shape
    vec_norm_ecef = _surf_norm_1d(lon.ravel(), lat.ravel())
    return vec_norm_ecef.reshape(shp + (3,))


def _surf_norm_1d(double[:] lon, double[:] lat):
    """Compute surface normal unit vectors (for 1-dimensional data).

    Sources
    -------
    - https://en.wikipedia.org/wiki/N-vector"""

    cdef int len_0 = lon.shape[0]
    cdef int i
    cdef double sin_lon, cos_lon, sin_lat, cos_lat
    cdef float[:, :] vec_norm_ecef = np.empty((len_0, 3), dtype=np.float32)

    # Compute surface normals
    for i in prange(len_0, nogil=True, schedule="static"):
        sin_lon = sin(deg2rad(lon[i]))
        cos_lon = cos(deg2rad(lon[i]))
        sin_lat = sin(deg2rad(lat[i]))
        cos_lat = cos(deg2rad(lat[i]))
        vec_norm_ecef[i, 0] = cos_lat * cos_lon
        vec_norm_ecef[i, 1] = cos_lat * sin_lon
        vec_norm_ecef[i, 2] = sin_lat

    return np.asarray(vec_norm_ecef)


# -----------------------------------------------------------------------------

def north_dir(x_ecef, y_ecef, z_ecef, vec_norm_ecef, ellps):
    """Compute unit vectors pointing towards North.

    Computation unit vectors pointing towards North in earth-centered,
    earth-fixed (ECEF) coordinates. These vectors are perpendicular to surface
    normal unit vectors.

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF z-coordinates [metre]
    vec_norm_ecef : ndarray of float
        Array (at least two-dimensional; vector components must be stored in
        last dimension) with surface normal components in ECEF coordinates
        [metre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)

    Returns
    -------
    vec_north_ecef : ndarray of float
        Array (dimensions according to input; vector components are stored in
        last dimension) with north vector components in ECEF coordinates
        [metre]"""

    # Check arguments
    if (x_ecef.shape != y_ecef.shape) or (y_ecef.shape != z_ecef.shape) \
            or (z_ecef.shape != vec_norm_ecef
                                .shape[:(vec_norm_ecef.ndim - 1)]):
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")
    if ((x_ecef.dtype != "float64") or (y_ecef.dtype != "float64")
            or (z_ecef.dtype != "float64")
            or (vec_norm_ecef.dtype != "float32")):
        raise ValueError("Input array(s) has/have incorrect data type(s)")
    if ellps not in ("sphere", "GRS80", "WGS84"):
        raise ValueError("Unknown value for 'ellps'")

    # Wrapper for 1-dimensional function
    shp = x_ecef.shape
    vec_north_ecef = _north_dir_1d(x_ecef.ravel(), y_ecef.ravel(),
                                   z_ecef.ravel(),
                                   vec_norm_ecef.reshape(prod(shp), 3))
    return vec_north_ecef.reshape(shp + (3,))


def _north_dir_1d(double[:] x_ecef, double[:] y_ecef, double[:] z_ecef,
                  float[:, :] vec_norm_ecef, ellps):
    """Compute unit vectors pointing towards North (for 1-dimensional data).

    Sources
    -------
    - Geoid parameters r, a and f: PROJ"""

    cdef int len_0 = x_ecef.shape[0]
    cdef int i
    cdef double r, f, a, b
    cdef double np_x, np_y, np_z
    cdef double vec_nor_x, vec_nor_y, vec_nor_z
    cdef double dot_pr, vec_proj_x, vec_proj_y, vec_proj_z, norm
    cdef float[:, :] vec_north_ecef = np.empty((len_0, 3), dtype=np.float32)

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

        # Vector to North Pole
        vec_nor_x = (np_x - x_ecef[i])
        vec_nor_y = (np_y - y_ecef[i])
        vec_nor_z = (np_z - z_ecef[i])

        # Project vector to North Pole on surface normal plane
        dot_pr = ((vec_nor_x * vec_norm_ecef[i, 0])
                  + (vec_nor_y * vec_norm_ecef[i, 1])
                  + (vec_nor_z * vec_norm_ecef[i, 2]))
        vec_proj_x = vec_nor_x - dot_pr * vec_norm_ecef[i, 0]
        vec_proj_y = vec_nor_y - dot_pr * vec_norm_ecef[i, 1]
        vec_proj_z = vec_nor_z - dot_pr * vec_norm_ecef[i, 2]

        # Normalise vector
        norm = sqrt(vec_proj_x ** 2 + vec_proj_y ** 2 + vec_proj_z ** 2)
        vec_north_ecef[i, 0] = vec_proj_x / norm
        vec_north_ecef[i, 1] = vec_proj_y / norm
        vec_north_ecef[i, 2] = vec_proj_z / norm

    return np.asarray(vec_north_ecef)


# -----------------------------------------------------------------------------
# Auxiliary function(s)
# -----------------------------------------------------------------------------

cdef inline double deg2rad(double ang_in) nogil:
    """Convert degree to radian"""

    cdef double ang_out
    
    ang_out = ang_in * (M_PI / 180.0)
       
    return ang_out
