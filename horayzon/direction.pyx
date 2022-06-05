#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from libc.math cimport sin, cos, sqrt
from libc.math cimport M_PI
from cython.parallel import prange


# -----------------------------------------------------------------------------

def surf_norm(double[:, :] lon, double[:, :] lat):
    """Compute surface normal unit vectors.

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

    Sources
    -------
    - https://en.wikipedia.org/wiki/N-vector"""

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
    """Compute surface-perpendicular unit vectors pointing towards North.

    Computation surface-perpendicular unit vectors pointing towards North
    in earth-centered, earth-fixed (ECEF) coordinates.

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

    Sources
    -------
    - Geoid parameters r, a and f: PROJ"""

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


# -----------------------------------------------------------------------------
# Auxiliary function(s)
# -----------------------------------------------------------------------------

cdef inline double deg2rad(double ang_in) nogil:
    """Convert degree to radian"""

    cdef double ang_out
    
    ang_out = ang_in * (M_PI / 180.0)
       
    return ang_out
