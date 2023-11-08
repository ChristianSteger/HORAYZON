# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

cimport numpy as np
import numpy as np
import os


# -----------------------------------------------------------------------------
# Compute horizon for gridded domain
# -----------------------------------------------------------------------------

cdef extern from "horizon_comp.h":
    void horizon_gridded_comp(
            float* vert_grid,
            int dem_dim_0, int dem_dim_1,
            float* vec_norm, float* vec_north,
            int offset_0, int offset_1,
            float* hori_buffer,
            int dim_in_0, int dim_in_1,
            int azim_num, float dist_search,
            float hori_acc, char* ray_algorithm, char* geom_type,
            float* vert_simp, int num_vert_simp,
            np.npy_int32* tri_ind_simp, int num_tri_simp,
            char* file_out,
            float* x_axis_val, float* y_axis_val,
            char* x_axis_name, char* y_axis_name, char* units,
            float hori_buffer_size_max,
            float elev_ang_low_lim,
            np.npy_uint8* mask, float hori_fill,
            float ray_org_elev)

def horizon_gridded(
        np.ndarray[np.float32_t, ndim = 1] vert_grid,
        int dem_dim_0, int dem_dim_1,
        np.ndarray[np.float32_t, ndim = 3] vec_norm,
        np.ndarray[np.float32_t, ndim = 3] vec_north,
        int offset_0, int offset_1,
        str file_out,
        np.ndarray[np.float32_t, ndim = 1] x_axis_val,
        np.ndarray[np.float32_t, ndim = 1] y_axis_val,
        str x_axis_name, str y_axis_name, str units,
        float dist_search,
        int azim_num=360,
        float hori_acc=0.25,
        str ray_algorithm="guess_constant",
        str geom_type="grid",
        np.ndarray[np.float32_t, ndim = 1]
        vert_simp=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        int num_vert_simp=1,
        np.ndarray[np.int32_t, ndim = 1]
        tri_ind_simp=np.array([0, 0, 0, 0], dtype=np.int32),
        int num_tri_simp=1,
        float hori_buffer_size_max=1.5,
        float elev_ang_low_lim = -15.0,
        np.ndarray[np.uint8_t, ndim = 2] mask=None,
        float hori_fill=0.0,
        float ray_org_elev=0.01):
    """Horizon computation for gridded domain.

    Computes horizon from a Digital Elevation Model (DEM) with Intel Embree
    high performance ray tracing kernels.

    Parameters
    ----------
    vert_grid : ndarray of float
        Array (one-dimensional) with vertices of DEM [metre]
    dem_dim_0 : int
        Dimension length of DEM in y-direction
    dem_dim_1 : int
        Dimension length of DEM in x-direction
    vec_norm : ndarray of float
        Array (three-dimensional) with surface normal components
        (y, x, components) [metre]
    vec_north : ndarray of float
        Array (three-dimensional) with north vector components
        (y, x, components) [metre]
    offset_0 : int
        Offset of inner domain in y-direction
    offset_1 : int
        Offset of inner domain in x-direction
    file_out : str
        Path and file name for output
    x_axis_val : ndarray of float
        Array (one-dimensional) with x-coordinates of inner domain
    y_axis_val : ndarray of float
        Array (one-dimensional) with y-coordinates of inner domain
    x_axis_name : str
        Name of x-axis
    y_axis_name : str
        Name of y-axis
    units : str
        Units of x- and y-axis
    dist_search : float
        Search distance for horizon [kilometre]
    azim_num : int
        Number of azimuth sectors
    hori_acc : float
        Accuracy of horizon computation [degree]
    ray_algorithm : str
        Algorithm for horizon detection (discrete_sampling, binary_search,
        guess_constant)
    geom_type : str
        Embree geometry type (triangle, quad, grid)
    vert_simp : ndarray of float
        Array (one-dimensional) with vertices of simplified outer DEM [metre]
    num_vert_simp : int
        Number of vertices of outer simplified DEM
    tri_ind_simp : ndarray of int
        Array (one-dimensional) with vertex indices of triangles
    num_tri_simp : int
        Number of triangles
    hori_buffer_size_max : float
        Maximal size of horizon buffer [Gigabyte]
    elev_ang_low_lim : float
        Lower limit for elevation angle search [degree]
    mask : ndarray of uint8
        Array (two-dimensional) with locations for which horizon is computed
    hori_fill : float
        Horizon fill values for masked locations
    ray_org_elev : float
        Vertical elevation of ray origin above surface [metre]"""

	# Check consistency and validity of input arguments
    if len(vert_grid) < (dem_dim_0 * dem_dim_1 * 3):
        raise ValueError("inconsistency between input arguments vert_grid, "
                         "dem_dim_0 and dem_dim_1")
    if ((offset_0 + vec_norm.shape[0] > dem_dim_0)
            or (offset_1 + vec_norm.shape[1] > dem_dim_1)):
        raise ValueError("inconsistency between input arguments dem_dim_0, "
                         "dem_dim_1, offset_0, offset_1 and vec_norm")
    if ((vec_norm.ndim != 3) or (vec_north.ndim != 3)
            or (vec_norm.shape[0] != vec_north.shape[0])
            or (vec_norm.shape[1] != vec_north.shape[1])
            or (vec_norm.shape[2] != vec_north.shape[2])):
        raise ValueError("dimension (lengths) of vec_norm and/or vec_north "
                         "is/are erroneous")
    if ray_algorithm not in ("discrete_sampling", "binary_search",
                             "guess_constant"):
        raise ValueError("invalid input argument for ray_algorithm")
    if geom_type not in ("triangle", "quad", "grid"):
        raise ValueError("invalid input argument for geom_type")
    if len(vert_simp) < (num_vert_simp * 3):
        raise ValueError("inconsistency between input arguments vert_simp "
                         "and num_vert_simp")
    if len(tri_ind_simp) < (num_tri_simp * 3):
        raise ValueError("inconsistency between input arguments tri_ind_simp "
                         "and num_tri_simp")
    if tri_ind_simp.max() > (num_vert_simp - 1):
        raise ValueError("triangle indices of simplified outer domain exceed "
                         "number of vertices")
    if not os.path.isdir("/".join(file_out.split("/")[:-1])):
        raise ValueError("output directory does not exist")
    if ((len(y_axis_val) != vec_norm.shape[0])
            or (len(x_axis_val) != vec_norm.shape[1])):
        raise ValueError("lengths of x_axis_val and/or y_axis_val is/are"
                         " inconsistent with dimension lengths of vec_norm")
    if hori_acc > 10.0:
        raise ValueError("limit of hori_acc (10 degree) is exceeded")
    if mask is None:
        mask = np.ones((vec_norm.shape[0], vec_norm.shape[1]), dtype=np.uint8)
    if (mask.shape[0] != vec_norm.shape[0]) \
            or (mask.shape[1] != vec_norm.shape[1]):
        raise ValueError("shape of mask is inconsistent with other input")
    if mask.dtype != "uint8":
        raise TypeError("data type of mask must be 'uint8'")
    if ray_org_elev < 0.005:
        raise TypeError("minimal allowed value for 'ray_org_elev' is 0.005 m")

    # Check size of input geometries
    if (dem_dim_0 > 32767) or (dem_dim_1 > 32767):
        raise ValueError("maximal allowed input length for dem_dim_0 and "
                         "dem_dim_1 is 32'767")
    if vert_simp.nbytes > (16.0 * 10 ** 9):
        raise ValueError("vertex buffer vert_simp is larger than 16 GB")
    # -> this check also ensures that length of vert_simp is smaller than
    #    2'147'483'647 -> otherwise, indices in tri_ind_simp are erroneous
    #    due to signed/unsigned 32bit integer

    # Ensure that passed arrays are contiguous in memory
    vert_grid = np.ascontiguousarray(vert_grid)
    vec_norm = np.ascontiguousarray(vec_norm)
    vec_north = np.ascontiguousarray(vec_north)
    vert_simp = np.ascontiguousarray(vert_simp)
    tri_ind_simp = np.ascontiguousarray(tri_ind_simp)

    # Convert input strings to bytes
    ray_algorithm_c = ray_algorithm.encode("utf-8")
    geom_type_c = geom_type.encode("utf-8")
    file_out_c = file_out.encode("utf-8")
    x_axis_name_c = x_axis_name.encode("utf-8")
    y_axis_name_c = y_axis_name.encode("utf-8")
    units_c = units.encode("utf-8")

    # Allocate horizon array
    cdef float hori_buffer_size = (vec_norm.shape[0] * vec_norm.shape[1] *
                                   azim_num * 4) / (10.0 ** 9.0)
    cdef np.int64_t hori_buffer_len
    if hori_buffer_size <= hori_buffer_size_max:
        print("Horizon buffer size is below specified limit")
        hori_buffer_len = vec_norm.shape[0] * vec_norm.shape[1] * azim_num
    else:
        print("Horizon buffer size is restricted")
        hori_buffer_len = np.int64((hori_buffer_size_max * 10 ** 9) / 4) + 100000
        # add some "safety" memory to the buffer (-> 100000)
    cdef np.ndarray[np.float32_t, ndim = 1, mode = "c"] \
        hori_buffer = np.empty(hori_buffer_len,  dtype=np.float32)
    hori_buffer.fill(np.nan)

    horizon_gridded_comp(
        &vert_grid[0],
        dem_dim_0, dem_dim_1,
        &vec_norm[0,0,0], &vec_north[0,0,0],
        offset_0, offset_1,
        &hori_buffer[0],
        vec_norm.shape[0], vec_norm.shape[1],
        azim_num, dist_search,
        hori_acc, ray_algorithm_c, geom_type_c,
        &vert_simp[0], num_vert_simp,
        &tri_ind_simp[0], num_tri_simp,
        file_out_c,
        &x_axis_val[0], &y_axis_val[0],
        x_axis_name_c, y_axis_name_c, units_c,
        hori_buffer_size_max,
        elev_ang_low_lim,
        &mask[0,0], hori_fill,
        ray_org_elev)

# -----------------------------------------------------------------------------
# Compute horizon for arbitrary locations
# -----------------------------------------------------------------------------

cdef extern from "horizon_comp.h":
    void horizon_locations_comp(
            float* vert_grid,
            int dem_dim_0, int dem_dim_1,
            float* coords,
            float* vec_norm, float* vec_north,
            float* hori_buffer,
            int num_loc,
            int azim_num, float dist_search,
            float hori_acc, char* ray_algorithm, char* geom_type,
            char* file_out,
            float* x_axis_val, float* y_axis_val,
            char* x_axis_name, char* y_axis_name, char* units,
            float elev_ang_low_lim,
            float* ray_org_elev,
            int hori_dist_out)

def horizon_locations(
        np.ndarray[np.float32_t, ndim = 1] vert_grid,
        int dem_dim_0, int dem_dim_1,
        np.ndarray[np.float32_t, ndim = 2] coords,
        np.ndarray[np.float32_t, ndim = 2] vec_norm,
        np.ndarray[np.float32_t, ndim = 2] vec_north,
        str file_out,
        np.ndarray[np.float32_t, ndim = 1] x_axis_val,
        np.ndarray[np.float32_t, ndim = 1] y_axis_val,
        str x_axis_name, str y_axis_name, str units,
        float dist_search,
        int azim_num=360,
        float hori_acc=0.25,
        str ray_algorithm="binary_search",
        str geom_type="grid",
        float elev_ang_low_lim = -89.98,
        np.ndarray[np.float32_t, ndim = 1] ray_org_elev \
        = np.array([0.01], dtype=np.float32),
        bint hori_dist_out=False):
    """Horizon computation for arbitrary locations.

    Computes horizon from a Digital Elevation Model (DEM) with Intel Embree
    high performance ray tracing kernels.

    Parameters
    ----------
    vert_grid : ndarray of float
        Array (one-dimensional) with vertices of DEM [metre]
    dem_dim_0 : int
        Dimension length of DEM in y-direction
    dem_dim_1 : int
        Dimension length of DEM in x-direction
    coords : ndarray of float
        Array (two-dimensional) with coordinates of locations
        (number of locations, x-/y-/z-coordinates)
    vec_norm : ndarray of float
        Array (two-dimensional) with surface normal components
        (number of locations, components) [metre]
    vec_north : ndarray of float
        Array (two-dimensional) with north vector components
        (number of locations, components) [metre]
    file_out : str
        Path and file name for output
    x_axis_val : ndarray of float
        Array (one-dimensional) with x-coordinates of locations
    y_axis_val : ndarray of float
        Array (one-dimensional) with y-coordinates of locations
    x_axis_name : str
        Name of x-axis
    y_axis_name : str
        Name of y-axis
    units: str
        Units of x- and y-axis
    dist_search : float
        Search distance for horizon [kilometre]
    azim_num : int
        Number of azimuth sectors
    hori_acc : float
        Accuracy of horizon computation [degree]
    ray_algorithm : str
        Algorithm for horizon detection (discrete_sampling, binary_search,
        guess_constant)
    geom_type : str
        Embree geometry type (triangle, quad, grid)
    elev_ang_low_lim : float
        Lower limit for elevation angle search [degree]
    ray_org_elev : ndarray of float
        Vertical elevation of ray origin above surface [metre]
    hori_dist_out : bool
        Option to output distance to horizon
    """

	# Check consistency and validity of input arguments
    if len(vert_grid) < (dem_dim_0 * dem_dim_1 * 3):
        raise ValueError("inconsistency between input arguments vert_grid, "
                         "dem_dim_0 and dem_dim_1")
    if ((coords.ndim != 2) or (coords.shape[0] != vec_norm.shape[0])
            or (coords.shape[1] !=3)):
        raise ValueError("'number of dimensions and/or dimension "
                         + "length(s) of 'coords' incorrect")
    if ((vec_norm.ndim != 2) or (vec_north.ndim != 2)
            or (vec_norm.shape[0] != vec_north.shape[0])
            or (vec_norm.shape[1] != vec_north.shape[1])):
        raise ValueError("dimension (lengths) of vec_norm and/or vec_north "
                         "is/are erroneous")
    if ray_algorithm not in ("discrete_sampling", "binary_search",
                             "guess_constant"):
        raise ValueError("invalid input argument for ray_algorithm")
    if geom_type not in ("triangle", "quad", "grid"):
        raise ValueError("invalid input argument for geom_type")
    if not os.path.isdir("/".join(file_out.split("/")[:-1])):
        raise ValueError("output directory does not exist")
    if len(y_axis_val) != vec_norm.shape[0]:
        raise ValueError("lengths of x_axis_val and/or y_axis_val is/are"
                         " inconsistent with dimension lengths of vec_norm")
    if hori_acc > 10.0:
        raise ValueError("limit of hori_acc (10 degree) is exceeded")
    if (len(ray_org_elev) != 1) and (len(ray_org_elev) != coords.shape[0]):
        raise ValueError("length of array 'ray_org_elev' must be either "
                         + "one or correspond to the number of locations")
    if ray_org_elev.min() < 0.005:
        raise TypeError("minimal allowed value for 'ray_org_elev' is 0.005 m")
    if hori_dist_out and (ray_algorithm == "guess_constant"):
        raise TypeError("horizon detection algorithm 'guess_constant' not "
                        + "implemented for horizon distance computation")

    # Check size of input geometries
    if (dem_dim_0 > 32767) or (dem_dim_1 > 32767):
        raise ValueError("maximal allowed input length for dem_dim_0 and "
                         "dem_dim_1 is 32'767")

    # Repeat array 'ray_org_elev' if necessary
    if len(ray_org_elev) != coords.shape[0]:
        ray_org_elev = np.repeat(ray_org_elev, coords.shape[0])

    # Ensure that passed arrays are contiguous in memory
    vert_grid = np.ascontiguousarray(vert_grid)
    coords = np.ascontiguousarray(coords)
    vec_norm = np.ascontiguousarray(vec_norm)
    vec_north = np.ascontiguousarray(vec_north)
    ray_org_elev = np.ascontiguousarray(ray_org_elev)

    # Convert input strings to bytes
    ray_algorithm_c = ray_algorithm.encode("utf-8")
    geom_type_c = geom_type.encode("utf-8")
    file_out_c = file_out.encode("utf-8")
    x_axis_name_c = x_axis_name.encode("utf-8")
    y_axis_name_c = y_axis_name.encode("utf-8")
    units_c = units.encode("utf-8")

    # Allocate horizon array
    cdef np.int64_t hori_buffer_len = vec_norm.shape[0] * azim_num
    cdef np.ndarray[np.float32_t, ndim = 1, mode = "c"] \
        hori_buffer = np.empty(hori_buffer_len,  dtype=np.float32)
    hori_buffer.fill(np.nan)

    horizon_locations_comp(
        &vert_grid[0],
        dem_dim_0, dem_dim_1,
        &coords[0,0],
        &vec_norm[0,0], &vec_north[0,0],
        &hori_buffer[0],
        vec_norm.shape[0],
        azim_num, dist_search,
        hori_acc, ray_algorithm_c, geom_type_c,
        file_out_c,
        &x_axis_val[0], &y_axis_val[0],
        x_axis_name_c, y_axis_name_c, units_c,
        elev_ang_low_lim,
        &ray_org_elev[0],
        hori_dist_out)
