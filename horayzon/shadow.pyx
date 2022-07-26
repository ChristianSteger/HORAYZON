# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

cimport numpy as np
import numpy as np
import os

cdef extern from "shadow_comp.h" namespace "shapes":
    cdef cppclass CppTerrain:
        CppTerrain()
        void initialise(float*, int, int, int, int, float*, float*,
                        int, int, float*, char*, unsigned char*, float, float)
        void shadow(float*, unsigned char*)
        void sw_dir_cor(float*, float*)

cdef class Terrain:

    cdef CppTerrain *thisptr

    def __cinit__(self):
        self.thisptr = new CppTerrain()

    def __dealloc__(self):
        del self.thisptr

    def initialise(self, np.ndarray[np.float32_t, ndim = 1] vert_grid,
                   int dem_dim_0, int dem_dim_1,
                   int offset_0, int offset_1,
                   np.ndarray[np.float32_t, ndim = 3] vec_tilt,
                   np.ndarray[np.float32_t, ndim = 3] vec_norm,
                   np.ndarray[np.float32_t, ndim = 2] surf_enl_fac,
                   str geom_type,
                   np.ndarray[np.uint8_t, ndim = 2] mask,
                   float sw_dir_cor_fill=np.nan,
                   float ang_max=89.0):
        """Initialise Terrain class with Digital Elevation Model (DEM) data.

        Initialise Terrain class with Digital Elevation Model (DEM) data and
        auxiliary quantities (vectors for horizontal/titled surface normals
        and surface enlargement factors).

        Parameters
        ----------
        vert_grid : ndarray of float
            Array (one-dimensional) with vertices of DEM [metre]
        dem_dim_0 : int
            Dimension length of DEM in y-direction
        dem_dim_1 : int
            Dimension length of DEM in x-direction
        offset_0 : int
            Offset of inner domain in y-direction
        offset_1 : int
            Offset of inner domain in x-direction
        vec_tilt : ndarray of float
            Array (three-dimensional) with titled surface normals
            (y, x, components) [metre]
        vec_norm : ndarray of float
            Array (three-dimensional) with surface normal components
            (y, x, components) [metre]
        surf_enl_fac : ndarray of float
            Array (three-dimensional) with surface enlargement factor
            (y, x) [-]
        geom_type : str
            Embree geometry type (triangle, quad, grid)
        mask : ndarray of uint8
            Array (two-dimensional) with grid cell mask (0: ignored,
            1: considered)
        sw_dir_cor_fill : float
            Shortwave correction factor fill value for masked grid cells
        ang_max: float
            Maximal angle between sun vector and tilted surface normal
            for which correction is computed. For larger angles, 'sw_dir_cor'
            is set to 0.0. 'ang_max' is also applied to restrict the maximal
            angle between the sun vector and the horizontal surface normal."""

        # Check consistency and validity of input arguments
        if len(vert_grid) < (dem_dim_0 * dem_dim_1 * 3):
            raise ValueError("inconsistency between input arguments "
                             + "'vert_grid', 'dem_dim_0' and 'dem_dim_1'")
        if ((offset_0 + vec_tilt.shape[0] > dem_dim_0)
                or (offset_1 + vec_tilt.shape[1] > dem_dim_1)):
            raise ValueError("inconsistency between input arguments "
                             + "'dem_dim_0', 'dem_dim_1', 'offset_0', "
                             + "'offset_1' and 'vec_norm'")
        if ((vec_tilt.ndim != 3) or (vec_norm.ndim != 3)
                or (surf_enl_fac.ndim != 2)
                or (vec_tilt.shape[0] != vec_norm.shape[0])
                or (vec_tilt.shape[1] != vec_norm.shape[1])
                or (vec_tilt.shape[2] != vec_norm.shape[2])
                or (vec_tilt.shape[0] != surf_enl_fac.shape[0])
                or (vec_tilt.shape[1] != surf_enl_fac.shape[1])):
            raise ValueError("Inconsistent/incorrect shape of 'vec_tilt', "
                             + "'vec_norm' and/or 'surf_enl_fac'")
        if ((not vert_grid.flags["C_CONTIGUOUS"])
                or (not vec_tilt.flags["C_CONTIGUOUS"])
                or (not vec_norm.flags["C_CONTIGUOUS"])
                or (not surf_enl_fac.flags["C_CONTIGUOUS"])):
            raise ValueError("not all input arrays are C-contiguous")
        if ((np.abs((vec_tilt ** 2).sum(axis=2) - 1.0).max() > 1.0e-5)
                or (np.abs((vec_norm ** 2).sum(axis=2) - 1.0).max() > 1.0e-5)):
            raise ValueError("Vectors in 'vec_tilt' and/or 'vec_norm' are "
                             + "not normalised")
        if geom_type not in ("triangle", "quad", "grid"):
            raise ValueError("invalid input argument for geom_type")
        if ((mask.shape[0] != vec_tilt.shape[0])
                or (mask.shape[1] != vec_tilt.shape[1])):
            raise ValueError("shape of mask is inconsistent with other input")
        if mask.dtype != "uint8":
            raise TypeError("data type of mask must be 'uint8'")
        if (ang_max < 85.0) or (ang_max > 89.99):
            raise TypeError("'ang_max' must be in the range [85.0, 89.99]")

        # Check size of input geometries
        if (dem_dim_0 > 32767) or (dem_dim_1 > 32767):
            raise ValueError("maximal allowed input length for dem_dim_0 and "
                             "dem_dim_1 is 32'767")

        self.thisptr.initialise(&vert_grid[0],
                                dem_dim_0, dem_dim_1,
                                offset_0, offset_1,
                                &vec_tilt[0,0,0],
                                &vec_norm[0,0,0],
                                vec_tilt.shape[0], vec_tilt.shape[1],
                                &surf_enl_fac[0,0],
                                geom_type.encode("utf-8"),
                                &mask[0,0],
                                sw_dir_cor_fill,
                                ang_max)

    def shadow(self, np.ndarray[np.float32_t, ndim = 1] sun_position,
                 np.ndarray[np.uint8_t, ndim = 2] shadow_buffer):
        """Compute shadow mask for specified sun position.

        Compute shadow mask for specified sun position with the following
        encoding: 0: illuminated, 1: self-shaded, 2: terrain-shaded.

        Parameters
        ----------
        sun_position : ndarray of float
            Array (one-dimensional) with position of sun in global ENU
            coordinates (x, y, z) [metre]
        shadow_buffer : ndarray of unsigned char (8-bit unsigned integer)
            Array (two-dimensional) with shadow mask (y, x) [-]"""

        # Check consistency and validity of input arguments
        if (sun_position.ndim != 1) or (sun_position.size != 3):
            raise ValueError("array 'sun_position' has incorrect shape")
        if not shadow_buffer.flags["C_CONTIGUOUS"]:
            raise ValueError("array 'shadow_buffer' is not C-contiguous")

        self.thisptr.shadow(&sun_position[0], &shadow_buffer[0,0])

    def sw_dir_cor(self, np.ndarray[np.float32_t, ndim = 1] sun_position,
                 np.ndarray[np.float32_t, ndim = 2] sw_dir_cor_buffer):
        """Compute shortwave correction factor for specified sun position.

        Compute correction factor for direct downward shortwave radiation
        (from 1D plane-parallel radiative transfer model) for specified sun
        position.

        Parameters
        ----------
        sun_position : ndarray of float
            Array (one-dimensional) with position of sun in global ENU
            coordinates (x, y, z) [metre]
        sw_dir_cor_buffer : ndarray of float
            Array (two-dimensional) with shortwave correction factor (y, x) [-]

        References
        ----------
        - Mueller, M. D., & Scherer, D. (2005): A Grid- and Subgrid-Scale
        Radiation Parameterization of Topographic Effects for Mesoscale
        Weather Forecast Models, Monthly Weather Review, 133(6), 1431-1442."""

        # Check consistency and validity of input arguments
        if (sun_position.ndim != 1) or (sun_position.size != 3):
            raise ValueError("array 'sun_position' has incorrect shape")
        if not sw_dir_cor_buffer.flags["C_CONTIGUOUS"]:
            raise ValueError("array 'sw_dir_cor_buffer' is not C-contiguous")

        self.thisptr.sw_dir_cor(&sun_position[0], &sw_dir_cor_buffer[0,0])