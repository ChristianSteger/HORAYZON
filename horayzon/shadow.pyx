# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

cimport numpy as np
import numpy as np
import os

cdef extern from "shadow_comp.h" namespace "shapes":
    cdef cppclass CppTerrain:
        CppTerrain()
        void initialise(float*, int, int, char*, int, int, float*, float*,
                        int, int, float*)
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
                   str geom_type,
                   int offset_0, int offset_1,
                   np.ndarray[np.float32_t, ndim = 3] vec_tilt,
                   np.ndarray[np.float32_t, ndim = 3] vec_norm,
                   int dim_in_0, int dim_in_1,
                   np.ndarray[np.float32_t, ndim = 2] surf_enl_fac):
        self.thisptr.initialise(&vert_grid[0],
                                dem_dim_0, dem_dim_1,
                                geom_type.encode("utf-8"),
                                offset_0, offset_1,
                                &vec_tilt[0,0,0],
                                &vec_norm[0,0,0],
                                dim_in_0, dim_in_1,
                                &surf_enl_fac[0,0])
    def shadow(self, np.ndarray[np.float32_t, ndim = 1] sun_position,
                 np.ndarray[np.uint8_t, ndim = 2] shaddow_buffer):
        self.thisptr.shadow(&sun_position[0], &shaddow_buffer[0,0])
    def sw_dir_cor(self, np.ndarray[np.float32_t, ndim = 1] sun_position,
                 np.ndarray[np.float32_t, ndim = 2] sw_dir_cor_buffer):
        self.thisptr.sw_dir_cor(&sun_position[0], &sw_dir_cor_buffer[0,0])