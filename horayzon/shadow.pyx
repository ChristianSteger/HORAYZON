# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

cimport numpy as np
import numpy as np
import os

cdef extern from "shadow_comp.h" namespace "shapes":
    cdef cppclass Rectangle:
        Rectangle(int, int, int, int) except +
        int x0, y0, x1, y1
        int getArea()
        void move(int, int)
        void initialise(float*, int, int, char*, int, int, float*, float*,
                        int, int)
        void shadow(float*, float*)

cdef class Terrain:
    cdef Rectangle *thisptr
    def __cinit__(self, int x0, int y0, int x1, int y1):
        self.thisptr = new Rectangle(x0, y0, x1, y1)
    def __dealloc__(self):
        del self.thisptr
    def getArea(self):
        return self.thisptr.getArea()
    def move(self, dx, dy):
        self.thisptr.move(dx, dy)
    def initialise(self, np.ndarray[np.float32_t, ndim = 1] vert_grid,
                   int dem_dim_0, int dem_dim_1,
                   str geom_type,
                   int offset_0, int offset_1,
                   np.ndarray[np.float32_t, ndim = 3] vec_tilt,
                   np.ndarray[np.float32_t, ndim = 3] vec_norm,
                   int dim_in_0, int dim_in_1):
        self.thisptr.initialise(&vert_grid[0],
                                dem_dim_0, dem_dim_1,
                                geom_type.encode("utf-8"),
                                offset_0, offset_1,
                                &vec_tilt[0,0,0],
                                &vec_norm[0,0,0],
                                dim_in_0, dim_in_1)

    def shadow(self, np.ndarray[np.float32_t, ndim = 1] sun_position,
                 np.ndarray[np.float32_t, ndim = 2] shaddow_buffer):
        self.thisptr.shadow(&sun_position[0], &shaddow_buffer[0,0])