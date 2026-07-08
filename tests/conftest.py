import numpy as np
import pytest

import horayzon as hray

HORIZON_ATOL = 2.0e-2


@pytest.fixture
def flat_planar_dem():
    """Small flat DEM with enough buffer around a 3x3 inner domain."""
    coord = np.linspace(-1000.0, 1000.0, 9, dtype=np.float32)
    x, y = np.meshgrid(coord, coord)
    z = np.zeros_like(x, dtype=np.float32)
    vert_grid = hray.auxiliary.rearrange_pad_buffer(x, y, z)
    return x, y, z, vert_grid


@pytest.fixture
def flat_vectors():
    def make(shape):
        vec_norm = np.zeros(shape + (3,), dtype=np.float32)
        vec_norm[..., 2] = 1.0
        vec_north = np.zeros(shape + (3,), dtype=np.float32)
        vec_north[..., 1] = 1.0
        return vec_norm, vec_north

    return make


@pytest.fixture
def horizon_atol():
    return HORIZON_ATOL
