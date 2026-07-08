import numpy as np
import pytest

import horayzon as hray


def test_rearrange_pad_buffer_preserves_vertices_and_aligns_buffer():
    x = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    y = x + 10.0
    z = x + 20.0

    buffer = hray.auxiliary.rearrange_pad_buffer(x, y, z)

    expected = np.hstack(
        (x.reshape(x.size, 1), y.reshape(y.size, 1), z.reshape(z.size, 1))
    ).ravel()
    np.testing.assert_array_equal(buffer[: expected.size], expected)
    assert buffer.dtype == np.float32
    assert buffer.nbytes % 16 == 0
    assert np.all(buffer[expected.size :] == 0.0)


def test_rearrange_pad_buffer_validates_shape_and_dtype():
    x = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros((2, 2), dtype=np.float32)
    z = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(TypeError):
        hray.auxiliary.rearrange_pad_buffer(x, y, z)

    with pytest.raises(ValueError):
        hray.auxiliary.rearrange_pad_buffer(x, y[:1], x)


def test_pad_buffer_requires_one_dimensional_numpy_array():
    with pytest.raises(ValueError):
        hray.auxiliary.pad_buffer([1.0, 2.0])

    with pytest.raises(ValueError):
        hray.auxiliary.pad_buffer(np.zeros((2, 2), dtype=np.float32))


def test_planar_grid_expands_domain_by_search_distance():
    domain = {"x_min": 100.0, "x_max": 200.0, "y_min": -50.0, "y_max": 50.0}

    outer = hray.domain.planar_grid(domain, dist_search=2.5)

    assert outer == {
        "x_min": -2400.0,
        "x_max": 2700.0,
        "y_min": -2550.0,
        "y_max": 2550.0,
    }


def test_curved_grid_sphere_expands_known_equatorial_domain():
    domain = {"lon_min": -1.0, "lon_max": 1.0, "lat_min": -1.0, "lat_max": 1.0}

    outer = hray.domain.curved_grid(domain, dist_search=10.0, ellps="sphere")

    assert outer["lon_min"] == pytest.approx(-1.089945902136039)
    assert outer["lon_max"] == pytest.approx(1.089945902136039)
    assert outer["lat_min"] == pytest.approx(-1.0899322029394807)
    assert outer["lat_max"] == pytest.approx(1.0899322029394807)


def test_domain_helpers_reject_invalid_domains():
    with pytest.raises(ValueError):
        hray.domain.planar_grid(
            {"x_min": 1.0, "x_max": 0.0, "y_min": 0.0, "y_max": 1.0}
        )

    with pytest.raises(ValueError):
        hray.domain.curved_grid(
            {"lon_min": 0.0, "lon_max": 1.0, "lat_min": 2.0, "lat_max": 1.0}
        )

    with pytest.raises(NotImplementedError):
        hray.domain.curved_grid(
            {"lon_min": 0.0, "lon_max": 1.0, "lat_min": 0.0, "lat_max": 1.0},
            ellps="unknown",
        )
