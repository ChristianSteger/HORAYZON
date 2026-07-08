import numpy as np
import pytest

import horayzon as hray


def test_slope_methods_return_vertical_normals_for_flat_plane():
    x_1d = np.arange(5, dtype=np.float32)
    y_1d = np.arange(5, dtype=np.float32)
    x, y = np.meshgrid(x_1d, y_1d)
    z = np.zeros_like(x, dtype=np.float32)

    for slope_func in (
        hray.topo_param.slope_plane_meth,
        hray.topo_param.slope_vector_meth,
    ):
        vec_tilt = slope_func(x, y, z)
        np.testing.assert_allclose(
            vec_tilt[2, 2], [0.0, 0.0, 1.0], atol=1.0e-6
        )
        assert np.isnan(vec_tilt[0, 0]).all()


def test_slope_methods_match_tilted_plane_normal():
    x_1d = np.arange(5, dtype=np.float32)
    y_1d = np.arange(5, dtype=np.float32)
    x, y = np.meshgrid(x_1d, y_1d)
    z = (0.1 * x).astype(np.float32)
    expected = np.array([-0.1, 0.0, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)

    for slope_func in (
        hray.topo_param.slope_plane_meth,
        hray.topo_param.slope_vector_meth,
    ):
        vec_tilt = slope_func(x, y, z)
        np.testing.assert_allclose(vec_tilt[2, 2], expected, atol=1.0e-6)


def test_slope_plane_method_leaves_singular_cells_as_nan():
    x = np.zeros((3, 3), dtype=np.float32)
    y = np.zeros((3, 3), dtype=np.float32)
    z = np.zeros((3, 3), dtype=np.float32)

    vec = hray.topo_param.slope_plane_meth(x, y, z)

    assert np.isnan(vec[1, 1]).all()


def test_unobstructed_flat_sky_metrics_are_analytic():
    azim = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    hori = np.zeros((2, 3, azim.size), dtype=np.float32)
    vec_tilt = np.zeros((2, 3, 3), dtype=np.float32)
    vec_tilt[..., 2] = 1.0

    svf = hray.topo_param.sky_view_factor(azim, hori, vec_tilt)
    vsf = hray.topo_param.visible_sky_fraction(azim, hori, vec_tilt)
    top = hray.topo_param.topographic_openness(azim, hori)

    np.testing.assert_allclose(svf, 1.0, atol=1.0e-6)
    np.testing.assert_allclose(vsf, 1.0, atol=1.0e-6)
    np.testing.assert_allclose(top, np.pi / 2.0, atol=1.0e-6)


def test_constant_horizon_metrics_match_closed_form_values():
    horizon_angle = np.float32(np.deg2rad(30.0))
    azim = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
    hori = np.full((1, 2, azim.size), horizon_angle, dtype=np.float32)
    vec_tilt = np.zeros((1, 2, 3), dtype=np.float32)
    vec_tilt[..., 2] = 1.0

    svf = hray.topo_param.sky_view_factor(azim, hori, vec_tilt)
    vsf = hray.topo_param.visible_sky_fraction(azim, hori, vec_tilt)
    top = hray.topo_param.topographic_openness(azim, hori)

    np.testing.assert_allclose(svf, np.cos(horizon_angle) ** 2, atol=1.0e-6)
    np.testing.assert_allclose(vsf, 1.0 - np.sin(horizon_angle), atol=1.0e-6)
    np.testing.assert_allclose(top, np.pi / 2.0 - horizon_angle, atol=1.0e-6)


def test_sky_metrics_reject_invalid_azimuth_and_tilt_vectors():
    azim = np.array([0.0], dtype=np.float32)
    hori = np.zeros((1, 1, 1), dtype=np.float32)
    vec_tilt = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)

    with pytest.raises(ValueError):
        hray.topo_param.sky_view_factor(azim, hori, vec_tilt)
    with pytest.raises(ValueError):
        hray.topo_param.visible_sky_fraction(azim, hori, vec_tilt)

    azim = np.array([0.0, np.pi], dtype=np.float32)
    hori = np.zeros((1, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        hray.topo_param.sky_view_factor(azim, hori, vec_tilt)

    with pytest.raises(ValueError):
        hray.topo_param.topographic_openness(
            np.array([], dtype=np.float32),
            np.zeros((1, 1, 0), dtype=np.float32),
        )
