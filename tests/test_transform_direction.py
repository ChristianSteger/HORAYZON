import numpy as np
import pytest

import horayzon as hray


def test_lonlat2ecef_sphere_cardinal_points():
    lon = np.array([0.0, 90.0, 0.0], dtype=np.float64)
    lat = np.array([0.0, 0.0, 90.0], dtype=np.float64)
    height = np.zeros(3, dtype=np.float32)

    x, y, z = hray.transform.lonlat2ecef(lon, lat, height, ellps="sphere")

    radius = 6370997.0
    np.testing.assert_allclose(x, [radius, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(y, [0.0, radius, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(z, [0.0, 0.0, radius], atol=1.0e-6)


def test_ecef2enu_origin_and_vector_orientation():
    lon = np.array([0.0], dtype=np.float64)
    lat = np.array([0.0], dtype=np.float64)
    height = np.array([0.0], dtype=np.float32)
    x, y, z = hray.transform.lonlat2ecef(lon, lat, height, ellps="sphere")
    transformer = hray.transform.TransformerEcef2enu(0.0, 0.0, "sphere")

    x_enu, y_enu, z_enu = hray.transform.ecef2enu(x, y, z, transformer)
    vec_enu = hray.transform.ecef2enu_vector(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32), transformer)

    np.testing.assert_allclose(x_enu, [0.0], atol=1.0e-6)
    np.testing.assert_allclose(y_enu, [0.0], atol=1.0e-6)
    np.testing.assert_allclose(z_enu, [0.0], atol=1.0e-6)
    np.testing.assert_allclose(vec_enu, [[0.0, 0.0, 1.0]], atol=1.0e-6)


def test_surface_normal_and_north_direction_at_equator():
    lon = np.array([0.0], dtype=np.float64)
    lat = np.array([0.0], dtype=np.float64)
    height = np.array([0.0], dtype=np.float32)
    x, y, z = hray.transform.lonlat2ecef(lon, lat, height, ellps="sphere")
    vec_norm = hray.direction.surf_norm(lon, lat)

    vec_north = hray.direction.north_dir(x, y, z, vec_norm, ellps="sphere")

    np.testing.assert_allclose(vec_norm, [[1.0, 0.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(vec_north, [[0.0, 0.0, 1.0]], atol=1.0e-6)


def test_swiss_projection_round_trip_is_stable_for_reference_location():
    lon = np.array([7.4386], dtype=np.float64)
    lat = np.array([46.9511], dtype=np.float64)
    h_wgs = np.array([550.0], dtype=np.float32)

    east, north, h_ch = hray.transform.wgs2swiss(lon, lat, h_wgs)
    lon_rt, lat_rt, h_wgs_rt = hray.transform.swiss2wgs(east, north, h_ch)

    np.testing.assert_allclose(lon_rt, lon, atol=5.0e-6)
    np.testing.assert_allclose(lat_rt, lat, atol=5.0e-6)
    np.testing.assert_allclose(h_wgs_rt, h_wgs, atol=1.0e-2)


def test_transform_functions_validate_dtype():
    lon = np.array([0.0], dtype=np.float32)
    lat = np.array([0.0], dtype=np.float64)
    height = np.array([0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        hray.transform.lonlat2ecef(lon, lat, height, ellps="sphere")
