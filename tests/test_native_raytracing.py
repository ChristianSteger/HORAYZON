import numpy as np
import pytest

import horayzon as hray


def test_horizon_gridded_flat_dem_returns_near_zero_horizon(
    flat_planar_dem, flat_vectors, horizon_atol
):
    _, _, z, vert_grid = flat_planar_dem
    vec_norm, vec_north = flat_vectors((3, 3))

    hori, azim = hray.horizon.horizon_gridded(
        vert_grid,
        z.shape[0],
        z.shape[1],
        vec_norm,
        vec_north,
        offset_0=3,
        offset_1=3,
        dist_search=1.0,
        azim_num=8,
        hori_acc=1.0,
        ray_algorithm="binary_search",
        elev_ang_low_lim=-10.0,
    )

    assert hori.shape == (3, 3, 8)
    assert azim.shape == (8,)
    np.testing.assert_allclose(
        azim, np.arange(8, dtype=np.float32) * (2.0 * np.pi / 8.0), atol=1.0e-6
    )
    assert np.isfinite(hori).all()
    np.testing.assert_allclose(hori, 0.0, atol=horizon_atol)


def test_horizon_gridded_respects_mask_fill(flat_planar_dem, flat_vectors):
    _, _, z, vert_grid = flat_planar_dem
    vec_norm, vec_north = flat_vectors((2, 2))
    mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    hori_fill = np.float32(-9.0)

    hori, _ = hray.horizon.horizon_gridded(
        vert_grid,
        z.shape[0],
        z.shape[1],
        vec_norm,
        vec_north,
        offset_0=3,
        offset_1=3,
        dist_search=1.0,
        azim_num=8,
        hori_acc=1.0,
        ray_algorithm="binary_search",
        elev_ang_low_lim=-10.0,
        mask=mask,
        hori_fill=hori_fill,
    )

    np.testing.assert_array_equal(hori[0, 1], np.full(8, hori_fill))
    assert np.isfinite(hori[mask == 1]).all()
    assert not np.any(hori[mask == 1] == hori_fill)


def test_horizon_gridded_rejects_invalid_native_inputs(
    flat_planar_dem, flat_vectors
):
    _, _, z, vert_grid = flat_planar_dem
    vec_norm, vec_north = flat_vectors((3, 3))

    with pytest.raises(ValueError):
        hray.horizon.horizon_gridded(
            vert_grid,
            z.shape[0],
            z.shape[1],
            vec_norm[:, :, :2],
            vec_north[:, :, :2],
            offset_0=3,
            offset_1=3,
            dist_search=1.0,
        )

    with pytest.raises(ValueError):
        hray.horizon.horizon_gridded(
            vert_grid,
            z.shape[0],
            z.shape[1],
            vec_norm,
            vec_north,
            offset_0=-1,
            offset_1=3,
            dist_search=1.0,
        )

    with pytest.raises(ValueError):
        hray.horizon.horizon_gridded(
            vert_grid,
            z.shape[0],
            z.shape[1],
            vec_norm,
            vec_north,
            offset_0=3,
            offset_1=3,
            dist_search=0.0,
        )


def test_horizon_locations_rejects_invalid_native_inputs(flat_planar_dem):
    _, _, z, vert_grid = flat_planar_dem
    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    vec_norm = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    vec_north = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError):
        hray.horizon.horizon_locations(
            vert_grid,
            z.shape[0],
            z.shape[1],
            coords,
            vec_norm[:, :2],
            vec_north[:, :2],
            dist_search=1.0,
        )

    with pytest.raises(ValueError):
        hray.horizon.horizon_locations(
            vert_grid,
            z.shape[0],
            z.shape[1],
            coords[:0],
            vec_norm[:0],
            vec_north[:0],
            dist_search=1.0,
        )

    with pytest.raises(ValueError):
        hray.horizon.horizon_locations(
            vert_grid,
            z.shape[0],
            z.shape[1],
            coords,
            vec_norm,
            vec_north,
            dist_search=0.0,
        )


def test_flat_terrain_shadow_and_shortwave_correction(flat_vectors):
    coord = np.linspace(-100.0, 100.0, 5, dtype=np.float32)
    x, y = np.meshgrid(coord, coord)
    z = np.zeros_like(x, dtype=np.float32)
    vert_grid = hray.auxiliary.rearrange_pad_buffer(x, y, z)
    vec_norm, _ = flat_vectors((3, 3))
    vec_tilt = vec_norm.copy()
    surf_enl_fac = np.ones((3, 3), dtype=np.float32)
    elevation = np.zeros((3, 3), dtype=np.float32)
    mask = np.ones((3, 3), dtype=np.uint8)

    terrain = hray.shadow.Terrain()
    terrain.initialise(
        vert_grid,
        z.shape[0],
        z.shape[1],
        offset_0=1,
        offset_1=1,
        vec_tilt=vec_tilt,
        vec_norm=vec_norm,
        surf_enl_fac=surf_enl_fac,
        elevation=elevation,
        mask=mask,
        geom_type="grid",
        refrac_cor=False,
        ang_max=89.99,
    )

    sun_position = np.array([0.0, 0.0, 1000.0], dtype=np.float32)
    shadow = np.empty((3, 3), dtype=np.uint8)
    sw_dir_cor = np.empty((3, 3), dtype=np.float32)
    terrain.shadow(sun_position, shadow)
    terrain.sw_dir_cor(sun_position, sw_dir_cor)

    np.testing.assert_array_equal(shadow, np.zeros((3, 3), dtype=np.uint8))
    np.testing.assert_allclose(sw_dir_cor, 1.0, atol=1.0e-6)


def test_terrain_rejects_uninitialized_and_wrong_output_shape(flat_vectors):
    terrain = hray.shadow.Terrain()
    sun_position = np.array([0.0, 0.0, 1000.0], dtype=np.float32)

    with pytest.raises(RuntimeError):
        terrain.shadow(sun_position, np.empty((1, 1), dtype=np.uint8))

    coord = np.linspace(-100.0, 100.0, 5, dtype=np.float32)
    x, y = np.meshgrid(coord, coord)
    z = np.zeros_like(x, dtype=np.float32)
    vert_grid = hray.auxiliary.rearrange_pad_buffer(x, y, z)
    vec_norm, _ = flat_vectors((3, 3))

    terrain.initialise(
        vert_grid,
        z.shape[0],
        z.shape[1],
        offset_0=1,
        offset_1=1,
        vec_tilt=vec_norm.copy(),
        vec_norm=vec_norm,
        surf_enl_fac=np.ones((3, 3), dtype=np.float32),
        elevation=np.zeros((3, 3), dtype=np.float32),
        mask=np.ones((3, 3), dtype=np.uint8),
        geom_type="grid",
    )

    with pytest.raises(ValueError):
        terrain.shadow(sun_position, np.empty((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        terrain.sw_dir_cor(sun_position, np.empty((3, 2), dtype=np.float32))
