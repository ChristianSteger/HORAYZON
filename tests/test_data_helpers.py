import numpy as np
import pytest
import xarray as xr

import horayzon as hray
import horayzon.ocean_masking as ocean_masking


def test_download_helpers_validate_local_path_and_mode(tmp_path):
    with pytest.raises(ValueError):
        hray.download.file("https://example.invalid/file.dat",
                           str(tmp_path / "missing") + "/")

    with pytest.raises(ValueError):
        hray.download.files(["https://example.invalid/file.dat"],
                            str(tmp_path) + "/", mode="invalid")


def test_geoid_undulation_validates_model_and_spatial_coverage():
    lon = np.array([0.0, 1.0], dtype=np.float64)
    lat = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(NotImplementedError):
        hray.geoid.undulation(lon, lat, geoid="unknown")

    with pytest.raises(ValueError):
        hray.geoid.undulation(np.array([181.0], dtype=np.float64),
                              lat, geoid="EGM96")


def test_load_dem_validation_helpers_and_preprocess(capsys):
    domain_geo = {"lon_min": 0.0, "lon_max": 1.0,
                  "lat_min": 0.0, "lat_max": 1.0}
    domain_planar = {"x_min": 0.0, "x_max": 1.0,
                     "y_min": 0.0, "y_max": 1.0}

    with pytest.raises(ValueError):
        hray.load_dem.srtm("missing.tif", domain_geo, engine="invalid")
    with pytest.raises(ValueError):
        hray.load_dem.dhm25("missing.asc", domain_planar, engine="invalid")
    with pytest.raises(ValueError):
        hray.load_dem.swissalti3d("/missing/", domain_planar,
                                  engine="invalid")
    with pytest.raises(ValueError):
        hray.load_dem.rema("missing.tif", domain_planar, engine="invalid")

    ds = xr.Dataset(coords={"lon": np.arange(4), "lat": np.arange(4)})
    trimmed = hray.load_dem.preprocess(ds)
    assert trimmed.sizes == {"lon": 4, "lat": 4}

    hray.load_dem.print_dem_info(np.array([[1.0, np.nan]], dtype=np.float32))
    output = capsys.readouterr().out
    assert "Size of loaded DEM domain: (1, 2)" in output
    assert "Warning: NaN values are present" in output


def test_ocean_masking_contours_and_distance_are_deterministic():
    lon = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    lat = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    mask_bin = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.uint8)

    contours = ocean_masking.coastline_contours(lon, lat, mask_bin)

    assert len(contours) == 1
    assert contours[0].shape == (5, 2)
    np.testing.assert_allclose(contours[0][0], [1.0, 0.5])

    x_ecef = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    y_ecef = np.zeros_like(x_ecef)
    z_ecef = np.zeros_like(x_ecef)
    mask_land = np.array([[True, False], [False, True]], dtype=bool)
    pts_ecef = np.array([[0.0, 0.0, 0.0],
                         [2.0, 0.0, 0.0]], dtype=np.float64)

    dist = ocean_masking.coastline_distance(x_ecef, y_ecef, z_ecef,
                                            mask_land, pts_ecef)

    np.testing.assert_allclose(dist[~mask_land], [1.0, 0.0])
    assert np.isnan(dist[mask_land]).all()


def test_ocean_masking_validates_inputs():
    lon = np.array([0.0, 1.0], dtype=np.float64)
    lat = np.array([0.0, 1.0], dtype=np.float64)
    mask_bad = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        ocean_masking.coastline_contours(lon, lat, mask_bad)

    x_ecef = np.zeros((2, 2), dtype=np.float64)
    mask_land = np.zeros((2, 2), dtype=np.uint8)
    pts_ecef = np.zeros((1, 3), dtype=np.float64)

    with pytest.raises(ValueError):
        ocean_masking.coastline_distance(x_ecef, x_ecef, x_ecef,
                                         mask_land, pts_ecef)
