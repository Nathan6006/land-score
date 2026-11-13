import numpy as np

from landscore.features.landscape import landscape_layers
from landscore.features.terrain import flow_accumulation, slope_aspect, terrain_features, tpi


def test_slope_of_tilted_plane():
    cell = 30.0
    rows, cols = np.indices((40, 40))
    dem = (cols * cell * 0.1).astype(float)  # rises 10% to the east
    slope, aspect = slope_aspect(dem, cell)
    assert np.allclose(slope[5:-5, 5:-5], np.degrees(np.arctan(0.1)), atol=1e-4)
    # Downslope faces west (3*pi/2).
    assert np.allclose(aspect[5:-5, 5:-5], 1.5 * np.pi, atol=1e-6)


def test_aspect_north_facing():
    rows, cols = np.indices((30, 30))
    dem = rows * 3.0  # higher to the south (row index grows southwards) -> faces north
    _, aspect = slope_aspect(dem.astype(float), 30.0)
    assert np.allclose(np.cos(aspect[5:-5, 5:-5]), 1.0, atol=1e-6)


def test_flow_accumulates_downhill():
    rows, cols = np.indices((20, 20))
    dem = (100 - cols).astype(float)  # everything drains east
    acc = flow_accumulation(dem)
    assert acc[10, 0] == 1
    assert acc[10, -1] == 20


def test_tpi_valley_negative():
    rows, cols = np.indices((61, 61))
    dem = np.abs(cols - 30).astype(float) * 2  # V-shaped valley along the middle column
    t = tpi(dem, 30.0, 300)
    assert t[30, 30] < 0 and t[30, 5] > 0


def test_terrain_features_handles_nan():
    dem = np.random.default_rng(0).normal(300, 2, (50, 50))
    dem[:5, :5] = np.nan
    f = terrain_features(dem, 30.0)
    assert set(f) >= {"slope_deg", "twi", "tpi_300", "tpi_1000", "northness"}
    assert np.isnan(f["twi"][0, 0]) and np.isfinite(f["twi"][25, 25])


def test_landscape_distance_and_percent():
    lc = np.full((50, 50), 82, dtype=np.uint8)
    lc[:, :10] = 41  # forest strip on the west
    out = landscape_layers(lc, (41, 42, 43), 30.0)
    assert out["dist_forest_m"][25, 10] == 30
    assert out["dist_forest_m"][25, 20] == 330
    assert out["forest_pct_100m"][25, 45] == 0
    assert out["forest_pct_100m"][25, 2] == 100
    assert 0 < out["forest_pct_500m"][25, 20] < 100
