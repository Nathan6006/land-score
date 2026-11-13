import math

from landscore.data.grid import NLCD_ORIGIN_X, NLCD_ORIGIN_Y, Grid, snap_grid


def test_snap_grid_aligned_and_covering():
    b = (1_250_017.3, 2_190_011.9, 1_282_444.0, 2_236_999.5)
    g = snap_grid(b)
    xmin, ymin, xmax, ymax = g.bounds
    assert xmin <= b[0] and ymin <= b[1] and xmax >= b[2] and ymax >= b[3]
    for x in (xmin, xmax):
        assert math.isclose(((x - NLCD_ORIGIN_X) / 30) % 1, 0, abs_tol=1e-9)
    for y in (ymin, ymax):
        assert math.isclose(((NLCD_ORIGIN_Y - y) / 30) % 1, 0, abs_tol=1e-9)
    assert xmax - xmin < b[2] - b[0] + 60


def test_grid_roundtrip(tmp_path):
    g = snap_grid((0, 0, 3000, 3000))
    g.save(tmp_path / "g.json")
    assert Grid.load(tmp_path / "g.json") == g
    assert g.profile("uint8")["width"] == g.width
