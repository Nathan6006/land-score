import numpy as np
import pytest

from landscore.config import label_config
from landscore.labels.abandonment import Status, build_labels, find_abandonment_year, summarize

LC_YEARS = np.arange(1985, 2025)
CAN_YEARS = np.arange(1985, 2024)
FOREST, GRASS, PASTURE, CROP, DEV, WATER = 41, 71, 81, 82, 22, 11


@pytest.fixture
def cfg():
    return label_config()


def series(spec: dict[tuple[int, int], int], default=GRASS) -> np.ndarray:
    """Build one pixel's land cover series from {(first_year, last_year): class}."""
    s = np.full(len(LC_YEARS), default, dtype=np.uint8)
    for (a, b), c in spec.items():
        s[(LC_YEARS >= a) & (LC_YEARS <= b)] = c
    return s


def canopy(start=0, end=60, rise_from=1995, rise_to=2010) -> np.ndarray:
    c = np.full(len(CAN_YEARS), start, dtype=np.float32)
    ramp = (CAN_YEARS >= rise_from) & (CAN_YEARS <= rise_to)
    c[ramp] = np.linspace(start, end, ramp.sum())
    c[CAN_YEARS > rise_to] = end
    return c.astype(np.uint8)


def run(lcs, cans, cfg):
    return build_labels(np.stack(lcs, 1), LC_YEARS, np.stack(cans, 1), CAN_YEARS, cfg)


def test_abandonment_year_simple(cfg):
    s = series({(1985, 1994): PASTURE})  # farmed through 1994, grass from 1995
    assert find_abandonment_year(s[:, None], LC_YEARS, cfg)[0] == 1995


def test_short_fallow_is_rotation_not_abandonment(cfg):
    # Two years of grass then farmed again: rotation. Real abandonment later in 2000.
    s = series({(1985, 1994): CROP, (1997, 1999): CROP})
    assert find_abandonment_year(s[:, None], LC_YEARS, cfg)[0] == 2000


def test_needs_enough_farmed_years_before(cfg):
    s = series({(1993, 1994): CROP})  # only 2 farmed years
    assert find_abandonment_year(s[:, None], LC_YEARS, cfg)[0] == 0


def test_outside_window_not_counted(cfg):
    s = series({(1985, 2010): CROP})  # stopped in 2011, after the 2005 window end
    assert find_abandonment_year(s[:, None], LC_YEARS, cfg)[0] == 0


def test_labeled_positive_and_negative(cfg):
    lc = series({(1985, 1994): PASTURE})
    r = run([lc, lc], [canopy(end=70), canopy(end=5)], cfg)
    assert list(r.status) == [Status.LABELED, Status.LABELED]
    assert list(r.label_binary) == [1, 0]
    assert r.canopy_outcome[0] == 70  # outcome year 1995 + 15 = 2010
    assert r.abandon_year[0] == 1995


def test_outcome_measured_at_lag(cfg):
    lc = series({(1985, 1999): CROP})  # A = 2000, outcome 2015
    c = np.zeros(len(CAN_YEARS), np.uint8)
    c[CAN_YEARS == 2015] = 44
    c[CAN_YEARS == 2016] = 99
    r = run([lc], [c], cfg)
    assert r.canopy_outcome[0] == 44


def test_converted_and_refarmed_dropped(cfg):
    dev = series({(1985, 1994): CROP, (2003, 2024): DEV})
    water = series({(1985, 1994): CROP, (2000, 2024): WATER})
    refarm = series({(1985, 1994): CROP, (2002, 2006): PASTURE})
    later_dev = series({(1985, 1994): CROP, (2015, 2024): DEV})  # after outcome year 2010: ok
    r = run([dev, water, refarm, later_dev], [canopy()] * 4, cfg)
    assert list(r.status) == [Status.CONVERTED, Status.CONVERTED, Status.REFARMED, Status.LABELED]


def test_never_farmed_and_wooded_start(cfg):
    forest = series({}, default=FOREST)
    lc = series({(1985, 1994): PASTURE})
    r = run([forest, lc], [canopy(start=80, end=80), canopy(start=40)], cfg)
    assert list(r.status) == [Status.NEVER_FARMED, Status.WOODED_AT_START]


def test_canopy_nodata(cfg):
    lc = series({(1985, 1994): PASTURE})
    c = canopy()
    c[CAN_YEARS == 2010] = 255
    r = run([lc], [c], cfg)
    assert r.status[0] == Status.NO_CANOPY_DATA
    assert r.label_binary[0] == -1


def test_landcover_nodata(cfg):
    r = run([np.full(len(LC_YEARS), 250, np.uint8)], [canopy()], cfg)
    assert r.status[0] == Status.NODATA


def test_summary_counts(cfg):
    lc = series({(1985, 1994): PASTURE})
    r = run([lc, lc, series({}, default=FOREST)], [canopy(end=70), canopy(end=0), canopy()], cfg)
    s = summarize(r)
    assert (
        s["labeled"] == 2 and s["positive"] == 1 and s["negative"] == 1 and s["never_farmed"] == 1
    )


def test_year_axis_must_be_consecutive(cfg):
    with pytest.raises(ValueError):
        find_abandonment_year(np.zeros((3, 1), np.uint8), [1990, 1991, 1993], cfg)
