import numpy as np
import pandas as pd

from landscore.data import soil


def test_codes():
    assert soil.drainage_code("Well drained") == 3
    assert soil.drainage_code("Very poorly drained") == 7
    assert np.isnan(soil.drainage_code(None))
    assert soil.hydgrp_code("B/D") == 4 and soil.hydgrp_code("A") == 1
    assert np.isnan(soil.hydgrp_code(""))


def test_horizon_weighted_top_30cm():
    h = pd.DataFrame(
        {
            "cokey": [1, 1, 1],
            "hzdept_r": [0, 20, 50],
            "hzdepb_r": [20, 50, 100],
            "ph": [5.0, 6.5, 8.0],  # 20 cm of 5.0 + 10 cm of 6.5 inside 0-30 cm
        }
    )
    w = soil.horizon_weighted(h, ["ph"])
    assert np.isclose(w.loc[1, "ph"], (20 * 5.0 + 10 * 6.5) / 30)


def test_horizon_weighted_skips_missing():
    h = pd.DataFrame(
        {"cokey": [1, 1], "hzdept_r": [0, 10], "hzdepb_r": [10, 30], "ph": [np.nan, 6.0]}
    )
    assert soil.horizon_weighted(h, ["ph"]).loc[1, "ph"] == 6.0


def test_dominant_component():
    df = pd.DataFrame(
        {"mukey": [1, 1, 2], "cokey": [10, 11, 20], "comppct_r": [30, 60, 100], "v": [1, 2, 3]}
    )
    assert list(soil.dominant(df)["v"]) == [2, 3]


def test_rasterize_lookup():
    t = pd.DataFrame({"drainage_class": [3.0, 6.0]}, index=pd.Index([100, 300], name="mukey"))
    mukey = np.array([[100, 300], [0, 200]], dtype=np.int32)
    r = soil.rasterize(mukey, t)["drainage_class"]
    assert r[0, 0] == 3 and r[0, 1] == 6 and np.isnan(r[1, 0]) and np.isnan(r[1, 1])
