"""Species fit: a lookup table, not a model. Matches native trees to soil drainage and pH."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

DRAINAGE_NAMES = {
    1: "excessively drained",
    2: "somewhat excessively drained",
    3: "well drained",
    4: "moderately well drained",
    5: "somewhat poorly drained",
    6: "poorly drained",
    7: "very poorly drained",
}


def species_table() -> pd.DataFrame:
    with (
        resources.files("landscore.report.data").joinpath("species.csv").open(encoding="utf-8") as f
    ):
        return pd.read_csv(f)


def species_fit(
    drainage: np.ndarray, ph: np.ndarray, min_fraction: float = 0.5, ph_slack: float = 0.3
) -> pd.DataFrame:
    """Rank species by the fraction of plot pixels whose drainage and pH they tolerate.

    drainage: ordinal drainage class per pixel (1-7, NaN unknown); ph: pH per pixel (NaN unknown).
    Unknown pH is treated as compatible; unknown drainage pixels are ignored.
    """
    d = np.asarray(drainage, float).ravel()
    p = np.asarray(ph, float).ravel()
    keep = ~np.isnan(d)
    d, p = d[keep], p[keep]
    t = species_table()
    if d.size == 0:
        return t.assign(fit_fraction=np.nan).iloc[0:0]
    fr = []
    for _, s in t.iterrows():
        ok_d = (d >= s.drainage_min) & (d <= s.drainage_max)
        ok_p = np.isnan(p) | ((p >= s.ph_min - ph_slack) & (p <= s.ph_max + ph_slack))
        fr.append(float((ok_d & ok_p).mean()))
    t["fit_fraction"] = fr
    t = t[t.fit_fraction >= min_fraction].sort_values("fit_fraction", ascending=False)
    return t.reset_index(drop=True)
