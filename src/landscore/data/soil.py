"""Soil properties per map unit from USDA Soil Data Access (SDA), and rasterizing them.

One row per map unit key (mukey). Component-level values use the dominant component (the one
with the largest share of the map unit). Horizon values are thickness-weighted over 0-30 cm.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from landscore.data import sources as S

DRAINAGE = {
    "excessively drained": 1,
    "somewhat excessively drained": 2,
    "well drained": 3,
    "moderately well drained": 4,
    "somewhat poorly drained": 5,
    "poorly drained": 6,
    "very poorly drained": 7,
}
HYDGRP = {"A": 1, "B": 2, "C": 3, "D": 4}
SOIL_FEATURES = (
    "drainage_class",
    "aws_0_100",
    "wt_depth_min",
    "ph_0_30",
    "sand_0_30",
    "clay_0_30",
    "om_0_30",
    "hydgrp",
    "seedling_mortality",
    "site_index",
)


def sda_query(sql: str, tries: int = 4) -> pd.DataFrame:
    """Run one SQL query against SDA (anonymous) and return a DataFrame of strings."""
    for i in range(tries):
        try:
            r = requests.post(
                S.SDA_URL, json={"query": sql, "format": "JSON+COLUMNNAME"}, timeout=180
            )
            r.raise_for_status()
            rows = r.json().get("Table", []) if r.text.strip() else []
            return pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise AssertionError("unreachable")


def _chunks(keys, n=400):
    keys = [int(k) for k in keys]
    for i in range(0, len(keys), n):
        yield ",".join(map(str, keys[i : i + n]))


def _query_all(template: str, keys) -> pd.DataFrame:
    parts = [sda_query(template.format(keys=c)) for c in _chunks(keys)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def drainage_code(name) -> float:
    return DRAINAGE.get(str(name).strip().lower(), np.nan) if name is not None else np.nan


def hydgrp_code(name) -> float:
    """A=1 .. D=4. Dual groups such as 'B/D' use the undrained (second) letter."""
    if name is None or not str(name).strip():
        return np.nan
    return HYDGRP.get(str(name).strip().split("/")[-1].upper(), np.nan)


def horizon_weighted(
    h: pd.DataFrame, cols: list[str], top: float = 0, bottom: float = 30
) -> pd.DataFrame:
    """Thickness-weighted mean of horizon columns over a depth slice, per cokey."""
    h = h.copy()
    h["thick"] = (h["hzdepb_r"].clip(upper=bottom) - h["hzdept_r"].clip(lower=top)).clip(lower=0)
    out = {}
    for c in cols:
        w = h["thick"].where(h[c].notna(), 0)
        num = (h[c].fillna(0) * w).groupby(h["cokey"]).sum()
        den = w.groupby(h["cokey"]).sum()
        out[c] = num / den.where(den > 0)
    return pd.DataFrame(out)


def dominant(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows of the largest component per mukey (ties: lowest cokey)."""
    top = (
        df[["mukey", "cokey", "comppct_r"]]
        .drop_duplicates()
        .sort_values(["mukey", "comppct_r", "cokey"], ascending=[True, False, True])
        .drop_duplicates("mukey")
    )
    return df[df["cokey"].isin(top["cokey"])]


def mapunit_table(mukeys) -> pd.DataFrame:
    """Query SDA and return one row per mukey with the columns in SOIL_FEATURES."""
    out = pd.DataFrame(index=pd.Index(sorted(int(k) for k in mukeys), name="mukey"))

    agg = _query_all(
        "SELECT mukey, drclassdcd, aws0100wta, wtdepannmin, hydgrpdcd FROM muaggatt WHERE mukey IN ({keys})",
        out.index,
    )
    if len(agg):
        agg["mukey"] = agg["mukey"].astype(int)
        agg = agg.set_index("mukey")
        out["drainage_class"] = agg["drclassdcd"].map(drainage_code)
        out["aws_0_100"] = pd.to_numeric(agg["aws0100wta"])
        # No water table recorded within the survey depth: treat as deep (200 cm).
        out["wt_depth_min"] = pd.to_numeric(agg["wtdepannmin"]).fillna(200.0).reindex(out.index)
        out["hydgrp"] = agg["hydgrpdcd"].map(hydgrp_code)

    hz = _query_all(
        "SELECT c.mukey, c.cokey, c.comppct_r, h.hzdept_r, h.hzdepb_r, h.ph1to1h2o_r, "
        "h.sandtotal_r, h.claytotal_r, h.om_r FROM component c JOIN chorizon h ON h.cokey = c.cokey "
        "WHERE c.mukey IN ({keys}) AND h.hzdept_r < 30",
        out.index,
    )
    if len(hz):
        hz = hz.apply(pd.to_numeric)
        hz = dominant(hz)
        w = horizon_weighted(hz, ["ph1to1h2o_r", "sandtotal_r", "claytotal_r", "om_r"])
        w["mukey"] = hz.drop_duplicates("cokey").set_index("cokey")["mukey"]
        w = w.set_index("mukey")
        for src, dst in (
            ("ph1to1h2o_r", "ph_0_30"),
            ("sandtotal_r", "sand_0_30"),
            ("claytotal_r", "clay_0_30"),
            ("om_r", "om_0_30"),
        ):
            out[dst] = w[src]

    sm = _query_all(
        "SELECT c.mukey, c.cokey, c.comppct_r, ci.interphr FROM component c "
        "JOIN cointerp ci ON ci.cokey = c.cokey WHERE c.mukey IN ({keys}) "
        f"AND ci.mrulename = '{S.SEEDLING_MORTALITY_RULE}' AND ci.ruledepth = 0 "
        "AND ci.interphr IS NOT NULL",
        out.index,
    )
    if len(sm):
        sm = dominant(sm.apply(pd.to_numeric))
        out["seedling_mortality"] = sm.groupby("mukey")["interphr"].first()

    si = _query_all(
        "SELECT c.mukey, c.cokey, c.comppct_r, f.siteindex_r FROM component c "
        "JOIN coforprod f ON f.cokey = c.cokey WHERE c.mukey IN ({keys}) AND f.siteindex_r IS NOT NULL",
        out.index,
    )
    if len(si):
        si = dominant(si.apply(pd.to_numeric))
        out["site_index"] = si.groupby("mukey")["siteindex_r"].max()

    for c in SOIL_FEATURES:
        if c not in out:
            out[c] = np.nan
    return out[list(SOIL_FEATURES)].astype("float32")


def rasterize(mukey: np.ndarray, table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map each pixel's mukey to its table values. Unknown keys and 0 become NaN."""
    keys = table.index.to_numpy()
    pos = np.searchsorted(keys, mukey)
    pos = np.clip(pos, 0, len(keys) - 1)
    hit = keys[pos] == mukey
    out = {}
    for c in table.columns:
        vals = table[c].to_numpy(np.float32)
        out[c] = np.where(hit, vals[pos], np.nan).astype(np.float32)
    return out
