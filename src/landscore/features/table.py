"""`landscore features`: static feature rasters + the sampled training table (Parquet)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from landscore.config import label_config, load_config
from landscore.data import layers as L
from landscore.features.landscape import landscape_layers
from landscore.features.registry import FEATURE_NAMES, LANDSCAPE_OFFSET, check_table
from landscore.features.terrain import _fft_convolve, disk_kernel, terrain_features
from landscore.labels.abandonment import Status

LANDSCAPE_NAMES = ("dist_forest_m", "forest_pct_100m", "forest_pct_500m")


def build_static(county: str) -> dict[str, np.ndarray]:
    """Soil + terrain + climate on the county grid -> derived/<County>/static.tif."""
    raw = L.raw_dir(county)
    g = L.grid(county)
    layers: dict[str, np.ndarray] = {}
    for name in ("soil.tif", "climate.tif"):
        if (raw / name).exists():
            layers |= L.read_named(raw / name)
        else:
            print(f"warning: {name} missing for {county}; those features will be NaN")
    dem = L.read_named(raw / "dem.tif")["elevation"]
    layers |= terrain_features(dem, g.res)
    layers = {k: v for k, v in layers.items() if k in FEATURE_NAMES}
    L.write_named(L.derived_dir(county) / "static.tif", g, layers)
    return layers


def build_landscape_now(county: str) -> None:
    """Landscape features from the newest land cover year (used by the plot report)."""
    raw = L.raw_dir(county)
    years = L.year_bands(raw / "landcover.tif")
    with rasterio.open(raw / "landcover.tif") as src:
        lc = src.read(len(years))
    lay = landscape_layers(lc, label_config().forest_classes, L.grid(county).res)
    forest = np.isin(lc, label_config().forest_classes).astype(np.float64)
    k = disk_kernel(1000 / L.grid(county).res)
    lay["forest_pct_1km"] = (100 * _fft_convolve(forest, k) / k.sum()).astype(np.float32)
    lay["landcover_now"] = lc.astype(np.float32)
    L.write_named(L.derived_dir(county) / "landscape_now.tif", L.grid(county), lay)


def build_table(county: str, static: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    cfg = load_config()
    lcfg = label_config()
    raw, out = L.raw_dir(county), L.derived_dir(county)
    g = L.grid(county)
    static = static if static is not None else L.read_named(out / "static.tif")
    lab = L.read_named(out / "labels.tif")
    rows, cols = np.nonzero(lab["status"] == Status.LABELED)
    n_max = cfg["sampling"]["n_pixels"]
    rng = np.random.default_rng(cfg["sampling"]["seed"])
    if len(rows) > n_max:
        pick = np.sort(rng.choice(len(rows), n_max, replace=False))
        rows, cols = rows[pick], cols[pick]
    a_year = lab["abandon_year"][rows, cols].astype(int)
    xs, ys = rasterio.transform.xy(g.affine, rows, cols)
    df = pd.DataFrame(
        {
            "county": county,
            "row": rows,
            "col": cols,
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "abandon_year": a_year,
            "landscape_year": a_year + LANDSCAPE_OFFSET,
            "canopy_start": lab["canopy_start"][rows, cols],
            "canopy_outcome": lab["canopy_outcome"][rows, cols],
            "label_binary": lab["label_binary"][rows, cols].astype(int),
        }
    )
    # Diagnostic only (not a feature): size of the connected patch of labeled pixels.
    # Single pixels and thin strips along a field edge are more likely mapping noise.
    patches, _ = ndimage.label(lab["status"] == Status.LABELED, structure=np.ones((3, 3)))
    sizes = np.bincount(patches.ravel())
    df["patch_px"] = sizes[patches[rows, cols]]
    for name, arr in static.items():
        df[name] = arr[rows, cols]
    # Landscape features: one land cover map per start year, A-1.
    lc_years = L.year_bands(raw / "landcover.tif")
    for n in (*LANDSCAPE_NAMES, "was_pasture"):
        df[n] = np.nan
    with rasterio.open(raw / "landcover.tif") as src:
        for y in sorted(df["landscape_year"].unique()):
            lc = src.read(lc_years.index(int(y)) + 1)
            lay = landscape_layers(lc, lcfg.forest_classes, g.res)
            sel = np.flatnonzero(df["landscape_year"].to_numpy() == y)
            r, c = rows[sel], cols[sel]
            for n in LANDSCAPE_NAMES:
                df.loc[sel, n] = lay[n][r, c]
            df.loc[sel, "was_pasture"] = (lc[r, c] == 81).astype(float)
    check_table(df)
    df.to_parquet(out / "training.parquet", index=False)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landscore features")
    ap.add_argument("--county", nargs="+", default=load_config()["study_area"]["pilot"])
    a = ap.parse_args(argv)
    for c in a.county:
        static = build_static(c)
        build_landscape_now(c)
        df = build_table(c, static)
        print(f"{c}: {len(df):,} rows, positive rate {df.label_binary.mean():.1%}")
        print(
            df[[f for f in FEATURE_NAMES if f in df]].describe().T[["mean", "min", "max"]].round(2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
