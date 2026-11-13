"""`landscore report plot.geojson`: score an offered plot and write a short report.

No LLM calls here (rule 1): the score comes from the saved trained model only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

from landscore import SITE_POTENTIAL_CAVEAT
from landscore.config import data_dir, load_config, outputs_dir
from landscore.data import layers as L
from landscore.report import flags as F
from landscore.report.species import DRAINAGE_NAMES, species_fit


def _pick_county(geom) -> str:
    for county in load_config()["study_area"]["counties"]:
        root = data_dir()
        if not (root / "raw" / county / "grid.json").exists():
            continue
        if not (root / "derived" / county / "static.tif").exists():
            continue
        g = L.grid(county)
        xmin, ymin, xmax, ymax = g.bounds
        b = geom.bounds
        if b[0] >= xmin and b[1] >= ymin and b[2] <= xmax and b[3] <= ymax:
            return county
    raise SystemExit(
        "The plot is outside every county that has been built. Fetch and build the county "
        "first (see docs/SETUP.md)."
    )


def score_plot(geojson: Path, model_path: Path | None = None) -> dict:
    plot = gpd.read_file(geojson).to_crs("EPSG:5070")
    geom = plot.union_all()
    county = _pick_county(geom)
    der = L.derived_dir(county)
    with rasterio.open(der / "static.tif") as src:
        win = from_bounds(*geom.bounds, transform=src.transform).round_offsets().round_lengths()
        win = rasterio.windows.Window(
            win.col_off, win.row_off, max(win.width, 1), max(win.height, 1)
        )
        transform = src.window_transform(win)
    static = L.read_named(der / "static.tif", window=win)
    now = L.read_named(der / "landscape_now.tif", window=win)
    shape = next(iter(static.values())).shape
    inside = ~geometry_mask([geom], out_shape=shape, transform=transform, all_touched=False)
    if not inside.any():  # very small plot: use pixels the polygon touches
        inside = ~geometry_mask([geom], out_shape=shape, transform=transform, all_touched=True)

    bundle = joblib.load(model_path or outputs_dir() / "models" / "model.joblib")
    feats = {**static, **now}
    feats["was_pasture"] = (now["landcover_now"] == 81).astype(np.float32)
    X = np.stack(
        [
            feats[f][inside] if f in feats else np.full(inside.sum(), np.nan)
            for f in bundle["features"]
        ],
        1,
    )
    p = bundle["model"].predict_proba(X)[:, 1] * 100
    score = np.full(shape, np.nan, np.float32)
    score[inside] = p

    drain = static.get("drainage_class", np.full(shape, np.nan))[inside]
    ph = static.get("ph_0_30", np.full(shape, np.nan))[inside]
    flags = [
        F.wet_soil(drain, static.get("wt_depth_min", np.full(shape, np.nan))[inside]),
        F.competing_vegetation(now["landcover_now"][inside].astype(int)),
        F.deer_pressure(float(np.nanmean(now["forest_pct_1km"][inside]))),
        F.steep_slope(static["slope_deg"][inside]),
    ]
    sp = species_fit(drain, ph)
    area_ha = geom.area / 1e4
    return {
        "plot": Path(geojson).stem,
        "county": county,
        "area_ha": round(area_ha, 2),
        "n_pixels": int(inside.sum()),
        "model": bundle["name"],
        "score_mean": round(float(np.mean(p)), 1),
        "score_p10": round(float(np.percentile(p, 10)), 1),
        "score_p90": round(float(np.percentile(p, 90)), 1),
        "share_score_ge_50": round(float((p >= 50).mean()), 3),
        "dominant_drainage": DRAINAGE_NAMES.get(int(np.nanmedian(drain)), "unknown")
        if np.isfinite(drain).any()
        else "unknown",
        "median_ph": round(float(np.nanmedian(ph)), 1) if np.isfinite(ph).any() else None,
        "flags": [f.__dict__ for f in flags],
        "species": sp[["common_name", "scientific_name", "fit_fraction", "notes"]]
        .head(8)
        .to_dict("records"),
        "caveat": SITE_POTENTIAL_CAVEAT,
        "_score": score,
        "_geom": geom,
        "_transform": transform,
    }


def _map(r: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = r["_score"]
    t = r["_transform"]
    ext = (t.c, t.c + t.a * s.shape[1], t.f + t.e * s.shape[0], t.f)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(s, extent=ext, cmap="RdYlGn", vmin=0, vmax=100, interpolation="nearest")
    gpd.GeoSeries([r["_geom"]]).boundary.plot(ax=ax, color="black", lw=1)
    ax.set_title(f"{r['plot']}: establishment score")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.75, label="score (0-100)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_report(r: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _map(r, out_dir / "score_map.png")
    clean = {k: v for k, v in r.items() if not k.startswith("_")}
    (out_dir / "report.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    flags = "\n".join(
        f"- **{f['name'].replace('_', ' ')}**: {'FLAGGED' if f['raised'] else 'not flagged'}. {f['detail']}"
        for f in r["flags"]
    )
    species = (
        "\n".join(
            f"| {s['common_name']} | *{s['scientific_name']}* | {s['fit_fraction']:.0%} | {s['notes']} |"
            for s in r["species"]
        )
        or "| (no match) | | | |"
    )
    md = f"""# Land Score report: {r["plot"]}

> **{r["caveat"]}**

**County:** {r["county"]}  |  **Area:** {r["area_ha"]} ha ({r["n_pixels"]} pixels of 30 m)

## Establishment score

**{r["score_mean"]:.0f} / 100** (plot average; middle 80% of pixels {r["score_p10"]:.0f}-{r["score_p90"]:.0f}).
{r["share_score_ge_50"]:.0%} of the plot scores 50 or more.

The score is the model's estimated chance (x100) that former farmland with these soil,
terrain, climate, and surrounding-forest conditions reached the canopy target on its own.
Model: `{r["model"]}`. See docs/RESULTS.md for how well this holds up on counties the model
never saw.

![score map](score_map.png)

## Species fit (lookup table, draft pending forester review)

Soil: mostly {r["dominant_drainage"]}, median pH {r["median_ph"]}.

| Species | Scientific name | Share of plot suited | Notes |
|---|---|---|---|
{species}

## Planning flags

{flags}

---
{r["caveat"]}
"""
    path = out_dir / "report.md"
    path.write_text(md, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landscore report")
    ap.add_argument("geojson", nargs="+", type=Path)
    ap.add_argument("--model", type=Path)
    a = ap.parse_args(argv)
    for gj in a.geojson:
        r = score_plot(gj, a.model)
        path = write_report(r, outputs_dir() / "reports" / gj.stem)
        print(f"{gj.stem}: score {r['score_mean']:.0f}/100 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
