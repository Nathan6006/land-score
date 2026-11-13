"""`landscore labels`: build label rasters and counts for counties, window by window."""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
from rasterio.windows import Window

from landscore.config import label_config, load_config, outputs_dir
from landscore.data import layers as L
from landscore.labels.abandonment import Status, build_labels, summarize

TILE = 512


def build_county(county: str) -> dict:
    cfg = label_config()
    raw, out = L.raw_dir(county), L.derived_dir(county)
    g = L.grid(county)
    inside = L.county_mask(county, g)
    lc_years = np.array(L.year_bands(raw / "landcover.tif"))
    can_years = np.array(L.year_bands(raw / "canopy.tif"))
    prof = g.profile("float32", count=len(L.LABEL_BANDS), nodata=np.nan)
    totals: dict[str, int] = {}
    by_year: dict[int, int] = {}
    with (
        rasterio.open(raw / "landcover.tif") as lc_src,
        rasterio.open(raw / "canopy.tif") as can_src,
        rasterio.open(out / "labels.tif", "w", **prof) as dst,
    ):
        for i, n in enumerate(L.LABEL_BANDS, start=1):
            dst.set_band_description(i, n)
        for r0 in range(0, g.height, TILE):
            for c0 in range(0, g.width, TILE):
                w = Window(c0, r0, min(TILE, g.width - c0), min(TILE, g.height - r0))
                lc = lc_src.read(window=w).reshape(len(lc_years), -1)
                can = can_src.read(window=w).reshape(len(can_years), -1)
                res = build_labels(lc, lc_years, can, can_years, cfg)
                m = inside[r0 : r0 + w.height, c0 : c0 + w.width].ravel()
                res.status[~m] = Status.NODATA
                res.label_binary[~m] = -1
                bands = [
                    res.status.astype(np.float32),
                    np.where(res.abandon_year > 0, res.abandon_year, np.nan),
                    res.canopy_start,
                    res.canopy_outcome,
                    np.where(res.label_binary >= 0, res.label_binary, np.nan),
                ]
                for i, b in enumerate(bands, start=1):
                    dst.write(np.asarray(b, np.float32).reshape(w.height, w.width), i, window=w)
                for k, v in summarize(res).items():
                    totals[k] = totals.get(k, 0) + v
                lab = res.labeled
                for y, n in zip(*np.unique(res.abandon_year[lab], return_counts=True), strict=True):
                    by_year[int(y)] = by_year.get(int(y), 0) + int(n)
    totals["labeled_by_abandon_year"] = dict(sorted(by_year.items()))
    totals["label_config"] = load_config()["labels"]
    (out / "label_counts.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    _quicklook(county)
    return totals


def _quicklook(county: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    d = L.read_named(
        L.derived_dir(county) / "labels.tif", ["status", "label_binary", "canopy_outcome"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    status = d["status"]
    cmap = ListedColormap(
        ["white", "#d9d9d9", "#bdbdbd", "#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#377eb8"]
    )
    axes[0].imshow(status, cmap=cmap, vmin=-0.5, vmax=7.5, interpolation="nearest")
    axes[0].set_title("Label status")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(s.value)) for s in Status]
    axes[0].legend(handles, [s.name.lower() for s in Status], fontsize=7, loc="lower left")
    im = axes[1].imshow(
        np.where(status == Status.LABELED, d["canopy_outcome"], np.nan),
        cmap="YlGn",
        vmin=0,
        vmax=100,
    )
    axes[1].set_title("Canopy % at outcome year (labeled pixels)")
    fig.colorbar(im, ax=axes[1], shrink=0.7)
    for ax in axes:
        ax.set_axis_off()
    p = outputs_dir() / "labels"
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{county}_labels.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landscore labels")
    ap.add_argument("--county", nargs="+", default=load_config()["study_area"]["pilot"])
    a = ap.parse_args(argv)
    for c in a.county:
        t = build_county(c)
        keys = [
            "labeled",
            "positive",
            "negative",
            "not_abandoned",
            "converted",
            "refarmed",
            "wooded_at_start",
            "never_farmed",
        ]
        print(c, {k: t[k] for k in keys})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
