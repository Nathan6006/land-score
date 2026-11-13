"""`landscore quicklook`: one PNG per fetched layer, for checking data by eye."""

from __future__ import annotations

import argparse

import numpy as np

from landscore.config import load_config, outputs_dir
from landscore.data import layers as L

# Official NLCD class colours.
NLCD_COLORS = {
    11: "#466b9f", 12: "#d1def8", 21: "#dec5c5", 22: "#d99282", 23: "#eb0000", 24: "#ab0000",
    31: "#b3ac9f", 41: "#68ab5f", 42: "#1c5f2c", 43: "#b5c58f", 52: "#ccb879", 71: "#dfdfc2",
    81: "#dcd939", 82: "#ab6c28", 90: "#b8d9eb", 95: "#6c9fb8",
}  # fmt: skip


def _nlcd_rgb(a: np.ndarray) -> np.ndarray:
    from matplotlib.colors import to_rgb

    rgb = np.ones((*a.shape, 3))
    for k, c in NLCD_COLORS.items():
        rgb[a == k] = to_rgb(c)
    return rgb


def county(county: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    raw = L.raw_dir(county)
    out = outputs_dir() / "quicklook"
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        p = out / f"{county}_{name}.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    for name, years in (("landcover", (1990, 2005, 2024)), ("canopy", (1990, 2005, 2020))):
        path = raw / f"{name}.tif"
        if not path.exists():
            continue
        all_years = L.year_bands(path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        with rasterio.open(path) as src:
            for ax, y in zip(axes, years, strict=True):
                if y not in all_years:
                    ax.set_axis_off()
                    continue
                a = src.read(all_years.index(y) + 1)
                if name == "landcover":
                    ax.imshow(_nlcd_rgb(a), interpolation="nearest")
                else:
                    im = ax.imshow(np.where(a <= 100, a, np.nan), cmap="YlGn", vmin=0, vmax=100)
                ax.set_title(f"{county} {name} {y}")
                ax.set_axis_off()
        if name == "canopy":
            fig.colorbar(im, ax=axes, shrink=0.6, label="canopy %")
        save(fig, name)

    for name in ("dem", "soil", "climate"):
        path = raw / f"{name}.tif"
        if not path.exists():
            continue
        bands = L.read_named(path)
        n = len(bands)
        cols = min(n, 4)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
        for ax, (b, a) in zip(axes.ravel(), bands.items(), strict=False):
            im = ax.imshow(a, cmap="viridis")
            ax.set_title(b, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.7)
        for ax in axes.ravel():
            ax.set_axis_off()
        save(fig, name)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landscore quicklook")
    ap.add_argument("--county", nargs="+", default=load_config()["study_area"]["pilot"])
    a = ap.parse_args(argv)
    for c in a.county:
        for p in county(c):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
