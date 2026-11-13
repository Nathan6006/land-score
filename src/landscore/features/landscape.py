"""Landscape context at the start year, computed from that year's land cover map.

For a pixel abandoned in year A, these use the Annual NLCD map of year A-1 (the last year it
was farmed), so they are known before regrowth began.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from landscore.features.terrain import _fft_convolve, disk_kernel


def landscape_layers(
    landcover: np.ndarray,
    forest_classes: tuple[int, ...],
    cell: float,
    nodata: tuple[int, ...] = (0, 250),
) -> dict[str, np.ndarray]:
    """Distance to forest and percent forest within 100 m / 500 m for one year's map."""
    forest = np.isin(landcover, forest_classes)
    valid = ~np.isin(landcover, nodata)
    if forest.any():
        dist = ndimage.distance_transform_edt(~forest, sampling=cell).astype(np.float32)
    else:
        dist = np.full(landcover.shape, np.inf, dtype=np.float32)
    out = {"dist_forest_m": dist}
    for r in (100, 500):
        k = disk_kernel(r / cell)
        f = _fft_convolve((forest & valid).astype(np.float64), k)
        n = _fft_convolve(valid.astype(np.float64), k)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(n > 0.5, 100.0 * f / n, np.nan)
        out[f"forest_pct_{r}m"] = np.clip(pct, 0, 100).astype(np.float32)
    for v in out.values():
        v[~valid] = np.nan
    return out
