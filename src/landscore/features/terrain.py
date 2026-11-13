"""Terrain features from a DEM on the 30 m grid.

All functions take a 2-D float array (metres, NaN = nodata) and the cell size in metres.
They run on a whole county tile (a few million pixels, tens of MB), never a full raster.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba ships with shap

    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)


def slope_aspect(dem: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (radians clockwise from north, downslope direction)."""
    z = _fill_nan(dem)
    dzdy, dzdx = np.gradient(z, cell)  # rows increase southwards
    # Downslope direction: east component -dz/dx, north component +dz/dy (row axis points south).
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.arctan2(-dzdx, dzdy) % (2 * np.pi)
    slope[np.isnan(dem)] = np.nan
    aspect[np.isnan(dem)] = np.nan
    return slope.astype(np.float32), aspect.astype(np.float32)


def tpi(dem: np.ndarray, cell: float, radius_m: float) -> np.ndarray:
    """Topographic position index: elevation minus mean elevation in a circle of radius_m."""
    k = disk_kernel(radius_m / cell)
    mean = _masked_mean(dem, k)
    out = dem - mean
    return out.astype(np.float32)


def flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """D8 flow accumulation (number of upslope cells, including the cell itself).

    Cells are visited from highest to lowest; each passes its total to its steepest lower
    neighbour. Pits and flats keep their water. Good enough for a wetness index on gentle
    glaciated terrain; not a hydrologic model.
    """
    z = _fill_nan(dem).astype(np.float64)
    h, w = z.shape
    pad = np.pad(z, 1, mode="edge")
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([np.hypot(dy, dx) for dy, dx in offsets])
    drops = np.stack(
        [
            (z - pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]) / d
            for (dy, dx), d in zip(offsets, dist, strict=True)
        ]
    )
    best = drops.argmax(axis=0)
    has_out = drops.max(axis=0) > 0
    dy = np.array([o[0] for o in offsets])[best]
    dx = np.array([o[1] for o in offsets])[best]
    rows, cols = np.indices((h, w))
    tr = np.clip(rows + dy, 0, h - 1)
    tc = np.clip(cols + dx, 0, w - 1)
    target = np.where(has_out, tr * w + tc, -1).ravel()
    order = np.argsort(-z.ravel(), kind="stable")
    return _accumulate(order, target).reshape(h, w).astype(np.float32)


@njit(cache=False)
def _accumulate(order: np.ndarray, target: np.ndarray) -> np.ndarray:
    acc = np.ones(target.size)
    for i in order:
        t = target[i]
        if t >= 0:
            acc[t] += acc[i]
    return acc


def twi(dem: np.ndarray, cell: float) -> np.ndarray:
    """Topographic wetness index ln(a / tan(b)), a = specific catchment area (m)."""
    slope, _ = slope_aspect(dem, cell)
    acc = flow_accumulation(dem)
    a = acc * cell  # contributing area per unit contour width
    tanb = np.tan(np.radians(np.maximum(np.nan_to_num(slope, nan=0.0), 0.1)))
    out = np.log(a / tanb).astype(np.float32)
    out[np.isnan(dem)] = np.nan
    return out


def terrain_features(dem: np.ndarray, cell: float) -> dict[str, np.ndarray]:
    slope, aspect = slope_aspect(dem, cell)
    flat = slope < 1.0  # aspect is meaningless on flat ground
    north = np.where(flat, 0.0, np.cos(aspect)).astype(np.float32)
    east = np.where(flat, 0.0, np.sin(aspect)).astype(np.float32)
    return {
        "elevation": dem.astype(np.float32),
        "slope_deg": slope,
        "northness": north,
        "eastness": east,
        "twi": twi(dem, cell),
        "tpi_300": tpi(dem, cell, 300),
        "tpi_1000": tpi(dem, cell, 1000),
    }


def disk_kernel(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return (x**2 + y**2 <= radius_px**2).astype(np.float32)


def _masked_mean(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(a)
    s = _fft_convolve(np.where(valid, a, 0.0), kernel)
    n = _fft_convolve(valid.astype(np.float64), kernel)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0.5, s / n, np.nan)


def _fft_convolve(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    if kernel.shape[0] <= 15:
        return ndimage.convolve(a.astype(np.float64), kernel.astype(np.float64), mode="constant")
    return fftconvolve(a.astype(np.float64), kernel.astype(np.float64), mode="same")


def _fill_nan(a: np.ndarray) -> np.ndarray:
    """Replace NaN with the nearest valid value (so gradients near edges stay finite)."""
    mask = np.isnan(a)
    if not mask.any():
        return a.astype(np.float64)
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype(np.float64)
