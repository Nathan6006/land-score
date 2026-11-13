"""Find abandoned farm pixels and measure the canopy that developed afterwards.

All functions work on stacks shaped (years, pixels) so they can run on one window of a
raster at a time. Nothing here reads files.

Definitions (values come from config/study.yaml, section `labels`):

* A pixel is *farmed* in a year if its Annual NLCD class is in `farm_classes`.
* The *abandonment year* A is the first year inside `abandon_window` such that the pixel was
  farmed for `min_farm_years` years in a row just before A, and not farmed for
  `min_fallow_years` years in a row starting at A.
* The *outcome year* is A + `outcome_lag`. The label is tree canopy cover (percent) in that year.
* A pixel is dropped (not labeled) if, between A and the outcome year, it was farmed again or
  became water/developed (a person decided its fate), if canopy was already above
  `max_start_canopy` in year A-1, or if canopy data are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from landscore.config import LabelConfig

CANOPY_NODATA_MIN = 101  # USFS TCC percent is 0-100; anything above is nodata/background.


class Status(IntEnum):
    """Why a pixel did or did not get a label. Stored as uint8 in label rasters."""

    NODATA = 0
    NEVER_FARMED = 1  # never farmed in any year before the end of the abandonment window
    NOT_ABANDONED = 2  # farmed, but no qualifying abandonment year in the window
    CONVERTED = 3  # became developed or water before the outcome year
    REFARMED = 4  # farmed again before the outcome year
    WOODED_AT_START = 5  # canopy already above max_start_canopy in year A-1
    NO_CANOPY_DATA = 6
    LABELED = 7


@dataclass
class LabelResult:
    status: np.ndarray  # uint8 (pixels,)
    abandon_year: np.ndarray  # int16, 0 where no abandonment
    canopy_start: np.ndarray  # float32, NaN where not labeled
    canopy_outcome: np.ndarray  # float32, NaN where not labeled
    label_binary: np.ndarray  # int8, -1 where not labeled

    @property
    def labeled(self) -> np.ndarray:
        return self.status == Status.LABELED


def find_abandonment_year(
    landcover: np.ndarray, lc_years: np.ndarray | list[int], cfg: LabelConfig
) -> np.ndarray:
    """Return the abandonment year for each pixel (0 if none).

    landcover: uint8 array (n_years, n_pixels), one row per year in `lc_years` (consecutive).
    """
    lc_years = np.asarray(lc_years)
    _check_consecutive(lc_years)
    farm = np.isin(landcover, cfg.farm_classes)
    y0 = int(lc_years[0])
    n_years = len(lc_years)
    out = np.zeros(landcover.shape[1], dtype=np.int16)
    unset = np.ones(landcover.shape[1], dtype=bool)
    m, f = cfg.min_farm_years, cfg.min_fallow_years
    for year in range(cfg.abandon_window[0], cfg.abandon_window[1] + 1):
        a = year - y0
        if a - m < 0 or a + f > n_years:
            continue  # not enough years of data on one side to judge this year
        farmed_before = farm[a - m : a].all(axis=0)
        fallow_after = ~farm[a : a + f].any(axis=0)
        hit = unset & farmed_before & fallow_after
        out[hit] = year
        unset &= ~hit
    return out


def build_labels(
    landcover: np.ndarray,
    lc_years: np.ndarray | list[int],
    canopy: np.ndarray,
    canopy_years: np.ndarray | list[int],
    cfg: LabelConfig,
    lc_nodata: int = 250,
) -> LabelResult:
    """Compute status, abandonment year, and canopy outcome for each pixel.

    landcover: uint8 (n_lc_years, n_pixels); canopy: uint8/float (n_canopy_years, n_pixels).
    Year arrays must be consecutive. Pixels where every land cover year is `lc_nodata`
    (or 0) get Status.NODATA.
    """
    lc_years = np.asarray(lc_years)
    canopy_years = np.asarray(canopy_years)
    _check_consecutive(lc_years)
    _check_consecutive(canopy_years)
    n = landcover.shape[1]
    if canopy.shape[1] != n:
        raise ValueError("landcover and canopy must have the same number of pixels")

    status = np.full(n, Status.NOT_ABANDONED, dtype=np.uint8)
    farm = np.isin(landcover, cfg.farm_classes)
    end_idx = min(cfg.abandon_window[1] - int(lc_years[0]) + 1, len(lc_years))
    status[~farm[:end_idx].any(axis=0)] = Status.NEVER_FARMED
    nodata = ((landcover == lc_nodata) | (landcover == 0)).all(axis=0)
    status[nodata] = Status.NODATA

    a_year = find_abandonment_year(landcover, lc_years, cfg)
    a_year[nodata] = 0
    cand = a_year > 0
    o_year = np.where(cand, a_year + cfg.outcome_lag, 0)

    # Outcome year must exist in both stacks; otherwise there is no canopy outcome.
    in_range = (
        cand
        & (o_year <= lc_years[-1])
        & (o_year <= canopy_years[-1])
        & (a_year - 1 >= canopy_years[0])
    )
    status[cand & ~in_range] = Status.NO_CANOPY_DATA

    # Between A and outcome year (inclusive): converted or refarmed?
    years_col = lc_years[:, None]
    window = (years_col >= a_year[None, :]) & (years_col <= o_year[None, :]) & in_range[None, :]
    converted = (np.isin(landcover, cfg.converted_classes) & window).any(axis=0)
    refarmed = (farm & window).any(axis=0)
    status[in_range & refarmed] = Status.REFARMED
    status[in_range & converted] = Status.CONVERTED  # conversion wins over refarming

    still = in_range & ~converted & ~refarmed
    canopy_start = np.full(n, np.nan, dtype=np.float32)
    canopy_out = np.full(n, np.nan, dtype=np.float32)
    idx = np.flatnonzero(still)
    if idx.size:
        cs = canopy[(a_year[idx] - 1 - canopy_years[0]).astype(int), idx].astype(np.float32)
        co = canopy[(o_year[idx] - canopy_years[0]).astype(int), idx].astype(np.float32)
        bad = (cs >= CANOPY_NODATA_MIN) | (co >= CANOPY_NODATA_MIN)
        status[idx[bad]] = Status.NO_CANOPY_DATA
        wooded = ~bad & (cs > cfg.max_start_canopy)
        status[idx[wooded]] = Status.WOODED_AT_START
        good = ~bad & ~wooded
        status[idx[good]] = Status.LABELED
        canopy_start[idx[good]] = cs[good]
        canopy_out[idx[good]] = co[good]

    label_bin = np.full(n, -1, dtype=np.int8)
    lab = status == Status.LABELED
    label_bin[lab] = (canopy_out[lab] >= cfg.binary_threshold).astype(np.int8)
    # a_year is kept for every abandonment candidate (labeled or not) for diagnostics.
    return LabelResult(status, a_year, canopy_start, canopy_out, label_bin)


def summarize(result: LabelResult) -> dict[str, int]:
    """Counts per status plus positive/negative counts for the binary label."""
    counts = {s.name.lower(): int((result.status == s).sum()) for s in Status}
    lab = result.labeled
    counts["positive"] = int((result.label_binary[lab] == 1).sum())
    counts["negative"] = int((result.label_binary[lab] == 0).sum())
    return counts


def _check_consecutive(years: np.ndarray) -> None:
    if len(years) and not np.array_equal(years, np.arange(years[0], years[0] + len(years))):
        raise ValueError("year axis must be consecutive with no gaps")
