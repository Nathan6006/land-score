"""Planning flags: simple, documented rules. Not part of the trained model.

Current land cover (most recent Annual NLCD year) is allowed here because flags describe the
plot today. These inputs are never used as model features (rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HERBACEOUS = (71, 81, 82)  # grassland, pasture/hay, cultivated crops
SHRUB = (52,)


@dataclass
class Flag:
    name: str
    raised: bool
    detail: str


def _frac(mask: np.ndarray, valid: np.ndarray) -> float:
    n = valid.sum()
    return float((mask & valid).sum() / n) if n else 0.0


def wet_soil(drainage: np.ndarray, wt_depth_cm: np.ndarray, area_frac: float = 0.25) -> Flag:
    valid = ~np.isnan(drainage)
    wet = (drainage >= 6) | (np.nan_to_num(wt_depth_cm, nan=999) < 30)
    f = _frac(wet, valid | ~np.isnan(wt_depth_cm))
    return Flag(
        "wet_soil",
        f >= area_frac,
        f"{f:.0%} of the plot is poorly/very poorly drained or has a water table within 30 cm. "
        "Choose wet-tolerant species there and avoid planting in standing water.",
    )


def steep_slope(slope_deg: np.ndarray, threshold: float = 15.0, area_frac: float = 0.2) -> Flag:
    valid = ~np.isnan(slope_deg)
    f = _frac(slope_deg > threshold, valid)
    return Flag(
        "steep_slope",
        f >= area_frac,
        f"{f:.0%} of the plot is steeper than {threshold:.0f} degrees. Plan for erosion control "
        "and safe access for planting crews.",
    )


def competing_vegetation(current_landcover: np.ndarray, area_frac: float = 0.5) -> Flag:
    valid = current_landcover > 0
    herb = _frac(np.isin(current_landcover, HERBACEOUS), valid)
    shrub = _frac(np.isin(current_landcover, SHRUB), valid)
    raised = herb + shrub >= area_frac
    return Flag(
        "competing_vegetation",
        raised,
        f"Latest land cover: {herb:.0%} grass/pasture/crop, {shrub:.0%} shrub. Dense grass and "
        "shrubs compete hard with seedlings; plan mowing, mulch, or other weed control for at "
        "least the first three years.",
    )


def deer_pressure(forest_pct_1km: float, low: float = 20.0, high: float = 70.0) -> Flag:
    raised = low <= forest_pct_1km <= high
    return Flag(
        "deer_pressure",
        raised,
        f"Forest covers {forest_pct_1km:.0f}% of land within 1 km. Mixed forest and open land is "
        "prime deer habitat, so browse is likely; plan tree tubes or fencing. "
        "(Rough proxy: Northeast Ohio deer pressure is high almost everywhere.)",
    )
