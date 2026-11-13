"""The list of model features and the leakage rule that guards them.

Rule (CLAUDE.md, rule 2): never use a feature measured after regrowth started.

Every feature is either
* STATIC: an inherent site property that regrowth cannot change (soil, terrain, long-term
  climate). Only the groups in STATIC_GROUPS may be static. Vegetation, imagery, or
  embeddings can never be static, because they describe what grew.
* DYNAMIC: measured in a specific year, given as an offset from the abandonment year A.
  The offset must be negative, i.e. the measurement year is before regrowth began.

`check_registry` enforces this for the definitions. `check_table` enforces it for an actual
training table, row by row, using the stored measurement-year columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

STATIC_GROUPS = frozenset({"soil", "terrain", "climate"})
FORBIDDEN_GROUPS = frozenset({"imagery", "embedding", "current_vegetation"})


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    kind: Literal["static", "dynamic"]
    description: str
    offset_years: int | None = None  # dynamic only: measurement year = A + offset_years
    year_column: str | None = None  # dynamic only: table column holding the measurement year
    categorical: bool = False


# Landscape features are computed from the land cover map of year A-1 (last farmed year).
LANDSCAPE_OFFSET = -1

FEATURES: tuple[Feature, ...] = (
    # Soil (gNATSGO). Survey ratings describe the soil, not what is growing on it.
    Feature(
        "drainage_class",
        "soil",
        "static",
        "Drainage class, dominant condition, ordinal 1 (excessively) to 7 (very poorly)",
    ),
    Feature("aws_0_100", "soil", "static", "Available water storage 0-100 cm (cm)"),
    Feature("wt_depth_min", "soil", "static", "Annual minimum depth to water table (cm)"),
    Feature("ph_0_30", "soil", "static", "pH (1:1 water), 0-30 cm, dominant component"),
    Feature("sand_0_30", "soil", "static", "Sand percent, 0-30 cm, dominant component"),
    Feature("clay_0_30", "soil", "static", "Clay percent, 0-30 cm, dominant component"),
    Feature("om_0_30", "soil", "static", "Organic matter percent, 0-30 cm, dominant component"),
    Feature("hydgrp", "soil", "static", "Hydrologic soil group, ordinal A=1 .. D=4"),
    Feature(
        "seedling_mortality",
        "soil",
        "static",
        "Survey potential seedling mortality rating, 0 (low) to 1 (high)",
    ),
    Feature(
        "site_index",
        "soil",
        "static",
        "Survey forest site index, highest of listed species (ft at base age)",
    ),
    # Terrain (3DEP).
    Feature("elevation", "terrain", "static", "Elevation (m)"),
    Feature("slope_deg", "terrain", "static", "Slope (degrees)"),
    Feature("northness", "terrain", "static", "cos(aspect)"),
    Feature("eastness", "terrain", "static", "sin(aspect)"),
    Feature("twi", "terrain", "static", "Topographic wetness index ln(a / tan b)"),
    Feature("tpi_300", "terrain", "static", "Elevation minus mean elevation within 300 m (m)"),
    Feature("tpi_1000", "terrain", "static", "Elevation minus mean elevation within 1000 m (m)"),
    # Climate normals.
    Feature("ppt_annual", "climate", "static", "Mean annual precipitation (mm), 30-year normal"),
    Feature("tmean_annual", "climate", "static", "Mean annual temperature (C), 30-year normal"),
    # Landscape at the start (land cover of year A-1).
    Feature(
        "dist_forest_m",
        "landscape",
        "dynamic",
        "Distance to nearest forest pixel (m) in year A-1",
        LANDSCAPE_OFFSET,
        "landscape_year",
    ),
    Feature(
        "forest_pct_100m",
        "landscape",
        "dynamic",
        "Percent forest within 100 m in year A-1",
        LANDSCAPE_OFFSET,
        "landscape_year",
    ),
    Feature(
        "forest_pct_500m",
        "landscape",
        "dynamic",
        "Percent forest within 500 m in year A-1",
        LANDSCAPE_OFFSET,
        "landscape_year",
    ),
    Feature(
        "was_pasture",
        "landscape",
        "dynamic",
        "1 if class in year A-1 was pasture/hay (81), 0 if crops (82)",
        LANDSCAPE_OFFSET,
        "landscape_year",
    ),
)

FEATURE_NAMES = tuple(f.name for f in FEATURES)


class LeakageError(ValueError):
    pass


def check_registry(features: tuple[Feature, ...] = FEATURES) -> None:
    """Raise LeakageError if any feature definition could leak the outcome."""
    for f in features:
        if f.group in FORBIDDEN_GROUPS:
            raise LeakageError(f"{f.name}: group '{f.group}' may never be a model feature")
        if f.kind == "static":
            if f.group not in STATIC_GROUPS:
                raise LeakageError(
                    f"{f.name}: only {sorted(STATIC_GROUPS)} may be static; '{f.group}' must be dated"
                )
        elif f.kind == "dynamic":
            if f.offset_years is None or f.year_column is None:
                raise LeakageError(f"{f.name}: dynamic feature needs offset_years and year_column")
            if f.offset_years >= 0:
                raise LeakageError(
                    f"{f.name}: measured at A{f.offset_years:+d}, on/after regrowth started"
                )
        else:
            raise LeakageError(f"{f.name}: unknown kind {f.kind!r}")


def check_table(
    table: pd.DataFrame,
    features: tuple[Feature, ...] = FEATURES,
    start_col: str = "abandon_year",
) -> None:
    """Raise LeakageError if any row has a feature measured on/after its label start year."""
    check_registry(features)
    present = [f for f in features if f.name in table.columns]
    for f in present:
        if f.kind != "dynamic":
            continue
        if f.year_column not in table.columns:
            raise LeakageError(f"{f.name}: table lacks measurement-year column {f.year_column!r}")
        late = table[f.year_column] >= table[start_col]
        if late.any():
            raise LeakageError(
                f"{f.name}: {int(late.sum())} rows measured on/after the label start year"
            )
    unknown = [c for c in table.columns if c.startswith("feat_")]
    if unknown:
        raise LeakageError(f"unregistered feature-like columns: {unknown}")


def by_name(name: str) -> Feature:
    for f in FEATURES:
        if f.name == name:
            return f
    raise KeyError(name)
