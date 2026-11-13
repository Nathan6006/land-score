"""Where each per-county layer lives, and small raster helpers shared by pipeline stages.

Layout under the data folder:
  raw/<County>/grid.json          analysis grid (EPSG:5070, 30 m) covering county + buffer
  raw/<County>/boundary.geojson   county polygon (EPSG:5070)
  raw/<County>/landcover.tif      Annual NLCD, one band per year (descriptions = year)
  raw/<County>/canopy.tif         USFS TCC percent, one band per year
  raw/<County>/dem.tif            elevation (m), float32
  raw/<County>/soil.tif           soil properties, one band per feature (descriptions = name)
  raw/<County>/climate.tif        climate normals, one band per feature
  derived/<County>/labels.tif     status, abandon_year, canopy_start, canopy_outcome, label_binary
  derived/<County>/static.tif     all static features (soil + terrain + climate) on the grid
  derived/<County>/landscape_now.tif  landscape features from the latest land cover year
  derived/<County>/training.parquet   sampled training table
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize

from landscore.config import data_dir
from landscore.data.grid import Grid

LABEL_BANDS = ("status", "abandon_year", "canopy_start", "canopy_outcome", "label_binary")


def raw_dir(county: str) -> Path:
    p = data_dir() / "raw" / county
    p.mkdir(parents=True, exist_ok=True)
    return p


def derived_dir(county: str) -> Path:
    p = data_dir() / "derived" / county
    p.mkdir(parents=True, exist_ok=True)
    return p


def grid(county: str) -> Grid:
    return Grid.load(raw_dir(county) / "grid.json")


def band_names(path: Path) -> list[str]:
    with rasterio.open(path) as src:
        return [d or str(i + 1) for i, d in enumerate(src.descriptions)]


def read_named(path: Path, names: list[str] | None = None, window=None) -> dict[str, np.ndarray]:
    """Read bands by description. Returns float32 arrays with nodata -> NaN."""
    with rasterio.open(path) as src:
        desc = list(src.descriptions)
        names = names or [d for d in desc if d]
        out = {}
        for n in names:
            a = src.read(desc.index(n) + 1, window=window).astype(np.float32)
            if src.nodata is not None and not np.isnan(src.nodata):
                a[a == src.nodata] = np.nan
            out[n] = a
        return out


def write_named(path: Path, g: Grid, layers: dict[str, np.ndarray], dtype="float32", nodata=np.nan):
    prof = g.profile(dtype, count=len(layers), nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        for i, (name, a) in enumerate(layers.items(), start=1):
            dst.write(a.astype(dtype), i)
            dst.set_band_description(i, name)


def county_mask(county: str, g: Grid | None = None) -> np.ndarray:
    """Boolean mask of grid pixels whose centre lies inside the county boundary."""
    import geopandas as gpd

    g = g or grid(county)
    shp = gpd.read_file(raw_dir(county) / "boundary.geojson").to_crs(g.crs)
    return rasterize(
        [(geom, 1) for geom in shp.geometry],
        out_shape=(g.height, g.width),
        transform=g.affine,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def year_bands(path: Path) -> list[int]:
    return [int(n) for n in band_names(path)]
