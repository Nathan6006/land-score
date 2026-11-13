"""`landscore fetch`: download and clip every input layer to a county's analysis grid.

Nothing here downloads a whole CONUS raster. Large sources are Cloud-Optimized GeoTIFFs read
through HTTP range requests, so only the county window crosses the network. Every output is
recorded in data/manifest.json. Finished layers are skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

from landscore.config import county_fips, data_dir, load_config
from landscore.data import layers as L
from landscore.data import manifest
from landscore.data import sources as S
from landscore.data.grid import Grid, snap_grid

GDAL_ENV = {
    "GDAL_HTTP_MAX_RETRY": "6",
    "GDAL_HTTP_RETRY_DELAY": "4",
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "GDAL_CACHEMAX": 256,
}
COG_ENV = GDAL_ENV | {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
ALL_LAYERS = ("boundary", "landcover", "canopy", "dem", "soil", "climate")


def _retry(fn, tries: int = 4, wait: float = 5.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry {i + 1}/{tries - 1} after error: {str(e)[:120]}", flush=True)
            time.sleep(wait * (i + 1))


def _download(url: str, dest: Path) -> Path:
    """Stream a small file to disk (used for the county boundary zip only)."""
    if dest.exists() and manifest.is_current(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    manifest.record(dest, [url])
    return dest


# ---------------------------------------------------------------- boundary + grid


def fetch_boundary(county: str, force: bool = False) -> Grid:
    import geopandas as gpd

    cfg = load_config()["study_area"]
    raw = L.raw_dir(county)
    gpath, bpath = raw / "grid.json", raw / "boundary.geojson"
    if not force and manifest.is_current(bpath) and gpath.exists():
        return Grid.load(gpath)
    zpath = _download(S.COUNTIES_URL, data_dir() / "raw" / "_shared" / "cb_2023_us_county_500k.zip")
    fips = county_fips()[county]
    shp = gpd.read_file(f"zip://{zpath}", where=f"GEOID = '{fips}'").to_crs(cfg["crs"])
    if len(shp) != 1:
        raise RuntimeError(f"expected one county for FIPS {fips}, got {len(shp)}")
    shp[["GEOID", "NAME", "geometry"]].to_file(bpath, driver="GeoJSON")
    xmin, ymin, xmax, ymax = shp.total_bounds
    b = cfg["buffer_m"]
    g = snap_grid((xmin - b, ymin - b, xmax + b, ymax + b), cfg["resolution"], crs=cfg["crs"])
    g.save(gpath)
    manifest.record(bpath, [S.COUNTIES_URL], fips=fips)
    manifest.record(gpath, [S.COUNTIES_URL], note="derived from boundary + buffer")
    print(f"{county}: grid {g.width} x {g.height} px ({g.width * g.height / 1e6:.2f} M)")
    return g


# ---------------------------------------------------------------- aligned window reads


def _is_conus_albers(crs) -> bool:
    d = crs.to_dict()
    return (
        d.get("proj") == "aea"
        and math.isclose(d.get("lat_1", 0), 29.5)
        and math.isclose(d.get("lat_2", 0), 45.5)
        and math.isclose(d.get("lat_0", 0), 23.0)
        and math.isclose(d.get("lon_0", 0), -96.0)
    )


def read_on_grid(src, g: Grid, resampling=Resampling.nearest, band: int = 1) -> np.ndarray:
    """Read one band of an open dataset onto the grid.

    If the source is already on the CONUS Albers 30 m grid (Annual NLCD, TCC) pixels are
    copied exactly with a window read. Otherwise the source is warped.
    """
    t = src.transform
    aligned = (
        _is_conus_albers(src.crs)
        and math.isclose(t.a, g.res)
        and math.isclose(-t.e, g.res)
        and abs(((g.affine.c - t.c) / g.res + 0.5) % 1 - 0.5) < 1e-3
        and abs(((t.f - g.affine.f) / g.res + 0.5) % 1 - 0.5) < 1e-3
    )
    if aligned:
        win = from_bounds(*g.bounds, transform=t).round_offsets().round_lengths()
        a = src.read(band, window=win, boundless=True, fill_value=src.nodata)
        if a.shape != (g.height, g.width):
            raise RuntimeError(f"window shape {a.shape} != grid {(g.height, g.width)}")
        return a
    dst = np.full((g.height, g.width), src.nodata if src.nodata is not None else 0, src.dtypes[0])
    reproject(
        rasterio.band(src, band),
        dst,
        dst_transform=g.affine,
        dst_crs=g.crs,
        dst_nodata=src.nodata,
        resampling=resampling,
    )
    return dst


def _write_years(path: Path, g: Grid, years: list[int], read_year, nodata: int, workers: int):
    prof = g.profile("uint8", count=len(years), nodata=nodata)
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as dst, ThreadPoolExecutor(workers) as pool:
        for i, (y, a) in enumerate(zip(years, pool.map(read_year, years), strict=True), start=1):
            dst.write(a.astype("uint8"), i)
            dst.set_band_description(i, str(y))
            print(f"  {path.stem} {y}", flush=True)
    tmp.replace(path)


# ---------------------------------------------------------------- land cover


def _nlcd_year(year: int, g: Grid) -> np.ndarray:
    def mirror():
        with rasterio.Env(**COG_ENV), rasterio.open(S.NLCD_MIRROR.format(year=year)) as src:
            return read_on_grid(src, g)

    def wcs():
        xmin, ymin, xmax, ymax = g.bounds
        url = S.NLCD_WCS.format(year=year, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with rasterio.MemoryFile(r.content) as mem, mem.open() as src:
            a = src.read(1)
        if a.shape != (g.height, g.width):
            raise RuntimeError(f"WCS returned shape {a.shape}")
        return a

    try:
        return _retry(mirror)
    except Exception as e:
        print(f"  NLCD mirror failed for {year} ({str(e)[:80]}); using MRLC WCS", flush=True)
        return _retry(wcs)


def fetch_landcover(county: str, g: Grid, force: bool = False) -> None:
    path = L.raw_dir(county) / "landcover.tif"
    if not force and manifest.is_current(path):
        return
    y0, y1 = load_config()["years"]["landcover"]
    years = list(range(y0, y1 + 1))
    _write_years(path, g, years, lambda y: _nlcd_year(y, g), S.NLCD_NODATA, workers=6)
    manifest.record(
        path,
        [S.NLCD_MIRROR.format(year="{1985..2024}")],
        product="Annual NLCD Collection 1.1 land cover (LndCov), C1V1",
        fallback=S.NLCD_WCS.split("?")[0],
    )


# ---------------------------------------------------------------- canopy


def _tcc_year(year: int, g: Grid) -> np.ndarray:
    url = f"/vsizip//vsicurl/{S.TCC_ZIP.format(year=year)}/{S.TCC_MEMBER.format(year=year)}"

    def read():
        with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src:
            return read_on_grid(src, g)

    return _retry(read)


def fetch_canopy(county: str, g: Grid, force: bool = False) -> None:
    path = L.raw_dir(county) / "canopy.tif"
    if not force and manifest.is_current(path):
        return
    y0, y1 = load_config()["years"]["canopy"]
    years = list(range(y0, y1 + 1))
    _write_years(path, g, years, lambda y: _tcc_year(y, g), S.TCC_NODATA, workers=4)
    manifest.record(
        path,
        [S.TCC_ZIP.format(year="{1985..2025}")],
        product="USFS Science Tree Canopy Cover v2025-6 (percent, 0-100; >100 = no data)",
    )


# ---------------------------------------------------------------- elevation


def _dem_tiles(g: Grid) -> list[str]:
    w, s, e, n = transform_bounds(g.crs, "EPSG:4269", *g.bounds, densify_pts=21)
    tiles = []
    for lat in range(math.floor(s), math.floor(n) + 1):
        for lon in range(math.floor(w), math.floor(e) + 1):
            tiles.append(f"n{lat + 1:02d}w{-lon:03d}")  # named by the NW corner
    return tiles


def fetch_dem(county: str, g: Grid, force: bool = False) -> None:
    path = L.raw_dir(county) / "dem.tif"
    if not force and manifest.is_current(path):
        return
    dem = np.full((g.height, g.width), np.nan, np.float32)
    urls = []
    for tile in _dem_tiles(g):
        url = S.DEM_TILE.format(tile=tile)

        def read(url=url):
            part = np.full((g.height, g.width), np.nan, np.float32)
            with rasterio.Env(**COG_ENV), rasterio.open(url) as src:
                reproject(
                    rasterio.band(src, 1),
                    part,
                    dst_transform=g.affine,
                    dst_crs=g.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
            return part

        part = _retry(read)
        dem = np.where(np.isnan(dem), part, dem)
        urls.append(url)
        print(f"  dem {tile}: {np.isfinite(part).mean():.0%} of grid", flush=True)
    L.write_named(path, g, {"elevation": dem})
    manifest.record(path, urls, product="USGS 3DEP 1/3 arc-second DEM, averaged to 30 m")


# ---------------------------------------------------------------- climate


def _pc_client():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(S.PC_STAC, modifier=planetary_computer.sign_inplace)


def fetch_climate(county: str, g: Grid, force: bool = False) -> None:
    path = L.raw_dir(county) / "climate.tif"
    if not force and manifest.is_current(path):
        return
    item = _retry(
        lambda: _pc_client().get_collection(S.CLIMATE_COLLECTION).get_item(S.CLIMATE_ITEM)
    )
    out = {}
    for name, key in (("ppt_annual", "prcp_norm"), ("tmean_annual", "tavg_norm")):
        dst = np.full((g.height, g.width), np.nan, np.float32)
        with rasterio.Env(**COG_ENV), rasterio.open(item.assets[key].href) as src:
            reproject(
                rasterio.band(src, 1),
                dst,
                dst_transform=g.affine,
                dst_crs=g.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        out[name] = dst
    L.write_named(path, g, out)
    manifest.record(
        path,
        [f"{S.PC_STAC}/collections/{S.CLIMATE_COLLECTION}/items/{S.CLIMATE_ITEM}"],
        product="NOAA nClimGrid 1991-2020 annual normals (prcp_norm mm, tavg_norm C), bilinear",
    )


# ---------------------------------------------------------------- soil


def fetch_soil(county: str, g: Grid, force: bool = False) -> None:
    from landscore.data import soil

    raw = L.raw_dir(county)
    path = raw / "soil.tif"
    if not force and manifest.is_current(path):
        return
    w, s, e, n = transform_bounds(g.crs, "EPSG:4326", *g.bounds, densify_pts=21)
    items = _retry(
        lambda: list(
            _pc_client().search(collections=[S.GNATSGO_COLLECTION], bbox=[w, s, e, n]).items()
        )
    )
    mukey = np.zeros((g.height, g.width), np.int32)
    for it in items:

        def read(it=it):
            part = np.zeros((g.height, g.width), np.int32)
            with rasterio.Env(**COG_ENV), rasterio.open(it.assets["mukey"].href) as src:
                reproject(
                    rasterio.band(src, 1),
                    part,
                    dst_transform=g.affine,
                    dst_crs=g.crs,
                    src_nodata=src.nodata,
                    dst_nodata=0,
                    resampling=Resampling.mode,
                )
            return part

        part = _retry(read)
        mukey = np.where(mukey == 0, part, mukey)
        print(f"  mukey {it.id}: {(part > 0).mean():.0%} of grid", flush=True)
    prof = g.profile("int32", nodata=0)
    with rasterio.open(raw / "soil_mukey.tif", "w", **prof) as dst:
        dst.write(mukey, 1)
        dst.set_band_description(1, "mukey")
    manifest.record(
        raw / "soil_mukey.tif",
        [f"{S.PC_STAC}/collections/{S.GNATSGO_COLLECTION}/items/{it.id}" for it in items],
        product="gNATSGO July 2020 map unit keys, 10 m, majority-resampled to 30 m",
    )
    keys = np.unique(mukey[mukey > 0])
    table = soil.mapunit_table(keys)
    table.to_parquet(raw / "soil_table.parquet")
    manifest.record(
        raw / "soil_table.parquet",
        [S.SDA_URL],
        product="USDA Soil Data Access query results per map unit",
    )
    missing = table["drainage_class"].isna().mean()
    print(f"  soil: {len(keys)} map units, {missing:.0%} without drainage class")
    L.write_named(path, g, soil.rasterize(mukey, table))
    manifest.record(
        path,
        [S.SDA_URL],
        product="soil properties per pixel from soil_mukey.tif + soil_table.parquet",
    )


# ---------------------------------------------------------------- CLI


def fetch_county(county: str, layers=ALL_LAYERS, force: bool = False) -> None:
    g = fetch_boundary(county, force and "boundary" in layers)
    steps = {
        "landcover": fetch_landcover,
        "canopy": fetch_canopy,
        "dem": fetch_dem,
        "soil": fetch_soil,
        "climate": fetch_climate,
    }
    for name, fn in steps.items():
        if name in layers:
            t = time.time()
            print(f"{county}: {name} ...", flush=True)
            fn(county, g, force)
            print(f"{county}: {name} done in {time.time() - t:.0f} s", flush=True)
    size = sum(p.stat().st_size for p in L.raw_dir(county).glob("*") if p.is_file())
    print(f"{county}: raw data {size / 1e6:.0f} MB in {L.raw_dir(county)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landscore fetch")
    ap.add_argument("--county", nargs="+", default=load_config()["study_area"]["pilot"])
    ap.add_argument("--layers", nargs="+", default=list(ALL_LAYERS), choices=ALL_LAYERS)
    ap.add_argument("--force", action="store_true", help="refetch even if the manifest matches")
    a = ap.parse_args(argv)
    for c in a.county:
        fetch_county(c, tuple(a.layers), a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
