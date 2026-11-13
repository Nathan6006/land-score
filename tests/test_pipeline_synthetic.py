"""End-to-end run on small synthetic rasters: labels -> features -> train -> report.

Two fake counties (named like real ones so the config lookup works) let the county-holdout
path run. Nothing here touches the network or the real data folder.
"""

import json

import numpy as np
import pytest
from shapely.geometry import box, mapping

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from landscore.data.grid import snap_grid  # noqa: E402

N = 90  # pixels per side
LC_YEARS = list(range(1985, 2025))
CAN_YEARS = list(range(1985, 2024))


def _write(path, g, bands: dict, dtype, nodata):
    prof = g.profile(dtype, count=len(bands), nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        for i, (name, a) in enumerate(bands.items(), start=1):
            dst.write(np.asarray(a, dtype), i)
            dst.set_band_description(i, str(name))


def make_county(root, county, x0, seed):
    rng = np.random.default_rng(seed)
    raw = root / "raw" / county
    raw.mkdir(parents=True)
    g = snap_grid((x0, 2_200_000, x0 + N * 30, 2_200_000 + N * 30))
    g.save(raw / "grid.json")
    b = g.bounds
    poly = box(b[0] + 60, b[1] + 60, b[2] - 60, b[3] - 60)
    gpd.GeoDataFrame({"name": [county]}, geometry=[poly], crs="EPSG:5070").to_file(
        raw / "boundary.geojson", driver="GeoJSON"
    )
    rows, cols = np.indices((g.height, g.width))
    forest = cols < 15  # forest strip on the west edge
    lc = np.full((len(LC_YEARS), g.height, g.width), 82, np.uint8)
    lc[:, forest] = 41
    # Fields abandoned in random years 1992-2003 on random 6x6 patches.
    a_year = np.zeros((g.height, g.width), int)
    for _ in range(120):
        r, c = rng.integers(0, g.height - 6), rng.integers(15, g.width - 6)
        a_year[r : r + 6, c : c + 6] = rng.integers(1992, 2004)
    for i, y in enumerate(LC_YEARS):
        lc[i][(a_year > 0) & (y >= a_year)] = 71
    dist = (cols - 15).clip(0) * 30.0
    wet = rng.uniform(1, 7, (g.height, g.width)).round()
    p = 1 / (1 + np.exp(-(2.5 - dist / 500 - 0.3 * (wet - 4))))
    final = np.where(rng.uniform(size=p.shape) < p, 70, 5)
    can = np.zeros((len(CAN_YEARS), g.height, g.width), np.uint8)
    can[:, forest] = 85
    for i, y in enumerate(CAN_YEARS):
        grow = (a_year > 0) & (y >= a_year + 3)
        can[i][grow] = final[grow]
    _write(raw / "landcover.tif", g, dict(zip(LC_YEARS, lc, strict=True)), "uint8", 250)
    _write(raw / "canopy.tif", g, dict(zip(CAN_YEARS, can, strict=True)), "uint8", 255)
    dem = 300 + rows * 0.5 + rng.normal(0, 0.3, rows.shape)
    _write(raw / "dem.tif", g, {"elevation": dem}, "float32", np.nan)
    soil = {
        "drainage_class": wet,
        "seedling_mortality": (wet >= 6).astype(float) * 2,
        "ph_0_30": np.full(rows.shape, 6.2),
        "wt_depth_min": np.where(wet >= 6, 15, 150).astype(float),
        "aws_0_100": rng.uniform(10, 20, rows.shape),
    }
    _write(raw / "soil.tif", g, soil, "float32", np.nan)
    _write(
        raw / "climate.tif",
        g,
        {"ppt_annual": np.full(rows.shape, 1000.0), "tmean_annual": np.full(rows.shape, 9.5)},
        "float32",
        np.nan,
    )
    return g


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("LANDSCORE_DATA_DIR", str(root))
    monkeypatch.setenv("LANDSCORE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    g = make_county(root, "Geauga", 1_280_010, 1)
    make_county(root, "Lake", 1_300_020, 2)

    from landscore.features.table import main as features_main
    from landscore.labels.build import main as labels_main
    from landscore.models.train import main as train_main
    from landscore.report.plot_report import main as report_main

    assert labels_main(["--county", "Geauga", "Lake"]) == 0
    counts = json.loads((root / "derived" / "Geauga" / "label_counts.json").read_text())
    assert counts["labeled"] > 500 and counts["positive"] > 50 and counts["negative"] > 50

    assert features_main(["--county", "Geauga", "Lake"]) == 0
    import pandas as pd

    t = pd.read_parquet(root / "derived" / "Geauga" / "training.parquet")
    assert (t["landscape_year"] < t["abandon_year"]).all()
    assert t["dist_forest_m"].notna().all()

    docs = tmp_path / "docs"
    assert train_main(["--no-shap", "--docs-dir", str(docs)]) == 0
    res = pd.read_csv(tmp_path / "outputs" / "models" / "cv_results.csv")
    assert {"county", "block", "random"} <= set(res.scheme)
    text = (docs / "RESULTS.md").read_text()
    assert "county" in text and "Verdict" in text
    rf = res[(res.task == "binary") & (res.scheme == "county") & (res.model == "random_forest")]
    assert rf["auc"].iloc[0] > 0.6

    b = g.bounds
    plot = box(b[0] + 300, b[1] + 900, b[0] + 1500, b[1] + 1800)
    gj = tmp_path / "plot_test.geojson"
    gj.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:5070"}},
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(plot)}],
            }
        )
    )
    assert report_main([str(gj)]) == 0
    rep = json.loads((tmp_path / "outputs" / "reports" / "plot_test" / "report.json").read_text())
    assert 0 <= rep["score_mean"] <= 100
    assert "not a promise" in rep["caveat"]
    md = (tmp_path / "outputs" / "reports" / "plot_test" / "report.md").read_text()
    assert "not a promise" in md and "Species fit" in md
