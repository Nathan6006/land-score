# Decisions

One entry per scope or design change: date, what changed, why, what was rejected.
Newest at the bottom.

## 2026-11-13 — Environment tool: uv

- **Chose:** [uv](https://docs.astral.sh/uv/) with `pyproject.toml` + committed `uv.lock`,
  Python 3.12.
- **Why:** One command (`uv sync --locked`) installs Python itself and all packages on macOS,
  Windows, and Linux with no system package manager. The geospatial stack now ships as binary
  wheels with GDAL/PROJ bundled (rasterio, pyogrio, pyproj, shapely), so nothing compiles.
  Smaller on disk than a conda environment, which matters on a nearly full laptop.
- **Rejected:** pixi/conda-forge (more robust for exotic GDAL drivers such as FileGDB, but a
  larger install and we only need GeoTIFF/COG, Parquet, GeoJSON); plain pip + venv (no
  cross-platform lockfile, no Python install).
- **Revisit if:** a needed GDAL driver is missing from the rasterio/pyogrio wheels.

## 2026-11-13 — Gradient boosting from scikit-learn, not LightGBM/XGBoost

- **Chose:** `sklearn.ensemble.HistGradientBoosting*` for the boosted-tree model.
- **Why:** LightGBM and XGBoost macOS wheels need OpenMP (`libomp`), normally installed with
  Homebrew. That breaks the "no system package manager, one command" rule for Karan's Mac.
  HistGradientBoosting is the same algorithm family (histogram-based GBDT) with no extra dependency.
- **Rejected:** LightGBM, XGBoost (libomp), CatBoost (large wheel).

## 2026-11-13 — CI on Linux for every PR; macOS/Windows on demand

- **Chose:** `ci.yml` (ruff + pytest on ubuntu) on every PR and push to main.
  `cross-platform.yml` (macOS + Windows) runs weekly and on manual dispatch.
- **Why:** Private repos get a free monthly Actions allowance; macOS minutes count 10x and
  Windows 2x. Running all three on every PR could exhaust the allowance and start costing money.
- **Rejected:** full OS matrix on every PR.

## 2026-11-13 — Commands are a Python CLI, Makefile is optional

- **Chose:** `uv run landscore <command>` as the primary interface; `Makefile` wraps it.
- **Why:** Windows has no `make` by default. A Python CLI behaves the same everywhere.

## 2026-11-13 — Soil, terrain, and climate normals are "static" features

- **Chose:** Treat soil survey properties, terrain, and 30-year climate normals as
  time-invariant site properties, even though the survey/DEM/normals were produced after 1990.
  Only these three groups may be static; everything vegetation-related must carry a
  measurement year earlier than the abandonment year (enforced in code and tests).
- **Why:** These describe the ground and climate, which regrowth on a field does not change in
  a way that encodes the outcome. The original plan lists them as allowed features.
- **Caveat recorded in METHODS:** soil organic matter in the top layer and survey site index
  could in principle reflect vegetation history; we will check SHAP for suspicious dominance
  and run an ablation without them.
- **Rejected:** dropping soil/terrain (would remove the signal the tool is meant to use).
