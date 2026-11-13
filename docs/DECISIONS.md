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

## 2026-11-13 — Annual NLCD read from a public mirror, MRLC WCS as fallback

- **Chose:** Windowed reads of the Annual NLCD C1V1 COGs from the source.coop mirror
  (`data.source.coop/kylebarron/usgs-landcover/...`), 1985–2024. If the mirror fails, the
  fetcher falls back to MRLC's own WCS (`www.mrlc.gov/geoserver/...`, version 1.0.0).
- **Why:** The official USGS bucket `s3://usgs-landcover` is requester-pays (needs an AWS
  account and costs money). The official zip bundles are ~1.4 GB per year (56 GB total).
- **Checked:** Mirror and MRLC WCS matched pixel for pixel for Geauga in 1985, 1990, 2005, 2024.
  The mirror takes ~20 s per county; the WCS takes ~5 min.
- **Risk:** The mirror is a third-party copy. The WCS fallback and the checksum manifest cover this.

## 2026-11-13 — Canopy: USFS Science TCC v2025-6 (1985–2025), not v2023-5

- **Chose:** Science TCC v2025-6, read as county windows straight out of the yearly zip files
  on the USFS raster gateway (`/vsizip//vsicurl/`), since the zips store an uncompressed COG.
- **Why:** v2023-5 CONUS files were withdrawn (HTTP 404). v2025-6 adds 2024–2025. Each zip is
  ~6 GB but only the county window (~1.5 MB per year) is transferred.
- **Rejected:** NLCD TCC (masks agriculture using land cover, which would be circular with our
  labels); the ArcGIS ImageServer for Science TCC (needs a login token).
- **Note:** value 254 appears in a few 1985–1990 pixels; anything above 100 is treated as no data.

## 2026-11-13 — Soil tables from USDA Soil Data Access, not Planetary Computer

- **Chose:** Map unit key raster from Planetary Computer gNATSGO (July 2020, 10 m,
  majority-resampled to 30 m). Attribute tables from Soil Data Access (SDA, current SSURGO).
- **Why:** Planetary Computer's `cointerp` and `chorizon` tables hold no Ohio rows.
- **Risk:** raster (2020) and tables (current) are different vintages. In Geauga 403 of 435
  map units matched; the rest are water/urban/unmatched and become missing values.
- **Seedling mortality:** uses the national rule "FOR - Potential Seedling Mortality", fuzzy
  rating 0 (low) to 1 (high), dominant component.

## 2026-11-13 — Climate normals: NOAA nClimGrid (5 km) instead of PRISM (800 m)

- **Chose:** NOAA 1991–2020 annual normals from Planetary Computer, bilinear to the grid.
- **Why:** PRISM blocks a client that downloads the same file twice in 24 hours, which breaks
  repeatable fetches on two machines and CI. PRISM's web service no longer serves normals.
- **Cost:** coarser grid. Within a county, climate varies little, so this is a weak feature
  until several counties are included.

## 2026-11-13 — Label definition widened after the pilot

- **Found:** Farmland in Geauga is very persistent in Annual NLCD. Of pixels farmed in 1990,
  77% were still pasture/hay in 2024 and ~10% had become developed. The original window
  (1990–2005) gave only 4,155 labeled pixels.
- **Chose:** `abandon_window` 1988–2009 (the widest the data allow with a 15-year lag and
  three farmed years before). Gives 5,545 labeled pixels in Geauga.
- **Chose:** `binary_threshold` 40% canopy (the median outcome), giving 52% positive. At 30%
  the classes were 69/31.
- **Rejected:** 10-year lag (more pixels, 5,890, but a shorter regrowth period); dropping the
  start-canopy rule (adds ~1,000 pixels that were already partly wooded).
- **Consequence:** one county cannot reach the planned 200,000-pixel sample. All labeled
  pixels are used. `sampling.n_pixels` now acts as a per-county cap.
- **Caveat:** labeled pixels sit close to forest (mean 47 m to the nearest forest pixel).
  Many are field edges, where a class change can be mapping noise as well as real regrowth.

## 2026-11-13 — Pilot uses spatial-block holdout; county holdout needs more counties

- With one county there is no county to hold out. RESULTS.md reports 5 km block holdout and
  says so in bold. County holdout starts automatically once two or more counties are built.

## 2026-11-13 — Lidar check deferred

- 2019–2020 Ohio statewide lidar covers Geauga, but canopy height needs the point cloud.
  Planetary Computer has only a western sliver; USGS LAZ downloads were rate-limited and
  reading EPT needs PDAL (not pip-installable). Left as issue #17.

## 2026-11-13 — main is not branch-protected

- GitHub branch protection is unavailable for private repos on the free plan. "No direct
  commits to main" is a convention enforced by habit and CI, not by GitHub.

## 2026-11-13 — All nine counties built; edge-noise filter left as an open choice

- **Done:** Cuyahoga, Lorain, Ashtabula, Lake, Portage, Summit, Trumbull, Medina added to
  Geauga. 49,563 labeled pixels. Leave-one-county-out is now the headline validation.
- **Found (docs/SENSITIVITY.md):** keeping only labeled patches of 4+ pixels raises
  county-holdout AUC for random forest from 0.671 to 0.706 (9+ pixels: 0.717), with 21,151
  pixels left. Pixels not touching forest still give 0.681. Single-pixel labels look like
  mapping noise, and the signal is not only a forest-edge artifact.
- **Not changed yet:** the shipped model still trains on all labeled pixels. Adopting a
  minimum patch size after seeing test scores is a form of tuning, so it should be a
  deliberate choice by Nathan and Karan (and ideally justified by looking at imagery of a
  sample of single-pixel labels), not something slipped in.
- **Lag and threshold:** 15-year lag with a 40% threshold stays. It is the most balanced
  (48% positive); a 10-year lag gives more pixels (56,390) but only 35% positive at 40%.
- **Binary vs regression:** binary is used for the score. Regression under county holdout is
  weak (random forest R² 0.13), while the binary ranking is more stable across counties
  (AUC 0.65–0.70 in every held-out county).
