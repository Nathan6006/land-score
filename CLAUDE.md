# Land Score — shared context for Claude Code sessions

**Read `docs/STATUS.md` first.** It says what was done last, what is in progress, and what to
do next. Update it at the end of every work session.

## Goal

Land Score rates an offered plot of land in Northeast Ohio for how easily trees should
establish there. It serves Roots of Tomorrow, a student-run nonprofit that plants native trees
on private land offered by landowners. The tool evaluates a given plot; it does not search for
land.

Input: a plot polygon (GeoJSON). Output:
1. **Establishment score** (trained model): 0–100 per 30 m pixel, a plot summary, and a map.
2. **Species fit** (lookup table, not a model): native trees suited to soil and drainage.
3. **Planning flags** (simple rules): wet soil, heavy competing vegetation, deer pressure, steep slope.

The score means *site potential*. Never present it as a promise that planted trees will
survive; survival depends on care after planting (weed control, deer protection). The caveat
text lives in `landscore.SITE_POTENTIAL_CAVEAT` and must appear in every report.

Method: no public dataset links planted-tree outcomes to Ohio locations, so the model learns
from a stand-in — former farmland (Annual NLCD crop/pasture, abandoned 1990–2005) and how much
canopy (USFS TCC) developed a fixed number of years later. Details: `docs/METHODS.md`.

## Rules that must not change

1. The trained model is the core. **No LLM calls anywhere in the scoring pipeline.**
2. **No leakage.** Never use a feature measured after regrowth started. Recent imagery or
   embeddings may describe current vegetation for the *flags* only, never as model features.
   Enforced by `landscore.features.registry` and `tests/test_leakage.py`.
3. **Always report county-holdout results** next to any accuracy claim.
4. **Never overstate.** If the model does not beat the simple baselines, say so plainly.

Anything else may change. Log every scope/design change in `docs/DECISIONS.md` (date, what,
why, what was rejected). Verify dataset names, URLs, and years before relying on them.

## Commands

```
uv sync --locked                      # install everything (one command)
uv run pytest                         # tests
uv run ruff check . && uv run ruff format --check .   # lint
uv run landscore fetch --county Geauga     # download + clip inputs, writes data/manifest.json
uv run landscore quicklook --county Geauga # PNG of each layer -> outputs/quicklook/
uv run landscore labels --county Geauga    # label rasters + counts -> data/derived/
uv run landscore features --county Geauga  # sampled training table (Parquet)
uv run landscore train                     # baselines + models -> docs/RESULTS.md, outputs/
uv run landscore report examples/plot_a.geojson   # plot report -> outputs/reports/
```
`make <target>` wraps these on macOS/Linux.

## Folder layout

```
config/study.yaml     study area, years, label definitions, sampling (single source of truth)
src/landscore/
  data/               download, clip to grid, manifest (source URL, date, sha256)
  labels/             abandonment detection, canopy outcome
  features/           registry (leakage rule), soil, terrain, climate, landscape, table
  models/             baselines, training, evaluation
  report/             per-plot score, species lookup, flags
notebooks/            exploration only — no pipeline logic
tests/                pytest; label logic + leakage rule must stay covered
docs/                 STATUS, DECISIONS, SETUP, METHODS, RESULTS
data/  outputs/       gitignored
```

## Conventions

- Python 3.12, managed by **uv**; dependencies pinned in `uv.lock`. Add packages with
  `uv add <pkg>` and commit the lockfile. No system package managers (no brew/apt/conda).
- Hardware: 16 GB laptops under memory pressure. **Never load a full raster.** Read windows,
  keep peak memory under ~6 GB, sample pixels into Parquet. Check free disk before downloads
  and state expected size. Ask before any single download >5 GB or total data >30 GB.
- Grid: EPSG:5070, 30 m, aligned to Annual NLCD. All layers are clipped to this grid per county.
- Random seeds come from `config/study.yaml`. No hidden randomness.
- Every downloaded file is recorded in `data/manifest.json` (URL, date, sha256, bytes).
- Data sources must not need an account. Google Earth Engine is optional only.
- Git: feature branches + pull requests. Never force-push. Never commit directly to `main`.
  CI (ruff + pytest) must pass before merge.
- Ruff for lint and format (line length 100). Type hints on public functions.
- Do not type passwords, create accounts, or accept terms on anyone's behalf.
