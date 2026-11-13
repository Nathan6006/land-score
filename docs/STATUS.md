# Status (handoff log)

Newest session at the top. Update at the end of every work session.

## 2026-11-13 — Claude Code session 2 (Nathan's machine)

**Works now**
- All nine counties fetched (0.9 GB raw, 1.5 GB with derived layers), labeled, and modeled.
- 49,563 labeled pixels, 47.9% positive (canopy >= 40% after 15 years).
- Leave-one-county-out AUC: random forest 0.671, gradient boosting 0.659, logistic 0.646,
  distance to forest 0.557, seedling mortality rating 0.493. Random split: 0.747 / 0.756.
  Every held-out county lands between 0.65 and 0.70. See `docs/RESULTS.md`.
- `landscore sensitivity` writes `docs/SENSITIVITY.md` (lag/threshold counts, edge-noise subsets).
- `docs/METHODS.md` written. Quick-look and label maps for Geauga checked by eye: fine.

**Not done / not verified**
- Quick-look maps for the other eight counties have not been generated or looked at.
- Minimum patch size for labels is an open decision (see DECISIONS, last entry).
- Species table still a draft (issue #21). Lidar check not built (issue #17).
- Human checks (issues #25, #26) not started.
- `pipeline.yml` untested (needs to be on `main` first).

**Blocked**
- PRs #27 and the phase-6 PR are open; Claude is not permitted to merge. Merge #27 first.

**Next**
1. Merge the two PRs in order.
2. Decide on the minimum patch size (look at a sample of single-pixel labels on imagery).
3. Issue #21: forester review of the species table; issue #25: blind ranking protocol.
4. Issue #17: lidar canopy-height check.

## 2026-11-13 — Claude Code session 1 (Nathan's machine)

**Works now (Geauga pilot, end to end)**
- `fetch` → `quicklook` → `labels` → `features` → `train` → `report` all run on real data.
- Pilot data is 83 MB on disk (not GBs): land cover 17, canopy 58, DEM 5, soil 4, climate 3.
  Fetch takes ~8 minutes, mostly canopy and DEM.
- Labels: 5,545 pixels (2,894 positive, 2,651 negative at canopy >= 40%).
- Models (5 km block holdout, NOT county holdout): random forest AUC 0.692, gradient
  boosting 0.684, logistic 0.669, distance-to-forest 0.581, seedling mortality rating 0.469.
  Random split: gradient boosting 0.761. See `docs/RESULTS.md`.
- CI green on Linux; macOS and Windows install + tests verified once via `cross-platform`.

**Not done / not verified**
- County holdout: needs at least two counties (Phase 6).
- Nobody has looked at the quick-look maps or label map by eye yet
  (`outputs/quicklook/`, `outputs/labels/Geauga_labels.png`).
- Species table is a draft from general references; needs forester review (issue #21).
- Lidar check not built (issue #17). METHODS.md is a stub (issue #24).
- `pipeline.yml` (run the pipeline on a GitHub runner) is untested; it can only be dispatched
  once it is on `main`.

**Blocked**
- PR #27 is open as a draft. Claude was not permitted to merge it; Nathan or Karan should
  review and merge.
- Nathan's laptop is nearly out of disk (~1 GB free). Adding the other eight counties needs
  ~1 GB. Free space, set `LANDSCORE_DATA_DIR` to another drive, or run on Karan's machine.

**Next**
1. Merge PR #27.
2. Issue #18: fetch Cuyahoga, Lorain, Ashtabula, rerun labels/features/train to get
   leave-one-county-out numbers.
3. Look at the label map; decide whether to drop pixels within one pixel of forest
   (edge noise) and log it.
4. Issue #9: label sensitivity (lag, threshold) across counties.
