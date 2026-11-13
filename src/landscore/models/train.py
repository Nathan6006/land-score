"""Train baselines and models, cross-validate, and write docs/RESULTS.md.

Validation schemes, reported side by side:
* county  - leave one county out (headline; needs >= 2 counties)
* block   - 5 km spatial blocks (used for the single-county pilot; NOT county holdout)
* random  - random K-fold (optimistic; shown so the gap is visible)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from landscore.config import REPO_ROOT, data_dir, load_config, outputs_dir
from landscore.features.registry import FEATURE_NAMES, check_table
from landscore.models import evaluate as ev
from landscore.models.zoo import fresh, model_specs, predict

TASKS = {"binary": "label_binary", "regression": "canopy_outcome"}


def load_tables(counties: list[str] | None = None) -> pd.DataFrame:
    paths = sorted((data_dir() / "derived").glob("*/training.parquet"))
    if counties:
        paths = [p for p in paths if p.parent.name in counties]
    if not paths:
        raise FileNotFoundError("no training tables; run `landscore features` first")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    check_table(df)
    return df


def schemes(df: pd.DataFrame, cfg: dict) -> dict[str, list]:
    v = cfg["validation"]
    out = {
        "random": list(ev.random_splits(len(df), v["n_folds"], v["seed"])),
        "block": list(
            ev.block_splits(df["x"], df["y"], v["block_size_m"], v["n_folds"], v["seed"])
        ),
    }
    if df["county"].nunique() >= 2:
        out = {"county": list(ev.county_splits(df["county"].to_numpy()))} | out
    return out


def cross_validate(df: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    rows, oof = [], {}
    specs = model_specs([c for c in FEATURE_NAMES if c in df.columns and df[c].notna().any()])
    for task, target in TASKS.items():
        y = df[target].to_numpy()
        for scheme, splits in schemes(df, cfg).items():
            for spec in specs:
                X = df[list(spec.features)].to_numpy(dtype=float)
                pred = np.full(len(df), np.nan)
                for tr, te in splits:
                    est = fresh(spec, task, seed).fit(X[tr], y[tr])
                    pred[te] = predict(est, X[te], task)
                m = (
                    ev.binary_metrics(y, pred)
                    if task == "binary"
                    else ev.regression_metrics(y, pred)
                )
                rows.append(
                    {"task": task, "scheme": scheme, "model": spec.name, "kind": spec.kind} | m
                )
                oof[(task, scheme, spec.name)] = pred
                print(f"{task:10s} {scheme:7s} {spec.name:30s} " + _fmt(m), flush=True)
    return pd.DataFrame(rows), oof


def _fmt(m: dict) -> str:
    keys = ["auc", "brier_skill"] if "auc" in m else ["r2", "spearman"]
    return " ".join(f"{k}={m.get(k, float('nan')):.3f}" for k in keys)


def fit_final(df: pd.DataFrame, seed: int, model_name: str, out: Path) -> Path:
    spec = next(
        s
        for s in model_specs([c for c in FEATURE_NAMES if c in df.columns])
        if s.name == model_name
    )
    est = fresh(spec, "binary", seed).fit(
        df[list(spec.features)].to_numpy(float), df["label_binary"]
    )
    path = out / "model.joblib"
    joblib.dump(
        {"model": est, "features": list(spec.features), "name": spec.name, "task": "binary"}, path
    )
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="landscore train")
    p.add_argument("--counties", nargs="*", help="limit to these counties (default: all built)")
    p.add_argument("--no-shap", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config()
    seed = cfg["sampling"]["seed"]
    df = load_tables(a.counties)
    df = df[df["label_binary"] >= 0].reset_index(drop=True)
    print(f"{len(df):,} labeled rows from {df['county'].nunique()} county(ies)")
    res, oof = cross_validate(df, cfg, seed)
    out = outputs_dir() / "models"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "cv_results.csv", index=False)
    headline = "county" if "county" in set(res["scheme"]) else "block"
    best = _best_main(res, headline)
    model_path = fit_final(df, seed, best, out)
    from landscore.models.figures import write_figures

    figs = write_figures(df, res, oof, headline, best, seed, shap=not a.no_shap)
    from landscore.models.results_md import write_results

    write_results(df, res, headline, best, figs, REPO_ROOT / "docs" / "RESULTS.md")
    (out / "summary.json").write_text(
        json.dumps({"headline": headline, "best": best, "model": str(model_path)}, indent=2)
    )
    print(f"best main model under {headline} holdout: {best}; saved {model_path}")
    return 0


def _best_main(res: pd.DataFrame, scheme: str) -> str:
    r = res[(res.task == "binary") & (res.scheme == scheme) & (res.kind == "main")]
    return str(r.sort_values("auc", ascending=False).iloc[0]["model"])


if __name__ == "__main__":
    raise SystemExit(main())
