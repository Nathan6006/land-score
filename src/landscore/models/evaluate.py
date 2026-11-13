"""Splits and metrics. County holdout is the headline; random split is shown for contrast."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold

Split = tuple[np.ndarray, np.ndarray]


def random_splits(n: int, n_folds: int, seed: int) -> Iterator[Split]:
    yield from KFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(n))


def block_ids(x: np.ndarray, y: np.ndarray, block_m: float) -> np.ndarray:
    bx = np.floor(np.asarray(x) / block_m).astype(np.int64)
    by = np.floor(np.asarray(y) / block_m).astype(np.int64)
    return bx * 1_000_003 + by


def block_splits(x, y, block_m: float, n_folds: int, seed: int) -> Iterator[Split]:
    """Spatial blocks assigned to folds at random (seeded), so neighbours stay together."""
    ids = block_ids(x, y, block_m)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % n_folds, strict=True))
    folds = np.array([fold_of[i] for i in ids])
    for k in range(n_folds):
        te = np.flatnonzero(folds == k)
        if te.size and te.size < len(ids):  # fewer blocks than folds: skip empty folds
            yield np.flatnonzero(folds != k), te


def county_splits(county: np.ndarray) -> Iterator[Split]:
    """Leave one county out. Needs at least two counties."""
    county = np.asarray(county)
    groups = np.unique(county)
    if len(groups) < 2:
        raise ValueError("county holdout needs at least two counties")
    yield from GroupKFold(len(groups)).split(np.zeros(len(county)), groups=county)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "base_rate": float(y.mean())}
    if len(np.unique(y)) < 2:
        return out | {
            "auc": np.nan,
            "ap": np.nan,
            "brier": float(brier_score_loss(y, p)),
            "log_loss": np.nan,
        }
    return out | {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_skill": float(
            1 - brier_score_loss(y, p) / brier_score_loss(y, np.full_like(p, y.mean()))
        ),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "n": int(len(y)),
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 else np.nan,
    }


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed rate vs mean predicted probability in equal-count bins."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
