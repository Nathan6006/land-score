"""Model definitions: simple baselines first, then the main tree models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from landscore.features.registry import FEATURE_NAMES


def _log_dist(x):
    return np.log1p(np.asarray(x, dtype=float))


@dataclass
class ModelSpec:
    name: str
    features: tuple[str, ...]
    make: Callable[[str, int], BaseEstimator]  # (task, seed) -> estimator
    kind: str  # "baseline" or "main"
    note: str = ""


def _linear(task: str, seed: int, transform=None):
    steps = [SimpleImputer(strategy="median")]
    if transform is not None:
        steps.append(FunctionTransformer(transform))
    steps.append(StandardScaler())
    est = (
        LogisticRegression(max_iter=2000, random_state=seed)
        if task == "binary"
        else LinearRegression()
    )
    return make_pipeline(*steps, est)


def _rf(task: str, seed: int):
    kw = dict(
        n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=seed
    )
    est = RandomForestClassifier(**kw) if task == "binary" else RandomForestRegressor(**kw)
    return make_pipeline(SimpleImputer(strategy="median"), est)


def _hgb(task: str, seed: int):
    kw = dict(
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    return (
        HistGradientBoostingClassifier(**kw)
        if task == "binary"
        else HistGradientBoostingRegressor(**kw)
    )


def model_specs(available: list[str] | tuple[str, ...] | None = None) -> list[ModelSpec]:
    """All models, restricted to features present in the table."""
    avail = tuple(f for f in FEATURE_NAMES if available is None or f in available)
    specs = [
        ModelSpec(
            "baseline_seedling_mortality",
            ("seedling_mortality",),
            lambda t, s: _linear(t, s),
            "baseline",
            "Soil survey seedling mortality rating alone",
        ),
        ModelSpec(
            "baseline_distance_to_forest",
            ("dist_forest_m",),
            lambda t, s: _linear(t, s, _log_dist),
            "baseline",
            "Distance to nearest forest at the start year alone",
        ),
        ModelSpec(
            "logistic_all",
            avail,
            lambda t, s: _linear(t, s),
            "baseline",
            "Linear/logistic on all features",
        ),
        ModelSpec("random_forest", avail, _rf, "main"),
        ModelSpec("gradient_boosting", avail, _hgb, "main"),
    ]
    return [m for m in specs if m.features and all(f in avail for f in m.features)]


def predict(est, X, task: str) -> np.ndarray:
    if task == "binary":
        return est.predict_proba(X)[:, 1]
    return est.predict(X)


def fresh(spec: ModelSpec, task: str, seed: int):
    return clone(spec.make(task, seed))
