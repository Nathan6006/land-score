import numpy as np
import pandas as pd
import pytest

from landscore.models import evaluate as ev
from landscore.models.results_md import verdict
from landscore.models.zoo import fresh, model_specs, predict


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(
        {
            "dist_forest_m": rng.exponential(300, n),
            "seedling_mortality": rng.integers(0, 3, n).astype(float),
            "slope_deg": rng.uniform(0, 10, n),
            "x": rng.uniform(0, 40_000, n),
            "y": rng.uniform(0, 40_000, n),
            "county": rng.choice(["A", "B", "C"], n),
        }
    )
    logit = 1.5 - 0.006 * d["dist_forest_m"] - 0.5 * d["seedling_mortality"]
    d["label_binary"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    d["canopy_outcome"] = np.clip(60 * (1 / (1 + np.exp(-logit))) + rng.normal(0, 10, n), 0, 100)
    return d


def test_block_splits_keep_blocks_together():
    d = synthetic()
    ids = ev.block_ids(d.x, d.y, 5000)
    for tr, te in ev.block_splits(d.x, d.y, 5000, 5, 0):
        assert not set(ids[tr]) & set(ids[te])


def test_county_splits_need_two_counties():
    with pytest.raises(ValueError):
        list(ev.county_splits(np.array(["A"] * 10)))
    folds = list(ev.county_splits(np.array(["A", "B", "C"] * 5)))
    assert len(folds) == 3


def test_models_learn_signal():
    d = synthetic()
    specs = {s.name: s for s in model_specs(list(d.columns))}
    assert {"baseline_distance_to_forest", "random_forest", "gradient_boosting"} <= set(specs)
    tr, te = np.arange(2000), np.arange(2000, 3000)
    for name in ("baseline_distance_to_forest", "gradient_boosting"):
        s = specs[name]
        X = d[list(s.features)].to_numpy(float)
        est = fresh(s, "binary", 0).fit(X[tr], d.label_binary[tr])
        m = ev.binary_metrics(d.label_binary[te], predict(est, X[te], "binary"))
        assert m["auc"] > 0.65, name


def test_verdict_says_plainly_when_baselines_win():
    res = pd.DataFrame(
        [
            {"task": "binary", "scheme": "county", "model": "b", "kind": "baseline", "auc": 0.70},
            {"task": "binary", "scheme": "county", "model": "m", "kind": "main", "auc": 0.705},
        ]
    )
    assert "do not meaningfully beat" in verdict(res, "county")


def test_calibration_table_bins():
    t = ev.calibration_table(np.r_[np.zeros(50), np.ones(50)], np.linspace(0, 1, 100), 5)
    assert len(t) == 5 and t["observed"].iloc[-1] == 1
