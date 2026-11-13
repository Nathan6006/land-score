"""Figures for RESULTS.md: model comparison, calibration, SHAP importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from landscore.config import REPO_ROOT  # noqa: E402
from landscore.models import evaluate as ev  # noqa: E402
from landscore.models.zoo import fresh, model_specs  # noqa: E402

FIG_DIR = REPO_ROOT / "docs" / "figures"


def write_figures(df, res, oof, headline, best, seed, shap=True) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = {"comparison": _comparison(res), "calibration": _calibration(df, oof, headline)}
    if shap:
        try:
            figs["shap"] = _shap(df, best, seed)
        except Exception as e:  # SHAP is diagnostic; never block results on it
            print(f"SHAP skipped: {e}")
    return figs


def _save(fig, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(path).relative_to(REPO_ROOT / "docs").as_posix()


def _comparison(res: pd.DataFrame) -> str:
    r = res[res.task == "binary"].pivot(index="model", columns="scheme", values="auc")
    order = [m for m in res["model"].unique() if m in r.index]
    r = r.loc[order]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cols = [c for c in ("county", "block", "random") if c in r.columns]
    w = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.barh(np.arange(len(r)) + i * w, r[c], height=w, label=c)
    ax.set_yticks(np.arange(len(r)) + w * (len(cols) - 1) / 2, r.index)
    ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("ROC AUC (canopy >= threshold)")
    ax.legend(title="validation", loc="lower right", fontsize=8)
    ax.invert_yaxis()
    return _save(fig, "model_comparison.png")


def _calibration(df, oof, headline) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    y = df["label_binary"].to_numpy()
    for (task, scheme, name), p in oof.items():
        if (
            task != "binary"
            or scheme != headline
            or name not in ("random_forest", "gradient_boosting", "logistic_all")
        ):
            continue
        t = ev.calibration_table(y, p)
        ax.plot(t["mean_pred"], t["observed"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraction with canopy >= threshold")
    ax.set_title(f"Calibration ({headline} holdout)")
    ax.legend(fontsize=8)
    return _save(fig, "calibration.png")


def _shap(df, best, seed, n=3000) -> str:
    import shap

    spec = next(s for s in model_specs(list(df.columns)) if s.name == best)
    X = df[list(spec.features)]
    est = fresh(spec, "binary", seed).fit(X.to_numpy(float), df["label_binary"])
    sample = X.sample(min(n, len(X)), random_state=seed)
    model = est[-1] if hasattr(est, "steps") else est
    Xs = (
        est[:-1].transform(sample.to_numpy(float))
        if hasattr(est, "steps")
        else sample.to_numpy(float)
    )
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes) -> positive class
        sv = sv[..., 1]
    plt.figure()
    shap.summary_plot(sv, pd.DataFrame(Xs, columns=sample.columns), show=False, max_display=15)
    fig = plt.gcf()
    pd.Series(np.abs(sv).mean(0), index=sample.columns).sort_values(ascending=False).to_csv(
        FIG_DIR / "shap_mean_abs.csv", header=["mean_abs_shap"]
    )
    return _save(fig, "shap_summary.png")
