"""Load the study configuration and per-machine settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "study.yaml"


@dataclass(frozen=True)
class LabelConfig:
    farm_classes: tuple[int, ...]
    converted_classes: tuple[int, ...]
    forest_classes: tuple[int, ...]
    abandon_window: tuple[int, int]
    min_farm_years: int
    min_fallow_years: int
    outcome_lag: int
    max_start_canopy: int
    binary_threshold: int

    @classmethod
    def from_dict(cls, d: dict) -> LabelConfig:
        return cls(
            farm_classes=tuple(d["farm_classes"]),
            converted_classes=tuple(d["converted_classes"]),
            forest_classes=tuple(d["forest_classes"]),
            abandon_window=tuple(d["abandon_window"]),
            min_farm_years=int(d["min_farm_years"]),
            min_fallow_years=int(d["min_fallow_years"]),
            outcome_lag=int(d["outcome_lag"]),
            max_start_canopy=int(d["max_start_canopy"]),
            binary_threshold=int(d["binary_threshold"]),
        )


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict:
    """Return the raw study config as a dict."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_config(cfg: dict | None = None) -> LabelConfig:
    return LabelConfig.from_dict((cfg or load_config())["labels"])


def county_fips(cfg: dict | None = None) -> dict[str, str]:
    return dict((cfg or load_config())["study_area"]["counties"])


def data_dir() -> Path:
    """Data folder. Set LANDSCORE_DATA_DIR in .env to keep data outside the repo."""
    load_dotenv(REPO_ROOT / ".env")
    p = Path(os.environ.get("LANDSCORE_DATA_DIR", REPO_ROOT / "data")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir() -> Path:
    load_dotenv(REPO_ROOT / ".env")
    p = Path(os.environ.get("LANDSCORE_OUTPUTS_DIR", REPO_ROOT / "outputs")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p
