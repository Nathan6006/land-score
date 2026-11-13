"""The leakage rule: no feature may be measured on or after the label's start year."""

import dataclasses

import pandas as pd
import pytest

from landscore.features.registry import (
    FEATURES,
    Feature,
    LeakageError,
    check_registry,
    check_table,
)


def test_shipped_registry_is_clean():
    check_registry()


def test_dynamic_feature_dated_at_or_after_start_fails():
    bad = Feature("canopy_now", "landscape", "dynamic", "x", offset_years=0, year_column="y")
    with pytest.raises(LeakageError):
        check_registry((bad,))
    later = dataclasses.replace(bad, offset_years=5)
    with pytest.raises(LeakageError):
        check_registry((later,))


def test_vegetation_cannot_pretend_to_be_static():
    with pytest.raises(LeakageError):
        check_registry((Feature("ndvi_2020", "landscape", "static", "x"),))
    with pytest.raises(LeakageError):
        check_registry((Feature("emb", "embedding", "static", "x"),))
    with pytest.raises(LeakageError):
        check_registry((Feature("img", "imagery", "dynamic", "x", -3, "y"),))


def _table(landscape_years, abandon_years):
    return pd.DataFrame(
        {
            "abandon_year": abandon_years,
            "landscape_year": landscape_years,
            "dist_forest_m": [10.0] * len(abandon_years),
            "slope_deg": [2.0] * len(abandon_years),
        }
    )


def test_table_rows_before_start_pass():
    check_table(_table([1994, 1999], [1995, 2000]))


def test_table_row_dated_after_start_fails():
    with pytest.raises(LeakageError):
        check_table(_table([1994, 2003], [1995, 2000]))


def test_table_row_dated_same_year_fails():
    with pytest.raises(LeakageError):
        check_table(_table([1995], [1995]))


def test_table_missing_year_column_fails():
    t = _table([1994], [1995]).drop(columns="landscape_year")
    with pytest.raises(LeakageError):
        check_table(t)


def test_every_dynamic_feature_is_before_start():
    for f in FEATURES:
        if f.kind == "dynamic":
            assert f.offset_years < 0, f.name
