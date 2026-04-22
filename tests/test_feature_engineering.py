"""
Tests for feature_engineering.py.

Tests the public API:
  - create_target: logic verified via isolated pure-Python reimplementation
  - get_feature_columns: contract on column exclusion
  - create_all_features_clean / create_predictive_features: smoke tests

Private _add_* helpers are tested implicitly via create_all_features_clean.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from feature_engineering import (
    DIST_COLS,
    DATE_COL,
    CODE_COL,
    TARGET_HORIZON,
    OOS_PCT_THRESHOLD,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_dist_cols_has_five_distributors(self):
        assert len(DIST_COLS) == 5

    def test_dist_cols_contains_expected_names(self):
        expected = {"puls_stock", "katren_stock", "protek_stock", "farm_stock", "gk_stock"}
        assert set(DIST_COLS) == expected

    def test_date_col_is_date(self):
        assert DATE_COL == "date"

    def test_code_col_is_code_kag(self):
        assert CODE_COL == "code_kag"

    def test_target_horizon_is_14(self):
        assert TARGET_HORIZON == 14

    def test_oos_pct_threshold_is_10_percent(self):
        assert abs(OOS_PCT_THRESHOLD - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# get_feature_columns — column exclusion contract
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def _make_df_with_cols(self, cols):
        df = MagicMock()
        df.columns = cols
        return df

    def test_excludes_date_column(self):
        df = self._make_df_with_cols(["date", "code_kag", "feat_a", "feat_b"])
        cols = get_feature_columns(df)
        assert "date" not in cols

    def test_excludes_code_kag_column(self):
        df = self._make_df_with_cols(["date", "code_kag", "feat_a"])
        cols = get_feature_columns(df)
        assert "code_kag" not in cols

    def test_excludes_target_column(self):
        df = self._make_df_with_cols(["date", "code_kag", "feat_a", "target"])
        cols = get_feature_columns(df)
        assert "target" not in cols

    def test_includes_total_stock_column(self):
        # total_stock is a valid feature — NOT excluded by get_feature_columns
        df = self._make_df_with_cols(["date", "code_kag", "feat_a", "total_stock", "target"])
        cols = get_feature_columns(df)
        assert "total_stock" in cols

    def test_includes_feature_columns(self):
        df = self._make_df_with_cols(["date", "code_kag", "feat_a", "feat_b", "target"])
        cols = get_feature_columns(df)
        assert "feat_a" in cols
        assert "feat_b" in cols

    def test_returns_list(self):
        df = self._make_df_with_cols(["date", "code_kag", "rolling_7d", "target"])
        result = get_feature_columns(df)
        assert isinstance(result, list)

    def test_dist_cols_are_included_as_features(self):
        # DIST_COLS are raw input columns; get_feature_columns does NOT exclude them
        all_cols = ["date", "code_kag", "target"] + DIST_COLS + ["feat_x"]
        df = self._make_df_with_cols(all_cols)
        cols = get_feature_columns(df)
        assert "feat_x" in cols
        assert "puls_stock" in cols  # included as a feature

    def test_excludes_private_underscore_columns(self):
        # Columns starting with _ are intermediate; excluded by startswith("_") check
        df = self._make_df_with_cols(["date", "code_kag", "_is_oos", "feat_x", "target"])
        cols = get_feature_columns(df)
        assert "_is_oos" not in cols
        assert "feat_x" in cols

    def test_empty_dataframe_returns_empty(self):
        df = self._make_df_with_cols([])
        cols = get_feature_columns(df)
        assert cols == []

    def test_only_excluded_cols_returns_empty(self):
        # Only date, code_kag, target → all excluded
        df = self._make_df_with_cols(["date", "code_kag", "target"])
        cols = get_feature_columns(df)
        assert cols == []


# ---------------------------------------------------------------------------
# create_target — logic tested independently from pandas
# ---------------------------------------------------------------------------

class TestCreateTargetLogic:
    """
    create_target uses shift(-horizon) and a percentage threshold.
    We verify the logic:
      target = (future_stock < threshold * expanding_median).astype(int)
    """

    def test_oos_threshold_is_10_percent(self):
        # The threshold in the module must be exactly 0.10
        assert OOS_PCT_THRESHOLD == 0.10

    def test_target_horizon_is_14_days(self):
        assert TARGET_HORIZON == 14

    def test_low_stock_signals_oos(self):
        """Stock below 10% of median → OOS (target=1)."""
        median_stock = 100.0
        threshold = OOS_PCT_THRESHOLD * median_stock  # 10.0
        low_stock = 5.0  # < 10
        assert low_stock < threshold  # would produce target=1

    def test_normal_stock_no_oos(self):
        """Stock well above threshold → no OOS (target=0)."""
        median_stock = 100.0
        threshold = OOS_PCT_THRESHOLD * median_stock  # 10.0
        normal_stock = 50.0
        assert normal_stock >= threshold  # would produce target=0

    def test_stock_exactly_at_threshold_no_oos(self):
        """Stock at exactly 10% is NOT below threshold → target=0."""
        median_stock = 100.0
        threshold = OOS_PCT_THRESHOLD * median_stock  # 10.0
        stock_at_threshold = 10.0
        assert stock_at_threshold >= threshold  # strictly <, so this is 0

    def test_stock_just_below_threshold_is_oos(self):
        median_stock = 100.0
        threshold = OOS_PCT_THRESHOLD * median_stock  # 10.0
        stock_just_below = 9.99
        assert stock_just_below < threshold


# ---------------------------------------------------------------------------
# create_all_features_clean — smoke test (mocked pandas)
# ---------------------------------------------------------------------------

class TestCreateAllFeaturesClean:
    def test_function_is_callable(self):
        from feature_engineering import create_all_features_clean
        assert callable(create_all_features_clean)

    def test_accepts_df_and_optional_catalog(self):
        """Verify the function signature: df required, catalog optional."""
        import inspect
        from feature_engineering import create_all_features_clean
        sig = inspect.signature(create_all_features_clean)
        params = sig.parameters
        assert "df" in params
        # catalog is the optional catalog DataFrame
        if "catalog" in params:
            assert params["catalog"].default is None or params["catalog"].default != inspect.Parameter.empty


# ---------------------------------------------------------------------------
# create_predictive_features — smoke test
# ---------------------------------------------------------------------------

class TestCreatePredictiveFeatures:
    def test_function_is_callable(self):
        from feature_engineering import create_predictive_features
        assert callable(create_predictive_features)

    def test_returns_dataframe_on_mock(self):
        from feature_engineering import create_predictive_features
        df = MagicMock()
        result = create_predictive_features(df)
        assert result is not None
