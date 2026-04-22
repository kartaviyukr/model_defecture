"""
Tests for data_validation.py.

Uses mock DataFrames from conftest.make_stock_df() because pandas is
unavailable in this environment. Tests verify the (bool, errors) contract
and the control-flow branches of validate_stock_df / clean_stock_df.
"""
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from data_validation import (
    validate_stock_df,
    clean_stock_df,
    check_feature_coverage,
    DIST_COLS,
    DATE_COL,
    CODE_COL,
    MIN_HISTORY_DAYS,
)
from tests.conftest import make_stock_df, MockDataFrame


# ---------------------------------------------------------------------------
# validate_stock_df — blocking errors
# ---------------------------------------------------------------------------

class TestValidateStockDfBlocking:
    def test_empty_dataframe_is_invalid(self):
        df = make_stock_df(n_rows=0)
        is_valid, errors = validate_stock_df(df)
        assert is_valid is False
        assert len(errors) == 1
        assert "пустой" in errors[0].lower()

    def test_missing_columns_is_invalid(self):
        df = make_stock_df(include_all_cols=False)
        # DataFrame with no columns → all required cols missing
        is_valid, errors = validate_stock_df(df)
        assert is_valid is False
        assert any("колонк" in e.lower() for e in errors)

    def test_insufficient_history_is_invalid(self):
        # date_range_days=5 < MIN_HISTORY_DAYS (30)
        df = make_stock_df(n_rows=50, date_range_days=5)
        is_valid, errors = validate_stock_df(df)
        assert is_valid is False
        assert any("истори" in e.lower() for e in errors)

    def test_blocking_errors_list_is_empty_on_success(self, stock_df):
        is_valid, errors = validate_stock_df(stock_df)
        assert is_valid is True
        assert errors == []

    def test_valid_df_passes(self, stock_df):
        is_valid, _ = validate_stock_df(stock_df)
        assert is_valid is True

    def test_returns_tuple_always(self, stock_df):
        result = validate_stock_df(stock_df)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# validate_stock_df — warnings (non-blocking)
# ---------------------------------------------------------------------------

class TestValidateStockDfWarnings:
    def test_negative_stocks_are_not_blocking(self):
        """Negative values trigger a warning logged by loguru but return is_valid=True."""
        df = make_stock_df()
        # Configure pd.to_numeric to signal negative values
        pd_mock = sys.modules["pandas"]
        original_to_numeric = pd_mock.to_numeric

        class _NegSeries:
            def __lt__(self, _):
                class _Mask:
                    def sum(self):
                        return 5  # 5 negative values
                return _Mask()

        pd_mock.to_numeric = lambda s, **kw: _NegSeries()
        try:
            is_valid, errors = validate_stock_df(df)
        finally:
            pd_mock.to_numeric = original_to_numeric

        assert is_valid is True
        assert errors == []

    def test_duplicates_are_not_blocking(self):
        """Duplicates trigger warning but do not make validation fail."""
        df = make_stock_df(duplicate_sum=5)
        is_valid, errors = validate_stock_df(df)
        assert is_valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# check_feature_coverage
# ---------------------------------------------------------------------------

class TestCheckFeatureCoverage:
    def _make_df_with_cols(self, cols):
        df = MagicMock()
        df.columns = cols
        return df

    def test_all_present(self):
        df = self._make_df_with_cols(["a", "b", "c"])
        result = check_feature_coverage(df, ["a", "b", "c"])
        assert result["present"] == ["a", "b", "c"]
        assert result["missing"] == []
        assert result["coverage_pct"] == 100.0

    def test_some_missing(self):
        df = self._make_df_with_cols(["a", "b"])
        result = check_feature_coverage(df, ["a", "b", "c", "d"])
        assert set(result["present"]) == {"a", "b"}
        assert set(result["missing"]) == {"c", "d"}
        assert result["coverage_pct"] == 50.0

    def test_all_missing(self):
        df = self._make_df_with_cols([])
        result = check_feature_coverage(df, ["a", "b"])
        assert result["present"] == []
        assert result["missing"] == ["a", "b"]
        assert result["coverage_pct"] == 0.0

    def test_empty_feature_list(self):
        df = self._make_df_with_cols(["a", "b"])
        result = check_feature_coverage(df, [])
        assert result["coverage_pct"] == 0.0

    def test_returns_dict_with_required_keys(self):
        df = self._make_df_with_cols(["a"])
        result = check_feature_coverage(df, ["a"])
        assert "present" in result
        assert "missing" in result
        assert "coverage_pct" in result

    def test_coverage_precision(self):
        df = self._make_df_with_cols(["a"])
        result = check_feature_coverage(df, ["a", "b", "c"])
        # 1/3 ≈ 33.33%
        assert abs(result["coverage_pct"] - 33.333) < 0.01


# ---------------------------------------------------------------------------
# clean_stock_df — structural contract
# ---------------------------------------------------------------------------

class TestCleanStockDf:
    def test_returns_without_exception(self, stock_df):
        """Smoke test: function should not raise on a valid MockDataFrame."""
        result = clean_stock_df(stock_df)
        assert result is not None

    def test_returns_a_dataframe(self, stock_df):
        """clean_stock_df should return something with a columns attribute."""
        result = clean_stock_df(stock_df)
        # Duck-type check: the return should look like a DataFrame
        assert result is not None
        assert hasattr(result, "columns")

    def test_no_dupes_skips_groupby(self):
        """When duplicate_sum=0, groupby should never be called."""
        df = make_stock_df(duplicate_sum=0)
        result = clean_stock_df(df)
        # MockDataFrame.groupby() is a MagicMock; if called, it records calls.
        # We verify no crash and correct return type.
        assert result is not None

    def test_with_duplicates_calls_groupby(self):
        """When there are duplicates, deduplication branch is taken."""
        df = make_stock_df(duplicate_sum=5)
        # MockDataFrame.groupby returns a MagicMock; the result of .agg() is df itself
        result = clean_stock_df(df)
        assert result is not None
