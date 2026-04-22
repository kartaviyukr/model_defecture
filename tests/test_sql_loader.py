"""
Tests for sql_loader.py.

Focus: _validate_date() — the SQL injection prevention guard.
load_stock_gk/load_competitors/load_sales/load_catalog are DB-bound
and tested only for their date-validation behavior (no real DB needed).
"""
import pytest

# Access private _validate_date via the module
import sql_loader
from sql_loader import _validate_date


# ---------------------------------------------------------------------------
# _validate_date
# ---------------------------------------------------------------------------

class TestValidateDate:
    """Validates the YYYY-MM-DD regex guard against SQL injection."""

    # --- Valid dates ---

    def test_accepts_valid_date(self):
        _validate_date("2025-01-15", "start_date")  # must not raise

    def test_accepts_year_boundary_start(self):
        _validate_date("2025-01-01", "end_date")

    def test_accepts_year_boundary_end(self):
        _validate_date("2025-12-31", "end_date")

    def test_accepts_leap_day_format(self):
        # Regex only checks format, not calendar validity
        _validate_date("2024-02-29", "start_date")

    # --- Invalid dates that must raise ValueError ---

    def test_rejects_sql_injection_or(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            _validate_date("2025-01-01' OR '1'='1", "start_date")

    def test_rejects_sql_injection_union(self):
        with pytest.raises(ValueError):
            _validate_date("2025-01-01; DROP TABLE foo; --", "start_date")

    def test_rejects_wrong_separator(self):
        with pytest.raises(ValueError):
            _validate_date("2025/01/15", "start_date")

    def test_rejects_dd_mm_yyyy_format(self):
        with pytest.raises(ValueError):
            _validate_date("15-01-2025", "start_date")

    def test_rejects_missing_dashes(self):
        with pytest.raises(ValueError):
            _validate_date("20250115", "start_date")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            _validate_date("", "start_date")

    def test_rejects_partial_date(self):
        with pytest.raises(ValueError):
            _validate_date("2025-01", "start_date")

    def test_rejects_with_time(self):
        with pytest.raises(ValueError):
            _validate_date("2025-01-15 12:00:00", "start_date")

    def test_error_message_contains_param_name(self):
        with pytest.raises(ValueError, match="my_param"):
            _validate_date("bad-date", "my_param")

    def test_error_message_contains_received_value(self):
        bad = "2025/01/01"
        with pytest.raises(ValueError, match=bad):
            _validate_date(bad, "start_date")

    def test_rejects_extra_leading_space(self):
        with pytest.raises(ValueError):
            _validate_date(" 2025-01-15", "start_date")

    def test_rejects_extra_trailing_space(self):
        with pytest.raises(ValueError):
            _validate_date("2025-01-15 ", "start_date")


# ---------------------------------------------------------------------------
# load_stock_gk / load_competitors / load_sales — date validation only
# ---------------------------------------------------------------------------

class TestLoadFunctionsDateValidation:
    """Verify that load_* functions reject invalid dates before touching the DB."""

    def test_load_stock_gk_rejects_bad_start(self):
        with pytest.raises(ValueError):
            sql_loader.load_stock_gk("bad-date", "2025-12-31")

    def test_load_stock_gk_rejects_bad_end(self):
        with pytest.raises(ValueError):
            sql_loader.load_stock_gk("2025-01-01", "not-a-date")

    def test_load_competitors_rejects_bad_start(self):
        with pytest.raises(ValueError):
            sql_loader.load_competitors("2025/01/01", "2025-12-31")

    def test_load_sales_rejects_bad_start(self):
        with pytest.raises(ValueError):
            sql_loader.load_sales("", "2025-12-31")
