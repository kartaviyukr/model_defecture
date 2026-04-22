"""
Tests for excel_report.py.

_risk_level is pure Python — fully tested.
save_excel_report is tested at the smoke level (mocked pandas/openpyxl).
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from excel_report import _risk_level, COLORS, REPORT_COLUMNS, COLUMN_NAMES_RU


# ---------------------------------------------------------------------------
# _risk_level
# ---------------------------------------------------------------------------

class TestRiskLevel:
    # Boundary at 0.80 — critical
    def test_score_0_80_is_critical(self):
        assert _risk_level(0.80) == "critical"

    def test_score_above_0_80_is_critical(self):
        assert _risk_level(0.95) == "critical"
        assert _risk_level(1.00) == "critical"

    # Boundary at 0.60 — high
    def test_score_0_60_is_high(self):
        assert _risk_level(0.60) == "high"

    def test_score_between_0_60_and_0_80_is_high(self):
        assert _risk_level(0.70) == "high"
        assert _risk_level(0.799) == "high"

    # Boundary at 0.40 — medium
    def test_score_0_40_is_medium(self):
        assert _risk_level(0.40) == "medium"

    def test_score_between_0_40_and_0_60_is_medium(self):
        assert _risk_level(0.50) == "medium"
        assert _risk_level(0.599) == "medium"

    # Low
    def test_score_below_0_40_is_low(self):
        assert _risk_level(0.39) == "low"
        assert _risk_level(0.00) == "low"

    def test_score_0_is_low(self):
        assert _risk_level(0.0) == "low"

    # Edge cases
    def test_exactly_0_79_is_high_not_critical(self):
        assert _risk_level(0.799999) == "high"

    def test_exactly_0_40_is_medium_not_high(self):
        # 0.40 < 0.60 so high branch not taken; 0.40 >= 0.40 so medium is returned
        assert _risk_level(0.40) == "medium"

    def test_returns_string(self):
        assert isinstance(_risk_level(0.5), str)

    def test_all_levels_are_in_colors_dict(self):
        for score in [0.90, 0.65, 0.45, 0.10]:
            level = _risk_level(score)
            assert level in COLORS, f"Level '{level}' not in COLORS"


# ---------------------------------------------------------------------------
# REPORT_COLUMNS / COLUMN_NAMES_RU — config sanity
# ---------------------------------------------------------------------------

class TestReportConfig:
    def test_report_columns_is_list(self):
        assert isinstance(REPORT_COLUMNS, list)
        assert len(REPORT_COLUMNS) > 0

    def test_column_names_ru_covers_all_report_columns(self):
        missing = [c for c in REPORT_COLUMNS if c not in COLUMN_NAMES_RU]
        assert missing == [], f"Missing RU names for: {missing}"

    def test_colors_has_all_risk_levels(self):
        for level in ("critical", "high", "medium", "low"):
            assert level in COLORS

    def test_colors_has_header_keys(self):
        assert "header_bg" in COLORS
        assert "header_font" in COLORS


# ---------------------------------------------------------------------------
# save_excel_report — smoke test with mocked workbook
# ---------------------------------------------------------------------------

class TestSaveExcelReport:
    def test_creates_reports_dir(self, tmp_path):
        from excel_report import save_excel_report
        predictions = MagicMock()
        predictions.sort_values.return_value = predictions
        predictions.head.return_value = predictions
        predictions.iterrows.return_value = iter([])

        reports_dir = str(tmp_path / "new_reports")
        try:
            save_excel_report(predictions, "2025-01-01", reports_dir)
        except Exception:
            pass  # openpyxl write may raise; we only care about dir creation
        assert os.path.isdir(reports_dir)

    def test_returns_string_path(self, tmp_path):
        from excel_report import save_excel_report
        predictions = MagicMock()
        predictions.sort_values.return_value = predictions
        predictions.head.return_value = predictions
        predictions.iterrows.return_value = iter([])
        result = save_excel_report(predictions, "2025-01-01", str(tmp_path))
        assert isinstance(result, str)
        assert "2025-01-01" in result
        assert result.endswith(".xlsx")
