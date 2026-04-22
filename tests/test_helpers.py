"""Tests for helpers.py — pure Python functions."""
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest

from helpers import normalize_kag_code, today_str, load_config, ensure_dirs


# ---------------------------------------------------------------------------
# normalize_kag_code
# ---------------------------------------------------------------------------

class TestNormalizeKagCode:
    def test_float_string_strips_dot_zero(self):
        assert normalize_kag_code("28031.0") == "28031"

    def test_integer_string_unchanged(self):
        assert normalize_kag_code("28031") == "28031"

    def test_none_returns_empty_string(self):
        assert normalize_kag_code(None) == ""

    def test_integer_input_converted(self):
        assert normalize_kag_code(28031) == "28031"

    def test_float_input_strips_dot_zero(self):
        # str(28031.0) == "28031.0"
        assert normalize_kag_code(28031.0) == "28031"

    def test_whitespace_stripped(self):
        assert normalize_kag_code("  28031.0  ") == "28031"

    def test_empty_string_returns_empty(self):
        assert normalize_kag_code("") == ""

    def test_string_ending_dot_zero_trimmed(self):
        assert normalize_kag_code("ABC.0") == "ABC"

    def test_string_without_dot_zero_preserved(self):
        assert normalize_kag_code("ABC.1") == "ABC.1"

    def test_zero_code(self):
        assert normalize_kag_code("0.0") == "0"

    def test_large_code(self):
        assert normalize_kag_code("1234567890.0") == "1234567890"


# ---------------------------------------------------------------------------
# today_str
# ---------------------------------------------------------------------------

class TestTodayStr:
    def test_default_format_is_yyyy_mm_dd(self):
        result = today_str()
        # Must be exactly YYYY-MM-DD: 10 characters
        assert len(result) == 10
        parts = result.split("-")
        assert len(parts) == 3
        year, month, day = parts
        assert len(year) == 4
        assert len(month) == 2
        assert len(day) == 2

    def test_custom_format(self):
        result = today_str("%Y%m%d")
        assert len(result) == 8
        assert result.isdigit()

    def test_result_is_valid_date(self):
        result = today_str()
        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed.year >= 2025

    def test_year_only_format(self):
        result = today_str("%Y")
        assert result.isdigit()
        assert len(result) == 4

    def test_custom_separator(self):
        result = today_str("%d/%m/%Y")
        assert "/" in result


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

SAMPLE_YAML = """
paths:
  data: data/raw
  models: models
  logs: logs
  reports: reports
database:
  server: localhost
  driver: ODBC Driver 18 for SQL Server
  database: dwh_price
  username: user
  password: pass
"""


class TestLoadConfig:
    def test_loads_yaml_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(SAMPLE_YAML, encoding="utf-8")
        config = load_config(str(cfg_file))
        assert config["paths"]["data"] == "data/raw"
        assert config["database"]["server"] == "localhost"

    def test_env_variable_substitution(self, tmp_path):
        yaml_with_var = "database:\n  server: ${TEST_SERVER_HOST}\n"
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_with_var, encoding="utf-8")
        with patch.dict(os.environ, {"TEST_SERVER_HOST": "prod-server"}):
            config = load_config(str(cfg_file))
        assert config["database"]["server"] == "prod-server"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# ensure_dirs
# ---------------------------------------------------------------------------

class TestEnsureDirs:
    def test_creates_all_path_directories(self, tmp_path):
        config = {
            "paths": {
                "data": str(tmp_path / "data" / "raw"),
                "models": str(tmp_path / "models"),
                "logs": str(tmp_path / "logs"),
            }
        }
        ensure_dirs(config)
        assert (tmp_path / "data" / "raw").is_dir()
        assert (tmp_path / "models").is_dir()
        assert (tmp_path / "logs").is_dir()

    def test_idempotent_existing_dirs(self, tmp_path):
        """Calling ensure_dirs twice should not raise."""
        existing = tmp_path / "existing"
        existing.mkdir()
        config = {"paths": {"a": str(existing)}}
        ensure_dirs(config)
        assert existing.is_dir()
