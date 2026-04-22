"""
Mock unavailable packages (pandas, numpy, catboost, loguru, etc.)
so source modules can be imported and tested in a restricted environment.

All sys.modules entries are inserted at conftest load time — before pytest
collects any test file — so every subsequent `import X` in source code
resolves to our mock.
"""
import os
import sys
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# loguru
# ---------------------------------------------------------------------------
_loguru = types.ModuleType("loguru")
_loguru.logger = MagicMock()
sys.modules["loguru"] = _loguru

# ---------------------------------------------------------------------------
# dotenv (python-dotenv)
# ---------------------------------------------------------------------------
_dotenv = types.ModuleType("dotenv")
_dotenv.load_dotenv = MagicMock()
sys.modules["dotenv"] = _dotenv

# ---------------------------------------------------------------------------
# pyodbc
# ---------------------------------------------------------------------------
_pyodbc = types.ModuleType("pyodbc")
_pyodbc.connect = MagicMock()
_pyodbc.Error = Exception
sys.modules["pyodbc"] = _pyodbc

# ---------------------------------------------------------------------------
# numpy
# ---------------------------------------------------------------------------
_numpy = MagicMock()
_numpy.__name__ = "numpy"
sys.modules["numpy"] = _numpy

# ---------------------------------------------------------------------------
# pandas — minimal functional stubs for the operations our source code uses
# ---------------------------------------------------------------------------
_pandas = MagicMock()
_pandas.__name__ = "pandas"


class _FakeNumericSeries:
    """Supports (series < 0).sum() → int for data_validation tests."""
    def __init__(self, neg_count: int = 0):
        self._neg_count = neg_count

    def __lt__(self, _):
        count = self._neg_count
        class _BoolMask:
            def sum(self):
                return count
        return _BoolMask()

    def fillna(self, val):
        return self

    def clip(self, lower=None):
        return self


# pd.to_datetime: return the series unchanged so tests can configure max/min
_pandas.to_datetime = lambda series, **kw: series
# pd.to_numeric: return a FakeNumericSeries with 0 negative values by default
_pandas.to_numeric = lambda series, **kw: _FakeNumericSeries(0)
# pd.DataFrame constructor
_pandas.DataFrame = MagicMock()
# pd.concat
_pandas.concat = MagicMock(return_value=MagicMock())
# pd.date_range
_pandas.date_range = MagicMock(return_value=[])
# pd.Timestamp
_pandas.Timestamp = MagicMock(side_effect=lambda x: x)

sys.modules["pandas"] = _pandas

# ---------------------------------------------------------------------------
# catboost
# ---------------------------------------------------------------------------
_catboost = types.ModuleType("catboost")
_catboost.CatBoostClassifier = MagicMock()
sys.modules["catboost"] = _catboost

# ---------------------------------------------------------------------------
# sklearn
# ---------------------------------------------------------------------------
_sklearn = types.ModuleType("sklearn")
_sklearn_metrics = types.ModuleType("sklearn.metrics")
_sklearn_metrics.roc_auc_score = MagicMock(return_value=0.85)
_sklearn_metrics.average_precision_score = MagicMock(return_value=0.55)
_sklearn_metrics.precision_recall_curve = MagicMock(
    return_value=([0.9, 0.8, 0.7], [0.5, 0.6, 0.7], [0.3, 0.5])
)
sys.modules["sklearn"] = _sklearn
sys.modules["sklearn.metrics"] = _sklearn_metrics
_sklearn.metrics = _sklearn_metrics

# ---------------------------------------------------------------------------
# optuna
# ---------------------------------------------------------------------------
_optuna = MagicMock()
_optuna_samplers = MagicMock()
_optuna.logging = MagicMock()
_optuna.logging.WARNING = 30
sys.modules["optuna"] = _optuna
sys.modules["optuna.samplers"] = _optuna_samplers
_optuna.samplers = _optuna_samplers

# ---------------------------------------------------------------------------
# openpyxl
# ---------------------------------------------------------------------------
_openpyxl = types.ModuleType("openpyxl")
_openpyxl.styles = types.ModuleType("openpyxl.styles")
_openpyxl.styles.Alignment = MagicMock()
_openpyxl.styles.Font = MagicMock()
_openpyxl.styles.PatternFill = MagicMock()
_openpyxl.utils = types.ModuleType("openpyxl.utils")
_openpyxl.utils.get_column_letter = lambda n: chr(64 + n)  # A, B, C, ...
sys.modules["openpyxl"] = _openpyxl
sys.modules["openpyxl.styles"] = _openpyxl.styles
sys.modules["openpyxl.utils"] = _openpyxl.utils

# ---------------------------------------------------------------------------
# Proper class-based DataFrame mock
# (avoids MagicMock's special-method dispatch issues where df[key] passes
#  the mock instance as first arg to any plain function set as __getitem__)
# ---------------------------------------------------------------------------
import pytest

_DIST_COLS = ["puls_stock", "katren_stock", "protek_stock", "farm_stock", "gk_stock"]
_ALL_COLS = ["date", "code_kag"] + _DIST_COLS


class _BoolSeries:
    """Result of df.duplicated() — supports .sum() and .any()."""
    def __init__(self, n: int):
        self._n = n

    def sum(self) -> int:
        return self._n

    def any(self) -> bool:
        return self._n > 0


class _DateSeries:
    """Date column: max()/min() return real datetimes so timedelta works."""
    def __init__(self, min_dt: datetime, max_dt: datetime):
        self._min = min_dt
        self._max = max_dt

    def max(self) -> datetime:
        return self._max

    def min(self) -> datetime:
        return self._min

    def nunique(self) -> int:
        return 2


class _StockSeries:
    """Stock column — supports ops validate_stock_df uses."""
    def nunique(self) -> int:
        return 5

    def fillna(self, val):
        return self

    def clip(self, lower=None):
        return self


class MockDataFrame:
    """
    Minimal DataFrame substitute for validate_stock_df / clean_stock_df tests.
    Uses normal Python __getitem__/__setitem__ so they receive only the key.
    """
    def __init__(
        self,
        n_rows: int = 100,
        date_range_days: int = 60,
        columns: list = None,
        duplicate_sum: int = 0,
    ):
        if columns is None:
            columns = list(_ALL_COLS)
        self.columns = columns
        self._n_rows = n_rows
        self._dup_sum = duplicate_sum
        min_dt = datetime(2025, 1, 1)
        max_dt = min_dt + timedelta(days=date_range_days)
        self._date_series = _DateSeries(min_dt, max_dt)

    def __len__(self) -> int:
        return self._n_rows

    def __getitem__(self, key):
        if key == "date":
            return self._date_series
        return _StockSeries()

    def __setitem__(self, key, value):
        pass

    def __contains__(self, key) -> bool:
        return key in self.columns

    def copy(self) -> "MockDataFrame":
        c = MockDataFrame(
            n_rows=self._n_rows,
            date_range_days=0,
            columns=list(self.columns),
            duplicate_sum=self._dup_sum,
        )
        c._date_series = self._date_series
        return c

    def duplicated(self, cols=None) -> _BoolSeries:
        return _BoolSeries(self._dup_sum)

    def sort_values(self, *args, **kwargs) -> "MockDataFrame":
        return self

    def reset_index(self, *args, **kwargs) -> "MockDataFrame":
        return self

    def groupby(self, *args, **kwargs):
        grp = MagicMock()
        grp.agg.return_value = self
        return grp


def make_stock_df(
    n_rows: int = 100,
    date_range_days: int = 60,
    include_all_cols: bool = True,
    duplicate_sum: int = 0,
) -> MockDataFrame:
    """Build a MockDataFrame for validate_stock_df / clean_stock_df tests."""
    cols = list(_ALL_COLS) if include_all_cols else []
    return MockDataFrame(
        n_rows=n_rows,
        date_range_days=date_range_days,
        columns=cols,
        duplicate_sum=duplicate_sum,
    )


@pytest.fixture
def stock_df() -> MockDataFrame:
    """Valid 100-row MockDataFrame with 60-day window."""
    return make_stock_df()
