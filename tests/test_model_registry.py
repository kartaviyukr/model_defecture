"""
Tests for model_registry.py.

list_models and get_latest_model use the real filesystem (tmp_path fixture).
save_model and load_model use a mocked CatBoostClassifier.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_registry import list_models, get_latest_model, save_model, load_model


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_empty_when_dir_does_not_exist(self, tmp_path):
        result = list_models(str(tmp_path / "nonexistent"))
        assert result == []

    def test_empty_when_dir_has_no_cbm_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("x")
        result = list_models(str(tmp_path))
        assert result == []

    def test_returns_stems_of_cbm_files(self, tmp_path):
        (tmp_path / "catboost_oos_20250115.cbm").write_bytes(b"")
        (tmp_path / "catboost_oos_20250201.cbm").write_bytes(b"")
        result = list_models(str(tmp_path))
        assert set(result) == {"catboost_oos_20250115", "catboost_oos_20250201"}

    def test_ignores_non_cbm_files(self, tmp_path):
        (tmp_path / "model.json").write_text("{}")
        (tmp_path / "model.cbm").write_bytes(b"")
        result = list_models(str(tmp_path))
        assert result == ["model"]

    def test_returns_sorted_list(self, tmp_path):
        (tmp_path / "catboost_oos_20250301.cbm").write_bytes(b"")
        (tmp_path / "catboost_oos_20250101.cbm").write_bytes(b"")
        (tmp_path / "catboost_oos_20250201.cbm").write_bytes(b"")
        result = list_models(str(tmp_path))
        # lexicographic sort (YYYYMMDD sorts correctly)
        assert result == [
            "catboost_oos_20250101",
            "catboost_oos_20250201",
            "catboost_oos_20250301",
        ]


# ---------------------------------------------------------------------------
# get_latest_model
# ---------------------------------------------------------------------------

class TestGetLatestModel:
    def test_returns_none_for_empty_dir(self, tmp_path):
        result = get_latest_model(str(tmp_path))
        assert result is None

    def test_returns_single_model(self, tmp_path):
        (tmp_path / "catboost_oos_20250115.cbm").write_bytes(b"")
        result = get_latest_model(str(tmp_path))
        assert result == "catboost_oos_20250115"

    def test_returns_latest_by_date(self, tmp_path):
        (tmp_path / "catboost_oos_20250101.cbm").write_bytes(b"")
        (tmp_path / "catboost_oos_20250301.cbm").write_bytes(b"")
        (tmp_path / "catboost_oos_20250201.cbm").write_bytes(b"")
        result = get_latest_model(str(tmp_path))
        assert result == "catboost_oos_20250301"

    def test_skips_files_without_valid_date(self, tmp_path):
        """Models with non-date suffixes sort last (datetime.min), so latest is the valid one."""
        (tmp_path / "catboost_oos_20250115.cbm").write_bytes(b"")
        (tmp_path / "my_custom_model.cbm").write_bytes(b"")
        result = get_latest_model(str(tmp_path))
        assert result == "catboost_oos_20250115"

    def test_date_ordering_not_lexicographic_edge_case(self, tmp_path):
        """20251231 must beat 20260101 only if parsed correctly."""
        (tmp_path / "catboost_oos_20251231.cbm").write_bytes(b"")
        (tmp_path / "catboost_oos_20260101.cbm").write_bytes(b"")
        result = get_latest_model(str(tmp_path))
        assert result == "catboost_oos_20260101"

    def test_date_parsing_via_get_latest_model(self, tmp_path):
        """get_latest_model uses datetime parsing; verify it picks June over January."""
        (tmp_path / "catboost_oos_20250615.cbm").write_bytes(b"")
        (tmp_path / "catboost_oos_20250101.cbm").write_bytes(b"")
        result = get_latest_model(str(tmp_path))
        assert result == "catboost_oos_20250615"

    def test_bad_name_model_loses_to_dated_model(self, tmp_path):
        """Model with non-date suffix sorts last (datetime.min) so dated one wins."""
        (tmp_path / "catboost_oos_20250115.cbm").write_bytes(b"")
        (tmp_path / "my_custom_model.cbm").write_bytes(b"")
        result = get_latest_model(str(tmp_path))
        assert result == "catboost_oos_20250115"


# ---------------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------------

class TestSaveModel:
    def _fake_model(self):
        m = MagicMock()
        m.get_params.return_value = {"iterations": 500}
        return m

    def test_creates_cbm_file(self, tmp_path):
        model = self._fake_model()
        save_model(model, ["f1", "f2"], {"roc_auc": 0.85}, {"balanced": 0.5},
                   "test_model", str(tmp_path))
        model.save_model.assert_called_once_with(str(tmp_path / "test_model.cbm"))

    def test_creates_metadata_json(self, tmp_path):
        model = self._fake_model()
        save_model(model, ["f1"], {"roc_auc": 0.9, "pr_auc": 0.6},
                   {"balanced": 0.4}, "m1", str(tmp_path))
        meta_path = tmp_path / "m1_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["metrics"]["roc_auc"] == 0.9
        assert meta["n_features"] == 1

    def test_creates_features_json(self, tmp_path):
        model = self._fake_model()
        features = ["feat_a", "feat_b", "feat_c"]
        save_model(model, features, {}, {}, "m1", str(tmp_path))
        feat_path = tmp_path / "m1_features.json"
        assert feat_path.exists()
        data = json.loads(feat_path.read_text())
        assert data["features"] == features
        assert data["n_features"] == 3

    def test_creates_thresholds_csv(self, tmp_path):
        """pd.DataFrame().to_csv() is mocked so the file isn't written, but
        we verify that pd.DataFrame was called with the expected data."""
        import sys
        pd_mock = sys.modules["pandas"]
        call_args_store = []
        original_df = pd_mock.DataFrame

        def capture_df(data, *a, **kw):
            call_args_store.append(data)
            return MagicMock()

        pd_mock.DataFrame = capture_df
        try:
            model = self._fake_model()
            save_model(model, ["f1"], {}, {"balanced": 0.35, "precision": 0.70},
                       "m1", str(tmp_path))
        finally:
            pd_mock.DataFrame = original_df

        assert len(call_args_store) == 1
        # The data should contain dicts with "type" and "threshold" keys
        data = call_args_store[0]
        assert any(d["type"] == "balanced" for d in data)

    def test_returns_cbm_path(self, tmp_path):
        model = self._fake_model()
        result = save_model(model, [], {}, {}, "m1", str(tmp_path))
        assert result == str(tmp_path / "m1.cbm")

    def test_creates_models_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "new_models"
        model = self._fake_model()
        save_model(model, [], {}, {}, "m1", str(new_dir))
        assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def _create_model_artifacts(self, tmp_path, name="m1", features=None, metrics=None,
                                thresholds=None):
        """Write the 4 artifacts that load_model expects."""
        if features is None:
            features = ["f1", "f2"]
        if metrics is None:
            metrics = {"roc_auc": 0.85}
        if thresholds is None:
            thresholds = {"balanced": 0.35, "precision": 0.70, "recall": 0.20}

        # .cbm file (must exist)
        (tmp_path / f"{name}.cbm").write_bytes(b"fake_model_bytes")

        # metadata
        meta = {
            "name": name, "saved_at": "2025-01-15T10:00:00",
            "metrics": metrics, "params": {}, "n_features": len(features),
            "model_path": str(tmp_path / f"{name}.cbm"),
        }
        (tmp_path / f"{name}_metadata.json").write_text(json.dumps(meta))

        # thresholds CSV
        csv_rows = "\n".join([f"{k},{v}" for k, v in thresholds.items()])
        (tmp_path / f"{name}_thresholds.csv").write_text("type,threshold\n" + csv_rows)

        # features JSON
        (tmp_path / f"{name}_features.json").write_text(
            json.dumps({"features": features, "n_features": len(features)})
        )

    def test_raises_file_not_found_when_cbm_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Модель не найдена"):
            load_model("nonexistent", str(tmp_path))

    def test_loads_features(self, tmp_path):
        self._create_model_artifacts(tmp_path, features=["feat_x", "feat_y"])
        with patch("model_registry.CatBoostClassifier") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = load_model("m1", str(tmp_path))
        assert result["features"] == ["feat_x", "feat_y"]

    def test_loads_metadata(self, tmp_path):
        self._create_model_artifacts(tmp_path, metrics={"roc_auc": 0.91})
        with patch("model_registry.CatBoostClassifier") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = load_model("m1", str(tmp_path))
        assert result["metadata"]["metrics"]["roc_auc"] == 0.91

    def test_loads_thresholds_as_dict(self, tmp_path):
        """Use a real CSV reader to work around mocked pd.read_csv."""
        import csv as _csv

        thresholds = {"balanced": 0.42, "precision": 0.71, "recall": 0.22}
        self._create_model_artifacts(tmp_path, thresholds=thresholds)

        def real_read_csv(path, **kw):
            """Read a simple type,threshold CSV and return a duck-typed DataFrame."""
            with open(path, newline="") as f:
                rows = list(_csv.DictReader(f))

            class _DfProxy:
                def __getitem__(self, key):
                    if key == "type":
                        return [r["type"] for r in rows]
                    if key == "threshold":
                        return [float(r["threshold"]) for r in rows]
                    return []
            return _DfProxy()

        import sys
        pd_mock = sys.modules["pandas"]
        original_read_csv = pd_mock.read_csv
        pd_mock.read_csv = real_read_csv
        try:
            with patch("model_registry.CatBoostClassifier") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = load_model("m1", str(tmp_path))
        finally:
            pd_mock.read_csv = original_read_csv

        assert abs(result["thresholds"]["balanced"] - 0.42) < 1e-6
        assert abs(result["thresholds"]["precision"] - 0.71) < 1e-6
