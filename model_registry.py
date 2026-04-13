"""
Реестр моделей: сохранение и загрузка артефактов CatBoost.

Для каждой модели сохраняются 4 файла:
  {name}.cbm                 — бинарная модель CatBoost
  {name}_metadata.json       — метрики, параметры, даты обучения
  {name}_thresholds.csv      — пороги классификации (balanced/precision/recall)
  {name}_features.json       — упорядоченный список признаков
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger


def save_model(
    model: CatBoostClassifier,
    features: list,
    metrics: dict,
    thresholds: dict,
    name: str,
    models_dir: str = "models",
) -> str:
    """
    Сохранить модель и все артефакты на диск.

    Args:
        model:       Обученная модель CatBoostClassifier
        features:    Упорядоченный список имён признаков
        metrics:     Словарь метрик {"roc_auc": ..., "pr_auc": ...}
        thresholds:  Словарь порогов {"balanced": ..., "precision": ..., "recall": ...}
        name:        Базовое имя файлов (без расширения), например "catboost_oos_20250126"
        models_dir:  Директория для сохранения

    Returns:
        Путь к .cbm файлу модели
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    base = os.path.join(models_dir, name)

    # --- Модель ---
    model_path = f"{base}.cbm"
    model.save_model(model_path)
    logger.info(f"Модель сохранена: {model_path}")

    # --- Метаданные ---
    metadata = {
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "metrics": metrics,
        "params": model.get_params(),
        "n_features": len(features),
        "model_path": model_path,
    }
    meta_path = f"{base}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Метаданные сохранены: {meta_path}")

    # --- Пороги ---
    thr_df = pd.DataFrame([
        {"type": k, "threshold": v} for k, v in thresholds.items()
    ])
    thr_path = f"{base}_thresholds.csv"
    thr_df.to_csv(thr_path, index=False)
    logger.info(f"Пороги сохранены: {thr_path}")

    # --- Признаки ---
    feat_path = f"{base}_features.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump({"features": features, "n_features": len(features)}, f,
                  ensure_ascii=False, indent=2)
    logger.info(f"Признаки сохранены: {feat_path}")

    return model_path


def load_model(name: str, models_dir: str = "models") -> dict:
    """
    Загрузить модель и все артефакты с диска.

    Args:
        name:       Базовое имя модели (без расширения)
        models_dir: Директория с моделями

    Returns:
        dict с ключами:
            "model"      — CatBoostClassifier
            "features"   — list[str] признаков
            "metadata"   — dict с метриками и параметрами
            "thresholds" — dict {"balanced": float, ...}
    """
    base = os.path.join(models_dir, name)

    model_path = f"{base}.cbm"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model = CatBoostClassifier()
    model.load_model(model_path)
    logger.info(f"Модель загружена: {model_path}")

    meta_path = f"{base}_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    thr_path = f"{base}_thresholds.csv"
    thr_df = pd.read_csv(thr_path)
    thresholds = dict(zip(thr_df["type"], thr_df["threshold"]))

    feat_path = f"{base}_features.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        feat_data = json.load(f)
    features = feat_data["features"]

    logger.info(
        f"Артефакты загружены: {len(features)} признаков, "
        f"метрики: {metadata.get('metrics', {})}"
    )
    return {
        "model": model,
        "features": features,
        "metadata": metadata,
        "thresholds": thresholds,
    }


def list_models(models_dir: str = "models") -> list:
    """Вернуть список имён доступных моделей (по .cbm файлам)."""
    p = Path(models_dir)
    if not p.exists():
        return []
    models = sorted(f.stem for f in p.glob("*.cbm"))
    logger.info(f"Доступные модели: {models}")
    return models


def get_latest_model(models_dir: str = "models") -> Optional[str]:
    """Вернуть имя последней (лексикографически) модели."""
    models = list_models(models_dir)
    if not models:
        logger.warning(f"Нет доступных моделей в '{models_dir}'")
        return None
    latest = models[-1]
    logger.info(f"Последняя модель: {latest}")
    return latest
