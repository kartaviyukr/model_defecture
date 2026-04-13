"""
Инференс модели: прогноз риска дефектуры на последнюю доступную дату.

Пайплайн:
  1. Загрузка модели и артефактов из model_registry
  2. Валидация и очистка входных данных
  3. Feature engineering
  4. Фильтрация по последней дате
  5. Инференс CatBoost — вероятности риска OOS
  6. Формирование итогового DataFrame с рейтингом

Запуск:
    python predict.py
    python predict.py --model catboost_oos_20260413
    python predict.py --data data/raw --threshold 0.65 --top-k 100
"""

import argparse
import glob
import os

import pandas as pd
from loguru import logger

from data_validation import check_feature_coverage, clean_stock_df, validate_stock_df
from feature_engineering import (
    create_all_features_clean,
    create_predictive_features,
)
from helpers import load_config
from logger import setup_logger
from model_registry import get_latest_model, load_model


def predict(
    df_stock: pd.DataFrame,
    model_name: str = None,
    config_path: str = "configs/config.yaml",
    threshold: float = None,
    top_k: int = None,
) -> pd.DataFrame:
    """
    Сформировать прогноз риска дефектуры для продуктов на последнюю дату.

    Args:
        df_stock:    DataFrame с остатками по всем доступным датам
                     (история нужна для корректного расчёта rolling-признаков)
        model_name:  Имя модели (без расширения); если None — берётся последняя
        config_path: Путь к конфигу
        threshold:   Порог is_alert; если None — берётся из конфига (0.70)
        top_k:       Топ-N продуктов по риску; если None — из конфига (200)

    Returns:
        DataFrame, отсортированный по убыванию risk_score, с колонками:
            risk_rank, code_kag, date, risk_score, is_alert,
            total_stock, n_zero, days_of_stock, prediction_date, model_name,
            + все признаки модели
    """
    config = load_config(config_path)
    pred_cfg = config.get("prediction", {})

    threshold = threshold if threshold is not None else pred_cfg.get("alert_threshold", 0.70)
    top_k = top_k if top_k is not None else pred_cfg.get("top_k", 200)
    models_dir = config["paths"].get("models", "models")

    # 1. Загрузка модели
    model_name = model_name or get_latest_model(models_dir)
    if model_name is None:
        raise ValueError(
            "Модель не найдена. Сначала запустите train.py для обучения модели."
        )
    artifacts = load_model(model_name, models_dir)
    model = artifacts["model"]
    features = artifacts["features"]

    # 2. Валидация и очистка
    is_valid, errors = validate_stock_df(df_stock)
    if not is_valid:
        raise ValueError(f"Ошибки валидации входных данных:\n" + "\n".join(errors))
    df_clean = clean_stock_df(df_stock)

    # 3. Feature engineering
    df_feat = create_all_features_clean(df_clean)
    df_feat = create_predictive_features(df_feat)

    # 4. Последняя дата в данных
    last_date = df_feat["date"].max()
    df_last = df_feat[df_feat["date"] == last_date].copy()
    logger.info(f"Прогноз на дату: {last_date.date()}, продуктов: {len(df_last)}")

    # 5. Проверка покрытия признаков
    coverage = check_feature_coverage(df_last, features)
    for f in coverage["missing"]:
        df_last[f] = -999.0

    X = df_last[features].fillna(-999)

    # 6. Инференс
    y_prob = model.predict_proba(X)[:, 1]

    # 7. Сборка результата
    out_base_cols = ["code_kag", "date", "total_stock", "n_zero", "days_of_stock"]
    available_base = [c for c in out_base_cols if c in df_last.columns]
    result = df_last[available_base].copy().reset_index(drop=True)

    result["risk_score"] = y_prob
    result["is_alert"] = result["risk_score"] >= threshold
    result["risk_rank"] = (
        result["risk_score"].rank(ascending=False, method="first").astype(int)
    )
    result["prediction_date"] = last_date.date()
    result["model_name"] = model_name

    # Добавляем все признаки (для диагностики)
    for col in features:
        if col not in result.columns:
            result[col] = df_last[col].values

    result = result.sort_values("risk_rank").reset_index(drop=True)

    if top_k and top_k > 0:
        result = result.head(top_k)

    n_alerts = int(result["is_alert"].sum())
    logger.info(
        f"Прогноз завершён: {len(result)} продуктов, "
        f"алертов: {n_alerts} (порог={threshold})"
    )
    return result


# ---------------------------------------------------------------------------
# CLI точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Прогноз риска дефектуры лекарственных препаратов"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Путь к YAML-конфигу"
    )
    parser.add_argument(
        "--model", default=None,
        help="Имя модели (без расширения). По умолчанию — последняя сохранённая."
    )
    parser.add_argument(
        "--data", default=None,
        help="Директория с CSV-файлами или путь к одному CSV"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Порог вероятности для флага is_alert (по умолчанию из конфига)"
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Вывести топ-N продуктов по риску"
    )
    args = parser.parse_args()

    setup_logger()
    config = load_config(args.config)

    if args.data:
        if os.path.isfile(args.data):
            df = pd.read_csv(args.data)
        else:
            chunks = sorted(glob.glob(os.path.join(args.data, "*.csv")))
            df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True)
    else:
        data_dir = config["paths"].get("data", "data/raw")
        chunks = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True)

    result = predict(
        df,
        model_name=args.model,
        config_path=args.config,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    display_cols = [
        c for c in
        ["risk_rank", "code_kag", "risk_score", "is_alert", "total_stock", "days_of_stock", "n_red_flags"]
        if c in result.columns
    ]
    print(result[display_cols].head(20).to_string(index=False))
