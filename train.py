"""
Обучение модели CatBoost для прогнозирования дефектуры (OOS).

Пайплайн:
  1. Загрузка CSV-файлов из data/raw/
  2. Валидация и очистка данных
  3. Feature engineering (155+ признаков)
  4. Создание целевой переменной (OOS в следующие 14 дней)
  5. Временной сплит с gap-периодом (предотвращение утечки данных)
  6. Опциональный подбор гиперпараметров через Optuna
  7. Обучение финальной модели CatBoost
  8. Оценка на тестовой выборке (ROC-AUC, PR-AUC)
  9. Сохранение модели и артефактов

Запуск:
    python train.py
    python train.py --tune              # с подбором гиперпараметров
    python train.py --config my.yaml   # кастомный конфиг
"""

import argparse
import glob
import os

import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from data_validation import clean_stock_df, validate_stock_df
from feature_engineering import (
    create_all_features_clean,
    create_predictive_features,
    create_target,
    get_feature_columns,
)
from helpers import ensure_dirs, load_config, today_str
from hyperparameter_tuning import compute_thresholds, tune_catboost
from logger import setup_logger
from model_registry import save_model


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_csv_data(data_dir: str) -> pd.DataFrame:
    """Загрузить все CSV-файлы из директории."""
    chunks = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not chunks:
        raise FileNotFoundError(f"CSV файлы не найдены в '{data_dir}'")

    logger.info(f"Загрузка {len(chunks)} CSV файлов из {data_dir}")
    dfs = [pd.read_csv(f) for f in chunks]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Загружено {len(df):,} строк из {len(chunks)} файлов")
    return df


# ---------------------------------------------------------------------------
# Временной сплит
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    train_cutoff: str,
    gap_end: str,
) -> tuple:
    """
    Разбить данные на train и test с gap-периодом между ними.

    Train:  date <= train_cutoff
    Gap:    train_cutoff < date <= gap_end  (исключается из обеих выборок)
    Test:   date > gap_end

    Gap нужен, чтобы временные признаки обучающей выборки не «видели» будущее.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train = df[df["date"] <= train_cutoff].copy()
    test = df[df["date"] > gap_end].copy()

    logger.info(
        f"Временной сплит:\n"
        f"  Train: {len(train):,} строк  (до {train_cutoff})\n"
        f"  Gap:   {(df['date'] > train_cutoff).sum() - len(test):,} строк "
        f"({train_cutoff} — {gap_end})  [исключены]\n"
        f"  Test:  {len(test):,} строк  (после {gap_end})"
    )
    return train, test


# ---------------------------------------------------------------------------
# Основной пайплайн обучения
# ---------------------------------------------------------------------------

def train(
    config_path: str = "configs/config.yaml",
    tune: bool = False,
) -> str:
    """
    Запустить полный пайплайн обучения.

    Args:
        config_path: Путь к YAML-конфигу
        tune:        Если True — запустить Optuna перед финальным обучением

    Returns:
        Имя сохранённой модели (например, "catboost_oos_20260413")
    """
    config = load_config(config_path)
    setup_logger(config["paths"].get("logs", "logs") + "/train.log")
    ensure_dirs(config)

    logger.info("=" * 60)
    logger.info("СТАРТ ОБУЧЕНИЯ")
    logger.info("=" * 60)

    # 1. Загрузка данных
    data_dir = config["paths"].get("data", "data/raw")
    df_raw = load_csv_data(data_dir)

    # 2. Валидация и очистка
    is_valid, errors = validate_stock_df(df_raw)
    if not is_valid:
        raise ValueError(f"Ошибки валидации входных данных:\n" + "\n".join(errors))
    df_clean = clean_stock_df(df_raw)

    # 3. Feature engineering
    df_feat = create_all_features_clean(df_clean)
    df_feat = create_predictive_features(df_feat)

    # 4. Целевая переменная
    pred_cfg = config.get("prediction", {})
    df_feat = create_target(
        df_feat,
        horizon=pred_cfg.get("horizon_days", 14),
        threshold=pred_cfg.get("oos_pct_threshold", 0.10),
    )

    # Убираем строки в конце датасета, где таргет не определён:
    # create_target() делает shift(-horizon), поэтому последние `horizon` дней
    # каждого продукта не имеют известного будущего → NaN в target.
    n_before = len(df_feat)
    df_feat = df_feat.dropna(subset=["target"])
    n_dropped = n_before - len(df_feat)
    logger.info(f"Удалено строк с NaN target (хвост горизонта): {n_dropped:,}")

    pos_rate = df_feat["target"].mean()
    logger.info(
        f"Итоговый датасет: {len(df_feat):,} строк, "
        f"позитивных: {pos_rate:.1%}"
    )

    # 5. Временной сплит
    model_cfg = config.get("model", {})
    train_cutoff = model_cfg.get("train_cutoff", "2025-12-10")
    gap_end = model_cfg.get("gap_end", "2025-12-23")

    df_train, df_test = temporal_split(df_feat, train_cutoff, gap_end)

    feature_cols = get_feature_columns(df_feat)
    # Заполняем пропуски значением -999: CatBoost умеет с ним работать,
    # а у нас пропуски возникают только на первых строках продукта,
    # где rolling-окна ещё не накопили достаточно истории (min_periods).
    X_train = df_train[feature_cols].fillna(-999)
    y_train = df_train["target"].astype(int)
    X_test = df_test[feature_cols].fillna(-999)
    y_test = df_test["target"].astype(int)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    imbalance_ratio = n_neg / max(n_pos, 1)
    logger.info(
        f"Баланс классов (train): 0={n_neg:,}, 1={n_pos:,}, "
        f"imbalance_ratio={imbalance_ratio:.2f}"
    )

    # 6. Гиперпараметры
    if tune:
        logger.info("Запуск Optuna для подбора гиперпараметров...")
        best_params = tune_catboost(
            X_train, y_train, X_test, y_test,
            n_trials=model_cfg.get("n_trials", 30),
            imbalance_ratio=imbalance_ratio,
            random_seed=model_cfg.get("random_seed", 42),
        )
    else:
        best_params = {
            "iterations":        model_cfg.get("iterations", 637),
            "learning_rate":     model_cfg.get("learning_rate", 0.0153),
            "depth":             model_cfg.get("depth", 10),
            "l2_leaf_reg":       model_cfg.get("l2_leaf_reg", 1.2605),
            "min_child_samples": model_cfg.get("min_child_samples", 189),
            "subsample":         model_cfg.get("subsample", 0.9474),
            "colsample_bylevel": model_cfg.get("colsample_bylevel", 0.7989),
            "scale_pos_weight":  model_cfg.get("scale_pos_weight", 6.7879),
        }
        logger.info("Используются параметры из конфига (без Optuna)")

    # 7. Финальное обучение
    logger.info("Обучение финальной модели CatBoost...")
    final_params = {
        **best_params,
        "eval_metric":           "AUC",
        "early_stopping_rounds": model_cfg.get("early_stopping_rounds", 50),
        "random_seed":           model_cfg.get("random_seed", 42),
        "verbose":               100,
        "allow_writing_files":   False,
    }

    model = CatBoostClassifier(**final_params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # 8. Оценка
    y_prob_test = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob_test)
    pr_auc = average_precision_score(y_test, y_prob_test)
    baseline_pr = float(y_test.mean())

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC:  {pr_auc:.4f}  (baseline: {baseline_pr:.4f})")

    metrics = {
        "roc_auc":         round(roc_auc, 4),
        "pr_auc":          round(pr_auc, 4),
        "baseline_pr_auc": round(baseline_pr, 4),
        "train_rows":      int(len(X_train)),
        "test_rows":       int(len(X_test)),
        "positive_rate":   round(float(y_train.mean()), 4),
        "train_cutoff":    train_cutoff,
        "gap_end":         gap_end,
        "n_features":      len(feature_cols),
    }

    # 9. Пороги классификации
    thresholds = compute_thresholds(y_test, y_prob_test)

    # 10. Сохранение
    model_name = f"catboost_oos_{today_str('%Y%m%d')}"
    models_dir = config["paths"].get("models", "models")
    save_model(model, feature_cols, metrics, thresholds, model_name, models_dir)

    logger.info("=" * 60)
    logger.info(f"ОБУЧЕНИЕ ЗАВЕРШЕНО: {model_name}")
    logger.info("=" * 60)
    return model_name


# ---------------------------------------------------------------------------
# CLI точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение модели прогнозирования дефектуры"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Путь к YAML-конфигу (по умолчанию: configs/config.yaml)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Запустить Optuna для подбора гиперпараметров (30 трайлов)"
    )
    args = parser.parse_args()
    train(args.config, args.tune)
