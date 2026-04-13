"""
Валидация и очистка входных данных перед feature engineering и обучением.
"""

from typing import List, Tuple

import pandas as pd
from loguru import logger

DIST_COLS = ["puls_stock", "katren_stock", "protek_stock", "farm_stock", "gk_stock"]
DATE_COL = "date"
CODE_COL = "code_kag"
MIN_HISTORY_DAYS = 30


def validate_stock_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Валидация DataFrame с остатками по дистрибьюторам.

    Разделяет проблемы на два уровня:
    - Блокирующие ошибки (is_valid=False): отсутствие колонок, пустой датасет,
      нечитаемая дата, недостаточная история — пайплайн не может продолжить.
    - Предупреждения: отрицательные остатки, дубликаты — будут исправлены
      в clean_stock_df(), не блокируют выполнение.

    Returns:
        (is_valid, blocking_errors) — errors содержит только блокирующие проблемы.
    """
    blocking_errors: List[str] = []
    dates = None

    # --- Пустой датасет ---
    if len(df) == 0:
        blocking_errors.append("DataFrame пустой")
        logger.error("Валидация ПРОВАЛЕНА: DataFrame пустой")
        return False, blocking_errors

    # --- Обязательные колонки ---
    required_cols = [DATE_COL, CODE_COL] + DIST_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        blocking_errors.append(f"Отсутствуют обязательные колонки: {missing}")

    # --- Тип даты ---
    if DATE_COL in df.columns:
        try:
            dates = pd.to_datetime(df[DATE_COL])
        except Exception:
            blocking_errors.append(f"Колонка '{DATE_COL}' не конвертируется в datetime")

    # Дальнейшие проверки теряют смысл без корректных колонок и дат
    if blocking_errors:
        for e in blocking_errors:
            logger.error(f"Валидация ПРОВАЛЕНА: {e}")
        return False, blocking_errors

    # --- Минимальный период истории ---
    date_range = (dates.max() - dates.min()).days
    if date_range < MIN_HISTORY_DAYS:
        blocking_errors.append(
            f"Недостаточно истории: {date_range} дней (минимум {MIN_HISTORY_DAYS})"
        )

    if blocking_errors:
        for e in blocking_errors:
            logger.error(f"Валидация ПРОВАЛЕНА: {e}")
        return False, blocking_errors

    # --- Предупреждения (не блокируют, но важно знать) ---
    for col in DIST_COLS:
        if col in df.columns:
            neg_count = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
            if neg_count > 0:
                logger.warning(
                    f"Отрицательные значения в '{col}': {neg_count} строк — "
                    f"будут заменены на 0 в clean_stock_df()"
                )

    dupes = df.duplicated([DATE_COL, CODE_COL]).sum()
    if dupes > 0:
        logger.warning(
            f"Дубликаты (date, code_kag): {dupes} строк — "
            f"будут агрегированы суммированием в clean_stock_df()"
        )

    logger.info(
        f"Валидация пройдена: {len(df):,} строк, "
        f"{df[CODE_COL].nunique()} продуктов, "
        f"период {dates.min().date()} — {dates.max().date()}"
    )
    return True, []


def clean_stock_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка DataFrame:
    - Конвертация типов
    - Замена NaN и отрицательных остатков на 0
    - Дедупликация по (date, code_kag) суммированием остатков
    - Сортировка по [code_kag, date]

    Args:
        df: Сырой DataFrame с остатками

    Returns:
        Очищенный DataFrame
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    for col in DIST_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # Дедупликация — суммируем остатки по (date, code_kag)
    if df.duplicated([DATE_COL, CODE_COL]).any():
        agg: dict = {col: "sum" for col in DIST_COLS if col in df.columns}
        other_cols = [
            c for c in df.columns
            if c not in [DATE_COL, CODE_COL] + DIST_COLS
        ]
        for c in other_cols:
            agg[c] = "first"
        df = df.groupby([DATE_COL, CODE_COL], as_index=False).agg(agg)
        logger.info(f"После дедупликации: {len(df):,} строк")

    df = df.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    logger.info(f"Данные очищены: {len(df):,} строк, {df[CODE_COL].nunique()} продуктов")
    return df


def check_feature_coverage(df_feat: pd.DataFrame, feature_list: list) -> dict:
    """
    Проверить, все ли ожидаемые признаки присутствуют в DataFrame.

    Args:
        df_feat:      DataFrame после feature engineering
        feature_list: Список ожидаемых признаков (из model_registry)

    Returns:
        {"present": [...], "missing": [...], "coverage_pct": float}
    """
    present = [f for f in feature_list if f in df_feat.columns]
    missing = [f for f in feature_list if f not in df_feat.columns]
    coverage = len(present) / len(feature_list) * 100 if feature_list else 0.0

    if missing:
        logger.warning(
            f"Отсутствуют {len(missing)}/{len(feature_list)} признаков "
            f"({coverage:.1f}% покрытие). Будут заполнены -999."
        )
    else:
        logger.info(f"Все {len(feature_list)} признаков присутствуют (100%)")

    return {"present": present, "missing": missing, "coverage_pct": coverage}
