"""
Feature engineering для прогнозирования дефектуры (OOS) лекарственных препаратов.

Создаёт 155+ признаков из временных рядов остатков по дистрибьюторам:
  - Скорость продаж / поставок (per-distributor и агрегированная)
  - Rolling-статистики (7, 14, 30-дневные окна)
  - Дни запаса и runway-флаги
  - Риск-флаги (низкий остаток, нулевой остаток, аномалии)
  - Синхронизация дистрибьюторов (HHI, CV, falling/rising)
  - Стресс спроса/предложения (net_flow, supply_deficit)
  - Категорийный и МНН-контекст
  - Временные признаки (с циклическим кодированием)
  - Составные сигналы риска
"""

import numpy as np
import pandas as pd
from loguru import logger

DIST_COLS = ["puls_stock", "katren_stock", "protek_stock", "farm_stock", "gk_stock"]
DATE_COL = "date"
CODE_COL = "code_kag"
TARGET_HORIZON = 14       # дней для прогноза OOS
OOS_PCT_THRESHOLD = 0.10  # OOS если остаток < 10% расширяющейся медианы


# ---------------------------------------------------------------------------
# Создание целевой переменной
# ---------------------------------------------------------------------------

def create_target(
    df: pd.DataFrame,
    horizon: int = TARGET_HORIZON,
    threshold: float = OOS_PCT_THRESHOLD,
) -> pd.DataFrame:
    """
    Создать бинарный таргет: станет ли продукт OOS в течение `horizon` дней?

    OOS определяется как: total_stock < threshold * expanding_median(total_stock)

    Args:
        df:        DataFrame с колонками [date, code_kag, total_stock]
                   (total_stock должен быть создан до вызова)
        horizon:   Горизонт прогноза в днях (по умолчанию 14)
        threshold: Доля от медианы ниже которой считаем OOS (по умолчанию 0.10)

    Returns:
        DataFrame с новой колонкой 'target' (0/1)
    """
    df = df.copy()
    df = df.sort_values([CODE_COL, DATE_COL])

    if "total_stock" not in df.columns:
        df["total_stock"] = df[[c for c in DIST_COLS if c in df.columns]].sum(axis=1)

    if "_product_median_exp" not in df.columns:
        df["_product_median_exp"] = df.groupby(CODE_COL)["total_stock"].transform(
            lambda x: x.expanding().median()
        )

    df["_is_oos"] = (df["total_stock"] < threshold * df["_product_median_exp"]).astype(int)

    # Смотрим вперёд на horizon шагов: был ли OOS хотя бы раз?
    df["target"] = (
        df.groupby(CODE_COL)["_is_oos"]
        .transform(lambda x: x.shift(-horizon).rolling(horizon, min_periods=1).max())
    ).fillna(0).astype(int)

    pos_rate = df["target"].mean()
    logger.info(f"Таргет создан: горизонт={horizon}д, OOS_порог={threshold}, "
                f"позитивных={pos_rate:.1%}")
    return df


# ---------------------------------------------------------------------------
# Основная функция feature engineering
# ---------------------------------------------------------------------------

def create_all_features_clean(
    df: pd.DataFrame,
    catalog: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Создать все 155+ признаков для модели OOS.

    Args:
        df:      DataFrame с остатками по дистрибьюторам
                 Обязательные колонки: [date, code_kag,
                 puls_stock, katren_stock, protek_stock, farm_stock, gk_stock]
        catalog: Опциональный справочник с [code_kag, mnn, NM_F, NM_DT]

    Returns:
        DataFrame со всеми добавленными признаками
    """
    logger.info("Запуск feature engineering...")
    df = df.copy()
    df = df.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    available_dists = [c for c in DIST_COLS if c in df.columns]
    for col in available_dists:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # ------------------------------------------------------------------
    # 1. Базовые агрегаты
    # ------------------------------------------------------------------
    df["total_stock"] = df[available_dists].sum(axis=1)
    df["full_oos"] = (df["total_stock"] == 0).astype(int)

    df["_product_median_exp"] = df.groupby(CODE_COL)["total_stock"].transform(
        lambda x: x.expanding().median()
    )
    df["_product_max_exp"] = df.groupby(CODE_COL)["total_stock"].transform(
        lambda x: x.expanding().max()
    )

    # ------------------------------------------------------------------
    # 2. Скорость продаж/поставок по каждому дистрибьютору
    # ------------------------------------------------------------------
    for dist in available_dists:
        prev = df.groupby(CODE_COL)[dist].transform(lambda x: x.shift(1))
        diff = prev - df[dist]  # >0: снижение = продажи, <0: рост = поставка
        prev_safe = prev.replace(0, np.nan)

        df[f"{dist}_sales_pct"] = (diff / prev_safe).clip(-1, 10).fillna(0)
        df[f"{dist}_supply_pct"] = ((-diff).clip(lower=0) / prev_safe).clip(0, 10).fillna(0)
        df[f"{dist}_falling"] = (diff > 0).astype(int)
        df[f"{dist}_rising"] = (diff < 0).astype(int)

    sales_cols = [f"{d}_sales_pct" for d in available_dists]
    supply_cols = [f"{d}_supply_pct" for d in available_dists]

    df["avg_sales_pct"] = df[sales_cols].mean(axis=1)
    df["avg_supply_pct"] = df[supply_cols].mean(axis=1)
    df["max_sales_pct"] = df[sales_cols].max(axis=1)

    # ------------------------------------------------------------------
    # 3. Rolling-статистики (7, 14, 30 дней)
    # ------------------------------------------------------------------
    grp = df.groupby(CODE_COL)

    for window in [7, 14, 30]:
        df[f"avg_sales_pct_{window}d"] = grp["avg_sales_pct"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean()
        )
        df[f"sales_cv_{window}d"] = grp["avg_sales_pct"].transform(
            lambda x, w=window: (
                x.rolling(w, min_periods=2).std()
                / (x.rolling(w, min_periods=2).mean().abs() + 1e-9)
            )
        ).fillna(0)
        df[f"oos_count_{window}d"] = grp["full_oos"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).sum()
        )
        df[f"had_oos_{window}d"] = (df[f"oos_count_{window}d"] > 0).astype(int)

    df["oos_rate_30d"] = df["oos_count_30d"] / 30.0

    for lag in [7, 14]:
        prev_stock = grp["total_stock"].transform(lambda x, l=lag: x.shift(l))
        prev_safe = prev_stock.replace(0, np.nan)
        df[f"stock_change_{lag}d_pct"] = (
            (df["total_stock"] - prev_stock) / prev_safe
        ).clip(-1, 10).fillna(0)

    # Частота поставок
    total_supply = df[supply_cols].sum(axis=1)
    df["_had_supply"] = (total_supply > 0.01).astype(int)
    df["supply_freq_30d"] = grp["_had_supply"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

    # Дней с последней поставки (кумулятивный счётчик)
    df["days_since_supply"] = grp["_had_supply"].transform(
        lambda x: x.groupby(x.cumsum()).cumcount()
    )
    avg_supply_interval = grp["days_since_supply"].transform(
        lambda x: x.rolling(90, min_periods=1).mean()
    )
    df["supply_interval_ratio"] = (
        df["days_since_supply"] / (avg_supply_interval + 1)
    ).clip(0, 10)
    df["supply_overdue"] = (df["supply_interval_ratio"] > 1.5).astype(int)

    # ------------------------------------------------------------------
    # 4. Скорость и ускорение продаж
    # ------------------------------------------------------------------
    sales_7d = grp["avg_sales_pct"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    sales_14d = grp["avg_sales_pct"].transform(lambda x: x.rolling(14, min_periods=1).mean())

    df["sales_velocity"] = (sales_7d / (sales_14d + 1e-9)).clip(0, 10)
    df["sales_accelerating"] = (df["sales_velocity"] > 1.3).astype(int)
    df["sales_acceleration"] = grp["sales_velocity"].transform(
        lambda x: x.diff().fillna(0)
    ).clip(-5, 5)

    # ------------------------------------------------------------------
    # 5. Дни запаса (runway)
    # ------------------------------------------------------------------
    safe_sales_7d = df["avg_sales_pct_7d"].replace(0, np.nan)
    df["days_of_stock"] = (1.0 / safe_sales_7d).clip(0, 365).fillna(365)

    df["days_of_stock_cat"] = pd.cut(
        df["days_of_stock"],
        bins=[0, 3, 7, 14, 30, 365, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
        right=True,
    ).astype(float)

    df["runway_critical"] = (df["days_of_stock"] < 7).astype(int)
    df["runway_warning"] = (df["days_of_stock"] < 14).astype(int)
    df["runway_caution"] = (df["days_of_stock"] < 30).astype(int)

    for dist in available_dists:
        s = df[f"{dist}_sales_pct"].replace(0, np.nan)
        df[f"{dist}_runway"] = (1.0 / s).clip(0, 365).fillna(365)

    runway_cols = [f"{d}_runway" for d in available_dists]
    df["min_dist_runway"] = df[runway_cols].min(axis=1)
    df["n_dist_runway_critical"] = (df[runway_cols] < 7).sum(axis=1)

    # ------------------------------------------------------------------
    # 6. Флаги риска по остаткам
    # ------------------------------------------------------------------
    df["n_zero"] = (df[available_dists] == 0).sum(axis=1)

    for dist in available_dists:
        med = df["_product_median_exp"].replace(0, np.nan)
        df[f"{dist}_is_low"] = (df[dist] < 0.10 * med).astype(int)
        df[f"{dist}_is_critical"] = (df[dist] < 0.05 * med).astype(int)

    low_cols = [f"{d}_is_low" for d in available_dists]
    crit_cols = [f"{d}_is_critical" for d in available_dists]
    df["n_low"] = df[low_cols].sum(axis=1)
    df["n_critical"] = df[crit_cols].sum(axis=1)

    # Z-score аномалии остатка
    stock_mean_30 = grp["total_stock"].transform(lambda x: x.rolling(30, min_periods=2).mean())
    stock_std_30 = grp["total_stock"].transform(lambda x: x.rolling(30, min_periods=2).std())
    df["stock_zscore"] = (
        (df["total_stock"] - stock_mean_30) / (stock_std_30 + 1e-9)
    ).clip(-5, 5).fillna(0)
    df["stock_anomaly_low"] = (df["stock_zscore"] < -2).astype(int)

    # Z-score аномалии продаж
    sales_mean_30 = grp["avg_sales_pct"].transform(lambda x: x.rolling(30, min_periods=2).mean())
    sales_std_30 = grp["avg_sales_pct"].transform(lambda x: x.rolling(30, min_periods=2).std())
    df["sales_zscore"] = (
        (df["avg_sales_pct"] - sales_mean_30) / (sales_std_30 + 1e-9)
    ).clip(-5, 5).fillna(0)
    df["sales_spike"] = (df["sales_zscore"] > 2).astype(int)

    # ------------------------------------------------------------------
    # 7. Синхронизация дистрибьюторов
    # ------------------------------------------------------------------
    falling_cols = [f"{d}_falling" for d in available_dists]
    rising_cols = [f"{d}_rising" for d in available_dists]

    df["n_dist_falling"] = df[falling_cols].sum(axis=1)
    df["n_dist_rising"] = df[rising_cols].sum(axis=1)
    df["all_falling"] = (df["n_dist_falling"] == len(available_dists)).astype(int)

    if len(available_dists) > 1:
        dist_vals = df[available_dists]
        dist_mean = dist_vals.mean(axis=1)
        dist_std = dist_vals.std(axis=1)
        df["stock_cv_cross"] = (dist_std / (dist_mean + 1e-9)).clip(0, 10).fillna(0)

        dist_sum = dist_vals.sum(axis=1).replace(0, np.nan)
        shares = dist_vals.div(dist_sum, axis=0).fillna(0)
        df["stock_hhi"] = (shares ** 2).sum(axis=1)

        for dist in available_dists:
            df[f"{dist}_share"] = (df[dist] / dist_sum).fillna(0)
    else:
        df["stock_cv_cross"] = 0.0
        df["stock_hhi"] = 1.0

    # ------------------------------------------------------------------
    # 8. Стресс спроса и предложения
    # ------------------------------------------------------------------
    df["supply_sales_ratio"] = (
        df["avg_supply_pct"] / (df["avg_sales_pct"] + 1e-9)
    ).clip(0, 10)
    df["supply_deficit"] = (df["avg_supply_pct"] < 0.8 * df["avg_sales_pct"]).astype(int)
    df["net_flow"] = df["avg_supply_pct"] - df["avg_sales_pct"]
    df["net_flow_7d"] = grp["net_flow"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["persistent_outflow"] = (df["net_flow_7d"] < -0.01).astype(int)

    # Средняя длительность OOS-эпизода за 90 дней
    df["avg_oos_duration"] = grp["full_oos"].transform(
        lambda x: x.rolling(90, min_periods=1).sum() / (
            x.rolling(90, min_periods=1).apply(
                lambda w: max(1, int(((w.values[1:] == 1) & (w.values[:-1] == 0)).sum()))
            ) + 1e-9
        )
    ).clip(0, 90).fillna(0)

    df["chronic_oos"] = (
        (df["oos_rate_30d"] > 0.10) | (df["avg_oos_duration"] > 3)
    ).astype(int)

    # ------------------------------------------------------------------
    # 9. События дистрибьюторов
    # ------------------------------------------------------------------
    for dist in available_dists:
        prev_val = grp[dist].transform(lambda x: x.shift(1))
        df[f"{dist}_went_zero_today"] = (
            (df[dist] == 0) & (prev_val > 0)
        ).astype(int)

    went_zero_cols = [f"{d}_went_zero_today" for d in available_dists]
    df["n_went_zero_today"] = df[went_zero_cols].sum(axis=1)

    for dist in available_dists:
        df[f"{dist}_first_zero_rate_60d"] = grp[f"{dist}_went_zero_today"].transform(
            lambda x: x.rolling(60, min_periods=1).mean()
        )

    # Серии последовательных падений / без поставок
    df["consecutive_falls"] = grp["all_falling"].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    df["consecutive_no_supply"] = grp["_had_supply"].transform(
        lambda x: (1 - x) * (
            (1 - x).groupby(((1 - x) != (1 - x).shift()).cumsum()).cumcount() + 1
        )
    )

    prev_n_zero = grp["n_zero"].transform(lambda x: x.shift(1))
    df["zeros_spreading"] = (df["n_zero"] > prev_n_zero).fillna(False).astype(int)

    # ------------------------------------------------------------------
    # 10. Категорийный / МНН контекст
    # ------------------------------------------------------------------
    if catalog is not None:
        merge_cols = [c for c in ["code_kag", "mnn", "NM_F", "NM_DT"] if c in catalog.columns]
        if len(merge_cols) > 1:
            df = df.merge(
                catalog[merge_cols].drop_duplicates("code_kag"),
                on="code_kag",
                how="left",
            )

    if "NM_F" in df.columns:
        df["cat_oos_rate_today"] = df.groupby([DATE_COL, "NM_F"])["full_oos"].transform("mean")
        df["cat_oos_rate_7d"] = grp["cat_oos_rate_today"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df["category_stress"] = (df["cat_oos_rate_today"] > 0.10).astype(int)
        df["cat_n_oos_today"] = df.groupby([DATE_COL, "NM_F"])["full_oos"].transform("sum")
        df["cat_oos_ratio"] = df.groupby([DATE_COL, "NM_F"])["full_oos"].transform("mean")

        prev_cat_oos = grp["cat_oos_rate_today"].transform(lambda x: x.shift(1))
        df["cat_oos_growth"] = (df["cat_oos_rate_today"] - prev_cat_oos).fillna(0)
        df["category_deteriorating"] = (df["cat_oos_growth"] > 0.02).astype(int)

    if "mnn" in df.columns:
        df["mnn_oos_rate"] = df.groupby([DATE_COL, "mnn"])["full_oos"].transform("mean")
        df["mnn_stress"] = (df["mnn_oos_rate"] > 0.15).astype(int)
        mnn_total = df.groupby([DATE_COL, "mnn"])["total_stock"].transform("sum")
        df["sku_share_in_mnn"] = (df["total_stock"] / (mnn_total + 1e-9)).clip(0, 1)

    # ------------------------------------------------------------------
    # 11. Временные признаки
    # ------------------------------------------------------------------
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["month"] = df[DATE_COL].dt.month
    df["day_of_month"] = df[DATE_COL].dt.day
    df["quarter"] = df[DATE_COL].dt.quarter

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * (df["quarter"] - 1) / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * (df["quarter"] - 1) / 4)

    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)

    # ------------------------------------------------------------------
    # 12. Составные сигналы риска
    # ------------------------------------------------------------------
    risk_flags = [
        "runway_critical", "sales_accelerating", "supply_deficit",
        "zeros_spreading", "persistent_outflow", "stock_anomaly_low",
    ]
    available_flags = [f for f in risk_flags if f in df.columns]
    df["n_red_flags"] = df[available_flags].sum(axis=1)

    df["critical_combination"] = (
        (df["days_of_stock"] < 14)
        & (df["days_since_supply"] > 7)
        & (df["sales_velocity"] > 1.2)
    ).astype(int)

    logger.info(f"Feature engineering завершён: {len(df.columns)} колонок итого")
    return df


def create_predictive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Дополнительные ранние предикторы OOS.
    Вызывается после create_all_features_clean.

    Добавляет:
    - Изменение дней запаса за 7/14 дней
    - Скользящее среднее количества красных флагов
    - Историческая вероятность OOS за 90 дней
    """
    df = df.copy()
    grp = df.groupby(CODE_COL)

    if "days_of_stock" in df.columns:
        dos_prev_7 = grp["days_of_stock"].transform(lambda x: x.shift(7))
        df["dos_change_7d"] = (df["days_of_stock"] - dos_prev_7).clip(-365, 365).fillna(0)
        dos_prev_14 = grp["days_of_stock"].transform(lambda x: x.shift(14))
        df["dos_change_14d"] = (df["days_of_stock"] - dos_prev_14).clip(-365, 365).fillna(0)

    if "n_red_flags" in df.columns:
        df["red_flags_trend_7d"] = grp["n_red_flags"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

    if "full_oos" in df.columns:
        df["oos_prob_hist_90d"] = grp["full_oos"].transform(
            lambda x: x.rolling(90, min_periods=1).mean()
        )

    logger.debug("Предиктивные признаки добавлены")
    return df


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Вернуть список признаков (исключить служебные, целевые и идентификаторы).
    """
    exclude = {
        DATE_COL, CODE_COL,
        "target", "target_future_oos", "target_multiclass",
        "_is_oos", "_product_median_exp", "_product_max_exp", "_had_supply",
        "NM_F", "mnn", "NM_DT",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]
