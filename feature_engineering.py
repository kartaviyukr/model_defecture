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

Публичный API:
  create_target()             — бинарный таргет OOS на N дней вперёд
  create_all_features_clean() — все признаки (вызывает приватные _add_* функции)
  create_predictive_features() — дополнительные ранние предикторы
  get_feature_columns()       — список признаков без служебных колонок
"""

import numpy as np
import pandas as pd
from loguru import logger

DIST_COLS = ["puls_stock", "katren_stock", "protek_stock", "farm_stock", "gk_stock"]
DATE_COL = "date"
CODE_COL = "code_kag"

# Горизонт прогноза: считаем продукт в зоне риска, если OOS наступит
# в течение следующих 14 дней — достаточно для упреждающего заказа.
TARGET_HORIZON = 14

# Порог дефектуры: остаток < 10% исторической медианы продукта.
# Использует расширяющуюся медиану (expanding), чтобы не заглядывать в будущее.
OOS_PCT_THRESHOLD = 0.10


# ===========================================================================
# Целевая переменная
# ===========================================================================

def create_target(
    df: pd.DataFrame,
    horizon: int = TARGET_HORIZON,
    threshold: float = OOS_PCT_THRESHOLD,
) -> pd.DataFrame:
    """
    Создать бинарный таргет: попадёт ли продукт в дефектуру в течение `horizon` дней?

    Дефектура: total_stock < threshold * expanding_median(total_stock).
    Смотрим вперёд на `horizon` шагов: если хотя бы раз — таргет = 1.

    Args:
        df:        DataFrame, отсортированный по [code_kag, date].
                   Колонка total_stock должна уже существовать.
        horizon:   Горизонт прогноза в днях.
        threshold: Доля от медианы, ниже которой считаем дефектуру.

    Returns:
        DataFrame с новыми колонками '_is_oos' и 'target'.
    """
    df = df.copy().sort_values([CODE_COL, DATE_COL])

    if "total_stock" not in df.columns:
        df["total_stock"] = df[[c for c in DIST_COLS if c in df.columns]].sum(axis=1)

    if "_product_median_exp" not in df.columns:
        df["_product_median_exp"] = df.groupby(CODE_COL)["total_stock"].transform(
            lambda x: x.expanding().median()
        )

    df["_is_oos"] = (df["total_stock"] < threshold * df["_product_median_exp"]).astype(int)

    # shift(-horizon): смотрим вперёд; rolling(horizon).max(): хотя бы один OOS
    df["target"] = (
        df.groupby(CODE_COL)["_is_oos"]
        .transform(lambda x: x.shift(-horizon).rolling(horizon, min_periods=1).max())
    ).fillna(0).astype(int)

    logger.info(
        f"Таргет создан: горизонт={horizon}д, порог={threshold}, "
        f"позитивных={df['target'].mean():.1%}"
    )
    return df


# ===========================================================================
# Приватные функции-помощники (вызываются только из create_all_features_clean)
# ===========================================================================

def _add_base_aggregates(df: pd.DataFrame, dists: list) -> pd.DataFrame:
    """Базовые агрегаты: суммарный остаток, флаг полного OOS, расширяющиеся статистики."""
    df["total_stock"] = df[dists].sum(axis=1)
    df["full_oos"] = (df["total_stock"] == 0).astype(int)

    grp = df.groupby(CODE_COL)["total_stock"]
    df["_product_median_exp"] = grp.transform(lambda x: x.expanding().median())
    df["_product_max_exp"] = grp.transform(lambda x: x.expanding().max())
    return df


def _add_velocity_features(df: pd.DataFrame, dists: list) -> pd.DataFrame:
    """
    Скорость продаж и поставок по каждому дистрибьютору.

    sales_pct  = (prev - curr) / prev  → >0 означает снижение остатка (продажи)
    supply_pct = (curr - prev) / prev  → >0 означает рост остатка (поставка)
    """
    for dist in dists:
        prev = df.groupby(CODE_COL)[dist].transform(lambda x: x.shift(1))
        delta = prev - df[dist]
        prev_safe = prev.replace(0, np.nan)

        df[f"{dist}_sales_pct"] = (delta / prev_safe).clip(-1, 10).fillna(0)
        df[f"{dist}_supply_pct"] = ((-delta).clip(lower=0) / prev_safe).clip(0, 10).fillna(0)
        df[f"{dist}_falling"] = (delta > 0).astype(int)
        df[f"{dist}_rising"] = (delta < 0).astype(int)

    sales_cols = [f"{d}_sales_pct" for d in dists]
    supply_cols = [f"{d}_supply_pct" for d in dists]
    df["avg_sales_pct"] = df[sales_cols].mean(axis=1)
    df["avg_supply_pct"] = df[supply_cols].mean(axis=1)
    df["max_sales_pct"] = df[sales_cols].max(axis=1)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-статистики по 7, 14, 30-дневным окнам и лаговые изменения."""
    grp = df.groupby(CODE_COL)

    for window in [7, 14, 30]:
        w = window  # захват значения в замыкании
        df[f"avg_sales_pct_{w}d"] = grp["avg_sales_pct"].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        )
        df[f"sales_cv_{w}d"] = grp["avg_sales_pct"].transform(
            lambda x, w=w: (
                x.rolling(w, min_periods=2).std()
                / (x.rolling(w, min_periods=2).mean().abs() + 1e-9)
            )
        ).fillna(0)
        df[f"oos_count_{w}d"] = grp["full_oos"].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).sum()
        )
        df[f"had_oos_{w}d"] = (df[f"oos_count_{w}d"] > 0).astype(int)

    df["oos_rate_30d"] = df["oos_count_30d"] / 30.0

    for lag in [7, 14]:
        prev_stock = grp["total_stock"].transform(lambda x, l=lag: x.shift(l))
        prev_safe = prev_stock.replace(0, np.nan)
        df[f"stock_change_{lag}d_pct"] = (
            (df["total_stock"] - prev_stock) / prev_safe
        ).clip(-1, 10).fillna(0)

    return df


def _add_supply_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Частота поставок и дни с момента последней поставки."""
    supply_cols = [c for c in df.columns if c.endswith("_supply_pct")]
    df["_had_supply"] = (df[supply_cols].sum(axis=1) > 0.01).astype(int)

    grp = df.groupby(CODE_COL)
    df["supply_freq_30d"] = grp["_had_supply"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

    # Считаем дни с последней поставки через cumsum групп
    df["days_since_supply"] = grp["_had_supply"].transform(
        lambda x: x.groupby(x.cumsum()).cumcount()
    )
    avg_interval = grp["days_since_supply"].transform(
        lambda x: x.rolling(90, min_periods=1).mean()
    )
    df["supply_interval_ratio"] = (
        df["days_since_supply"] / (avg_interval + 1)
    ).clip(0, 10)
    df["supply_overdue"] = (df["supply_interval_ratio"] > 1.5).astype(int)

    return df


def _add_sales_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Скорость и ускорение продаж (соотношение темпа 7д к 14д)."""
    grp = df.groupby(CODE_COL)
    sales_7d = grp["avg_sales_pct"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    sales_14d = grp["avg_sales_pct"].transform(lambda x: x.rolling(14, min_periods=1).mean())

    df["sales_velocity"] = (sales_7d / (sales_14d + 1e-9)).clip(0, 10)
    df["sales_accelerating"] = (df["sales_velocity"] > 1.3).astype(int)
    df["sales_acceleration"] = grp["sales_velocity"].transform(
        lambda x: x.diff().fillna(0)
    ).clip(-5, 5)
    return df


def _add_runway_features(df: pd.DataFrame, dists: list) -> pd.DataFrame:
    """Дни запаса (runway) на уровне всего продукта и каждого дистрибьютора."""
    safe_sales = df["avg_sales_pct_7d"].replace(0, np.nan)
    df["days_of_stock"] = (1.0 / safe_sales).clip(0, 365).fillna(365)

    df["days_of_stock_cat"] = pd.cut(
        df["days_of_stock"],
        bins=[0, 3, 7, 14, 30, 365, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
        right=True,
    ).astype(float)

    df["runway_critical"] = (df["days_of_stock"] < 7).astype(int)
    df["runway_warning"] = (df["days_of_stock"] < 14).astype(int)
    df["runway_caution"] = (df["days_of_stock"] < 30).astype(int)

    for dist in dists:
        s = df[f"{dist}_sales_pct"].replace(0, np.nan)
        df[f"{dist}_runway"] = (1.0 / s).clip(0, 365).fillna(365)

    runway_cols = [f"{d}_runway" for d in dists]
    df["min_dist_runway"] = df[runway_cols].min(axis=1)
    df["n_dist_runway_critical"] = (df[runway_cols] < 7).sum(axis=1)
    return df


def _add_risk_flags(df: pd.DataFrame, dists: list) -> pd.DataFrame:
    """Флаги риска по уровню остатков и статистические аномалии."""
    df["n_zero"] = (df[dists] == 0).sum(axis=1)

    med = df["_product_median_exp"].replace(0, np.nan)
    for dist in dists:
        df[f"{dist}_is_low"] = (df[dist] < 0.10 * med).astype(int)
        df[f"{dist}_is_critical"] = (df[dist] < 0.05 * med).astype(int)

    df["n_low"] = df[[f"{d}_is_low" for d in dists]].sum(axis=1)
    df["n_critical"] = df[[f"{d}_is_critical" for d in dists]].sum(axis=1)

    grp = df.groupby(CODE_COL)

    stock_mean = grp["total_stock"].transform(lambda x: x.rolling(30, min_periods=2).mean())
    stock_std = grp["total_stock"].transform(lambda x: x.rolling(30, min_periods=2).std())
    df["stock_zscore"] = (
        (df["total_stock"] - stock_mean) / (stock_std + 1e-9)
    ).clip(-5, 5).fillna(0)
    df["stock_anomaly_low"] = (df["stock_zscore"] < -2).astype(int)

    sales_mean = grp["avg_sales_pct"].transform(lambda x: x.rolling(30, min_periods=2).mean())
    sales_std = grp["avg_sales_pct"].transform(lambda x: x.rolling(30, min_periods=2).std())
    df["sales_zscore"] = (
        (df["avg_sales_pct"] - sales_mean) / (sales_std + 1e-9)
    ).clip(-5, 5).fillna(0)
    df["sales_spike"] = (df["sales_zscore"] > 2).astype(int)

    return df


def _add_distributor_sync(df: pd.DataFrame, dists: list) -> pd.DataFrame:
    """Синхронизация дистрибьюторов: одновременное падение, концентрация (HHI), CV."""
    df["n_dist_falling"] = df[[f"{d}_falling" for d in dists]].sum(axis=1)
    df["n_dist_rising"] = df[[f"{d}_rising" for d in dists]].sum(axis=1)
    df["all_falling"] = (df["n_dist_falling"] == len(dists)).astype(int)

    if len(dists) > 1:
        dist_vals = df[dists]
        dist_mean = dist_vals.mean(axis=1)
        dist_std = dist_vals.std(axis=1)
        df["stock_cv_cross"] = (dist_std / (dist_mean + 1e-9)).clip(0, 10).fillna(0)

        dist_sum = dist_vals.sum(axis=1).replace(0, np.nan)
        shares = dist_vals.div(dist_sum, axis=0).fillna(0)
        df["stock_hhi"] = (shares ** 2).sum(axis=1)

        for dist in dists:
            df[f"{dist}_share"] = (df[dist] / dist_sum).fillna(0)
    else:
        df["stock_cv_cross"] = 0.0
        df["stock_hhi"] = 1.0

    return df


def _add_supply_demand_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Стресс спроса/предложения: дефицит поставок, устойчивый отток, хроническая дефектура."""
    df["supply_sales_ratio"] = (
        df["avg_supply_pct"] / (df["avg_sales_pct"] + 1e-9)
    ).clip(0, 10)
    df["supply_deficit"] = (df["avg_supply_pct"] < 0.8 * df["avg_sales_pct"]).astype(int)
    df["net_flow"] = df["avg_supply_pct"] - df["avg_sales_pct"]

    grp = df.groupby(CODE_COL)
    df["net_flow_7d"] = grp["net_flow"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["persistent_outflow"] = (df["net_flow_7d"] < -0.01).astype(int)

    # Средняя длительность OOS-эпизода: сумма OOS-дней / количество эпизодов за 90д
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

    return df


def _add_distributor_events(df: pd.DataFrame, dists: list) -> pd.DataFrame:
    """События дистрибьюторов: переходы в ноль, серии падений, распространение нулей."""
    grp = df.groupby(CODE_COL)

    for dist in dists:
        prev_val = grp[dist].transform(lambda x: x.shift(1))
        df[f"{dist}_went_zero_today"] = (
            (df[dist] == 0) & (prev_val > 0)
        ).astype(int)
        df[f"{dist}_first_zero_rate_60d"] = grp[f"{dist}_went_zero_today"].transform(
            lambda x: x.rolling(60, min_periods=1).mean()
        )

    df["n_went_zero_today"] = df[[f"{d}_went_zero_today" for d in dists]].sum(axis=1)

    # Количество дней падения подряд у всех дистрибьюторов одновременно
    df["consecutive_falls"] = grp["all_falling"].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    # Количество дней без поставок подряд
    df["consecutive_no_supply"] = grp["_had_supply"].transform(
        lambda x: (1 - x) * (
            (1 - x).groupby(((1 - x) != (1 - x).shift()).cumsum()).cumcount() + 1
        )
    )

    prev_n_zero = grp["n_zero"].transform(lambda x: x.shift(1))
    df["zeros_spreading"] = (df["n_zero"] > prev_n_zero).fillna(False).astype(int)

    return df


def _add_category_context(
    df: pd.DataFrame, catalog: pd.DataFrame = None
) -> pd.DataFrame:
    """Категорийный и МНН-контекст: стресс группы, доля SKU в МНН."""
    if catalog is not None:
        merge_cols = [c for c in ["code_kag", "mnn", "NM_F", "NM_DT"] if c in catalog.columns]
        if len(merge_cols) > 1:
            df = df.merge(
                catalog[merge_cols].drop_duplicates("code_kag"),
                on="code_kag",
                how="left",
            )

    grp = df.groupby(CODE_COL)

    if "NM_F" in df.columns:
        df["cat_oos_rate_today"] = df.groupby([DATE_COL, "NM_F"])["full_oos"].transform("mean")
        df["cat_oos_rate_7d"] = grp["cat_oos_rate_today"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df["category_stress"] = (df["cat_oos_rate_today"] > 0.10).astype(int)
        df["cat_n_oos_today"] = df.groupby([DATE_COL, "NM_F"])["full_oos"].transform("sum")
        df["cat_oos_ratio"] = df.groupby([DATE_COL, "NM_F"])["full_oos"].transform("mean")

        prev_cat = grp["cat_oos_rate_today"].transform(lambda x: x.shift(1))
        df["cat_oos_growth"] = (df["cat_oos_rate_today"] - prev_cat).fillna(0)
        df["category_deteriorating"] = (df["cat_oos_growth"] > 0.02).astype(int)

    if "mnn" in df.columns:
        df["mnn_oos_rate"] = df.groupby([DATE_COL, "mnn"])["full_oos"].transform("mean")
        df["mnn_stress"] = (df["mnn_oos_rate"] > 0.15).astype(int)
        mnn_total = df.groupby([DATE_COL, "mnn"])["total_stock"].transform("sum")
        df["sku_share_in_mnn"] = (df["total_stock"] / (mnn_total + 1e-9)).clip(0, 1)

    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Временные признаки с циклическим синус/косинус кодированием."""
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
    return df


def _add_composite_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Составные сигналы риска: счётчик флагов и критическая комбинация."""
    individual_flags = [
        "runway_critical", "sales_accelerating", "supply_deficit",
        "zeros_spreading", "persistent_outflow", "stock_anomaly_low",
    ]
    present_flags = [f for f in individual_flags if f in df.columns]
    df["n_red_flags"] = df[present_flags].sum(axis=1)

    # Критическая комбинация: мало запаса + долго нет поставок + ускорение продаж
    df["critical_combination"] = (
        (df["days_of_stock"] < 14)
        & (df["days_since_supply"] > 7)
        & (df["sales_velocity"] > 1.2)
    ).astype(int)

    return df


# ===========================================================================
# Публичный API
# ===========================================================================

def create_all_features_clean(
    df: pd.DataFrame,
    catalog: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Создать все 155+ признаков для модели OOS.

    Оркестрирует последовательный вызов приватных _add_* функций.
    Порядок важен: каждая функция может опираться на колонки,
    созданные предыдущими.

    Args:
        df:      DataFrame с остатками. Обязательные колонки:
                 [date, code_kag, puls_stock, katren_stock,
                  protek_stock, farm_stock, gk_stock]
        catalog: Опциональный справочник [code_kag, mnn, NM_F, NM_DT]

    Returns:
        DataFrame со всеми признаками.
    """
    logger.info("Запуск feature engineering...")

    df = df.copy().sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Нормализуем остатки: NaN и отрицательные → 0
    dists = [c for c in DIST_COLS if c in df.columns]
    for col in dists:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    df = _add_base_aggregates(df, dists)
    df = _add_velocity_features(df, dists)
    df = _add_rolling_features(df)
    df = _add_supply_frequency(df)
    df = _add_sales_dynamics(df)
    df = _add_runway_features(df, dists)
    df = _add_risk_flags(df, dists)
    df = _add_distributor_sync(df, dists)
    df = _add_supply_demand_stress(df)
    df = _add_distributor_events(df, dists)
    df = _add_category_context(df, catalog)
    df = _add_temporal_features(df)
    df = _add_composite_signals(df)

    logger.info(f"Feature engineering завершён: {len(df.columns)} колонок итого")
    return df


def create_predictive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Дополнительные ранние предикторы OOS.
    Вызывается после create_all_features_clean.

    Добавляет:
    - Изменение дней запаса за 7/14 дней назад
    - Скользящее среднее числа красных флагов (тренд)
    - Историческая вероятность OOS за 90 дней
    """
    df = df.copy()
    grp = df.groupby(CODE_COL)

    if "days_of_stock" in df.columns:
        for lag in [7, 14]:
            prev = grp["days_of_stock"].transform(lambda x, l=lag: x.shift(l))
            df[f"dos_change_{lag}d"] = (df["days_of_stock"] - prev).clip(-365, 365).fillna(0)

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


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Вернуть список признаков для модели.
    Исключает служебные, целевые и идентификационные колонки.
    """
    exclude = {
        DATE_COL, CODE_COL,
        "target", "target_future_oos", "target_multiclass",
        "_is_oos", "_product_median_exp", "_product_max_exp", "_had_supply",
        "NM_F", "mnn", "NM_DT",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]
