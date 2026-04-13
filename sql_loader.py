"""
Загрузка данных из MSSQL-баз данных DWH и Cube.

ВАЖНО: Перед использованием укажите реальные имена таблиц.
Найдите строки с комментарием # TODO и замените заглушки.

Источники:
  db_dwh  — база dwh_price  (остатки ГК, конкуренты, продажи)
  db_cube — база cube        (справочник МНН/КАГ/групп)
"""

import re
from functools import lru_cache

import pandas as pd
from loguru import logger

from helpers import load_config
from db_connector import DatabaseConnector

# Паттерн для валидации дат (YYYY-MM-DD) перед подстановкой в SQL
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(value: str, param_name: str) -> None:
    """
    Проверить, что строка имеет формат YYYY-MM-DD.
    Предотвращает SQL-инъекции через параметры дат.
    """
    if not _DATE_RE.match(value):
        raise ValueError(
            f"Параметр '{param_name}' должен быть в формате YYYY-MM-DD, "
            f"получено: {value!r}"
        )


@lru_cache(maxsize=1)
def _get_connectors():
    """
    Создать коннекторы к БД и закешировать их на время жизни процесса.
    lru_cache гарантирует единственный вызов load_config() и однократное
    создание объектов DatabaseConnector.
    """
    config = load_config()
    db_dwh = DatabaseConnector(config)                   # dwh_price из конфига
    db_cube = DatabaseConnector(config, db_name="cube")  # cube (явно)
    return db_dwh, db_cube


def load_stock_gk(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загрузить остатки ГК из dwh_price.

    Таблица: [dbo].[AI_stock_farm_market_table]
    Ожидаемые колонки: Дата, code_kag, puls_stock, katren_stock,
                       protek_stock, farm_stock, gk_stock

    Args:
        start_date: Начало периода включительно (YYYY-MM-DD)
        end_date:   Конец периода не включая (YYYY-MM-DD)
    """
    _validate_date(start_date, "start_date")
    _validate_date(end_date, "end_date")

    db_dwh, _ = _get_connectors()
    query = f"""
        SELECT *
        FROM [dbo].[AI_stock_farm_market_table] WITH (NOLOCK)
        WHERE [Дата] >= '{start_date}'
          AND [Дата] <  '{end_date}'
        OPTION (MAXDOP 15)
    """
    logger.info(f"Загрузка остатков ГК за {start_date} — {end_date}")
    return db_dwh.read_sql(query)


def load_competitors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загрузить остатки конкурентов из dwh_price.

    # TODO: Укажите реальное имя таблицы вместо заглушки ниже.
    """
    _validate_date(start_date, "start_date")
    _validate_date(end_date, "end_date")

    db_dwh, _ = _get_connectors()
    query = f"""
        SELECT *
        FROM [dbo].[УКАЖИТЕ_ТАБЛИЦУ_КОНКУРЕНТОВ] WITH (NOLOCK)
        WHERE [Дата] >= '{start_date}'
          AND [Дата] <  '{end_date}'
    """
    logger.info(f"Загрузка конкурентов за {start_date} — {end_date}")
    return db_dwh.read_sql(query)


def load_sales(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загрузить продажи / отгрузки из dwh_price.

    # TODO: Укажите реальное имя таблицы вместо заглушки ниже.
    """
    _validate_date(start_date, "start_date")
    _validate_date(end_date, "end_date")

    db_dwh, _ = _get_connectors()
    query = f"""
        SELECT *
        FROM [dbo].[УКАЖИТЕ_ТАБЛИЦУ_ПРОДАЖ] WITH (NOLOCK)
        WHERE [Дата] >= '{start_date}'
          AND [Дата] <  '{end_date}'
    """
    logger.info(f"Загрузка продаж за {start_date} — {end_date}")
    return db_dwh.read_sql(query)


def load_catalog() -> pd.DataFrame:
    """
    Загрузить справочник МНН/КАГ/групп из базы cube.

    # TODO: Укажите реальное имя таблицы вместо заглушки ниже.
    """
    _, db_cube = _get_connectors()
    query = "SELECT * FROM [dbo].[УКАЖИТЕ_ТАБЛИЦУ_СПРАВОЧНИКА]"
    logger.info("Загрузка справочника из cube")
    return db_cube.read_sql(query)