# src/etl/sql_loader.py

from src.utils.helpers import load_config
from src.utils.db_connector import DatabaseConnector
from loguru import logger

# Загружаем конфиг и создаём коннекторы
config = load_config()
db_dwh = DatabaseConnector(config)                    # dwh_price (из конфига)
db_cube = DatabaseConnector(config, db_name="cube")   # cube (явно)


def load_stock_gk(start_date: str, end_date: str):
    """Остатки ГК из dwh_price."""
    query = f"""
        SELECT *
        FROM [dbo].[AI_stock_farm_market_table] WITH (NOLOCK)
        WHERE [Дата] >= '{start_date}'
          AND [Дата] < '{end_date}'
        OPTION (MAXDOP 15)
    """
    logger.info(f"Загрузка остатков ГК за {start_date} — {end_date}")
    return db_dwh.read_sql(query)


def load_competitors(start_date: str, end_date: str):
    """Остатки конкурентов из dwh_price."""
    query = f"""
        SELECT *
        FROM [dbo].[твоя_таблица_конкурентов] WITH (NOLOCK)
        WHERE [Дата] >= '{start_date}'
          AND [Дата] < '{end_date}'
    """
    logger.info(f"Загрузка конкурентов за {start_date} — {end_date}")
    return db_dwh.read_sql(query)


def load_sales(start_date: str, end_date: str):
    """Продажи / отгрузки из dwh_price."""
    query = f"""
        SELECT *
        FROM [dbo].[твоя_таблица_продаж] WITH (NOLOCK)
        WHERE [Дата] >= '{start_date}'
          AND [Дата] < '{end_date}'
    """
    logger.info(f"Загрузка продаж за {start_date} — {end_date}")
    return db_dwh.read_sql(query)


def load_catalog():
    """Справочник МНН/КАГ/группы из cube."""
    query = "SELECT * FROM [dbo].[твоя_таблица_справочника]"
    logger.info("Загрузка справочника из cube")
    return db_cube.read_sql(query)