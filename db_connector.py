from contextlib import contextmanager

import pyodbc
import pandas as pd
from loguru import logger


class DatabaseConnector:
    def __init__(self, config, db_name=None):
        db = config["database"]
        # если передали db_name — используем его, иначе берём из конфига
        database = db_name or db["database"]
        self._conn_str = (
            f"DRIVER={{{db['driver']}}};"
            f"SERVER={db['server']};"
            f"DATABASE={database};"
            f"UID={db['username']};"
            f"PWD={db['password']};"
            f"TrustServerCertificate=yes;"
            f"Encrypt=Optional;"
            f"Connection Timeout=30;"
        )
        logger.debug(f"DB connector создан для {db['server']}/{database}")

    @contextmanager
    def connect(self):
        conn = None
        try:
            conn = pyodbc.connect(self._conn_str, autocommit=True)
            logger.debug("Соединение с БД открыто")
            yield conn
        except pyodbc.Error as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            raise
        finally:
            if conn:
                conn.close()
                logger.debug("Соединение с БД закрыто")

    def read_sql(self, query, params=None):
        with self.connect() as conn:
            logger.info(f"Выполняю запрос ({len(query)} символов)")
            df = pd.read_sql(query, conn, params=params)
            logger.info(f"Загружено {len(df):,} строк, {len(df.columns)} столбцов")
            return df