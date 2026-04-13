"""
Вспомогательные функции, используемые во всех модулях.

Использование:
    from src.utils.helpers import load_config, normalize_kag_code
    config = load_config()
"""

import os
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Загрузить YAML-конфиг и подставить переменные из .env.

    ${DB_SERVER} в YAML → значение DB_SERVER из .env
    """
    load_dotenv()  # читаем .env в os.environ

    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Подставляем ${VAR} → значения из окружения
    for key, value in os.environ.items():
        raw = raw.replace(f"${{{key}}}", value)

    config = yaml.safe_load(raw)
    logger.debug(f"Конфиг загружен из {config_path}")
    return config


def ensure_dirs(config: dict) -> None:
    """Создать все директории из config['paths'], если не существуют."""
    for key, path in config["paths"].items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Директория проверена: {path}")


def normalize_kag_code(code) -> str:
    """
    Нормализовать KAG-код: '28031.0' → '28031', None → ''.

    Проблема: SQL иногда отдаёт float, pandas делает '28031.0'.
    Все модули должны работать с одним форматом — строка без .0
    """
    if code is None:
        return ""
    s = str(code).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def today_str(fmt: str = "%Y-%m-%d") -> str:
    """Сегодняшняя дата строкой. По умолчанию '2025-01-15'."""
    return datetime.now().strftime(fmt)
