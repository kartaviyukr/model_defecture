"""
Настройка логирования через loguru.

Использование в любом модуле:
    from loguru import logger
    logger.info("Сообщение")

Инициализация один раз в main.py:
    from src.utils.logger import setup_logger
    setup_logger()
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_file: str = "logs/defectura.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Настроить loguru: вывод в консоль + файл с ротацией."""

    # Убираем дефолтный хендлер
    logger.remove()

    # Консоль — цветной вывод
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # Файл — с ротацией и сжатием
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )

    logger.info("Логгер инициализирован")
