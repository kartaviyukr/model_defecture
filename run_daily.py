"""
Ежедневный оркестратор системы прогнозирования дефектуры.

Запускайте этот скрипт один раз в день (например, через cron или Task Scheduler).
Скрипт полностью автономен: загружает данные, строит прогноз, сохраняет отчёт,
отправляет уведомление.

Режимы запуска
--------------
  python run_daily.py                  # стандартный ежедневный прогноз
  python run_daily.py --retrain        # переобучить модель перед прогнозом
  python run_daily.py --retrain --tune # переобучить с подбором гиперпараметров
  python run_daily.py --source db      # загрузить данные из БД (sql_loader)
  python run_daily.py --source csv     # загрузить данные из CSV (по умолчанию)
  python run_daily.py --dry-run        # проверка без сохранения и email

Пример cron (каждый день в 07:00):
  0 7 * * 1-5 cd /path/to/model_defecture && python run_daily.py >> logs/cron.log 2>&1

Пример Windows Task Scheduler:
  Action: python C:\\path\\to\\model_defecture\\run_daily.py
  Trigger: Daily at 07:00
"""

import argparse
import glob
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Импорты модулей проекта
# ---------------------------------------------------------------------------
from helpers import load_config, ensure_dirs, today_str
from logger import setup_logger
from data_validation import validate_stock_df, clean_stock_df
from feature_engineering import (
    create_all_features_clean,
    create_predictive_features,
)
from model_registry import get_latest_model, load_model, list_models
from predict import predict
from excel_report import save_excel_report, generate_summary_report
from email_sender import send_alert_email, send_training_complete_email


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

CONFIG_PATH = "configs/config.yaml"
LOCK_FILE = ".run_daily.lock"       # блокировка от двойного запуска
MIN_DAYS_HISTORY = 90               # минимум дней истории для прогноза
RETRAIN_INTERVAL_DAYS = 30          # переобучать модель раз в N дней (авто-режим)


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_data_from_csv(config: dict) -> pd.DataFrame:
    """Загрузить все CSV-файлы из директории data/raw/."""
    data_dir = config["paths"].get("data", "data/raw")
    chunks = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if not chunks:
        raise FileNotFoundError(
            f"CSV файлы не найдены в '{data_dir}'.\n"
            f"Положите файлы с остатками в эту директорию или используйте --source db."
        )

    logger.info(f"Загрузка {len(chunks)} CSV файлов из {data_dir}")
    dfs = [pd.read_csv(f) for f in chunks]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Загружено: {len(df):,} строк, {df.shape[1]} колонок")
    return df


def load_data_from_db(config: dict, history_days: int = 180) -> pd.DataFrame:
    """
    Загрузить актуальные данные из MSSQL через sql_loader.

    Загружает историю за последние `history_days` дней — этого достаточно
    для расчёта всех rolling-признаков (максимальное окно — 90 дней).
    """
    from sql_loader import load_stock_gk, load_catalog

    end_date = today_str()
    start_date = (datetime.now() - timedelta(days=history_days)).strftime("%Y-%m-%d")

    logger.info(f"Загрузка из БД за {start_date} — {end_date}")
    df_stock = load_stock_gk(start_date, end_date)

    # Попробуем подгрузить справочник (необязательно — если таблица не настроена, пропускаем)
    try:
        catalog = load_catalog()
        logger.info(f"Справочник загружен: {len(catalog)} позиций")
        return df_stock, catalog
    except Exception as e:
        logger.warning(f"Справочник не загружен (используем только остатки): {e}")
        return df_stock, None


# ---------------------------------------------------------------------------
# Проверка необходимости переобучения
# ---------------------------------------------------------------------------

def should_retrain(config: dict) -> bool:
    """
    Проверить, нужно ли автоматически переобучить модель.

    Переобучаем если:
    - Нет ни одной сохранённой модели
    - Последняя модель старше RETRAIN_INTERVAL_DAYS дней
    """
    models_dir = config["paths"].get("models", "models")
    models = list_models(models_dir)

    if not models:
        logger.info("Нет сохранённых моделей — требуется обучение")
        return True

    latest = models[-1]
    # Имя модели: catboost_oos_YYYYMMDD → извлекаем дату
    try:
        date_part = latest.split("_")[-1]  # "20260413"
        model_date = datetime.strptime(date_part, "%Y%m%d")
        age_days = (datetime.now() - model_date).days
        if age_days >= RETRAIN_INTERVAL_DAYS:
            logger.info(
                f"Модель '{latest}' создана {age_days} дней назад "
                f"(интервал переобучения: {RETRAIN_INTERVAL_DAYS} дней) — требуется переобучение"
            )
            return True
        logger.info(f"Модель '{latest}' актуальна (возраст: {age_days} дней)")
        return False
    except (ValueError, IndexError):
        logger.warning(f"Не удалось определить дату модели '{latest}' — пропускаем автопереобучение")
        return False


# ---------------------------------------------------------------------------
# Переобучение модели
# ---------------------------------------------------------------------------

def run_retrain(config: dict, config_path: str, tune: bool, dry_run: bool) -> str | None:
    """Переобучить модель. Возвращает имя новой модели или None при ошибке."""
    if dry_run:
        logger.info("[DRY-RUN] Переобучение пропущено")
        return None

    logger.info("=" * 60)
    logger.info("ПЕРЕОБУЧЕНИЕ МОДЕЛИ")
    logger.info("=" * 60)

    try:
        from train import train
        model_name = train(config_path=config_path, tune=tune)

        # Отправить уведомление о завершении обучения
        email_cfg = config.get("email", {})
        if email_cfg.get("enabled", False):
            recipients = email_cfg.get("recipients", [])
            if recipients:
                artifacts = load_model(model_name, config["paths"].get("models", "models"))
                send_training_complete_email(
                    to=recipients,
                    model_name=model_name,
                    metrics=artifacts["metadata"].get("metrics", {}),
                    config=config,
                )

        return model_name

    except Exception as e:
        logger.error(f"Ошибка переобучения: {e}")
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Основная функция ежедневного прогноза
# ---------------------------------------------------------------------------

def run_daily_prediction(
    config: dict,
    config_path: str,
    source: str,
    model_name: str | None,
    dry_run: bool,
) -> dict:
    """
    Выполнить ежедневный прогноз и сформировать отчёт.

    Returns:
        dict с ключами: success, n_alerts, report_path, model_name, date
    """
    result = {
        "success": False,
        "n_alerts": 0,
        "report_path": None,
        "model_name": model_name,
        "date": today_str(),
    }

    # 1. Загрузка данных
    catalog = None
    logger.info(f"Источник данных: {source.upper()}")

    if source == "db":
        try:
            data = load_data_from_db(config)
            if isinstance(data, tuple):
                df_raw, catalog = data
            else:
                df_raw = data
        except Exception as e:
            logger.error(f"Ошибка загрузки из БД: {e}")
            logger.info("Пробуем резервный источник — CSV...")
            df_raw = load_data_from_csv(config)
    else:
        df_raw = load_data_from_csv(config)

    # 2. Валидация
    is_valid, errors = validate_stock_df(df_raw)
    if not is_valid:
        logger.error(f"Данные не прошли валидацию:\n" + "\n".join(errors))
        return result
    df_clean = clean_stock_df(df_raw)

    # 3. Прогноз
    logger.info("Запуск прогноза...")
    try:
        predictions = predict(
            df_stock=df_clean,
            model_name=model_name,
            config_path=config_path,
        )
    except Exception as e:
        logger.error(f"Ошибка прогноза: {e}")
        logger.debug(traceback.format_exc())
        return result

    n_alerts = int(predictions["is_alert"].sum())
    used_model = predictions["model_name"].iloc[0] if "model_name" in predictions.columns else "?"
    result["n_alerts"] = n_alerts
    result["model_name"] = used_model

    logger.info(
        f"Прогноз завершён: {len(predictions)} продуктов, "
        f"алертов: {n_alerts}, модель: {used_model}"
    )

    if dry_run:
        logger.info("[DRY-RUN] Сохранение и email пропущены")
        result["success"] = True
        return result

    # 4. Excel-отчёт
    reports_dir = config["paths"].get("reports", "reports")
    date_str = today_str()
    try:
        report_path = save_excel_report(predictions, date_str, reports_dir)
        result["report_path"] = report_path
    except Exception as e:
        logger.error(f"Ошибка создания Excel-отчёта: {e}")
        logger.debug(traceback.format_exc())

    # 5. Email-уведомление
    email_cfg = config.get("email", {})
    if email_cfg.get("enabled", False):
        recipients = email_cfg.get("recipients", [])
        if recipients:
            try:
                send_alert_email(
                    to=recipients,
                    date_str=date_str,
                    n_alerts=n_alerts,
                    report_path=result["report_path"],
                    config=config,
                )
            except Exception as e:
                logger.error(f"Ошибка отправки email: {e}")
        else:
            logger.warning("Email включён, но список recipients пуст")
    else:
        logger.debug("Email отключён (email.enabled: false в конфиге)")

    result["success"] = True
    return result


# ---------------------------------------------------------------------------
# Блокировка от параллельного запуска
# ---------------------------------------------------------------------------

class RunLock:
    """Простой файловый lock: предотвращает одновременный запуск двух экземпляров."""

    def __enter__(self):
        if Path(LOCK_FILE).exists():
            # Проверим, не завис ли старый процесс (lock старше 3 часов)
            age = datetime.now().timestamp() - Path(LOCK_FILE).stat().st_mtime
            if age < 3 * 3600:
                logger.error(
                    f"Оркестратор уже запущен (lock-файл: {LOCK_FILE}, "
                    f"возраст: {age/60:.0f} мин). Выход."
                )
                sys.exit(1)
            else:
                logger.warning(f"Устаревший lock-файл (возраст {age/3600:.1f} ч) — удаляем")
        Path(LOCK_FILE).write_text(str(os.getpid()))
        return self

    def __exit__(self, *args):
        try:
            Path(LOCK_FILE).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Итоговый вывод в лог
# ---------------------------------------------------------------------------

def _print_summary(step_results: dict, elapsed_sec: float) -> None:
    """Вывести итоговую сводку в лог."""
    retrain = step_results.get("retrain")
    predict = step_results.get("predict", {})

    lines = [
        "",
        "=" * 60,
        "СВОДКА ЕЖЕДНЕВНОГО ЗАПУСКА",
        "=" * 60,
        f"  Дата:             {predict.get('date', today_str())}",
        f"  Время выполнения: {elapsed_sec:.1f} сек",
        f"  Переобучение:     {'да — ' + retrain if retrain else 'нет'}",
        f"  Модель:           {predict.get('model_name', '—')}",
        f"  Алертов:          {predict.get('n_alerts', '—')}",
        f"  Excel-отчёт:      {predict.get('report_path') or '—'}",
        f"  Статус:           {'УСПЕХ ✓' if predict.get('success') else 'ОШИБКА ✗'}",
        "=" * 60,
    ]
    for line in lines:
        logger.info(line)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ежедневный оркестратор прогнозирования дефектуры",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python run_daily.py                      # стандартный запуск
  python run_daily.py --retrain            # с переобучением
  python run_daily.py --retrain --tune     # с переобучением + Optuna
  python run_daily.py --source db          # данные из MSSQL
  python run_daily.py --dry-run            # тест без сохранения
        """,
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH,
        help=f"Путь к YAML-конфигу (по умолчанию: {CONFIG_PATH})"
    )
    parser.add_argument(
        "--source", choices=["csv", "db"], default="csv",
        help="Источник данных: csv — файлы из data/raw/, db — MSSQL через sql_loader"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Переобучить модель перед прогнозом"
    )
    parser.add_argument(
        "--auto-retrain", action="store_true",
        help=f"Переобучить автоматически если модель старше {RETRAIN_INTERVAL_DAYS} дней"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="При переобучении запустить Optuna (только с --retrain или --auto-retrain)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Принудительно использовать указанную модель (имя без расширения)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Выполнить все шаги, но не сохранять файлы и не отправлять email"
    )
    args = parser.parse_args()

    # --- Инициализация ---
    config = load_config(args.config)
    ensure_dirs(config)
    log_dir = config["paths"].get("logs", "logs")
    setup_logger(
        log_file=os.path.join(log_dir, "defectura.log"),
        level="INFO",
    )

    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info(f"ЗАПУСК ОРКЕСТРАТОРА — {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        logger.info("*** РЕЖИМ DRY-RUN — файлы не сохраняются ***")
    logger.info("=" * 60)

    step_results = {}

    with RunLock():
        # ---- Шаг 1: переобучение (если нужно) ----
        do_retrain = args.retrain or (
            args.auto_retrain and should_retrain(config)
        )

        if do_retrain:
            new_model = run_retrain(
                config=config,
                config_path=args.config,
                tune=args.tune,
                dry_run=args.dry_run,
            )
            step_results["retrain"] = new_model
            # Используем только что обученную модель для прогноза
            model_name = new_model or args.model
        else:
            model_name = args.model

        # ---- Шаг 2: ежедневный прогноз ----
        try:
            pred_result = run_daily_prediction(
                config=config,
                config_path=args.config,
                source=args.source,
                model_name=model_name,
                dry_run=args.dry_run,
            )
        except Exception as e:
            logger.critical(f"Критическая ошибка на шаге прогноза: {e}")
            logger.debug(traceback.format_exc())
            pred_result = {"success": False, "date": today_str()}

        step_results["predict"] = pred_result

        # ---- Итог ----
        elapsed = (datetime.now() - start_time).total_seconds()
        _print_summary(step_results, elapsed)

        if not pred_result.get("success"):
            sys.exit(1)  # ненулевой код — сигнал для cron/планировщика об ошибке


if __name__ == "__main__":
    main()
