"""
Пакетный прогноз: предсказания дефектуры для диапазона дат.

Для каждой даты из диапазона:
  - Берёт историю данных до этой даты включительно
  - Запускает predict() с feature engineering
  - Опционально сохраняет Excel-отчёт
  - Опционально отправляет email-уведомление

Запуск:
    python batch_predict.py --start 2026-01-01 --end 2026-01-31
    python batch_predict.py --daily                # прогноз на сегодня
    python batch_predict.py --daily --email        # с email-уведомлением
"""

import argparse
import glob
import os
import traceback

import pandas as pd
from loguru import logger

from email_sender import send_alert_email
from excel_report import save_excel_report
from helpers import load_config, ensure_dirs, today_str
from logger import setup_logger
from predict import predict


def batch_predict(
    df_stock: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_name: str = None,
    config_path: str = "configs/config.yaml",
    save_reports: bool = True,
    send_email: bool = False,
) -> pd.DataFrame:
    """
    Пакетный прогноз для диапазона дат.

    Для каждой даты из [start_date, end_date]:
      - Срез данных до этой даты (история для rolling-признаков)
      - Инференс модели на эту дату
      - Сохранение Excel-отчёта (если save_reports=True)
      - Отправка email (если send_email=True и email настроен)

    Args:
        df_stock:     Полный DataFrame остатков (должен охватывать весь период)
        start_date:   Начало диапазона прогноза (YYYY-MM-DD)
        end_date:     Конец диапазона прогноза (YYYY-MM-DD)
        model_name:   Имя модели; если None — берётся последняя
        config_path:  Путь к конфигу
        save_reports: Сохранять Excel-отчёты в reports/
        send_email:   Отправлять email-уведомления

    Returns:
        Объединённый DataFrame всех прогнозов со всех дат
    """
    config = load_config(config_path)
    ensure_dirs(config)

    df_stock = df_stock.copy()
    df_stock["date"] = pd.to_datetime(df_stock["date"])

    dates = pd.date_range(start_date, end_date, freq="D")
    all_dates_in_data = set(df_stock["date"].dt.normalize().unique())

    logger.info(
        f"Пакетный прогноз: {start_date} — {end_date} "
        f"({len(dates)} дней в диапазоне)"
    )

    reports_dir = config["paths"].get("reports", "reports")
    email_cfg = config.get("email", {})

    all_results = []

    for pred_date in dates:
        pred_date_norm = pd.Timestamp(pred_date).normalize()

        # Проверяем наличие этой даты в данных
        if pred_date_norm not in all_dates_in_data:
            logger.debug(f"Дата {pred_date.date()} отсутствует в данных — пропускаем")
            continue

        # История до pred_date включительно
        df_slice = df_stock[df_stock["date"] <= pred_date].copy()

        if len(df_slice) == 0:
            logger.warning(f"Нет данных для {pred_date.date()}")
            continue

        logger.info(f"Прогноз для {pred_date.date()}...")
        try:
            result = predict(
                df_slice,
                model_name=model_name,
                config_path=config_path,
            )
        except FileNotFoundError as e:
            # Нет модели — дальнейшая обработка дат бессмысленна
            logger.error(f"Модель не найдена, прерываем пакетный прогноз: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка прогноза для {pred_date.date()}: {e}")
            logger.debug(traceback.format_exc())
            continue

        result["batch_date"] = pred_date.date()
        all_results.append(result)

        if save_reports:
            date_str = pred_date.strftime("%Y-%m-%d")
            report_path = save_excel_report(result, date_str, reports_dir)

            if send_email and email_cfg.get("enabled", False):
                recipients = email_cfg.get("recipients", [])
                if recipients:
                    send_alert_email(
                        to=recipients,
                        date_str=date_str,
                        n_alerts=int(result["is_alert"].sum()),
                        report_path=report_path,
                        config=config,
                    )

    if not all_results:
        logger.warning("Прогнозы не получены — нет подходящих дат в данных")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(f"Пакетный прогноз завершён: {len(combined):,} строк за {len(all_results)} дат")
    return combined


def run_daily(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Ежедневный прогноз: запустить batch_predict для сегодняшней даты.
    Данные загружаются из директории, указанной в конфиге (paths.data).
    """
    config = load_config(config_path)
    setup_logger(config["paths"].get("logs", "logs") + "/predict.log")
    ensure_dirs(config)

    data_dir = config["paths"].get("data", "data/raw")
    chunks = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not chunks:
        raise FileNotFoundError(f"Нет CSV файлов в '{data_dir}'")

    df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True)
    today = today_str()

    return batch_predict(
        df,
        start_date=today,
        end_date=today,
        config_path=config_path,
        save_reports=True,
        send_email=config.get("email", {}).get("enabled", False),
    )


# ---------------------------------------------------------------------------
# CLI точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Пакетный прогноз риска дефектуры лекарственных препаратов"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Путь к YAML-конфигу"
    )
    parser.add_argument(
        "--model", default=None,
        help="Имя модели (без расширения)"
    )
    parser.add_argument(
        "--start", default=None,
        help="Дата начала прогноза (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default=None,
        help="Дата конца прогноза (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--daily", action="store_true",
        help="Запустить ежедневный прогноз (на сегодняшнюю дату)"
    )
    parser.add_argument(
        "--email", action="store_true",
        help="Отправить email-уведомления"
    )
    parser.add_argument(
        "--no-reports", action="store_true",
        help="Не сохранять Excel-отчёты"
    )
    args = parser.parse_args()

    setup_logger()

    if args.daily:
        run_daily(args.config)
    else:
        if not args.start or not args.end:
            parser.error("Укажите --start и --end, или используйте --daily")

        config = load_config(args.config)
        data_dir = config["paths"].get("data", "data/raw")
        chunks = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not chunks:
            raise FileNotFoundError(f"Нет CSV файлов в '{data_dir}'")

        df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True)

        batch_predict(
            df,
            start_date=args.start,
            end_date=args.end,
            model_name=args.model,
            config_path=args.config,
            save_reports=not args.no_reports,
            send_email=args.email,
        )
