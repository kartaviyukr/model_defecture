"""
Отправка email-уведомлений с Excel-отчётами по дефектуре.

Поддерживает:
  - HTML-тело письма с кратким резюме
  - Вложение Excel-файла отчёта
  - SMTP с STARTTLS (порт 587)
  - Конфигурацию через config.yaml или переменные окружения

Функции:
  send_alert_email()            — уведомление о ежедневных алертах
  send_training_complete_email() — уведомление о завершении обучения
"""

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

from loguru import logger


def send_alert_email(
    to: List[str],
    date_str: str,
    n_alerts: int,
    report_path: Optional[str] = None,
    config: Optional[dict] = None,
) -> bool:
    """
    Отправить email с ежедневным отчётом по алертам дефектуры.

    Args:
        to:          Список email-адресов получателей
        date_str:    Дата прогноза (YYYY-MM-DD) для темы письма
        n_alerts:    Количество позиций с флагом is_alert=True
        report_path: Путь к Excel-файлу (прикрепляется к письму, если существует)
        config:      Словарь конфигурации (весь config или секция email)

    Returns:
        True если письмо успешно отправлено, False при ошибке
    """
    smtp_cfg = _get_smtp_config(config)
    if smtp_cfg is None:
        logger.warning("Email не настроен — пропускаем отправку. "
                       "Проверьте SMTP_HOST, EMAIL_SENDER, EMAIL_PASSWORD в .env")
        return False

    subject = f"[Дефектура] {date_str} — {n_alerts} позиций под риском OOS"
    body = _build_alert_body(date_str, n_alerts, report_path)

    msg = _build_message(
        sender=smtp_cfg["sender"],
        recipients=to,
        subject=subject,
        body_html=body,
        attachment_path=report_path,
    )

    return _send(msg, smtp_cfg, to)


def send_training_complete_email(
    to: List[str],
    model_name: str,
    metrics: dict,
    config: Optional[dict] = None,
) -> bool:
    """
    Отправить уведомление о завершении обучения новой модели.

    Args:
        to:         Список email-адресов получателей
        model_name: Имя обученной модели
        metrics:    Словарь метрик {"roc_auc": ..., "pr_auc": ...}
        config:     Конфигурационный словарь

    Returns:
        True при успешной отправке
    """
    smtp_cfg = _get_smtp_config(config)
    if smtp_cfg is None:
        return False

    subject = f"[Дефектура] Модель обучена: {model_name}"
    body = _build_training_body(model_name, metrics)

    msg = _build_message(
        sender=smtp_cfg["sender"],
        recipients=to,
        subject=subject,
        body_html=body,
    )

    return _send(msg, smtp_cfg, to)


# ---------------------------------------------------------------------------
# Внутренние вспомогательные функции
# ---------------------------------------------------------------------------

def _get_smtp_config(config: Optional[dict]) -> Optional[dict]:
    """Извлечь SMTP-параметры из конфига или переменных окружения."""
    cfg = (config or {}).get("email", {})

    host = cfg.get("smtp_host") or os.environ.get("SMTP_HOST", "")
    port = int(cfg.get("smtp_port", os.environ.get("SMTP_PORT", 587)))
    sender = cfg.get("sender") or os.environ.get("EMAIL_SENDER", "")
    password = cfg.get("password") or os.environ.get("EMAIL_PASSWORD", "")

    if not all([host, sender, password]):
        return None

    return {
        "host": host,
        "port": port,
        "sender": sender,
        "password": password,
    }


def _build_message(
    sender: str,
    recipients: List[str],
    subject: str,
    body_html: str,
    attachment_path: Optional[str] = None,
) -> MIMEMultipart:
    """Собрать MIME-сообщение с HTML-телом и опциональным вложением."""
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    if attachment_path and Path(attachment_path).exists():
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=Path(attachment_path).name)
        part["Content-Disposition"] = (
            f'attachment; filename="{Path(attachment_path).name}"'
        )
        msg.attach(part)
        logger.debug(f"Прикреплён файл: {attachment_path}")

    return msg


def _send(msg: MIMEMultipart, smtp_cfg: dict, recipients: List[str]) -> bool:
    """Отправить письмо через SMTP с STARTTLS."""
    host = smtp_cfg["host"]
    port = smtp_cfg["port"]
    sender = smtp_cfg["sender"]
    password = smtp_cfg["password"]

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info(f"Email отправлен на {recipients}")
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error(f"SMTP: ошибка аутентификации для {sender} на {host}:{port}")
    except smtplib.SMTPConnectError:
        logger.error(f"SMTP: не удалось подключиться к {host}:{port}")
    except smtplib.SMTPRecipientsRefused as e:
        logger.error(f"SMTP: получатели отклонены: {e}")
    except Exception as e:
        logger.error(f"SMTP: неожиданная ошибка: {e}")
    return False


def _build_alert_body(date_str: str, n_alerts: int, report_path: Optional[str]) -> str:
    """HTML-тело письма с ежедневным отчётом."""
    if n_alerts > 50:
        risk_color = "#D32F2F"
        risk_label = "ВЫСОКИЙ"
    elif n_alerts > 20:
        risk_color = "#F57C00"
        risk_label = "СРЕДНИЙ"
    else:
        risk_color = "#1976D2"
        risk_label = "НИЗКИЙ"

    attachment_note = (
        f'<p>&#128206; Во вложении Excel-отчёт: '
        f'<b>{Path(report_path).name}</b></p>'
        if report_path and Path(report_path).exists()
        else "<p>Excel-отчёт не был сформирован.</p>"
    )

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 640px; margin: 0 auto; color: #333;">

      <h2 style="color: #2F5496; border-bottom: 2px solid #2F5496; padding-bottom: 8px;">
        Мониторинг дефектуры &mdash; {date_str}
      </h2>

      <div style="background: #f5f5f5; border-radius: 8px; padding: 20px; margin: 16px 0;
                  border-left: 6px solid {risk_color};">
        <p style="margin: 0; font-size: 15px;">Уровень риска:
          <span style="color: {risk_color}; font-weight: bold;">{risk_label}</span>
        </p>
        <p style="margin: 8px 0 0; font-size: 28px; font-weight: bold; color: {risk_color};">
          {n_alerts} позиций под риском OOS
        </p>
        <p style="margin: 4px 0 0; font-size: 13px; color: #666;">
          Горизонт прогноза: 14 дней. Порог алерта: ≥ 0.70
        </p>
      </div>

      {attachment_note}

      <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;">
      <p style="color: #999; font-size: 11px; margin: 0;">
        Письмо сформировано автоматически системой прогнозирования дефектуры.<br>
        Для отписки или изменения настроек уведомлений обратитесь к администратору.
      </p>

    </body>
    </html>
    """


def _build_training_body(model_name: str, metrics: dict) -> str:
    """HTML-тело письма о завершении обучения."""
    roc = metrics.get("roc_auc", "—")
    pr = metrics.get("pr_auc", "—")
    baseline = metrics.get("baseline_pr_auc", "—")
    train_rows = metrics.get("train_rows", "—")
    pos_rate = metrics.get("positive_rate", "—")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    rows = [
        ("Название модели",   model_name),
        ("ROC-AUC",           fmt(roc)),
        ("PR-AUC",            fmt(pr)),
        ("Baseline PR-AUC",   fmt(baseline)),
        ("Строк обучения",    f"{train_rows:,}" if isinstance(train_rows, int) else str(train_rows)),
        ("Доля позитивов",    fmt(pos_rate)),
    ]

    table_rows = "".join(
        f"""<tr style="background: {'#f9f9f9' if i % 2 else 'white'}">
              <td style="padding:8px 12px; border:1px solid #ddd;"><b>{k}</b></td>
              <td style="padding:8px 12px; border:1px solid #ddd;">{v}</td>
            </tr>"""
        for i, (k, v) in enumerate(rows)
    )

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 640px; margin: 0 auto; color: #333;">

      <h2 style="color: #2F5496; border-bottom: 2px solid #2F5496; padding-bottom: 8px;">
        Новая модель успешно обучена
      </h2>

      <table style="border-collapse: collapse; width: 100%; margin: 16px 0;">
        {table_rows}
      </table>

      <p style="color: #999; font-size: 11px;">
        Модель сохранена и готова к использованию для прогнозирования.
      </p>

    </body>
    </html>
    """
