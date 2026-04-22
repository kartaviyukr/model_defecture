"""
Tests for email_sender.py.

Pure-Python functions (_get_smtp_config, _build_alert_body, _build_training_body)
are tested with full assertions.
_send is tested with a mocked smtplib.SMTP.
"""
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from email_sender import (
    _get_smtp_config,
    _build_alert_body,
    _build_training_body,
    _build_message,
    _send,
    send_alert_email,
    send_training_complete_email,
)


# ---------------------------------------------------------------------------
# _get_smtp_config
# ---------------------------------------------------------------------------

class TestGetSmtpConfig:
    def test_returns_none_when_no_config_and_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            result = _get_smtp_config(None)
        assert result is None

    def test_returns_none_when_host_missing(self):
        with patch.dict(os.environ, {"EMAIL_SENDER": "s@x.com", "EMAIL_PASSWORD": "pw"}, clear=True):
            result = _get_smtp_config(None)
        assert result is None

    def test_returns_none_when_sender_missing(self):
        with patch.dict(os.environ, {"SMTP_HOST": "smtp.example.com", "EMAIL_PASSWORD": "pw"}, clear=True):
            result = _get_smtp_config(None)
        assert result is None

    def test_returns_none_when_password_missing(self):
        with patch.dict(os.environ, {"SMTP_HOST": "smtp.example.com", "EMAIL_SENDER": "s@x.com"}, clear=True):
            result = _get_smtp_config(None)
        assert result is None

    def test_reads_from_env_vars(self):
        env = {
            "SMTP_HOST": "smtp.example.com",
            "EMAIL_SENDER": "sender@x.com",
            "EMAIL_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            result = _get_smtp_config(None)
        assert result is not None
        assert result["host"] == "smtp.example.com"
        assert result["sender"] == "sender@x.com"
        assert result["password"] == "secret"
        assert result["port"] == 587  # default

    def test_reads_port_from_env(self):
        env = {
            "SMTP_HOST": "smtp.example.com",
            "EMAIL_SENDER": "s@x.com",
            "EMAIL_PASSWORD": "pw",
            "SMTP_PORT": "465",
        }
        with patch.dict(os.environ, env, clear=True):
            result = _get_smtp_config(None)
        assert result["port"] == 465

    def test_reads_from_config_dict(self):
        config = {
            "email": {
                "smtp_host": "mail.corp.com",
                "smtp_port": 2525,
                "sender": "noreply@corp.com",
                "password": "pa$$",
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            result = _get_smtp_config(config)
        assert result["host"] == "mail.corp.com"
        assert result["port"] == 2525
        assert result["sender"] == "noreply@corp.com"

    def test_config_takes_precedence_over_env(self):
        config = {
            "email": {
                "smtp_host": "config-host",
                "sender": "config@x.com",
                "password": "cfg-pw",
            }
        }
        env = {
            "SMTP_HOST": "env-host",
            "EMAIL_SENDER": "env@x.com",
            "EMAIL_PASSWORD": "env-pw",
        }
        with patch.dict(os.environ, env, clear=True):
            result = _get_smtp_config(config)
        assert result["host"] == "config-host"

    def test_returns_none_when_config_has_no_email_key_and_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            result = _get_smtp_config({"paths": {"data": "data/raw"}})
        assert result is None


# ---------------------------------------------------------------------------
# _build_alert_body
# ---------------------------------------------------------------------------

class TestBuildAlertBody:
    def test_contains_date(self):
        body = _build_alert_body("2025-03-15", 10, None)
        assert "2025-03-15" in body

    def test_contains_alert_count(self):
        body = _build_alert_body("2025-03-15", 42, None)
        assert "42" in body

    def test_high_alert_count_shows_high_risk(self):
        body = _build_alert_body("2025-03-15", 51, None)
        assert "ВЫСОКИЙ" in body

    def test_medium_alert_count_shows_medium_risk(self):
        body = _build_alert_body("2025-03-15", 30, None)
        assert "СРЕДНИЙ" in body

    def test_low_alert_count_shows_low_risk(self):
        body = _build_alert_body("2025-03-15", 5, None)
        assert "НИЗКИЙ" in body

    def test_boundary_51_is_high(self):
        body = _build_alert_body("2025-01-01", 51, None)
        assert "ВЫСОКИЙ" in body

    def test_boundary_50_is_medium(self):
        body = _build_alert_body("2025-01-01", 50, None)
        assert "СРЕДНИЙ" in body

    def test_boundary_21_is_medium(self):
        body = _build_alert_body("2025-01-01", 21, None)
        assert "СРЕДНИЙ" in body

    def test_boundary_20_is_low(self):
        body = _build_alert_body("2025-01-01", 20, None)
        assert "НИЗКИЙ" in body

    def test_no_report_path_shows_no_attachment(self):
        body = _build_alert_body("2025-01-01", 5, None)
        assert "Excel-отчёт не был сформирован" in body

    def test_nonexistent_report_path_shows_no_attachment(self):
        body = _build_alert_body("2025-01-01", 5, "/no/such/file.xlsx")
        assert "Excel-отчёт не был сформирован" in body

    def test_existing_report_path_shows_filename(self, tmp_path):
        report = tmp_path / "report.xlsx"
        report.write_bytes(b"")
        body = _build_alert_body("2025-01-01", 5, str(report))
        assert "report.xlsx" in body

    def test_returns_html_string(self):
        body = _build_alert_body("2025-01-01", 0, None)
        assert "<html>" in body
        assert "</html>" in body


# ---------------------------------------------------------------------------
# _build_training_body
# ---------------------------------------------------------------------------

class TestBuildTrainingBody:
    def test_contains_model_name(self):
        body = _build_training_body("catboost_oos_20250115", {"roc_auc": 0.85})
        assert "catboost_oos_20250115" in body

    def test_contains_roc_auc(self):
        body = _build_training_body("m", {"roc_auc": 0.8765})
        assert "0.8765" in body

    def test_contains_pr_auc(self):
        body = _build_training_body("m", {"roc_auc": 0.80, "pr_auc": 0.6543})
        assert "0.6543" in body

    def test_handles_missing_metrics_gracefully(self):
        body = _build_training_body("m", {})
        # Missing values should render as "—"
        assert "—" in body

    def test_formats_train_rows_with_thousands(self):
        body = _build_training_body("m", {"train_rows": 150000})
        # Python formats 150000 as "150,000"
        assert "150,000" in body

    def test_returns_html(self):
        body = _build_training_body("m", {})
        assert "<html>" in body
        assert "</html>" in body


# ---------------------------------------------------------------------------
# _build_message
# ---------------------------------------------------------------------------

class TestBuildMessage:
    def test_returns_mime_multipart(self):
        msg = _build_message(
            sender="from@x.com",
            recipients=["to@x.com"],
            subject="Test",
            body_html="<p>Hello</p>",
        )
        assert isinstance(msg, MIMEMultipart)

    def test_subject_set(self):
        msg = _build_message("f@x.com", ["t@x.com"], "My Subject", "<p></p>")
        assert msg["Subject"] == "My Subject"

    def test_from_set(self):
        msg = _build_message("sender@x.com", ["to@x.com"], "S", "<p></p>")
        assert msg["From"] == "sender@x.com"

    def test_to_set_single(self):
        msg = _build_message("f@x.com", ["to@x.com"], "S", "<p></p>")
        assert "to@x.com" in msg["To"]

    def test_to_set_multiple(self):
        msg = _build_message("f@x.com", ["a@x.com", "b@x.com"], "S", "<p></p>")
        assert "a@x.com" in msg["To"]
        assert "b@x.com" in msg["To"]

    def test_attachment_added_when_file_exists(self, tmp_path):
        report = tmp_path / "report.xlsx"
        report.write_bytes(b"XLS")
        msg = _build_message("f@x.com", ["t@x.com"], "S", "<p></p>",
                              attachment_path=str(report))
        # Multipart message should have 2 parts: body + attachment
        assert len(msg.get_payload()) == 2

    def test_no_attachment_when_file_missing(self):
        msg = _build_message("f@x.com", ["t@x.com"], "S", "<p></p>",
                              attachment_path="/no/such.xlsx")
        assert len(msg.get_payload()) == 1


# ---------------------------------------------------------------------------
# _send
# ---------------------------------------------------------------------------

class TestSend:
    def _make_smtp_cfg(self):
        return {"host": "smtp.x.com", "port": 587, "sender": "s@x.com", "password": "pw"}

    def test_returns_true_on_success(self):
        cfg = self._make_smtp_cfg()
        msg = _build_message("s@x.com", ["t@x.com"], "S", "<p></p>")
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = _send(msg, cfg, ["t@x.com"])
        assert result is True

    def test_returns_false_on_auth_error(self):
        cfg = self._make_smtp_cfg()
        msg = _build_message("s@x.com", ["t@x.com"], "S", "<p></p>")
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp_cls.side_effect = smtplib.SMTPAuthenticationError(535, b"auth failed")
            result = _send(msg, cfg, ["t@x.com"])
        assert result is False

    def test_returns_false_on_connect_error(self):
        cfg = self._make_smtp_cfg()
        msg = _build_message("s@x.com", ["t@x.com"], "S", "<p></p>")
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp_cls.side_effect = smtplib.SMTPConnectError(421, b"unavailable")
            result = _send(msg, cfg, ["t@x.com"])
        assert result is False


# ---------------------------------------------------------------------------
# send_alert_email (integration)
# ---------------------------------------------------------------------------

class TestSendAlertEmail:
    def test_returns_false_when_smtp_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            result = send_alert_email(["to@x.com"], "2025-01-01", 5, config={})
        assert result is False

    def test_sends_when_configured(self):
        config = {
            "email": {
                "smtp_host": "smtp.x.com",
                "sender": "s@x.com",
                "password": "pw",
            }
        }
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = send_alert_email(["to@x.com"], "2025-01-01", 10, config=config)
        assert result is True
