from __future__ import annotations

import logging

from backend.config import log as log_config


def test_setup_logging_short_format_uses_shared_structure(monkeypatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "short")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setattr(log_config, "_LOGGING_CONFIGURED", False)

    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    root_logger.handlers.clear()

    try:
        log_config.setup_logging("tests.logging")
        assert root_logger.handlers
        formatter = root_logger.handlers[0].formatter
        assert formatter is not None
        assert "%(name)s" not in formatter._fmt
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)


def test_app_log_config_configures_uvicorn_and_fastapi() -> None:
    config = log_config.get_app_log_config()

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        assert logger_name in config["loggers"]
        logger_cfg = config["loggers"][logger_name]
        assert logger_cfg["handlers"] == ["default"]
        assert logger_cfg["propagate"] is False


def test_setup_logging_colors_level_without_colorlog(monkeypatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "short")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(log_config, "_HAS_COLORLOG", False)
    monkeypatch.setattr(log_config, "_LOGGING_CONFIGURED", False)

    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    root_logger.handlers.clear()

    try:
        logger = log_config.setup_logging("tests.logging")
        handler = root_logger.handlers[0]
        assert handler.formatter is not None
        record = logger.makeRecord(
            name="tests.logging",
            level=logging.INFO,
            fn=__file__,
            lno=1,
            msg="message",
            args=(),
            exc_info=None,
        )
        formatted = handler.formatter.format(record)
        assert "\x1b[" in formatted
        assert "INFO" in formatted
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
