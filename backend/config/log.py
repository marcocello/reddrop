from __future__ import annotations

import logging
import logging.config
import os
import sys

try:
    import colorlog

    _HAS_COLORLOG = True
except Exception:
    colorlog = None
    _HAS_COLORLOG = False

LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

ANSI_LEVEL_COLORS = {
    "DEBUG": "\x1b[36m",
    "INFO": "\x1b[32m",
    "WARNING": "\x1b[33m",
    "ERROR": "\x1b[31m",
    "CRITICAL": "\x1b[1;31m",
}
ANSI_RESET = "\x1b[0m"

_LOGGING_CONFIGURED = False


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.user_email_prefix = ""
        return True


def _log_level() -> str:
    level = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    return level if level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"


def _is_short_format() -> bool:
    return (os.getenv("LOG_FORMAT") or "short").strip().lower() == "short"


def _use_color_output() -> bool:
    if (os.getenv("NO_COLOR") or "").strip():
        return False
    force_color = (os.getenv("FORCE_COLOR") or "").strip().lower()
    if force_color in {"0", "false", "no"}:
        return False
    return True


def _noise_level(base_level: str) -> str:
    return "ERROR" if base_level == "INFO" else base_level


class _AnsiLevelFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: str = "%",
        validate: bool = True,
        use_colors: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self._use_colors:
            color = ANSI_LEVEL_COLORS.get(original_levelname)
            if color:
                record.levelname = f"{color}{original_levelname}{ANSI_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _default_formatter_config(date_fmt: str) -> dict:
    if _HAS_COLORLOG:
        if _is_short_format():
            default_format = "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s"
        else:
            default_format = (
                "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(name)s - %(user_email_prefix)s%(message)s"
            )
        return {
            "()": colorlog.ColoredFormatter,
            "format": default_format,
            "datefmt": date_fmt,
            "log_colors": LOG_COLORS,
            "secondary_log_colors": {},
            "style": "%",
        }

    if _is_short_format():
        default_format = "%(asctime)s - %(levelname)s - %(message)s"
    else:
        default_format = "%(asctime)s - %(levelname)s - %(name)s - %(user_email_prefix)s%(message)s"
    return {
        "()": _AnsiLevelFormatter,
        "format": default_format,
        "datefmt": date_fmt,
        "use_colors": _use_color_output(),
    }


def get_app_log_config() -> dict:
    level = _log_level()
    date_fmt = "%H:%M:%S" if _is_short_format() else "%Y-%m-%d %H:%M:%S"
    noise_level = _noise_level(level)

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": _default_formatter_config(date_fmt)},
        "filters": {"context": {"()": _ContextFilter}},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
                "filters": ["context"],
            }
        },
        "root": {"handlers": ["default"], "level": level},
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": level, "propagate": False},
            "fastapi": {"handlers": ["default"], "level": level, "propagate": False},
            "apscheduler": {"handlers": ["default"], "level": noise_level, "propagate": False},
            "httpx": {"handlers": ["default"], "level": noise_level, "propagate": False},
            "httpcore": {"handlers": ["default"], "level": noise_level, "propagate": False},
            "openai": {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "openai._base_client": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }


def setup_logging(name: str) -> logging.Logger:
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        try:
            logging.config.dictConfig(get_app_log_config())
        except Exception:
            logging.basicConfig(
                level=getattr(logging, _log_level(), logging.INFO),
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                stream=sys.stdout,
            )
        _LOGGING_CONFIGURED = True

    logger = logging.getLogger(name)
    if not any(isinstance(f, _ContextFilter) for f in logger.filters):
        logger.addFilter(_ContextFilter())
    return logger
