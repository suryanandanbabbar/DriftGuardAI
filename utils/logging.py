from __future__ import annotations

import json
import logging
from typing import Any

from core.config import LoggingSettings, get_settings

STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in STANDARD_LOG_RECORD_FIELDS and not key.startswith("_")
        }
        if extra_fields:
            payload.update(extra_fields)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(logging_settings: LoggingSettings | None = None) -> None:
    settings = logging_settings or get_settings().logging
    resolved_log_level = getattr(logging, settings.level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(resolved_log_level)

    handler = logging.StreamHandler()
    if settings.structured:
        handler.setFormatter(StructuredFormatter(datefmt=settings.timestamp_format))
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt=settings.timestamp_format,
            ),
        )

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: Any,
) -> None:
    logger.log(level, message, extra=fields)
