import logging

from core.config import get_settings


def configure_logging(log_level: str | None = None) -> None:
    resolved_log_level = log_level or get_settings().log_level
    logging.basicConfig(
        level=getattr(logging, resolved_log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
