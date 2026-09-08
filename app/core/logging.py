"""
Structured logging setup for ReconFace.

Provides JSON-formatted log output in production and human-readable output
in development. Log level is controlled via the LOG_LEVEL env var.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Attach extra fields passed via `extra={...}`
        for key in ("request_id", "task_id", "source_url", "face_count"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


class DevFormatter(logging.Formatter):
    """Colorized, human-readable formatter for local development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return (
            f"{color}{timestamp} [{record.levelname:<8}]{self.RESET} "
            f"{record.name}: {record.getMessage()}"
        )


def setup_logging(level: str = "INFO", is_production: bool = False) -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        is_production: If True, use JSON formatter; otherwise use colorized dev output.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any pre-existing handlers to avoid duplicates on reload
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if is_production else DevFormatter())
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "onnxruntime", "insightface"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger scoped to the application namespace."""
    return logging.getLogger(f"reconface.{name}")
