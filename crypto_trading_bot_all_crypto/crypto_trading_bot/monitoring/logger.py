"""Logging with secret redaction.

Redaction is applied to the *final formatted output* (message + traceback), so
secrets cannot leak through exception text either.
"""
from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Mapping

REDACTED = "***REDACTED***"
_SECRET_NAME_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSPHRASE")
_KV_RE = re.compile(r"(?i)((?:api[_-]?key|secret|token|password|passphrase)\s*[=:]\s*)(\S+)")
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def collect_secret_values(env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    return [
        v for k, v in env.items()
        if v and len(v) >= 6 and any(h in k.upper() for h in _SECRET_NAME_HINTS)
    ]


class RedactingFormatter(logging.Formatter):
    def __init__(self, fmt: str = _FMT, secrets: Iterable[str] = ()) -> None:
        super().__init__(fmt)
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return _KV_RE.sub(rf"\1{REDACTED}", text)


def setup_logging(level: str = "INFO", log_dir: Path | str = "logs",
                  env: Mapping[str, str] | None = None) -> None:
    """Configure the 'bot' logger. Safe to call more than once."""
    formatter = RedactingFormatter(secrets=collect_secret_values(env))
    logger = logging.getLogger("bot")
    logger.setLevel(level.upper())
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(log_path / "bot.log", maxBytes=5_000_000, backupCount=5,
                                       encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"bot.{name}")
