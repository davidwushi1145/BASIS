"""
Structured Logging with UUID Request Tracing

Uses structlog for JSON-formatted logs with automatic request ID injection.
"""

import sys
import logging
import structlog
from datetime import datetime
from typing import Optional
from contextvars import ContextVar

from config import settings

# Context variable for request ID (thread-safe)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


def set_request_id(request_id: str):
    """Set request ID for current context"""
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """Get request ID from current context"""
    return request_id_var.get()


def add_request_id(logger, method_name, event_dict):
    """Processor to inject request ID into log events"""
    rid = get_request_id()
    if rid:
        event_dict['request_id'] = rid
    return event_dict


def configure_logging():
    """Configure structured logging"""

    # Determine log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            add_request_id,  # Custom processor for request ID
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()  # JSON output
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Initialize on module import
configure_logging()


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# File logging (if configured)
if settings.LOG_FILE:
    file_handler = logging.FileHandler(settings.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
