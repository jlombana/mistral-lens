"""Shared retry wrapper with exponential backoff for API calls.

Handles transient errors (rate limits, timeouts, server errors)
with configurable retry attempts and backoff parameters.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that trigger a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Exceptions that trigger a retry
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.ReadError,
    ConnectionError,
    TimeoutError,
)

# Default retry configuration
DEFAULT_MAX_RETRIES = 10
DEFAULT_BASE_DELAY = 2.0
DEFAULT_MAX_DELAY = 60.0


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"All {attempts} retry attempts exhausted. Last error: {last_error}"
        )


def _calculate_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Calculate delay with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.

    Returns:
        Delay in seconds with jitter applied.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.5)
    return delay + jitter


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: The exception to check.

    Returns:
        True if the error should trigger a retry.
    """
    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS_CODES
    return False


def retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> Callable:
    """Decorator for synchronous functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.
        max_delay: Maximum delay in seconds between retries.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable_error(e):
                        raise

                    last_error = e
                    if attempt < max_retries - 1:
                        delay = _calculate_delay(attempt, base_delay, max_delay)
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs — %s: %s",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            delay,
                            type(e).__name__,
                            e,
                        )
                        time.sleep(delay)

            raise RetryExhaustedError(max_retries, last_error)  # type: ignore[arg-type]

        return wrapper

    return decorator


def async_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> Callable:
    """Decorator for async functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.
        max_delay: Maximum delay in seconds between retries.

    Returns:
        Decorated async function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable_error(e):
                        raise

                    last_error = e
                    if attempt < max_retries - 1:
                        delay = _calculate_delay(attempt, base_delay, max_delay)
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs — %s: %s",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            delay,
                            type(e).__name__,
                            e,
                        )
                        await asyncio.sleep(delay)

            raise RetryExhaustedError(max_retries, last_error)  # type: ignore[arg-type]

        return wrapper  # type: ignore[return-value]

    return decorator
