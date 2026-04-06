"""
Async scheduler for sktime MCP.

Provides reliable coroutine scheduling from synchronous code
without relying on deprecated asyncio.get_event_loop().
"""

import asyncio
import logging
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any

logger = logging.getLogger(__name__)


class AsyncScheduler:
    """
    Manages a dedicated background event loop for scheduling coroutines
    from synchronous code paths.

    Uses a long-lived daemon thread with its own event loop, avoiding
    the deprecated ``asyncio.get_event_loop()`` pattern in sync contexts.
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start the background loop thread if not already running."""
        if self._loop is not None and self._loop.is_running():
            return self._loop

        with self._lock:
            # Double-check after acquiring lock
            if self._loop is not None and self._loop.is_running():
                return self._loop

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever,
                daemon=True,
                name="sktime-mcp-async-scheduler",
            )
            self._thread.start()
            return self._loop

    def schedule(self, coro: Coroutine[Any, Any, Any]) -> Future:
        """
        Schedule a coroutine on the background event loop.

        Returns a ``concurrent.futures.Future`` that resolves when the
        coroutine completes. Failures are logged via a done-callback.

        Args:
            coro: The coroutine to schedule.

        Returns:
            A Future representing the coroutine's result.
        """
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        future.add_done_callback(self._done_callback)
        return future

    @staticmethod
    def _done_callback(future: Future) -> None:
        """Log unhandled exceptions from background tasks."""
        try:
            exc = future.exception()
            if exc is not None:
                logger.error(
                    "Background task failed with unhandled exception: %s",
                    exc,
                    exc_info=exc,
                )
        except Exception:
            # Future was cancelled or callback itself errored; nothing to do.
            pass


_scheduler_instance: AsyncScheduler | None = None
_scheduler_lock = threading.Lock()


def get_async_scheduler() -> AsyncScheduler:
    """Return the singleton AsyncScheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        with _scheduler_lock:
            if _scheduler_instance is None:
                _scheduler_instance = AsyncScheduler()
    return _scheduler_instance
