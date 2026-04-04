"""Utilities for scheduling background coroutines safely."""

import asyncio
import logging
import threading
import time
from typing import Any, Coroutine

logger = logging.getLogger(__name__)

_loop_lock = threading.Lock()
_background_loop: asyncio.AbstractEventLoop | None = None
_background_thread: threading.Thread | None = None


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    """Run an event loop forever in a dedicated daemon thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Get a long-lived background event loop, creating it if needed."""
    global _background_loop, _background_thread

    with _loop_lock:
        if _background_loop is not None and _background_loop.is_running():
            return _background_loop

        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_loop_forever,
            args=(loop,),
            daemon=True,
            name="sktime-mcp-background-loop",
        )
        thread.start()

        # Wait briefly for loop startup.
        for _ in range(100):
            if loop.is_running():
                break
            time.sleep(0.01)

        _background_loop = loop
        _background_thread = thread
        return _background_loop


def _log_future_exception(future) -> None:
    """Log unhandled exceptions raised by background tasks."""
    try:
        exception = future.exception()
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Failed to retrieve background task exception")
        return

    if exception is not None:
        logger.error("Background task failed", exc_info=exception)


def schedule_coroutine(coro: Coroutine[Any, Any, Any]) -> None:
    """Schedule a coroutine from sync code without event-loop warnings."""
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        running_loop.create_task(coro)
        return

    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    future.add_done_callback(_log_future_exception)
