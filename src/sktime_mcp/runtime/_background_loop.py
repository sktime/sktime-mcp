"""
Background event loop for async job execution.

Problem: asyncio.run_coroutine_threadsafe() only works if event loop is
already running in another thread. Without this, coroutines submitted
via run_coroutine_threadsafe() sit in the queue forever.

Solution: Start a daemon thread that runs an event loop forever, and
submit coroutines to that shared loop.
"""

import asyncio
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class BackgroundLoop:
    """A shared background event loop running in a dedicated thread."""
    
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.Lock()
        self._start()
    
    def _start(self):
        """Start the background thread with a running event loop."""
        with self._lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="sktime-mcp-background-loop",
                daemon=True
            )
            self._thread.start()
            self._started = True
            logger.debug("Background event loop started")
    
    def _run_loop(self):
        """Run the event loop forever (called in background thread)."""
        try:
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Background loop error: {e}")
    
    def submit(self, coro):
        """Submit a coroutine to the background event loop."""
        if not self._started or self._loop is None:
            raise RuntimeError("Background loop not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
    
    def is_running(self) -> bool:
        """Check if the background loop is running."""
        return self._started and self._loop is not None and self._loop.is_running()


# Singleton instance
_bg_loop: Optional[BackgroundLoop] = None
_init_lock = threading.Lock()


def get_background_loop() -> BackgroundLoop:
    """Get the singleton BackgroundLoop instance."""
    global _bg_loop
    if _bg_loop is None:
        with _init_lock:
            if _bg_loop is None:
                _bg_loop = BackgroundLoop()
    return _bg_loop


def submit_async(coro):
    """Convenience function to submit a coroutine to the background loop."""
    return get_background_loop().submit(coro)
