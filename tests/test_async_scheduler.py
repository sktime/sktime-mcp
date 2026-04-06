"""
Tests for the async scheduler utility (Issue #65).

Verifies that the scheduler reliably schedules coroutines from
synchronous code without using deprecated asyncio.get_event_loop().
"""

import asyncio
import sys
import time
import warnings

import pytest

sys.path.insert(0, "src")

from sktime_mcp.runtime.async_scheduler import AsyncScheduler, get_async_scheduler


class TestAsyncScheduler:
    """Tests for the AsyncScheduler class."""

    def test_schedule_coroutine_completes(self):
        """Scheduled coroutine should run to completion."""
        scheduler = AsyncScheduler()

        async def simple_task():
            return 42

        future = scheduler.schedule(simple_task())
        result = future.result(timeout=5)
        assert result == 42

    def test_schedule_from_sync_no_deprecation_warning(self):
        """Scheduling from sync code should not emit event loop warnings."""
        scheduler = AsyncScheduler()

        async def noop():
            return True

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            future = scheduler.schedule(noop())
            future.result(timeout=5)

        event_loop_warnings = [
            w for w in caught if "no current event loop" in str(w.message).lower()
        ]
        assert len(event_loop_warnings) == 0, (
            f"Got event loop deprecation warnings: {event_loop_warnings}"
        )

    def test_schedule_exception_is_logged(self):
        """Failed coroutines should not raise in the caller; errors are logged."""
        scheduler = AsyncScheduler()

        async def failing_task():
            raise ValueError("test error")

        future = scheduler.schedule(failing_task())

        # The future should contain the exception
        with pytest.raises(ValueError, match="test error"):
            future.result(timeout=5)

    def test_multiple_coroutines(self):
        """Multiple coroutines should all complete."""
        scheduler = AsyncScheduler()
        results = []

        async def append_value(val):
            await asyncio.sleep(0.01)
            return val

        futures = [scheduler.schedule(append_value(i)) for i in range(5)]
        results = [f.result(timeout=5) for f in futures]
        assert sorted(results) == [0, 1, 2, 3, 4]

    def test_singleton_returns_same_instance(self):
        """get_async_scheduler should return the same instance."""
        s1 = get_async_scheduler()
        s2 = get_async_scheduler()
        assert s1 is s2


class TestAsyncToolIntegration:
    """Integration tests: async tools should use the scheduler without warnings."""

    def test_fit_predict_async_no_event_loop_warning(self):
        """fit_predict_async_tool should not emit event loop warnings."""
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        result = instantiate_estimator_tool("NaiveForecaster")
        if not result["success"]:
            pytest.skip("Could not instantiate NaiveForecaster")

        handle = result["handle"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            from sktime_mcp.tools.fit_predict import fit_predict_async_tool

            async_result = fit_predict_async_tool(handle, "airline", horizon=3)

        assert async_result["success"]
        assert "job_id" in async_result

        event_loop_warnings = [
            w for w in caught if "no current event loop" in str(w.message).lower()
        ]
        assert len(event_loop_warnings) == 0

    def test_load_data_source_async_no_event_loop_warning(self):
        """load_data_source_async_tool should not emit event loop warnings."""
        config = {
            "type": "pandas",
            "data": {"date": ["2020-01", "2020-02", "2020-03"], "value": [10, 20, 30]},
            "time_column": "date",
            "target_column": "value",
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            from sktime_mcp.tools.data_tools import load_data_source_async_tool

            result = load_data_source_async_tool(config)

        assert result["success"]
        assert "job_id" in result

        event_loop_warnings = [
            w for w in caught if "no current event loop" in str(w.message).lower()
        ]
        assert len(event_loop_warnings) == 0
