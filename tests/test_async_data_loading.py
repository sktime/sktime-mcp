"""
Test async data loading tool (Issue #6).

Verifies that load_data_source_async_tool schedules data loading
as a background job and resolves to a valid data_handle.
"""

import sys
import asyncio
import unittest

sys.path.insert(0, "src")

from sktime_mcp.tools.data_tools import load_data_source_async_tool
from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import get_job_manager, JobStatus


class TestAsyncDataLoadingTool(unittest.TestCase):
    """Test the tool-level function returns a job_id."""

    def test_returns_job_id(self):
        """Async load should return a job_id immediately."""
        config = {
            "type": "pandas",
            "data": {"date": ["2020-01", "2020-02", "2020-03"],
                     "value": [10, 20, 30]},
            "time_column": "date",
            "target_column": "value",
        }
        result = load_data_source_async_tool(config)
        self.assertTrue(result["success"])
        self.assertIn("job_id", result)
        self.assertEqual(result["source_type"], "pandas")


class TestAsyncDataLoadingExecutor(unittest.TestCase):
    """Test the executor-level async method directly."""

    def test_async_load_completes(self):
        """Async load should complete and produce a data_handle."""
        executor = get_executor()

        config = {
            "type": "pandas",
            "data": {"date": ["2020-01", "2020-02", "2020-03"],
                     "value": [10, 20, 30]},
            "time_column": "date",
            "target_column": "value",
        }
        result = asyncio.run(
            executor.load_data_source_async(config)
        )
        self.assertTrue(result["success"])
        self.assertIn("data_handle", result)

    def test_job_resolves_to_data_handle(self):
        """Job result should contain the data_handle."""
        executor = get_executor()
        job_manager = get_job_manager()

        config = {
            "type": "pandas",
            "data": {"date": ["2020-01", "2020-02", "2020-03"],
                     "value": [10, 20, 30]},
            "time_column": "date",
            "target_column": "value",
        }
        result = asyncio.run(
            executor.load_data_source_async(config)
        )

        # find the most recent data_loading job
        jobs = job_manager.list_jobs()
        data_jobs = [
            j for j in jobs if j.job_type == "data_loading"
        ]
        self.assertTrue(len(data_jobs) > 0)

        latest = data_jobs[0]
        self.assertEqual(latest.status, JobStatus.COMPLETED)
        self.assertIsNotNone(latest.result)
        self.assertIn("data_handle", latest.result)

    def test_data_handle_usable(self):
        """Data handle from async load should work for fit_predict."""
        executor = get_executor()

        config = {
            "type": "pandas",
            "data": {"date": ["2020-01", "2020-02", "2020-03"],
                     "value": [10, 20, 30]},
            "time_column": "date",
            "target_column": "value",
        }
        result = asyncio.run(
            executor.load_data_source_async(config)
        )
        handle = result["data_handle"]

        # verify the handle exists
        self.assertIn(
            handle, executor._data_handles
        )


class TestAsyncDataLoadingErrors(unittest.TestCase):
    """Test error handling for async data loading."""

    def test_invalid_config(self):
        """Bad config should fail the job."""
        executor = get_executor()

        config = {"type": "nonexistent_source"}

        result = asyncio.run(
            executor.load_data_source_async(config)
        )
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
