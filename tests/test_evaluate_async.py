"""Tests for evaluate_tool async execution."""

import asyncio

import pytest
from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager
from sktime_mcp.tools.evaluate import evaluate_tool


async def _wait_for_job(job_manager, job_id: str, timeout: float = 30.0):
    """Poll job status until terminal or timeout."""
    deadline = int(timeout / 0.05)
    for _ in range(deadline):
        job = job_manager.get_job(job_id)
        if job is not None and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return job
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")


async def test_run_async_returns_job_id_and_completes():
    """evaluate_tool(run_async=True) schedules a job that completes with the expected result."""
    executor = get_executor()
    job_manager = get_job_manager()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})

    try:
        result = evaluate_tool(
            estimator_handle=handle,
            y="airline",
            cv_folds=2,
            run_async=True,
        )
        assert result["success"]
        assert "job_id" in result
        assert result["status"] == "running"

        job = await _wait_for_job(job_manager, result["job_id"])
        assert job.status is JobStatus.COMPLETED, job.errors
        assert job.result is not None
        assert "fold_results" in job.result
        assert "metrics" in job.result
        assert "summary" in job.result
        assert len(job.result["fold_results"]) == 2

    finally:
        executor._handle_manager.release_handle(handle)


async def test_evaluate_async_direct():
    """executor.evaluate_async completes and returns fold_results, metrics, summary."""
    executor = get_executor()
    job_manager = get_job_manager()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    job_id = job_manager.create_job(
        job_type="evaluate",
        estimator_handle=handle,
        estimator_name="NaiveForecaster",
        total_steps=3,
    )

    try:
        result = await executor.evaluate_async(
            handle_id=handle,
            y="airline",
            cv_folds=2,
            job_id=job_id,
        )
        assert result["success"]
        assert "fold_results" in result
        assert "metrics" in result
        assert "summary" in result
        assert len(result["fold_results"]) == 2

    finally:
        executor._handle_manager.release_handle(handle)


async def test_evaluate_async_bad_handle():
    """executor.evaluate_async surfaces a handle-not-found error on the job."""
    executor = get_executor()
    job_manager = get_job_manager()
    job_id = job_manager.create_job(
        job_type="evaluate",
        estimator_handle="does_not_exist",
        estimator_name="Unknown",
        total_steps=3,
    )

    result = await executor.evaluate_async(
        handle_id="does_not_exist",
        y="airline",
        cv_folds=2,
        job_id=job_id,
    )
    assert not result["success"]
    assert "Handle not found" in result["error"]
    job = job_manager.get_job(job_id)
    assert job is not None and job.status is JobStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
