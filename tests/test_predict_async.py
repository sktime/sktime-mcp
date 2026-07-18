"""Tests for predict_tool async execution."""

import asyncio

import pytest
from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager
from sktime_mcp.tools.fit_predict import predict_tool


async def _wait_for_job(job_manager, job_id: str, timeout: float = 30.0):
    """Poll job status until terminal or timeout."""
    deadline = int(timeout / 0.05)
    for _ in range(deadline):
        job = job_manager.get_job(job_id)
        if job is not None and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return job
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")


def _fitted_handle():
    """Create and fit a NaiveForecaster handle on the airline dataset."""
    executor = get_executor()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    data = executor.load_dataset("airline")
    fit_res = executor.fit(handle, y=data["data"], fh=list(range(1, 4)))
    assert fit_res["success"]
    return handle


async def test_run_async_returns_job_id_and_completes():
    """predict_tool(run_async=True) schedules a job that completes with predictions."""
    executor = get_executor()
    job_manager = get_job_manager()
    handle = _fitted_handle()

    try:
        result = predict_tool(
            estimator_handle=handle,
            horizon=3,
            run_async=True,
        )
        assert result["success"]
        assert "job_id" in result
        assert result["status"] == "running"

        job = await _wait_for_job(job_manager, result["job_id"])
        assert job.status is JobStatus.COMPLETED, job.errors
        assert job.result is not None
        assert "predictions" in job.result
        assert len(job.result["predictions"]) == 3

    finally:
        executor._handle_manager.release_handle(handle)


async def test_predict_async_direct():
    """executor.predict_async completes and returns predictions."""
    executor = get_executor()
    job_manager = get_job_manager()
    handle = _fitted_handle()
    job_id = job_manager.create_job(
        job_type="predict",
        estimator_handle=handle,
        estimator_name="NaiveForecaster",
        total_steps=2,
    )

    try:
        result = await executor.predict_async(
            handle_id=handle,
            horizon=3,
            job_id=job_id,
        )
        assert result["success"]
        assert "predictions" in result
        assert len(result["predictions"]) == 3

    finally:
        executor._handle_manager.release_handle(handle)


async def test_predict_async_unfitted_fails_job():
    """predict_async on an unfitted handle surfaces a failure on the job."""
    executor = get_executor()
    job_manager = get_job_manager()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    job_id = job_manager.create_job(
        job_type="predict",
        estimator_handle=handle,
        estimator_name="NaiveForecaster",
        total_steps=2,
    )

    try:
        result = await executor.predict_async(
            handle_id=handle,
            horizon=3,
            job_id=job_id,
        )
        assert not result["success"]
        job = job_manager.get_job(job_id)
        assert job is not None and job.status is JobStatus.FAILED

    finally:
        executor._handle_manager.release_handle(handle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
