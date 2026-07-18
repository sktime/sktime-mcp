"""Tests for real background-job cancellation and task retention (X-4 + P7-2)."""

import asyncio

import pytest

from sktime_mcp.runtime.jobs import JobStatus, get_job_manager


async def test_cancel_job_cancels_registered_task():
    """cancel_job cancels the running asyncio task, not just the status."""
    job_manager = get_job_manager()
    job_id = job_manager.create_job("fit", "handle", "ARIMA")
    job_manager.update_job(job_id, status=JobStatus.RUNNING)

    started = asyncio.Event()

    async def long_running():
        started.set()
        await asyncio.sleep(60)  # long enough to be cancelled mid-flight

    task = asyncio.create_task(long_running())
    job_manager.register_task(job_id, task)
    await started.wait()

    assert job_manager.cancel_job(job_id) is True

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert job_manager.get_job(job_id).status is JobStatus.CANCELLED

    job_manager.delete_job(job_id)


async def test_task_reference_cleared_on_completion():
    """The retained task reference is dropped once the task finishes."""
    job_manager = get_job_manager()
    job_id = job_manager.create_job("fit", "handle", "ARIMA")

    async def quick():
        return "done"

    task = asyncio.create_task(quick())
    job_manager.register_task(job_id, task)
    assert job_id in job_manager._tasks

    await task
    await asyncio.sleep(0)  # let the done callback run

    assert job_id not in job_manager._tasks
    job_manager.delete_job(job_id)


async def test_task_exception_is_surfaced(caplog):
    """A fire-and-forget task exception is logged instead of silently swallowed."""
    job_manager = get_job_manager()
    job_id = job_manager.create_job("fit", "handle", "ARIMA")

    async def boom():
        raise ValueError("kaboom")

    task = asyncio.create_task(boom())
    job_manager.register_task(job_id, task)

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            await task
        await asyncio.sleep(0)  # let the done callback run

    assert "kaboom" in caplog.text
    job_manager.delete_job(job_id)


async def test_cancel_completed_job_returns_false():
    """A job that already finished cannot be cancelled."""
    job_manager = get_job_manager()
    job_id = job_manager.create_job("fit", "handle", "ARIMA")
    job_manager.update_job(job_id, status=JobStatus.COMPLETED)

    assert job_manager.cancel_job(job_id) is False

    job_manager.delete_job(job_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
