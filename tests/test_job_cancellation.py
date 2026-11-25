"""Tests for real background-job cancellation and task retention (X-4 + P7-2)."""

import asyncio
import threading

import pytest

from sktime_mcp.runtime.jobs import JobStatus, get_job_manager


async def test_cancel_running_fit_keeps_server_responsive():
    """Cancelling a real worker must not join it on the server's event loop."""
    from sktime.forecasting.naive import NaiveForecaster

    from sktime_mcp.runtime.handles import get_handle_manager
    from sktime_mcp.tools.fit_predict import fit_tool

    from .test_mcp_wire import _body, _connected_session

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class HeldNaive(NaiveForecaster):
        def _fit(self, y, X, fh):
            started.set()
            # Bound the fixture even if the old blocking shutdown regresses.
            release.wait(timeout=3)
            try:
                return super()._fit(y, X, fh)
            finally:
                finished.set()

    handles = get_handle_manager()
    jobs = get_job_manager()
    handle = handles.create_handle("HeldNaive", HeldNaive(), params={})
    submitted = fit_tool(handle, y_dataset="airline", run_async=True)
    job_id = submitted["job_id"]
    task = jobs._tasks[job_id]
    try:
        assert await asyncio.to_thread(started.wait, 3)
        async with _connected_session() as session:
            cancelled = _body(await session.call_tool("cancel_job", {"job_id": job_id}))
            assert cancelled["success"]
            with pytest.raises(asyncio.CancelledError):
                await task
            status = _body(await session.call_tool("check_job_status", {"job_id": job_id}))
            assert status["status"] == "cancelled"
            assert not finished.is_set(), "Server waited for the fit worker before responding"
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        # fit() marks the handle after _fit returns; drain its worker before cleanup.
        for _ in range(300):
            if handles.is_fitted(handle):
                break
            await asyncio.sleep(0.01)
        assert jobs.get_job(job_id).status is JobStatus.CANCELLED
        assert jobs.get_job(job_id).result is None
        handles.release_handle(handle)
        jobs.delete_job(job_id)


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
