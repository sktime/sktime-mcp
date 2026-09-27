"""Regression tests for #556 item 24 / 24b (audit F-38, #65):

* every ``run_async=True`` tool called from plain synchronous code (no running
  event loop) returns a structured error, creates no job and leaves no
  un-awaited coroutine behind;
* inside a loop, ``fit``/``predict``/``evaluate`` with a missing estimator
  handle fail fast with the handle error instead of scheduling a job that
  fails a second later;
* no coroutine in ``src/`` uses ``asyncio.get_event_loop()``.
"""

import asyncio
import gc
import warnings
from pathlib import Path

import pytest
from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager, start_background_job
from sktime_mcp.tools.data_tools import load_data_source_tool
from sktime_mcp.tools.evaluate import evaluate_tool
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool

INLINE = {
    "type": "pandas",
    "data": {"date": ["2020-01", "2020-02", "2020-03"], "value": [1.0, 2.0, 3.0]},
    "time_column": "date",
    "target_column": "value",
}


@pytest.fixture
def handle():
    executor = get_executor()
    hid = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    yield hid
    executor._handle_manager.release_handle(hid)


def _job_ids():
    return set(get_job_manager().jobs)


def _call_without_loop(fn):
    """Call *fn* from sync code, returning (result, recorded warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
        gc.collect()  # force finalisation of any coroutine left behind
    return result, [str(w.message) for w in caught]


def _assert_no_loop_error(result, warned, jobs_before):
    assert result["success"] is False
    assert "job_id" not in result
    assert "event loop" in result["error"]
    assert "run_async=false" in result["error"]
    assert _job_ids() == jobs_before, "no job may be created without a loop"
    assert not [w for w in warned if "never awaited" in w], warned


# ---------------------------------------------------------------------------
# No running loop: structured error, no job, no un-awaited coroutine
# ---------------------------------------------------------------------------


class TestNoRunningLoop:
    def test_load_data_source(self):
        before = _job_ids()
        result, warned = _call_without_loop(lambda: load_data_source_tool(INLINE, run_async=True))
        _assert_no_loop_error(result, warned, before)
        assert "monitor progress" not in str(result)

    def test_fit(self, handle):
        before = _job_ids()
        result, warned = _call_without_loop(
            lambda: fit_tool(estimator_handle=handle, y_dataset="airline", run_async=True)
        )
        _assert_no_loop_error(result, warned, before)

    def test_predict(self, handle):
        before = _job_ids()
        result, warned = _call_without_loop(
            lambda: predict_tool(estimator_handle=handle, horizon=3, run_async=True)
        )
        _assert_no_loop_error(result, warned, before)

    def test_evaluate(self, handle):
        before = _job_ids()
        result, warned = _call_without_loop(
            lambda: evaluate_tool(estimator_handle=handle, y="airline", cv_folds=2, run_async=True)
        )
        _assert_no_loop_error(result, warned, before)

    def test_helper_does_not_build_coroutine_without_loop(self):
        calls = []

        def make_coro(job_id):
            calls.append(job_id)
            raise AssertionError("factory must not be called without a loop")

        before = _job_ids()
        result = start_background_job(
            get_job_manager(), make_coro, job_type="fit", estimator_handle="est_x"
        )
        assert result["success"] is False
        assert calls == []
        assert _job_ids() == before

    def test_sync_path_still_works(self):
        result = load_data_source_tool(INLINE, run_async=False)
        assert result["success"], result
        from sktime_mcp.tools.data_tools import release_data_handle_tool

        release_data_handle_tool(result["data_handle"])


# ---------------------------------------------------------------------------
# Running loop: missing handle fails fast, valid handle schedules a job
# ---------------------------------------------------------------------------


async def _wait(job_id, timeout=30.0):
    jm = get_job_manager()
    for _ in range(int(timeout / 0.05)):
        job = jm.get_job(job_id)
        if job is not None and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return job
        await asyncio.sleep(0.05)
    raise TimeoutError(job_id)


class TestMissingHandleInsideLoop:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: fit_tool(estimator_handle="est_nope", y_dataset="airline", run_async=True),
            lambda: predict_tool(estimator_handle="est_nope", horizon=3, run_async=True),
            lambda: evaluate_tool(
                estimator_handle="est_nope", y="airline", cv_folds=2, run_async=True
            ),
        ],
        ids=["fit", "predict", "evaluate"],
    )
    async def test_unknown_handle_fails_fast(self, call):
        before = _job_ids()
        result = call()
        await asyncio.sleep(0)  # would let a wrongly-scheduled task start
        assert result["success"] is False
        assert "job_id" not in result
        assert "Handle not found: est_nope" in result["error"]
        assert _job_ids() == before

    async def test_evicted_handle_is_distinguished(self):
        executor = get_executor()
        executor._handle_manager._evicted.append("est_gone")
        try:
            before = _job_ids()
            result = predict_tool(estimator_handle="est_gone", horizon=3, run_async=True)
            assert result["success"] is False
            assert "evicted" in result["error"]
            assert _job_ids() == before
        finally:
            executor._handle_manager._evicted.remove("est_gone")

    async def test_valid_handle_schedules_and_completes(self, handle):
        result = fit_tool(estimator_handle=handle, y_dataset="airline", run_async=True)
        assert result["success"], result
        assert result["status"] == "running"
        job = await _wait(result["job_id"])
        assert job.status is JobStatus.COMPLETED, job.errors

    async def test_load_data_source_schedules_and_completes(self):
        result = load_data_source_tool(INLINE, run_async=True)
        assert result["success"], result
        assert result["status"] == "running"
        assert "monitor progress" in result["message"]
        job = await _wait(result["job_id"])
        assert job.status is JobStatus.COMPLETED, job.errors
        from sktime_mcp.tools.data_tools import release_data_handle_tool

        release_data_handle_tool(job.result["data_handle"])


# ---------------------------------------------------------------------------
# No asyncio.get_event_loop() left in the package
# ---------------------------------------------------------------------------


def test_no_get_event_loop_in_src():
    import sktime_mcp

    root = Path(sktime_mcp.__file__).parent
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if "get_event_loop" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
