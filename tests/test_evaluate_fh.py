"""Tests for evaluate's ``fh`` and ``step_length`` parameters (#556 item 8, F-07/F-53).

``evaluate`` used to hard-wire ``ExpandingWindowSplitter(step_length=1, fh=[1])``,
so every fold scored a single one-step-ahead point and a multi-step evaluation
was not expressible. It also echoed ``cv_folds_requested`` even when
``initial_window`` had overridden it.
"""

import pandas as pd
import pytest
from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager
from sktime_mcp.tools.evaluate import evaluate_tool

AIRLINE_N = 144


@pytest.fixture
def handle():
    executor = get_executor()
    h = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    try:
        yield h
    finally:
        executor._handle_manager.release_handle(h)


def _train_windows(result):
    return [int(f["len_train_window"]) for f in result["fold_results"]]


def _metric(result):
    (name,) = [k for k in result["metrics"] if k.startswith("test_")]
    return result["metrics"][name]


class TestForecastHorizon:
    def test_fh_int_scores_multi_step_folds(self, handle):
        """fh=12 with cv_folds=3 gives 3 folds whose cutoffs each leave 12 held-out points."""
        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3, fh=12)
        assert result["success"], result.get("error")
        assert result["cv_folds_run"] == 3
        assert result["fh"] == list(range(1, 13))
        assert result["step_length"] == 1
        # the last fold trains on n-12 points and scores the final 12; earlier
        # folds step back by one
        assert _train_windows(result) == [AIRLINE_N - 14, AIRLINE_N - 13, AIRLINE_N - 12]
        assert result["initial_window"] == AIRLINE_N - 14
        assert _train_windows(result)[-1] + 12 == AIRLINE_N

    def test_fh_int_changes_metric_vs_one_step(self, handle):
        """A 12-step-ahead evaluation must not silently score one step ahead."""
        one_step = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3)
        twelve = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3, fh=12)
        assert one_step["success"] and twelve["success"]
        assert one_step["fh"] == [1]
        assert _train_windows(one_step) == [AIRLINE_N - 3, AIRLINE_N - 2, AIRLINE_N - 1]
        assert _metric(one_step) != pytest.approx(_metric(twelve))

    def test_fh_list(self, handle):
        """fh=[1, 6, 12] scores exactly those steps; max(fh) governs the fold layout."""
        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3, fh=[1, 6, 12])
        assert result["success"], result.get("error")
        assert result["fh"] == [1, 6, 12]
        assert result["cv_folds_run"] == 3
        assert _train_windows(result) == [AIRLINE_N - 14, AIRLINE_N - 13, AIRLINE_N - 12]

    def test_fh_list_is_deduplicated_and_sorted(self, handle):
        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=2, fh=[3, 1, 3])
        assert result["success"], result.get("error")
        assert result["fh"] == [1, 3]


class TestStepLength:
    def test_step_length_reduces_fold_count(self, handle):
        """With initial_window=120 on airline, step_length=6 runs 4 folds instead of 24."""
        dense = evaluate_tool(estimator_handle=handle, y="airline", initial_window=120)
        sparse = evaluate_tool(
            estimator_handle=handle, y="airline", initial_window=120, step_length=6
        )
        assert dense["success"] and sparse["success"]
        assert dense["cv_folds_run"] == AIRLINE_N - 120  # 24 one-step folds
        # cutoffs at 119, 125, 131, 137 -> 4 folds
        assert sparse["cv_folds_run"] == 4
        assert sparse["step_length"] == 6
        assert _train_windows(sparse) == [120, 126, 132, 138]

    def test_step_length_with_cv_folds_keeps_exact_fold_count(self, handle):
        """Without initial_window, cv_folds folds are run at the requested spacing."""
        result = evaluate_tool(
            estimator_handle=handle, y="airline", cv_folds=3, fh=12, step_length=6
        )
        assert result["success"], result.get("error")
        assert result["cv_folds_run"] == 3
        assert result["cv_folds_requested"] == 3
        # last cutoff leaves 12 points; earlier cutoffs 6 and 12 steps back
        assert _train_windows(result) == [AIRLINE_N - 24, AIRLINE_N - 18, AIRLINE_N - 12]


class TestValidation:
    @pytest.mark.parametrize("bad_fh", [0, -1, [], [0, 2], [1, -6], "12", True, 2.5])
    def test_invalid_fh_rejected(self, handle, bad_fh):
        result = evaluate_tool(estimator_handle=handle, y="airline", fh=bad_fh)
        assert not result["success"]
        assert "fh" in result["error"]

    @pytest.mark.parametrize("bad_step", [0, -3, 1.5, "2", False])
    def test_invalid_step_length_rejected(self, handle, bad_step):
        result = evaluate_tool(estimator_handle=handle, y="airline", step_length=bad_step)
        assert not result["success"]
        assert "step_length" in result["error"]

    def test_series_too_short_for_fh_names_minimum_length(self, handle):
        """cv_folds=3, fh=12, step_length=1 needs 12 + 2 + 1 = 15 points; a 10-point series fails."""
        executor = get_executor()
        y = pd.Series(range(10), index=pd.period_range("2000-01", periods=10, freq="M"))
        executor._data_handles["test_short_dh"] = {"y": y}
        try:
            result = evaluate_tool(estimator_handle=handle, y="test_short_dh", cv_folds=3, fh=12)
            assert not result["success"]
            assert "at least 15 observations" in result["error"]
            assert "10" in result["error"]
        finally:
            executor._data_handles.pop("test_short_dh", None)

    def test_initial_window_must_leave_room_for_fh(self, handle):
        """initial_window=140 with fh=12 leaves only 4 points after the first cutoff."""
        result = evaluate_tool(estimator_handle=handle, y="airline", initial_window=140, fh=12)
        assert not result["success"]
        assert "initial_window" in result["error"]
        assert str(AIRLINE_N - 12) in result["error"]


class TestResponseShape:
    def test_no_cv_folds_requested_when_initial_window_set(self, handle):
        """cv_folds is ignored with initial_window, so it must not be echoed back (F-53)."""
        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3, initial_window=140)
        assert result["success"], result.get("error")
        assert "cv_folds_requested" not in result
        assert result["initial_window"] == 140
        assert result["fh"] == [1]
        assert result["step_length"] == 1
        assert result["cv_folds_run"] == 4

    def test_cv_folds_requested_echoed_without_initial_window(self, handle):
        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=2)
        assert result["success"], result.get("error")
        assert result["cv_folds_requested"] == 2
        assert result["cv_folds_run"] == 2
        assert result["initial_window"] == AIRLINE_N - 2


class TestAsyncParity:
    async def test_run_async_matches_sync(self, handle):
        """The background job honours fh/step_length and shapes its result like the sync path."""
        sync = evaluate_tool(
            estimator_handle=handle, y="airline", initial_window=120, fh=[1, 6, 12], step_length=6
        )
        assert sync["success"], sync.get("error")

        job_manager = get_job_manager()
        started = evaluate_tool(
            estimator_handle=handle,
            y="airline",
            initial_window=120,
            fh=[1, 6, 12],
            step_length=6,
            run_async=True,
        )
        assert started["success"] and started["status"] == "running"
        job = await _wait_for_job(job_manager, started["job_id"])
        assert job.status is JobStatus.COMPLETED, job.errors
        res = job.result

        assert "cv_folds_requested" not in res
        for key in ("cv_folds_run", "initial_window", "fh", "step_length"):
            assert res[key] == sync[key], key
        assert _train_windows(res) == _train_windows(sync) == [120, 126, 132]
        assert res["metrics"] == sync["metrics"]

    async def test_run_async_series_too_short_fails_job(self, handle):
        executor = get_executor()
        y = pd.Series(range(10), index=pd.period_range("2000-01", periods=10, freq="M"))
        executor._data_handles["test_short_async_dh"] = {"y": y}
        try:
            started = evaluate_tool(
                estimator_handle=handle, y="test_short_async_dh", cv_folds=3, fh=12, run_async=True
            )
            assert started["success"]
            job = await _wait_for_job(get_job_manager(), started["job_id"])
            assert job.status is JobStatus.FAILED
            assert any("at least 15 observations" in e for e in job.errors)
        finally:
            executor._data_handles.pop("test_short_async_dh", None)


async def _wait_for_job(job_manager, job_id: str, timeout: float = 30.0):
    import asyncio

    for _ in range(int(timeout / 0.05)):
        job = job_manager.get_job(job_id)
        if job is not None and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return job
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")
