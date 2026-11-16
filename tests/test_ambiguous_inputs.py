"""Ambiguous or leaking data inputs must be refused, not silently resolved (#556 item 6).

Previously ``fit(y_handle=h, y_dataset="airline")`` was accepted and the
dataset silently won (same for the X pair, and in predict/update), and a
demo dataset without exogenous data used as X fell back to its *target*, so
``evaluate(y="airline", X="airline")`` regressed the series on itself.
"""

import contextlib

import pandas as pd
import pytest
from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import get_job_manager
from sktime_mcp.tools.evaluate import evaluate_tool
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool, update_tool

HANDLE = "ambig_y_only"


@pytest.fixture
def handle():
    executor = get_executor()
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    executor._data_handles[HANDLE] = {
        "y": pd.Series(range(12), index=idx, name="y", dtype=float),
        "X": None,
        "metadata": {"columns": ["y"]},
    }
    yield HANDLE
    executor._data_handles.pop(HANDLE, None)


@pytest.fixture
def forecaster():
    executor = get_executor()
    h = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    yield h
    with contextlib.suppress(KeyError):
        executor._handle_manager.release_handle(h)


def _assert_ambiguous(res, slot, handle):
    assert res["success"] is False, res
    err = res["error"]
    assert f"{slot}_handle" in err and f"{slot}_dataset" in err, err
    assert handle in err and "airline" in err, err


@pytest.mark.parametrize("tool", [fit_tool, predict_tool, update_tool])
@pytest.mark.parametrize("slot", ["y", "X"])
def test_handle_and_dataset_for_same_slot_is_refused(handle, forecaster, tool, slot):
    kwargs = {f"{slot}_handle": handle, f"{slot}_dataset": "airline"}
    res = tool(estimator_handle=forecaster, **kwargs)
    _assert_ambiguous(res, slot, handle)


@pytest.mark.parametrize("tool", [fit_tool, predict_tool])
@pytest.mark.parametrize("slot", ["y", "X"])
async def test_run_async_refuses_before_creating_a_job(handle, forecaster, tool, slot):
    job_manager = get_job_manager()
    n_before = len(job_manager.list_jobs())
    kwargs = {f"{slot}_handle": handle, f"{slot}_dataset": "airline"}
    res = tool(estimator_handle=forecaster, run_async=True, **kwargs)
    _assert_ambiguous(res, slot, handle)
    assert "job_id" not in res
    assert len(job_manager.list_jobs()) == n_before


def test_evaluate_dataset_as_its_own_x_is_refused(forecaster):
    res = evaluate_tool(estimator_handle=forecaster, y="airline", X="airline", cv_folds=2)
    assert res["success"] is False, res
    assert "airline" in res["error"] and "no exogenous" in res["error"], res["error"]


def test_fit_x_dataset_without_exog_errors_instead_of_using_target(handle, forecaster):
    res = fit_tool(estimator_handle=forecaster, y_handle=handle, X_dataset="airline")
    assert res["success"] is False, res
    assert "airline" in res["error"] and "no exogenous" in res["error"], res["error"]
