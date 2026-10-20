"""Exogenous data stored in a data handle must reach fit/predict/update/evaluate (#556 item 5, audit F-05).

Previously ``X_handle`` (and evaluate's ``X=<handle>``) resolved to the
handle's *target* series, and the X stored alongside a y_handle was never
used at all: a forecaster fitted with ``y_handle=h`` silently became
intercept-only, and ``evaluate(y=h, X=h)`` regressed the target on itself
(MAPE 0.0).
"""

import contextlib

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import YfromX

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager
from sktime_mcp.tools.evaluate import evaluate_tool
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool, update_tool
from sktime_mcp.tools.split_data import split_data_tool

FULL = "exog_full"
NO_X = "exog_stripped"


def _make_frames():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    promo = np.array([1.0 if i % 3 == 0 else 0.0 for i in range(24)])
    X = pd.DataFrame({"promo": promo}, index=idx)
    # sales is an exact linear function of promo, so a correctly wired
    # YfromX(LinearRegression()) reproduces it and an intercept-only fit cannot.
    y = pd.Series(100.0 + 40.0 * promo, index=idx, name="sales")
    return y, X


@pytest.fixture
def handles():
    executor = get_executor()
    y, X = _make_frames()
    executor._data_handles[FULL] = {
        "y": y,
        "X": X,
        "metadata": {"columns": ["sales"], "exog_columns": ["promo"]},
    }
    executor._data_handles[NO_X] = {"y": y.copy(), "X": None, "metadata": {"columns": ["sales"]}}
    split = split_data_tool(data_handle=FULL, test_size=0.25)
    assert split["success"], split
    created = [FULL, NO_X, split["train_handle"], split["test_handle"]]
    yield {"full": FULL, "no_x": NO_X, "train": split["train_handle"], "test": split["test_handle"]}
    for h in created:
        executor._data_handles.pop(h, None)


@pytest.fixture
def forecaster():
    executor = get_executor()
    handle = executor._handle_manager.create_handle("YfromX", YfromX(LinearRegression()), {})
    yield handle
    with contextlib.suppress(KeyError):
        executor._handle_manager.release_handle(handle)


def test_split_carries_x_into_both_halves(handles):
    executor = get_executor()
    train, test = executor._data_handles[handles["train"]], executor._data_handles[handles["test"]]
    assert list(train["X"].columns) == ["promo"] and len(train["X"]) == 18
    assert list(test["X"].columns) == ["promo"] and len(test["X"]) == 6
    assert train["X"].index.equals(train["y"].index)
    assert test["X"].index.equals(test["y"].index)


def test_fit_uses_handle_x_and_predict_x_handle_tracks_promo(handles, forecaster):
    executor = get_executor()
    fit_res = fit_tool(estimator_handle=forecaster, y_handle=handles["train"])
    assert fit_res["success"], fit_res
    assert fit_res["exogenous"]["handle"] == handles["train"]
    assert fit_res["exogenous"]["columns"] == ["promo"]

    pred = predict_tool(estimator_handle=forecaster, horizon=6, X_handle=handles["test"])
    assert pred["success"], pred
    values = np.array(list(pred["predictions"].values()), dtype=float)
    assert values.std() > 1.0, f"forecast is constant — stored X was ignored: {values}"
    expected = 100.0 + 40.0 * executor._data_handles[handles["test"]]["X"]["promo"].to_numpy()
    np.testing.assert_allclose(values, expected, atol=1e-6)


@pytest.mark.parametrize("tool", [fit_tool, predict_tool, update_tool])
def test_x_handle_without_exog_columns_errors_clearly(handles, forecaster, tool):
    kwargs = {"estimator_handle": forecaster, "X_handle": handles["no_x"]}
    if tool is not predict_tool:
        kwargs["y_handle"] = handles["train"]
    res = tool(**kwargs)
    assert res["success"] is False
    assert handles["no_x"] in res["error"]
    assert "no exogenous" in res["error"]
    assert "exog_columns" in res["error"]


def test_same_handle_for_y_and_x_is_refused(handles, forecaster):
    ev = evaluate_tool(
        estimator_handle=forecaster, y=handles["full"], X=handles["full"], cv_folds=2
    )
    assert ev["success"] is False
    assert handles["full"] in ev["error"]
    assert "same data handle" in ev["error"]

    fit_res = fit_tool(
        estimator_handle=forecaster, y_handle=handles["full"], X_handle=handles["full"]
    )
    assert fit_res["success"] is False
    assert "same data handle" in fit_res["error"]


def test_evaluate_auto_uses_handle_x(handles, forecaster):
    with_x = evaluate_tool(
        estimator_handle=forecaster,
        y=handles["full"],
        cv_folds=2,
        metric="MeanAbsolutePercentageError",
    )
    assert with_x["success"], with_x
    assert with_x["exogenous"]["handle"] == handles["full"]
    without_x = evaluate_tool(
        estimator_handle=forecaster,
        y=handles["no_x"],
        cv_folds=2,
        metric="MeanAbsolutePercentageError",
    )
    assert without_x["success"], without_x
    assert "exogenous" not in without_x
    mape_with = next(iter(with_x["metrics"].values()))
    mape_without = next(iter(without_x["metrics"].values()))
    assert mape_with < 1e-6, mape_with
    assert mape_without > 0.01, mape_without


def test_predict_without_future_x_after_fit_with_x_is_structured(handles, forecaster):
    fit_res = fit_tool(estimator_handle=forecaster, y_handle=handles["train"])
    assert fit_res["success"], fit_res
    pred = predict_tool(estimator_handle=forecaster, horizon=6)
    assert pred["success"] is False
    assert "fitted with exogenous" in pred["error"]
    assert "X_handle" in pred["error"]


def test_update_uses_handle_x(handles, forecaster):
    fit_res = fit_tool(estimator_handle=forecaster, y_handle=handles["train"])
    assert fit_res["success"], fit_res
    upd = update_tool(estimator_handle=forecaster, y_handle=handles["test"])
    assert upd["success"], upd
    assert upd["exogenous"]["handle"] == handles["test"]


async def test_fit_async_uses_handle_x(handles, forecaster):
    job_manager = get_job_manager()
    res = fit_tool(estimator_handle=forecaster, y_handle=handles["train"], run_async=True)
    assert res["success"], res
    for _ in range(600):
        job = job_manager.get_job(res["job_id"])
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        await __import__("asyncio").sleep(0.05)
    assert job.status is JobStatus.COMPLETED, job.errors

    pred = predict_tool(estimator_handle=forecaster, horizon=6, X_handle=handles["test"])
    assert pred["success"], pred
    values = np.array(list(pred["predictions"].values()), dtype=float)
    assert values.std() > 1.0, f"forecast is constant — stored X was ignored: {values}"
