"""Forecasts must be registered as data handles (#556 item 7, audit F-06).

Previously ``predict`` returned JSON only: nothing could plot, save or score a
forecast, although ``plot_series`` promised "train, test, forecasts", the
truncation note said "use save_data", and ``call_method`` on a metric with
``y_true_data_handle``/``y_pred_data_handle`` had nothing to reference.
"""

import asyncio
import contextlib

import pandas as pd
import pytest
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

import sktime_mcp.runtime.executor as executor_module
from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool
from sktime_mcp.tools.inspect_data import inspect_data_tool
from sktime_mcp.tools.plotting import plot_series_tool
from sktime_mcp.tools.save_data import save_data_tool
from sktime_mcp.tools.split_data import split_data_tool

AIRLINE = "predh_airline"
HORIZON = 6


@pytest.fixture
def split():
    """Airline as a data handle, split into train / 6-step test."""
    executor = get_executor()
    y = executor.load_dataset("airline")["y"]
    executor._data_handles[AIRLINE] = {
        "y": y,
        "X": None,
        "metadata": {"columns": ["Number of airline passengers"]},
        "validation": {},
        "config": {},
    }
    res = split_data_tool(data_handle=AIRLINE, fh=HORIZON)
    assert res["success"], res
    created = [AIRLINE, res["train_handle"], res["test_handle"]]
    yield {"train": res["train_handle"], "test": res["test_handle"], "cutoff": res["cutoff"]}
    for h in created:
        executor._data_handles.pop(h, None)


@pytest.fixture
def forecaster(split):
    """NaiveForecaster fitted on the train half; releases the estimator handle."""
    executor = get_executor()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    res = fit_tool(estimator_handle=handle, y_handle=split["train"])
    assert res["success"], res
    yield handle
    with contextlib.suppress(KeyError):
        executor._handle_manager.release_handle(handle)


@pytest.fixture
def released():
    """Collect prediction handles created by a test and release them afterwards."""
    handles: list[str] = []
    yield handles
    executor = get_executor()
    for h in handles:
        executor._data_handles.pop(h, None)


def _predict(forecaster, released, **kwargs):
    res = predict_tool(estimator_handle=forecaster, horizon=HORIZON, **kwargs)
    assert res["success"], res
    assert "prediction_handle" in res, res.keys()
    released.append(res["prediction_handle"])
    return res


# ---------------------------------------------------------------------------
# (1) every successful predict registers a data handle
# ---------------------------------------------------------------------------


def test_predict_registers_point_forecast_handle(split, forecaster, released):
    res = _predict(forecaster, released)
    executor = get_executor()
    handle = res["prediction_handle"]
    assert handle in executor._data_handles

    stored = executor._data_handles[handle]
    y = stored["y"]
    assert isinstance(y, pd.Series)
    assert len(y) == HORIZON
    # the stored series keeps a real time index (plottable), the JSON keys are its str form
    assert not pd.api.types.is_string_dtype(y.index)
    assert [str(i) for i in y.index] == list(res["predictions"].keys())
    assert stored["X"] is None

    meta = stored["metadata"]
    assert meta["source"] == "prediction"
    assert meta["estimator_handle"] == forecaster
    assert meta["mode"] == "predict"
    assert meta["horizon"] == HORIZON
    assert meta["cutoff"] == split["cutoff"]
    assert meta["rows"] == HORIZON

    # response is otherwise unchanged
    assert res["mode"] == "predict"
    assert res["horizon"] == HORIZON
    assert len(res["predictions"]) == HORIZON


@pytest.mark.parametrize(
    ("mode", "key", "extra"),
    [
        ("predict_interval", "intervals", {"coverage": 0.9}),
        ("predict_quantiles", "quantiles", {"alpha": [0.1, 0.9]}),
        ("predict_var", "predictions", {}),
    ],
)
def test_other_modes_register_flattened_frame(split, forecaster, released, mode, key, extra):
    res = _predict(forecaster, released, mode=mode, **extra)
    stored = get_executor()._data_handles[res["prediction_handle"]]
    y = stored["y"]
    assert isinstance(y, pd.DataFrame)
    assert len(y) == HORIZON
    assert not isinstance(y.columns, pd.MultiIndex)
    first_row = next(iter(res[key].values()))
    assert [str(c) for c in y.columns] == list(first_row.keys())
    assert stored["metadata"]["mode"] == mode
    assert stored["metadata"]["source"] == "prediction"


def test_each_predict_call_gets_its_own_handle(split, forecaster, released):
    a = _predict(forecaster, released)["prediction_handle"]
    b = _predict(forecaster, released)["prediction_handle"]
    assert a != b


def test_prediction_handle_listed_with_other_handles(split, forecaster, released):
    handle = _predict(forecaster, released)["prediction_handle"]
    listed = get_executor().list_data_handles()
    entry = next(h for h in listed["handles"] if h["handle"] == handle)
    assert entry["metadata"]["source"] == "prediction"


# ---------------------------------------------------------------------------
# (2) truncation note points at the handle
# ---------------------------------------------------------------------------


def test_truncation_note_names_prediction_handle(split, forecaster, released, monkeypatch):
    monkeypatch.setattr(executor_module, "_MAX_PREDICTION_ROWS", 3)
    res = _predict(forecaster, released)
    note = res["predictions_truncated"]
    assert note["shown"] == 3
    assert note["total"] == HORIZON
    assert len(res["predictions"]) == 3
    assert res["prediction_handle"] in note["note"]
    assert "save_data" in note["note"]
    # the handle still holds the untruncated forecast
    assert len(get_executor()._data_handles[res["prediction_handle"]]["y"]) == HORIZON


# ---------------------------------------------------------------------------
# (3) async path registers the handle too
# ---------------------------------------------------------------------------


async def _wait_for_job(job_manager, job_id: str, timeout: float = 30.0):
    for _ in range(int(timeout / 0.05)):
        job = job_manager.get_job(job_id)
        if job is not None and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return job
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")


async def test_async_predict_registers_handle(split, forecaster, released):
    job_manager = get_job_manager()
    res = predict_tool(estimator_handle=forecaster, horizon=HORIZON, run_async=True)
    assert res["success"], res
    job = await _wait_for_job(job_manager, res["job_id"])
    assert job.status is JobStatus.COMPLETED, job.errors

    handle = job.result.get("prediction_handle")
    assert handle, job.result.keys()
    released.append(handle)
    stored = get_executor()._data_handles[handle]
    assert len(stored["y"]) == HORIZON
    assert stored["metadata"]["source"] == "prediction"
    assert stored["metadata"]["estimator_handle"] == forecaster


# ---------------------------------------------------------------------------
# (4) the handle works with inspect_data, plot_series, save_data, call_method
# ---------------------------------------------------------------------------


def test_prediction_handle_plots_saves_and_scores(split, forecaster, released, tmp_path):
    executor = get_executor()
    pred = _predict(forecaster, released)["prediction_handle"]

    # inspect_data
    info = inspect_data_tool(pred)
    assert info["success"], info
    assert info["shape"][0] == HORIZON
    assert info["scitype"] == "Series"

    # plot_series: train + forecast on one figure
    png = tmp_path / "forecast.png"
    plot = plot_series_tool(
        data_handles=[split["train"], pred],
        labels=["train", "forecast"],
        path=str(png),
    )
    assert plot["success"], plot
    assert plot["n_series"] == 2
    assert png.exists() and png.stat().st_size > 0

    # save_data: the full forecast as csv
    csv = tmp_path / "forecast.csv"
    saved = save_data_tool(data_handle=pred, path=str(csv), format="csv")
    assert saved["success"], saved
    assert saved["rows"] == HORIZON
    assert len(pd.read_csv(csv)) == HORIZON

    # call_method on a metric with both sides injected from data handles
    metric = executor._handle_manager.create_handle(
        "MeanAbsolutePercentageError", MeanAbsolutePercentageError(), {}
    )
    try:
        scored = executor.call_method(
            handle_id=metric,
            method_name="__call__",
            kwargs={"y_true_data_handle": split["test"], "y_pred_data_handle": pred},
        )
    finally:
        with contextlib.suppress(KeyError):
            executor._handle_manager.release_handle(metric)
    assert scored["success"], scored
    mape = scored["result"]
    assert isinstance(mape, float)
    assert 0.0 < mape < 1.0
