"""update must require data and roll back on failure.

Previously update() with no data arguments returned success having done
nothing, and a REJECTED update left the live instance in sktime's
mid-update broken state — predict then failed with "has not been fitted
yet" while the handle manager still reported fitted=True.
"""

import pandas as pd
import pytest
from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.fit_predict import update_tool


@pytest.fixture
def fitted_handle():
    executor = get_executor()
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    y = pd.Series([float(100 + i) for i in range(24)], index=idx)
    executor._data_handles["upd_train"] = {"y": y}
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    fit_res = executor.fit(handle, y=y)
    assert fit_res["success"]
    yield handle
    executor._handle_manager.release_handle(handle)
    executor._data_handles.pop("upd_train", None)
    executor._data_handles.pop("upd_new", None)


def test_update_without_data_is_rejected(fitted_handle):
    result = update_tool(estimator_handle=fitted_handle)
    assert result["success"] is False
    assert "requires new data" in result["error"]


def test_failed_update_rolls_back_fitted_state(fitted_handle):
    executor = get_executor()
    # airline is PeriodIndex; the fitted series is DatetimeIndex — sktime
    # rejects the update deep inside, after mutating the instance
    result = update_tool(estimator_handle=fitted_handle, y_dataset="airline")
    assert result["success"] is False

    # the estimator must still predict from its original fitted state
    predict_res = executor.predict(fitted_handle, fh=[1, 2])
    assert predict_res["success"], (
        f"estimator was corrupted by the failed update: {predict_res.get('error')}"
    )


def test_successful_update_still_works(fitted_handle):
    executor = get_executor()
    idx = pd.date_range("2022-01-01", periods=3, freq="MS")
    executor._data_handles["upd_new"] = {"y": pd.Series([200.0, 210.0, 220.0], index=idx)}
    result = update_tool(estimator_handle=fitted_handle, y_handle="upd_new")
    assert result["success"], result

    predict_res = executor.predict(fitted_handle, fh=[1])
    assert predict_res["success"]
