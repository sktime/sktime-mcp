"""predict output-quality fixes (#536).

- NB-18: y forwarded to a forecaster raised a raw TypeError; now dropped with
  a warning.
- NB-21: predict_interval / predict_var dropped the time index (bare arrays);
  now index-keyed like predict.
- NB-22: an unbounded horizon flooded the response; now capped with a marker.
"""

import contextlib

import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool
from sktime_mcp.tools.instantiate import instantiate_tool


@pytest.fixture
def fitted_forecaster():
    res = instantiate_tool(spec="NaiveForecaster(sp=12)")
    handle = res["handle"]
    fit_tool(estimator_handle=handle, y_dataset="airline")
    yield handle
    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(handle)


def test_stray_y_is_dropped_with_warning(fitted_forecaster):
    # y_dataset on a forecaster used to raise "unexpected keyword argument 'y'"
    res = predict_tool(estimator_handle=fitted_forecaster, horizon=3, y_dataset="airline")
    assert res["success"], res
    assert len(res["predictions"]) == 3
    assert "warnings" in res
    assert any("y was ignored" in w for w in res["warnings"])


def test_predict_interval_is_index_keyed(fitted_forecaster):
    res = predict_tool(
        estimator_handle=fitted_forecaster, horizon=3, mode="predict_interval", coverage=0.8
    )
    assert res["success"], res
    intervals = res["intervals"]
    # keys are time periods, each mapping to a dict of bound -> value
    assert len(intervals) == 3
    first_key = next(iter(intervals))
    assert "-" in first_key or first_key.isdigit()  # a period/timestamp label
    assert isinstance(intervals[first_key], dict)
    assert any("lower" in col for col in intervals[first_key])
    assert any("upper" in col for col in intervals[first_key])


def test_predict_var_is_index_keyed(fitted_forecaster):
    res = predict_tool(estimator_handle=fitted_forecaster, horizon=2, mode="predict_var")
    assert res["success"], res
    preds = res["predictions"]
    assert len(preds) == 2
    assert all(isinstance(v, dict) for v in preds.values())


def test_large_horizon_is_capped(fitted_forecaster):
    res = predict_tool(estimator_handle=fitted_forecaster, horizon=1000)
    assert res["success"], res
    assert len(res["predictions"]) <= 500
    assert res["predictions_truncated"]["total"] == 1000
    assert res["predictions_truncated"]["shown"] == 500


def test_normal_horizon_not_capped(fitted_forecaster):
    res = predict_tool(estimator_handle=fitted_forecaster, horizon=12)
    assert res["success"]
    assert "predictions_truncated" not in res
    assert len(res["predictions"]) == 12
