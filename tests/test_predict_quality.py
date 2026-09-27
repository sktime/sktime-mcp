"""predict quality fixes (#556 items 18 and 25; audit F-23, F-24, bare-class spec).

- F-23: ``predict(mode="predict_proba")`` on a forecaster returned the
  ``repr()`` of the distribution object; now a numeric mean/var/quantile
  summary keyed by period, registered as a prediction handle. A forecaster
  without probabilistic support gets a structured error instead of a raw
  NotImplementedError.
- F-24: ``predict(horizon=100000)`` forecast every step before truncating the
  response; now ``SKTIME_MCP_MAX_HORIZON`` (default 10000) rejects the request
  before any work, sync and async alike.
- Bare class: ``instantiate(spec="NaiveForecaster")`` returned a handle
  wrapping the *class*; now rejected with a hint to call it.
"""

import contextlib

import pytest

from sktime_mcp.config import settings
from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.runtime.jobs import get_job_manager
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool
from sktime_mcp.tools.instantiate import instantiate_tool
from sktime_mcp.tools.split_data import split_data_tool

PERIODS = ["1961-01", "1961-02", "1961-03"]


def _fit(spec: str) -> str:
    res = instantiate_tool(spec=spec)
    assert res["success"], res
    fit = fit_tool(estimator_handle=res["handle"], y_dataset="airline")
    assert fit["success"], fit
    return res["handle"]


def _release(handle: str | None) -> None:
    if handle:
        with contextlib.suppress(KeyError):
            get_handle_manager().release_handle(handle)


@pytest.fixture
def released():
    """Data handles to drop after the test."""
    handles: list[str] = []
    yield handles
    executor = get_executor()
    for h in handles:
        executor._data_handles.pop(h, None)


@pytest.fixture
def theta():
    handle = _fit("ThetaForecaster(sp=12)")
    yield handle
    _release(handle)


@pytest.fixture
def naive():
    handle = _fit("NaiveForecaster(sp=12)")
    yield handle
    _release(handle)


# ---------------------------------------------------------------------------
# F-23: predict_proba is a numeric summary, not a repr string
# ---------------------------------------------------------------------------


def test_predict_proba_returns_numeric_summary(theta, released):
    res = predict_tool(estimator_handle=theta, horizon=3, mode="predict_proba")
    assert res["success"], res
    dist = res["predictions"]
    assert isinstance(dist, dict), dist
    assert dist["distribution"] == "Normal"

    assert list(dist["mean"]) == PERIODS
    assert list(dist["var"]) == PERIODS
    assert list(dist["quantiles"]) == PERIODS
    for period in PERIODS:
        assert isinstance(dist["mean"][period], float)
        assert isinstance(dist["var"][period], float)
        assert dist["var"][period] > 0
        q = dist["quantiles"][period]
        assert list(q) == ["0.05", "0.5", "0.95"]
        assert q["0.05"] < q["0.5"] < q["0.95"]
        assert q["0.5"] == pytest.approx(dist["mean"][period])

    # registered as a prediction handle with mean/var/quantile columns
    handle = res.get("prediction_handle")
    assert handle, res.keys()
    released.append(handle)
    stored = get_executor()._data_handles[handle]
    frame = stored["y"]
    assert len(frame) == 3
    assert [str(i) for i in frame.index] == PERIODS
    assert list(frame.columns) == [
        "Number of airline passengers_mean",
        "Number of airline passengers_var",
        "Number of airline passengers_0.05",
        "Number of airline passengers_0.5",
        "Number of airline passengers_0.95",
    ]
    assert stored["metadata"]["mode"] == "predict_proba"
    assert stored["metadata"]["source"] == "prediction"


def test_predict_proba_summary_is_json_serialisable(theta, released):
    import json

    from sktime_mcp.server import sanitize_for_json

    res = predict_tool(estimator_handle=theta, horizon=2, mode="predict_proba")
    assert res["success"], res
    released.append(res["prediction_handle"])
    text = json.dumps(sanitize_for_json(res))
    assert "Normal(columns=" not in text
    assert "mu=" not in text


def test_predict_proba_unsupported_is_structured_error():
    # TrendForecaster has capability:pred_int False (NaiveForecaster and
    # ThetaForecaster both support predict_proba in sktime >= 0.30).
    handle = _fit("TrendForecaster()")
    try:
        res = predict_tool(estimator_handle=handle, horizon=3, mode="predict_proba")
        assert res["success"] is False
        assert "TrendForecaster" in res["error"]
        assert "predict_proba" in res["error"]
        assert "predict_interval" in res["error"] or "mode='predict'" in res["error"]
        assert "open an issue" not in res["error"]
    finally:
        _release(handle)


# ---------------------------------------------------------------------------
# F-24: horizon above SKTIME_MCP_MAX_HORIZON is rejected before any work
# ---------------------------------------------------------------------------


def test_max_horizon_default_and_env_parsing(monkeypatch):
    monkeypatch.delenv("SKTIME_MCP_MAX_HORIZON", raising=False)
    assert settings.max_horizon == 10000
    monkeypatch.setenv("SKTIME_MCP_MAX_HORIZON", "20")
    assert settings.max_horizon == 20
    for bad in ("0", "-5", "abc"):
        monkeypatch.setenv("SKTIME_MCP_MAX_HORIZON", bad)
        assert settings.max_horizon == 10000, bad


def test_horizon_above_cap_rejected_sync(naive, monkeypatch):
    monkeypatch.setenv("SKTIME_MCP_MAX_HORIZON", "20")
    res = predict_tool(estimator_handle=naive, horizon=21)
    assert res["success"] is False
    assert "SKTIME_MCP_MAX_HORIZON" in res["error"]
    assert "20" in res["error"]
    assert "21" in res["error"]


def test_horizon_above_cap_rejected_async_without_job(naive, monkeypatch):
    monkeypatch.setenv("SKTIME_MCP_MAX_HORIZON", "20")
    job_manager = get_job_manager()
    before = {j.job_id for j in job_manager.list_jobs()}
    res = predict_tool(estimator_handle=naive, horizon=21, run_async=True)
    assert res["success"] is False
    assert "SKTIME_MCP_MAX_HORIZON" in res["error"]
    assert "job_id" not in res
    assert {j.job_id for j in job_manager.list_jobs()} == before


def test_horizon_at_cap_accepted(naive, released, monkeypatch):
    monkeypatch.setenv("SKTIME_MCP_MAX_HORIZON", "20")
    res = predict_tool(estimator_handle=naive, horizon=20)
    assert res["success"], res
    released.append(res["prediction_handle"])
    assert len(res["predictions"]) == 20
    assert "predictions_truncated" not in res


def test_split_data_fh_respects_cap(released, monkeypatch):
    monkeypatch.setenv("SKTIME_MCP_MAX_HORIZON", "20")
    executor = get_executor()
    y = executor.load_dataset("airline")["y"]
    executor._data_handles["pq_airline"] = {
        "y": y,
        "X": None,
        "metadata": {"columns": ["Number of airline passengers"]},
        "validation": {},
        "config": {},
    }
    released.append("pq_airline")
    res = split_data_tool(data_handle="pq_airline", fh=21)
    assert res["success"] is False
    assert "SKTIME_MCP_MAX_HORIZON" in res["error"]
    res = split_data_tool(data_handle="pq_airline", fh=[1, 21])
    assert res["success"] is False
    assert "SKTIME_MCP_MAX_HORIZON" in res["error"]
    res = split_data_tool(data_handle="pq_airline", fh=20)
    assert res["success"], res
    released.extend([res["train_handle"], res["test_handle"]])


# ---------------------------------------------------------------------------
# item 25: a bare class spec is rejected with a hint to call it
# ---------------------------------------------------------------------------


def test_bare_class_spec_rejected_with_hint():
    res = instantiate_tool(spec="NaiveForecaster")
    _release(res.get("handle"))
    assert res["success"] is False
    assert "NaiveForecaster()" in res["error"]
    assert "class" in res["error"]


def test_called_class_spec_still_instantiates():
    res = instantiate_tool(spec="NaiveForecaster()")
    try:
        assert res["success"], res
        assert res["estimator"] == "NaiveForecaster"
    finally:
        _release(res.get("handle"))


@pytest.mark.parametrize("spec", ["42", "[1, 2, 3]", "'just a string'", "None"])
def test_non_estimator_still_rejected(spec):
    res = instantiate_tool(spec=spec)
    assert res["success"] is False
    assert "did not produce an sktime estimator" in res["error"]
