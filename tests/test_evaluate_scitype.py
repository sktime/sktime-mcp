"""evaluate must reject non-forecasters and non-Series targets up front (#535).

Previously it attempted cross-validation on a transformer/int handle or on
Panel data, wasting time and (before the error_score fix) masking the result.
"""

import contextlib

import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.evaluate import evaluate_tool
from sktime_mcp.tools.instantiate import instantiate_tool


def _release(handle):
    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(handle)


def test_rejects_transformer():
    res = instantiate_tool(spec="Deseasonalizer()")
    handle = res["handle"]
    try:
        out = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3)
        assert not out["success"]
        assert "forecaster" in out["error"].lower()
        assert "transformer" in out["error"].lower()
    finally:
        _release(handle)


def test_rejects_non_estimator():
    res = instantiate_tool(spec="42")
    handle = res["handle"]
    try:
        out = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3)
        assert not out["success"]
        # int handle has no forecaster object_type -> rejected (here or by scitype)
        assert "forecaster" in out["error"].lower() or "series" in out["error"].lower()
    finally:
        _release(handle)


def test_rejects_classification_dataset_target():
    """basic_motions resolves to categorical labels — not a forecasting series."""
    res = instantiate_tool(spec="NaiveForecaster()")
    handle = res["handle"]
    try:
        out = evaluate_tool(estimator_handle=handle, y="basic_motions", cv_folds=3)
        assert not out["success"]
        err = out["error"].lower()
        assert "numeric" in err or "panel" in err or "series" in err or "classification" in err
    finally:
        _release(handle)


def test_forecaster_on_series_still_works():
    res = instantiate_tool(spec="NaiveForecaster(sp=12)")
    handle = res["handle"]
    try:
        out = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3)
        assert out["success"], out
        assert out["metrics"]
    finally:
        _release(handle)


def test_async_rejects_synchronously():
    """Invalid inputs must not burn a background job."""
    res = instantiate_tool(spec="Deseasonalizer()")
    handle = res["handle"]
    try:
        out = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3, run_async=True)
        assert not out["success"]
        assert "job_id" not in out
    finally:
        _release(handle)
