"""Tests for aligner fit with multi-series X (issue #491)."""

from sktime.datasets import load_airline

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.fit_predict import fit_tool
from sktime_mcp.tools.instantiate import instantiate_tool


def _two_airline_handles(executor):
    y = load_airline()
    h1, h2 = "data_align_a", "data_align_b"
    executor._register_data_handle(h1, {"y": y.iloc[:60], "X": None, "metadata": {}})
    executor._register_data_handle(h2, {"y": y.iloc[20:80], "X": None, "metadata": {}})
    return h1, h2


def test_aligner_fit_single_series_fails():
    """Single series cannot fit an aligner (issue #491)."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    out = fit_tool(estimator_handle=r["handle"], y_dataset="airline")
    assert not out["success"]
    assert "multiple series" in out["error"]
    assert "multiple values for argument 'X'" not in out["error"]


def test_aligner_fit_single_dataframe_fails():
    """Single flat DataFrame handle gets the same clear error as Series."""
    executor = get_executor()
    y = load_airline().to_frame()
    hid = "data_align_df"
    executor._register_data_handle(hid, {"y": y, "X": None, "metadata": {}})
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    try:
        out = fit_tool(estimator_handle=r["handle"], X_handle=hid)
        assert not out["success"]
        assert "multiple series" in out["error"]
        assert "df-list" not in out["error"]
    finally:
        executor._data_handles.pop(hid, None)
        executor._handle_manager.release_handle(r["handle"])


def test_aligner_fit_x_only_no_signature_clash():
    """X-only single series must not raise 'multiple values for X'."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    out = fit_tool(estimator_handle=r["handle"], X_dataset="airline")
    assert not out["success"]
    assert "multiple values for argument 'X'" not in out["error"]


def test_aligner_fit_list_of_handles():
    """List of X_handle ids builds df-list and fits."""
    executor = get_executor()
    h1, h2 = _two_airline_handles(executor)
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    try:
        out = fit_tool(estimator_handle=r["handle"], X_handle=[h1, h2])
        assert out["success"], out
        assert out.get("fitted") is True
        assert executor._handle_manager.is_fitted(r["handle"])
    finally:
        executor._data_handles.pop(h1, None)
        executor._data_handles.pop(h2, None)
        executor._handle_manager.release_handle(r["handle"])


def test_aligner_fit_list_of_datasets():
    """List of X_dataset names builds df-list and fits."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    out = fit_tool(estimator_handle=r["handle"], X_dataset=["airline", "airline"])
    assert out["success"], out
    assert out.get("fitted") is True


def test_aligner_fit_list_too_short():
    """Single-element X list is rejected before sktime."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    out = fit_tool(estimator_handle=r["handle"], X_handle=["only_one"])
    assert not out["success"]
    assert "at least two" in out["error"]


def test_aligner_call_method_list_data_handles():
    """call_method injects list of data handles into fit."""
    executor = get_executor()
    h1, h2 = _two_airline_handles(executor)
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    try:
        out = executor.call_method(r["handle"], "fit", {"X_data_handle": [h1, h2]})
        assert out["success"], out
        al = executor.call_method(r["handle"], "get_alignment", {})
        assert al["success"], al
        assert isinstance(al["result"], dict)
    finally:
        executor._data_handles.pop(h1, None)
        executor._data_handles.pop(h2, None)
        executor._handle_manager.release_handle(r["handle"])


def test_aligner_call_method_list_too_short():
    """call_method rejects empty/short multi-series lists like fit."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    executor = get_executor()
    out = executor.call_method(r["handle"], "fit", {"X_data_handle": []})
    assert not out["success"]
    assert "at least two" in out["error"]


def test_forecaster_fit_string_y_dataset():
    """Single-string y_dataset for forecasters is unchanged."""
    r = instantiate_tool("NaiveForecaster()")
    assert r["success"], r
    out = fit_tool(estimator_handle=r["handle"], y_dataset="airline")
    assert out["success"], out


def test_forecaster_same_dataset_uses_canonical_y_x():
    """Merge must keep main's y/X convention (longley target is TOTEMP)."""
    r = instantiate_tool("NaiveForecaster()")
    assert r["success"], r
    out = fit_tool(
        estimator_handle=r["handle"],
        y_dataset="longley",
        X_dataset="longley",
    )
    assert out["success"], out
    fitted_y = get_executor()._handle_manager.get_instance(r["handle"])._y
    assert fitted_y.ndim == 1 or fitted_y.shape[1] == 1, fitted_y.shape


def test_aligner_fit_unknown_dataset_in_list():
    """Unknown name in X_dataset list surfaces the load error, not a type error."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    out = fit_tool(
        estimator_handle=r["handle"],
        X_dataset=["airline", "no_such_dataset_zzz"],
    )
    assert not out["success"]
    assert "Unknown dataset: no_such_dataset_zzz" in out["error"]
    assert "available" in out


def test_aligner_fit_non_string_list_item():
    """Non-string X list items are rejected before sktime."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    out = fit_tool(estimator_handle=r["handle"], X_handle=[1, 2])
    assert not out["success"]
    assert "must be strings" in out["error"]


def test_aligner_call_method_list_datasets():
    """call_method injects a list of demo datasets into fit."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    executor = get_executor()
    out = executor.call_method(
        r["handle"],
        "fit",
        {"X_dataset": ["airline", "airline"]},
    )
    assert out["success"], out
    al = executor.call_method(r["handle"], "get_alignment", {})
    assert al["success"], al


def test_aligner_call_method_unknown_dataset_in_list():
    """Failed dataset load in a list returns available names, like a string kwarg."""
    r = instantiate_tool("AlignerNaive()")
    assert r["success"], r
    out = get_executor().call_method(
        r["handle"],
        "fit",
        {"X_dataset": ["airline", "no_such_dataset_zzz"]},
    )
    assert not out["success"]
    assert "Unknown dataset: no_such_dataset_zzz" in out["error"]
    assert "available" in out
    assert "unexpected keyword argument" not in out["error"]
