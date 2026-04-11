"""Tests for predict_interval MCP tool."""


def test_predict_interval_bad_handle():
    from sktime_mcp.tools.fit_predict import predict_interval_tool
    result = predict_interval_tool("bad_handle")
    assert result["success"] is False
    assert "error" in result


def test_predict_interval_not_fitted():
    from sktime_mcp.runtime.executor import get_executor
    from sktime_mcp.tools.fit_predict import predict_interval_tool
    executor = get_executor()
    r = executor.instantiate("ARIMA", {"order": [1, 1, 1]})
    result = predict_interval_tool(r["handle"])
    assert result["success"] is False


def test_predict_interval_full_workflow():
    from sktime_mcp.runtime.executor import get_executor
    from sktime_mcp.tools.fit_predict import predict_interval_tool
    executor = get_executor()
    r = executor.instantiate("ARIMA", {
        "order": [1, 1, 1],
        "suppress_warnings": True
    })
    assert r["success"]
    handle = r["handle"]
    fit = executor.fit_predict(handle, "airline", horizon=3)
    assert fit["success"]
    result = predict_interval_tool(handle, horizon=3, coverage=0.9)
    assert result["success"]
    assert result["coverage"] == 0.9
    assert result["horizon"] == 3
    for step, bounds in result["intervals"].items():
        assert "lower" in bounds
        assert "upper" in bounds
        assert bounds["upper"] >= bounds["lower"]