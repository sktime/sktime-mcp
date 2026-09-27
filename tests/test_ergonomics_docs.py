"""Error ergonomics and schema/docs drift (#556 items 26-27: F-66, F-39, F-26, F-44, F-45)."""

import asyncio
import contextlib

import pytest

from sktime_mcp.runtime.executor import _resolve_metric_scoring
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.server import list_tools
from sktime_mcp.tools.evaluate import evaluate_tool
from sktime_mcp.tools.instantiate import instantiate_tool


def _release(handle):
    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(handle)


# -- (a) factory functions in instantiate specs ------------------------------


def test_instantiate_factory_function_explains_and_suggests_class():
    res = instantiate_tool("make_reduction(LinearRegression(), window_length=3)")
    assert res["success"] is False
    err = res["error"]
    assert "make_reduction" in err
    assert "constructor" in err
    assert "RecursiveTabularRegressionForecaster" in err
    assert "YfromX" in err
    assert "is not defined" not in err


def test_instantiate_unknown_name_points_at_query_registry():
    res = instantiate_tool("NaiveForcaster(sp=12)")
    assert res["success"] is False
    assert "NaiveForcaster" in res["error"]
    assert "query_registry" in res["error"]


def test_instantiate_suggested_class_equivalent_works():
    res = instantiate_tool(
        "RecursiveTabularRegressionForecaster(estimator=LinearRegression(), window_length=3)"
    )
    assert res["success"], res
    _release(res["handle"])


# -- (b) evaluate metric aliases ---------------------------------------------


@pytest.mark.parametrize(
    ("alias", "class_name", "params"),
    [
        ("mape", "MeanAbsolutePercentageError", {"symmetric": False}),
        ("sMAPE", "MeanAbsolutePercentageError", {"symmetric": True}),
        ("mae", "MeanAbsoluteError", {}),
        ("mse", "MeanSquaredError", {"square_root": False}),
        ("RMSE", "MeanSquaredError", {"square_root": True}),
        ("mase", "MeanAbsoluteScaledError", {}),
        ("msle", "MeanSquaredLogError", {}),
        ("rmsse", "MeanSquaredScaledError", {"square_root": True}),
        ("meanabsoluteerror", "MeanAbsoluteError", {}),
        ("MeanAbsoluteError", "MeanAbsoluteError", {}),
    ],
)
def test_metric_aliases_resolve(alias, class_name, params):
    scoring = _resolve_metric_scoring(alias)
    assert scoring is not None, alias
    assert type(scoring).__name__ == class_name
    got = scoring.get_params()
    for k, v in params.items():
        assert got[k] == v, (alias, k)


def test_evaluate_accepts_mape_alias():
    h = instantiate_tool("NaiveForecaster()")["handle"]
    try:
        res = evaluate_tool(estimator_handle=h, y="airline", cv_folds=2, metric="mape")
        assert res["success"], res
        assert "test_MeanAbsolutePercentageError" in res["metrics"]
    finally:
        _release(h)


def test_evaluate_unknown_metric_lists_aliases():
    h = instantiate_tool("NaiveForecaster()")["handle"]
    try:
        res = evaluate_tool(estimator_handle=h, y="airline", cv_folds=2, metric="nope")
        assert res["success"] is False
        assert "Unknown metric: nope" in res["error"]
        assert "mape" in res["error"] and "rmsse" in res["error"]
    finally:
        _release(h)


def test_metric_schema_documents_aliases():
    tools = {t.name: t for t in asyncio.run(list_tools())}
    desc = tools["evaluate"].inputSchema["properties"]["metric"]["description"]
    for alias in ("mape", "smape", "mae", "mse", "rmse", "mase", "msle", "rmsse"):
        assert alias in desc
