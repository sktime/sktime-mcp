"""Error ergonomics and schema/docs drift (#556 items 26-27: F-66, F-39, F-26, F-44, F-45)."""

import contextlib

from sktime_mcp.runtime.handles import get_handle_manager
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
