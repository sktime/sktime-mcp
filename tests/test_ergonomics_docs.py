"""Error ergonomics and schema/docs drift (#556 items 26-27: F-66, F-39, F-26, F-44, F-45)."""

import asyncio
import contextlib
import json
import pathlib
import re

import pytest

from sktime_mcp.runtime.executor import _resolve_metric_scoring, get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.server import call_tool, list_tools
from sktime_mcp.tools.data_tools import load_data_source_tool, release_data_handle_tool
from sktime_mcp.tools.evaluate import evaluate_tool
from sktime_mcp.tools.inspect_data import inspect_data_tool
from sktime_mcp.tools.instantiate import instantiate_tool, release_handle_tool
from sktime_mcp.tools.list_available_data import list_available_data_tool


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


# -- (c) estimator vs data handle mix-ups ------------------------------------


@pytest.fixture
def data_handle():
    res = load_data_source_tool(
        {"type": "pandas", "data": {"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, "target_column": "v"}
    )
    assert res["success"], res
    yield res["data_handle"]
    get_executor().release_data_handle(res["data_handle"])


def test_release_handle_with_data_id_points_at_release_data_handle(data_handle):
    res = release_handle_tool(data_handle)
    assert res["success"] is False
    assert "data handle" in res["error"]
    assert "release_data_handle" in res["error"]
    # the data handle itself is untouched
    assert data_handle in get_executor()._data_handles


def test_release_handle_with_unknown_data_prefix_points_at_release_data_handle():
    res = release_handle_tool("data_never_existed")
    assert res["success"] is False
    assert "release_data_handle" in res["error"]


def test_call_method_with_data_id_explains_kwargs(data_handle):
    res = get_executor().call_method(data_handle, "fit", {})
    assert res["success"] is False
    assert "estimator handle" in res["error"]
    assert "_data_handle" in res["error"]


def test_release_data_handle_with_est_id_points_at_release_handle():
    h = instantiate_tool("NaiveForecaster()")["handle"]
    try:
        res = release_data_handle_tool(h)
        assert res["success"] is False
        assert "estimator handle" in res["error"]
        assert "release_handle" in res["error"]
        assert get_handle_manager().exists(h)
    finally:
        _release(h)


def test_inspect_data_with_est_id_points_at_estimator_tools():
    h = instantiate_tool("NaiveForecaster()")["handle"]
    try:
        res = inspect_data_tool(h)
        assert res["success"] is False
        assert "estimator handle" in res["error"]
    finally:
        _release(h)


# -- (i) undeclared tools are not dispatchable; unknown-tool shape -----------


def _call(name, arguments):
    out = asyncio.run(call_tool(name, arguments))
    return json.loads(out[0].text)


def test_unknown_tool_response_has_success_false():
    res = _call("no_such_tool", {})
    assert res == {"success": False, "error": "Unknown tool: no_such_tool"}


@pytest.mark.parametrize("name", ["list_data_sources", "auto_format_on_load"])
def test_undeclared_tools_are_not_dispatchable(name):
    declared = {t.name for t in asyncio.run(list_tools())}
    assert name not in declared
    executor = get_executor()
    before = executor._auto_format_enabled
    res = _call(name, {"enabled": False})
    assert res == {"success": False, "error": f"Unknown tool: {name}"}
    assert executor._auto_format_enabled == before  # no state mutation


def test_every_dispatchable_name_is_declared():
    # Every `name == "..."` branch in call_tool must be a declared tool.
    import inspect

    import sktime_mcp.server as server_module

    src = inspect.getsource(server_module.call_tool)
    dispatched = set(re.findall(r'name == "([a-z_]+)"', src))
    declared = {t.name for t in asyncio.run(list_tools())}
    assert dispatched == declared, dispatched ^ declared


# -- (h) list_available_data omits the filtered-out category -----------------


def test_list_available_data_demo_filter_omits_active_handles(data_handle):
    res = list_available_data_tool(is_demo=True)
    assert res["success"]
    assert "active_handles" not in res
    assert res["total"] == sum(len(v) for v in res["system_demos"].values())


def test_list_available_data_handles_filter_omits_system_demos(data_handle):
    res = list_available_data_tool(is_demo=False)
    assert res["success"]
    assert "system_demos" not in res
    assert data_handle in {h["handle"] for h in res["active_handles"]}
    assert res["total"] == len(res["active_handles"])


def test_list_available_data_unfiltered_has_both(data_handle):
    res = list_available_data_tool()
    assert "system_demos" in res and "active_handles" in res
    assert res["total"] == sum(len(v) for v in res["system_demos"].values()) + len(
        res["active_handles"]
    )


# -- (d)-(g) schema and docs drift -------------------------------------------

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_query_registry_schema_says_tags_are_paginated():
    tools = {t.name: t for t in asyncio.run(list_tools())}
    props = tools["query_registry"].inputSchema["properties"]
    for key in ("limit", "offset"):
        desc = props[key]["description"]
        assert "Ignored" not in desc, (key, desc)
        assert "task='tag'" in desc, (key, desc)


def test_transform_data_schema_does_not_advertise_np_ndarray():
    tools = {t.name: t for t in asyncio.run(list_tools())}
    tool = tools["transform_data"]
    to_mtype = tool.inputSchema["properties"]["to_mtype"]["description"]
    assert "np.ndarray" not in tool.description
    assert "np.ndarray" not in to_mtype
    assert "pd.Series" in to_mtype and "pd-multiindex" in to_mtype


def test_tool_reference_documents_url_source_with_url_key():
    text = (_ROOT / "docs" / "source" / "tool-reference.md").read_text()
    url_row = next(line for line in text.splitlines() if line.startswith("| `url` |"))
    assert "`url`" in url_row.split("|")[2]
    assert "`path`" not in url_row


def test_readme_tables_list_all_tools_and_config_vars():
    text = (_ROOT / "README.md").read_text()
    declared = {t.name for t in asyncio.run(list_tools())}
    missing = {name for name in declared if f"`{name}`" not in text}
    assert not missing, missing
    for var in ("SKTIME_MCP_MAX_DATA_HANDLES", "SKTIME_MCP_MAX_RESPONSE_TOKENS"):
        assert f"| `{var}` |" in text, var
