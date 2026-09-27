"""Regression tests for describe_component name resolution (#556 item 19 / audit F-25, #341).

``describe_component`` used to resolve ``name`` through ``sktime.registry.craft``
(an ``eval``), so ``describe_component("int")`` succeeded with module
``builtins.int`` and an expression with side effects was executed. Resolution
is now restricted to registry names and dotted import paths of
``skbase``/``sktime`` ``BaseObject`` subclasses; everything else is a
structured error with a did-you-mean suggestion. Non-string / empty names get
a structured error instead of an ``AttributeError``.
"""

import pytest

from sktime_mcp.tools.describe_component import describe_component_tool

SUGGESTION = "Use query_registry"


def _assert_structured_error(result):
    assert result["success"] is False
    assert isinstance(result["error"], str) and result["error"]
    assert SUGGESTION in result["suggestion"]


@pytest.mark.parametrize("builtin_name", ["int", "dict"])
def test_builtins_are_not_components(builtin_name):
    result = describe_component_tool(builtin_name)
    _assert_structured_error(result)
    assert builtin_name in result["error"]


def test_expression_with_side_effect_is_not_evaluated(tmp_path):
    marker = tmp_path / "pwned"
    expr = f"type(__import__('pathlib').Path({str(marker)!r}).write_text('x'))"
    result = describe_component_tool(expr)
    _assert_structured_error(result)
    assert not marker.exists(), "describe_component evaluated its name argument"


def test_import_of_non_baseobject_dotted_path_is_rejected():
    result = describe_component_tool("os.path.join")
    _assert_structured_error(result)


@pytest.mark.parametrize("bad_name", [None, 42, ["NaiveForecaster"], {"name": "ARIMA"}])
def test_non_string_name_is_structured_error(bad_name):
    result = describe_component_tool(bad_name)
    _assert_structured_error(result)
    assert "non-empty string" in result["error"]


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_name_is_structured_error(empty):
    result = describe_component_tool(empty)
    _assert_structured_error(result)
    assert "non-empty string" in result["error"]


def test_misspelt_name_gets_did_you_mean():
    result = describe_component_tool("ThetaForcaster")
    _assert_structured_error(result)
    assert "ThetaForecaster" in result["did_you_mean"]


def test_full_dotted_path_resolves():
    result = describe_component_tool("sktime.forecasting.theta.ThetaForecaster")
    assert result["success"] is True
    assert result["name"] == "ThetaForecaster"
    assert result["task"] == "forecaster"
    assert result["module"].startswith("sktime.forecasting.theta")
    assert result["module"].endswith(".ThetaForecaster")
    assert "sp" in result["parameters"]


def test_registry_name_still_works_with_parameters():
    result = describe_component_tool("NaiveForecaster")
    assert result["success"] is True
    assert result["name"] == "NaiveForecaster"
    assert result["task"] == "forecaster"
    assert result["module"].startswith("sktime.forecasting.naive")
    assert result["module"].endswith(".NaiveForecaster")
    assert set(result["parameters"]) >= {"strategy", "sp", "window_length"}
    assert result["parameters"]["strategy"]["default"] == "last"
    assert result["hyperparameters"] == result["parameters"]
    assert "tags" in result and "tag_explanations" in result and "docstring" in result


def test_case_insensitive_lookup_still_works():
    result = describe_component_tool("naiveforecaster")
    assert result["success"] is True
    assert result["name"] == "NaiveForecaster"


def test_non_forecaster_components_resolve():
    for name, task in [
        ("SlidingWindowSplitter", "splitter"),
        ("MeanAbsolutePercentageError", "metric"),
    ]:
        result = describe_component_tool(name)
        assert result["success"] is True, result
        assert result["task"] == task
