"""Guard against untyped inputSchema properties.

A property without a type constraint is serialized as a string by some MCP
clients, which silently breaks the parameter on the Python side. This is
invisible to unit tests that call the tool functions directly — it only
appears over the wire — so we lint the schemas instead.
"""

import asyncio

import pytest

from sktime_mcp.server import list_tools

# A property is considered typed if it has any of these constraint keywords.
_TYPE_KEYWORDS = {"type", "enum", "anyOf", "oneOf", "allOf", "const"}


def _iter_properties(schema, path):
    """Yield (path, property_schema) for every property, recursively."""
    if not isinstance(schema, dict):
        return
    for prop_name, prop_schema in schema.get("properties", {}).items():
        prop_path = f"{path}.{prop_name}"
        yield prop_path, prop_schema
        yield from _iter_properties(prop_schema, prop_path)
        if isinstance(prop_schema, dict):
            yield from _iter_properties(prop_schema.get("items"), f"{prop_path}[]")


def _all_tool_properties():
    tools = asyncio.run(list_tools())
    for tool in tools:
        yield from _iter_properties(tool.inputSchema, tool.name)


def test_every_schema_property_declares_a_type():
    untyped = [
        path
        for path, prop in _all_tool_properties()
        if isinstance(prop, dict) and not (_TYPE_KEYWORDS & prop.keys())
    ]
    assert not untyped, (
        "inputSchema properties without a type constraint (clients stringify "
        f"their values in transit): {untyped}"
    )


@pytest.mark.parametrize(
    ("tool_name", "prop", "expected_types"),
    [
        ("fit", "fh", ["integer", "array"]),
        ("predict", "coverage", ["number", "array"]),
        ("predict", "alpha", ["number", "array"]),
        ("split_data", "fh", ["integer", "array"]),
        ("plot_series", "markers", ["string", "array"]),
    ],
)
def test_previously_untyped_properties_are_typed(tool_name, prop, expected_types):
    tools = {t.name: t for t in asyncio.run(list_tools())}
    prop_schema = tools[tool_name].inputSchema["properties"][prop]
    assert prop_schema["type"] == expected_types


def test_query_registry_task_is_constrained_to_valid_scitypes():
    """The task filter rejects unknown scitypes at the schema level (#407).

    Catches the common 'forecasting' vs 'forecaster' mistake before the call
    reaches the server, instead of relying on the runtime error message.
    The enum mirrors the scitypes accepted by the runtime check in
    query_registry.py, plus the 'tag'/'tags' special case.
    """
    tools = {t.name: t for t in asyncio.run(list_tools())}
    task_schema = tools["query_registry"].inputSchema["properties"]["task"]
    assert task_schema["enum"] == [
        "aligner",
        "catalogue",
        "classifier",
        "clusterer",
        "dataset",
        "dataset_classification",
        "dataset_forecasting",
        "dataset_regression",
        "detector",
        "early_classifier",
        "estimator",
        "forecaster",
        "metric",
        "metric_detection",
        "metric_forecasting",
        "metric_forecasting_proba",
        "network",
        "object",
        "param_est",
        "reconciler",
        "regressor",
        "splitter",
        "transformer",
        "transformer-pairwise",
        "transformer-pairwise-panel",
        "tag",
        "tags",
    ]


def test_evaluate_cv_folds_has_minimum_one():
    """cv_folds must be a positive fold count (#407)."""
    tools = {t.name: t for t in asyncio.run(list_tools())}
    folds_schema = tools["evaluate"].inputSchema["properties"]["cv_folds"]
    assert folds_schema["minimum"] == 1
