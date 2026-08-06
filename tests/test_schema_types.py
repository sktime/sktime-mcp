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
