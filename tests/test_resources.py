"""MCP resources, exercised through a real client session."""

import json

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS

from sktime_mcp import resources
from sktime_mcp.server import sanitize_for_json
from sktime_mcp.tools.describe_component import describe_component_tool
from sktime_mcp.tools.list_available_data import list_available_data_tool
from sktime_mcp.tools.query_registry import query_registry_tool
from tests.test_mcp_wire import _connected_session


async def _read_json(uri: str) -> dict:
    async with _connected_session() as session:
        result = await session.read_resource(uri)
    assert len(result.contents) == 1
    contents = result.contents[0]
    assert contents.uri == uri
    assert contents.mime_type == "application/json"
    return json.loads(contents.text)


async def _read_error(uri: str) -> MCPError:
    async with _connected_session() as session:
        with pytest.raises(MCPError) as excinfo:
            await session.read_resource(uri)
    assert excinfo.value.code == INVALID_PARAMS
    return excinfo.value


async def test_capabilities_advertise_resources():
    async with _connected_session() as session:
        init = await session.initialize()
    assert init.capabilities.resources is not None


async def test_list_resources_and_templates():
    async with _connected_session() as session:
        listed = await session.list_resources()
        templates = await session.list_resource_templates()
    assert {r.uri for r in listed.resources} == {"sktime://tags", "sktime://datasets"}
    assert [t.uri_template for t in templates.resource_templates] == ["sktime://components/{name}"]
    assert all(r.mime_type == "application/json" for r in listed.resources)


async def test_tags_resource_returns_full_catalogue():
    body = await _read_json("sktime://tags")
    expected_total = query_registry_tool(task="tag")["total"]
    assert body["success"] is True
    assert body["total"] == expected_total == len(body["results"])
    # More than one default tool page, so pagination was followed.
    assert expected_total > 50


def test_tags_resource_merges_pages_without_duplicates(monkeypatch):
    monkeypatch.setattr(resources, "_TAG_PAGE_SIZE", 7)
    body = resources.read_resource("sktime://tags")
    names = [t["tag"] for t in body["results"]]
    assert len(names) == len(set(names)) == query_registry_tool(task="tag")["total"]


def test_tags_resource_surfaces_tool_failure(monkeypatch):
    monkeypatch.setattr(
        resources,
        "query_registry_tool",
        lambda **kwargs: {"success": False, "error": "registry unavailable"},
    )
    with pytest.raises(MCPError, match="registry unavailable"):
        resources.read_resource("sktime://tags")


async def test_datasets_resource_matches_tool():
    body = await _read_json("sktime://datasets")
    assert body == list_available_data_tool(is_demo=True)
    assert "airline" in body["system_demos"]["forecasting"]


async def test_component_resource_matches_tool():
    body = await _read_json("sktime://components/NaiveForecaster")
    assert body == sanitize_for_json(describe_component_tool("NaiveForecaster"))
    assert body["name"] == "NaiveForecaster"


async def test_unknown_component_is_an_error():
    error = await _read_error("sktime://components/NoSuchForecaster")
    assert "Unknown component class" in error.message


@pytest.mark.parametrize(
    "name", ["", "os;rm", "../ARIMA", "1ARIMA", "a..b", "ARIMA()", "sktime.ARIMA."]
)
def test_malformed_component_name_rejected_before_lookup(monkeypatch, name):
    def fail(name):
        raise AssertionError("resolver must not be called")

    monkeypatch.setattr(resources, "describe_component_tool", fail)
    with pytest.raises(MCPError, match="Malformed component name") as excinfo:
        resources.read_resource("sktime://components/" + name)
    assert excinfo.value.code == INVALID_PARAMS


async def test_malformed_component_name_over_the_wire():
    error = await _read_error("sktime://components/os;rm")
    assert "Malformed component name" in error.message


@pytest.mark.parametrize("uri", ["sktime://nope", "sktime://components", "file:///etc/passwd"])
async def test_unknown_uri_is_an_error(uri):
    error = await _read_error(uri)
    assert "Unknown resource" in error.message
