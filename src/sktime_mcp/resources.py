"""
MCP resources for sktime MCP.

Read-only views of the tag catalogue, the demo datasets and component
descriptions. Each one reuses an existing tool function, so a resource and the
matching tool always return the same JSON.
"""

import re
from typing import Any

from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, Resource, ResourceTemplate

from sktime_mcp.tools.describe_component import describe_component_tool
from sktime_mcp.tools.list_available_data import list_available_data_tool
from sktime_mcp.tools.query_registry import query_registry_tool

TAGS_URI = "sktime://tags"
DATASETS_URI = "sktime://datasets"
COMPONENT_URI_PREFIX = "sktime://components/"
JSON_MIME_TYPE = "application/json"

_TAG_PAGE_SIZE = 200
# Class names and dotted aliases only; anything else is rejected before lookup.
_COMPONENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


def get_resources() -> list[Resource]:
    """Return the fixed resources advertised by ``resources/list``."""
    return [
        Resource(
            name="tags",
            title="sktime capability tags",
            uri=TAGS_URI,
            description="Every capability tag in the registry, with its description.",
            mime_type=JSON_MIME_TYPE,
        ),
        Resource(
            name="datasets",
            title="sktime demo datasets",
            uri=DATASETS_URI,
            description="Built-in demo datasets grouped by task, as list_available_data returns.",
            mime_type=JSON_MIME_TYPE,
        ),
    ]


def get_resource_templates() -> list[ResourceTemplate]:
    """Return the templates advertised by ``resources/templates/list``."""
    return [
        ResourceTemplate(
            name="component",
            title="sktime component description",
            uri_template=COMPONENT_URI_PREFIX + "{name}",
            description="The describe_component result for a registered component class.",
            mime_type=JSON_MIME_TYPE,
        ),
    ]


def _read_all_tags() -> dict[str, Any]:
    tags: list[Any] = []
    offset = 0
    while True:
        page = query_registry_tool(task="tag", limit=_TAG_PAGE_SIZE, offset=offset)
        if not page["success"]:
            return page
        tags.extend(page["results"])
        if not page["has_more"]:
            return {"success": True, "results": tags, "total": len(tags)}
        offset += _TAG_PAGE_SIZE


def _read_component(name: str) -> dict[str, Any]:
    if not _COMPONENT_NAME.fullmatch(name):
        raise MCPError(INVALID_PARAMS, f"Malformed component name: {name!r}")
    return describe_component_tool(name)


def read_resource(uri: str) -> dict[str, Any]:
    """Return the JSON body of the resource at ``uri``.

    Raises ``MCPError`` with ``INVALID_PARAMS`` for an unknown URI, a malformed
    component name, or a lookup that fails (e.g. an unregistered component).
    """
    if uri == TAGS_URI:
        result = _read_all_tags()
    elif uri == DATASETS_URI:
        result = list_available_data_tool(is_demo=True)
    elif uri.startswith(COMPONENT_URI_PREFIX):
        result = _read_component(uri.removeprefix(COMPONENT_URI_PREFIX))
    else:
        raise MCPError(INVALID_PARAMS, f"Unknown resource: {uri}")

    if not result.get("success", False):
        raise MCPError(INVALID_PARAMS, result.get("error", f"Could not read {uri}"), data=result)
    return result
