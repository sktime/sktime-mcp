"""
Batch tool execution helpers for sktime MCP.

The MVP intentionally supports only read-only tools so agent workflows can
reduce round-trips without mutating server state.
"""

from collections.abc import Callable
from typing import Any

from sktime_mcp.tools.describe_estimator import describe_estimator_tool
from sktime_mcp.tools.list_available_data import list_available_data_tool
from sktime_mcp.tools.list_estimators import get_available_tags, list_estimators_tool


def run_tools_batch_tool(operations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Execute a list of read-only MCP operations in one call.

    Args:
        operations: List of operation dicts, each with:
            - tool: tool name
            - arguments: dict of arguments for that tool (optional)

    Returns:
        Dict with per-operation results and aggregate success.
    """
    if not isinstance(operations, list) or len(operations) == 0:
        return {
            "success": False,
            "error": "operations must be a non-empty list.",
        }

    dispatch: dict[str, Callable[..., dict[str, Any]]] = {
        "list_estimators": list_estimators_tool,
        "describe_estimator": describe_estimator_tool,
        "get_available_tags": get_available_tags,
        "list_available_data": list_available_data_tool,
    }

    results: list[dict[str, Any]] = []

    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            results.append(
                {
                    "index": index,
                    "success": False,
                    "error": "Operation must be an object with 'tool' and optional 'arguments'.",
                }
            )
            continue

        tool_name = operation.get("tool")
        arguments = operation.get("arguments", {})

        if not isinstance(tool_name, str) or not tool_name.strip():
            results.append(
                {
                    "index": index,
                    "success": False,
                    "error": "Operation field 'tool' must be a non-empty string.",
                }
            )
            continue

        if not isinstance(arguments, dict):
            results.append(
                {
                    "index": index,
                    "tool": tool_name,
                    "success": False,
                    "error": "Operation field 'arguments' must be an object.",
                }
            )
            continue

        fn = dispatch.get(tool_name)
        if fn is None:
            results.append(
                {
                    "index": index,
                    "tool": tool_name,
                    "success": False,
                    "error": (
                        "Unsupported tool for batch execution. "
                        "Allowed: list_estimators, describe_estimator, "
                        "get_available_tags, list_available_data."
                    ),
                }
            )
            continue

        try:
            item_result = fn(**arguments)
            results.append(
                {
                    "index": index,
                    "tool": tool_name,
                    "success": bool(item_result.get("success", True)),
                    "result": item_result,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "index": index,
                    "tool": tool_name,
                    "success": False,
                    "error": str(exc),
                }
            )

    return {
        "success": all(item.get("success", False) for item in results),
        "count": len(results),
        "results": results,
    }

