"""Inspect native sktime containers without changing them."""

from typing import Any

from sktime_mcp.data.contracts import describe_data, scientific_type
from sktime_mcp.runtime.executor import get_executor


def inspect_data_tool(data_handle: str) -> dict[str, Any]:
    executor = get_executor()
    if data_handle not in executor._data_handles:
        return {"success": False, **executor.data_handle_missing(data_handle)}
    record = executor._data_handles[data_handle]
    try:
        scitype = record.get("scitype") or scientific_type(record["y"])
        result = describe_data(record["y"], scitype)
        if record.get("X") is not None and record.get("role") != "X":
            result["exogenous"] = describe_data(record["X"], scitype)
            result["dtypes"].update({f"X:{k}": v for k, v in result["exogenous"]["dtypes"].items()})
            result["columns"].extend(f"X:{c}" for c in result["exogenous"]["columns"])
        return {
            "success": True,
            "data_handle": data_handle,
            "role": record.get("role", "y"),
            **result,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "error_type": type(exc).__name__}
