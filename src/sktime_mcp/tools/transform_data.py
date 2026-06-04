"""
Data transformation tool for sktime MCP.

Provides two actions:
  - "format": auto-fix frequency, duplicates, missing values (replaces format_time_series).
  - "convert": convert data between sktime mtypes using convert_to().
"""

import logging
import uuid
from typing import Any

import pandas as pd

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def transform_data_tool(
    data_handle: str,
    action: str = "format",
    auto_infer_freq: bool = True,
    fill_missing: bool = True,
    remove_duplicates: bool = True,
    to_mtype: str | None = None,
) -> dict[str, Any]:
    """Transform a data handle — either format it or convert its mtype.

    Supports two modes controlled by the `action` argument.

    Parameters
    ----------
    data_handle : str
        Handle ID of the loaded data to transform (from load_data_source).
    action : str, default="format"
        The transformation action to perform. Must be one of:
        - "format" : Auto-fix common time series issues like inferring frequency,
          removing duplicate timestamps, and filling missing values.
        - "convert" : Convert the data to a different sktime machine type (mtype).
    auto_infer_freq : bool, default=True
        (Format mode only) Infer and set frequency.
    fill_missing : bool, default=True
        (Format mode only) Forward/backward fill missing values.
    remove_duplicates : bool, default=True
        (Format mode only) Remove duplicate timestamps.
    to_mtype : str or None, default=None
        (Convert mode only) Target machine type string, e.g. "pd.DataFrame",
        "pd.Series", "np.ndarray".

    Returns
    -------
    dict
        Dictionary containing the new data handle and a list of applied changes:
        - "success" : bool
            True if the transformation succeeded, False otherwise.
        - "data_handle" : str
            The new unique data handle ID representing the transformed data.
        - "changes_applied" : list of str
            A list of human-readable changes that were applied to the data.
        - "metadata" : dict, optional
            Updated metadata for the new handle.
        - "error" : str, optional
            Error message if "success" is False.
    """
    if action not in ("format", "convert"):
        return {
            "success": False,
            "error": f"Unknown action '{action}'. Must be 'format' or 'convert'.",
        }

    if action == "convert" and not to_mtype:
        return {
            "success": False,
            "error": "The 'to_mtype' argument is required when action='convert'.",
        }

    executor = get_executor()

    if data_handle not in executor._data_handles:
        return {
            "success": False,
            "error": f"Data handle '{data_handle}' not found",
            "available_handles": list(executor._data_handles.keys()),
        }

    try:
        if action == "format":
            return _action_format(
                executor,
                data_handle,
                auto_infer_freq=auto_infer_freq,
                fill_missing=fill_missing,
                remove_duplicates=remove_duplicates,
            )
        else:
            return _action_convert(executor, data_handle, to_mtype)
    except Exception as e:
        logger.exception("Error transforming data")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# ---------------------------------------------------------------------------
# Action: format
# ---------------------------------------------------------------------------


def _action_format(
    executor: Any,
    data_handle: str,
    *,
    auto_infer_freq: bool,
    fill_missing: bool,
    remove_duplicates: bool,
) -> dict[str, Any]:
    """Delegate to the executor's existing format logic and wrap the result."""
    result = executor.format_data_handle(
        data_handle,
        auto_infer_freq=auto_infer_freq,
        fill_missing=fill_missing,
        remove_duplicates=remove_duplicates,
    )

    if not result.get("success"):
        return result

    # Build a human-readable list of changes
    changes_applied: list[str] = []
    changes = result.get("changes_made", {})

    if changes.get("duplicates_removed", 0) > 0:
        changes_applied.append(f"Removed {changes['duplicates_removed']} duplicate timestamps")
    if changes.get("frequency_set"):
        freq = changes.get("frequency", "?")
        changes_applied.append(f"Inferred and set frequency to '{freq}'")
    if changes.get("gaps_filled", 0) > 0:
        changes_applied.append(f"Filled {changes['gaps_filled']} gaps in the time index")
    if changes.get("missing_filled", 0) > 0:
        changes_applied.append(
            f"Filled {changes['missing_filled']} missing values (forward/backward fill)"
        )
    if not changes_applied:
        changes_applied.append("No changes needed — data was already clean")

    return {
        "success": True,
        "data_handle": result["data_handle"],
        "changes_applied": changes_applied,
        "metadata": result.get("metadata", {}),
    }


# ---------------------------------------------------------------------------
# Action: convert
# ---------------------------------------------------------------------------


def _action_convert(
    executor: Any,
    data_handle: str,
    to_mtype: str,
) -> dict[str, Any]:
    """Convert the data to a different sktime mtype."""
    data_info = executor._data_handles[data_handle]
    y = data_info["y"]

    try:
        from sktime.datatypes import convert_to
    except ImportError:
        return {
            "success": False,
            "error": "sktime.datatypes.convert_to is not available in this environment.",
        }

    original_mtype = type(y).__name__
    converted = convert_to(y, to_type=to_mtype)

    # Register as new handle
    new_handle = f"data_{uuid.uuid4().hex[:8]}"
    base_meta = data_info.get("metadata", {}).copy()
    base_meta["mtype"] = to_mtype
    base_meta["converted_from"] = original_mtype
    base_meta["parent_handle"] = data_handle

    # Determine y and X for the new handle
    if isinstance(converted, pd.DataFrame):
        new_y = converted
        new_X = None
    elif isinstance(converted, pd.Series):
        new_y = converted
        new_X = data_info.get("X")
    else:
        # numpy or other — wrap as Series for consistency
        new_y = converted
        new_X = None

    executor._register_data_handle(
        new_handle,
        {
            "y": new_y,
            "X": new_X,
            "metadata": base_meta,
            "validation": data_info.get("validation", {}),
            "config": data_info.get("config", {}),
        },
    )

    return {
        "success": True,
        "data_handle": new_handle,
        "changes_applied": [f"Converted from '{original_mtype}' to '{to_mtype}'"],
        "metadata": base_meta,
    }
