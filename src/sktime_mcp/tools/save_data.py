"""
Data persistence tool for sktime MCP.

Saves the data behind a handle to a local file in CSV, Parquet, or JSON format.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)

# Supported output formats and their pandas writer methods
_FORMAT_WRITERS = {
    "csv": "to_csv",
    "parquet": "to_parquet",
    "json": "to_json",
}


def save_data_tool(
    data_handle: str,
    path: str,
    format: str = "csv",
) -> dict[str, Any]:
    """
    Persist the data behind a handle to a local file.

    Supports CSV, Parquet, and JSON output formats.  The target
    directory is created automatically if it does not exist.

    Args:
        data_handle: Handle ID from load_data / split_data / transform_data
        path: Destination file path (e.g. ``"/tmp/forecast_output.csv"``).
              The file extension is **not** used to infer the format;
              use the ``format`` argument instead.
        format: Output format — ``"csv"`` (default), ``"parquet"``, or ``"json"``.

    Returns:
        Dictionary with:
        - success: bool
        - saved_path: str (absolute path to the written file)
        - format: str
        - rows: int
    """
    executor = get_executor()

    # --- validation --------------------------------------------------------
    if data_handle not in executor._data_handles:
        return {
            "success": False,
            "error": f"Data handle '{data_handle}' not found",
            "available_handles": list(executor._data_handles.keys()),
        }

    fmt = format.lower()
    if fmt not in _FORMAT_WRITERS:
        return {
            "success": False,
            "error": f"Unsupported format '{format}'. Choose from: {list(_FORMAT_WRITERS.keys())}",
        }

    data_info = executor._data_handles[data_handle]
    y = data_info["y"]
    X = data_info.get("X")

    try:
        # Combine y and X into a single DataFrame for export
        if isinstance(y, pd.Series):
            df = y.to_frame(name=y.name if y.name else "target")
        elif isinstance(y, pd.DataFrame):
            df = y.copy()
        else:
            # Best-effort: wrap in a DataFrame
            df = pd.DataFrame(y, columns=["target"])

        if X is not None and isinstance(X, pd.DataFrame):
            df = pd.concat([df, X], axis=1)

        # Ensure target directory exists
        abs_path = Path(path).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Write
        writer = getattr(df, _FORMAT_WRITERS[fmt])
        if fmt == "json":
            writer(str(abs_path), orient="records", date_format="iso", indent=2)
        elif fmt == "parquet":
            writer(str(abs_path))
        else:
            # CSV — include the index as a time column
            writer(str(abs_path))

        return {
            "success": True,
            "saved_path": str(abs_path),
            "format": fmt,
            "rows": len(df),
        }

    except Exception as e:
        logger.exception("Error saving data")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
