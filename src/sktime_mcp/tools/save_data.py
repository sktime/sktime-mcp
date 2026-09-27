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


def _index_to_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Move every index level into a column with a non-colliding name.

    Level names are kept when present; an unnamed time level becomes
    ``"time"`` and other unnamed levels ``"level_<i>"``. A name that is
    already a data column gets an ``_index`` suffix (then ``_index_2``, ...)
    instead of failing with "cannot insert time, already exists" (F-32).
    Returns the frame with a RangeIndex and the list of index column names,
    the last of which is the time level.
    """
    taken = {str(c) for c in df.columns}
    n_levels = df.index.nlevels
    names: list[str] = []
    for i, level_name in enumerate(df.index.names):
        if level_name is not None:
            base = str(level_name)
        else:
            base = "time" if i == n_levels - 1 else f"level_{i}"
        candidate, n = base, 1
        while candidate in taken:
            n += 1
            candidate = f"{base}_index" if n == 2 else f"{base}_index_{n - 1}"
        names.append(candidate)
        taken.add(candidate)
    return df.reset_index(names=names), names


def save_data_tool(
    data_handle: str,
    path: str,
    format: str = "csv",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist the data behind a handle to a local file.

    Supports CSV, Parquet, and JSON output formats. The target
    directory is created automatically if it does not exist.

    CSV and JSON write the time index as a regular column named after the
    index (``"time"`` when the index is unnamed; ``"time_index"`` if that
    name is already a data column). The name used is returned as
    ``time_column`` so the file can be loaded back with
    ``load_data_source(time_column=<time_column>)``. A MultiIndex (panel or
    hierarchical data) becomes one column per level, listed in
    ``index_columns``; ``time_column`` is the last level. Parquet keeps the
    index in the file itself, so it is loaded back without ``time_column``.

    Parameters
    ----------
    data_handle : str
        Handle ID of the data to save (from load_data_source, split_data, etc.).
    path : str
        Destination file path (e.g. "/tmp/forecast_output.csv").
        Note that the file extension is not used to infer the format;
        use the `format` argument instead.
    format : str, default="csv"
        Output format. Must be one of: "csv", "parquet", or "json".
    overwrite : bool, default=False
        If the target file already exists, the write is refused unless this
        is True, in which case the response reports ``overwritten: true``.

    Returns
    -------
    dict
        Dictionary containing success status and path information:

        - ``"success"`` (bool) -- True if the data was written successfully, False otherwise.
        - ``"saved_path"`` (str) -- Absolute path to the written file.
        - ``"format"`` (str) -- The format used to write the file.
        - ``"rows"`` (int) -- Number of rows written to the file.
        - ``"overwritten"`` (bool) -- True if an existing file was replaced.
        - ``"time_column"`` (str or None) -- Name of the column holding the time
          index (csv/json); None for parquet, which stores the index itself.
        - ``"index_columns"`` (list of str or None) -- All index level columns
          written (csv/json); a single-element list unless the index is a
          MultiIndex.
        - ``"error"`` (str, optional) -- Error message if "success" is False.
    """
    executor = get_executor()

    # --- validation --------------------------------------------------------
    if data_handle not in executor._data_handles:
        return {
            "success": False,
            **executor.data_handle_missing(data_handle),
        }

    fmt = format.lower()
    if fmt not in _FORMAT_WRITERS:
        return {
            "success": False,
            "error": f"Unsupported format '{format}'. Choose from: {list(_FORMAT_WRITERS.keys())}",
        }

    # Resolve (expanduser so "~/x" doesn't create a literal "~" dir) and guard
    # against silently clobbering an existing file (#539).
    abs_path = Path(path).expanduser().resolve()
    existed = abs_path.exists()
    if existed and not overwrite:
        return {
            "success": False,
            "error": (
                f"File already exists: '{abs_path}'. Pass overwrite=true to replace it, "
                "or choose a different path."
            ),
            "saved_path": str(abs_path),
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
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Write
        if fmt == "parquet":
            # Parquet stores the index in the file; pd.read_parquet restores it.
            df.to_parquet(str(abs_path))
            index_columns = None
        else:
            # csv/json: the index becomes explicit column(s) so the file
            # round-trips through load_data_source(time_column=...) (N-2).
            out_df, index_columns = _index_to_columns(df)
            if fmt == "json":
                # Period/Timestamp values are not JSON serialisable as-is.
                for col in index_columns:
                    out_df[col] = out_df[col].astype(str)
                out_df.to_json(str(abs_path), orient="records", indent=2)
            else:
                out_df.to_csv(str(abs_path), index=False)

        result = {
            "success": True,
            "saved_path": str(abs_path),
            "format": fmt,
            "rows": len(df),
            "overwritten": existed,
            "time_column": index_columns[-1] if index_columns else None,
            "index_columns": index_columns,
        }
        if fmt == "parquet":
            result["note"] = (
                "Parquet keeps the index in the file; load it back with "
                "load_data_source without time_column."
            )
        return result

    except Exception as e:
        logger.exception("Error saving data")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
