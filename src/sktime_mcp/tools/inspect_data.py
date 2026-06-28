"""
Data inspection tool for sktime MCP.

Provides rich metadata about loaded data handles including mtype,
scitype, shape, frequency, cutoff, missing values, and summary stats.
"""

import logging
from typing import Any

import pandas as pd

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def inspect_data_tool(data_handle: str) -> dict[str, Any]:
    """Inspect a loaded data handle and return rich metadata.

    Provides comprehensive information about the data behind a handle,
    including shape, column types, frequency, cutoff point, missing
    value counts, a preview (head), and summary statistics.

    Parameters
    ----------
    data_handle : str
        The unique handle ID for the loaded data source (from load_data_source).

    Returns
    -------
    dict
        Dictionary containing detailed metadata and summary statistics:
        - "success" : bool
            True if the data handle was found and inspected successfully.
        - "data_handle" : str
            The inspected data handle ID.
        - "mtype" : str
            The format mtype (e.g. 'pd.Series', 'pd.DataFrame').
        - "scitype" : str
            The sktime scientific type (e.g. 'Series', 'Panel').
        - "shape" : list of int
            Shape list: [rows, columns] or [rows].
        - "columns" : list of str
            Names of all variables (including exogenous features if present).
        - "dtypes" : dict
            Mapping of column names to string names of their data types.
        - "index_names" : list of str
            List of names of the index levels.
        - "freq" : str or None
            Inferred or declared frequency of the time index.
        - "cutoff" : str or None
            Cutoff timestamp/integer index indicating the end of the history.
        - "n_missing" : int
            Total count of missing values across the entire dataset.
        - "head" : dict
            Preview of the first 5 rows of data.
        - "summary_stats" : dict
            Statistical summary metrics (mean, std, min, max, etc.) per column.
        - "error" : str, optional
            Error message if "success" is False.
    """
    executor = get_executor()

    if data_handle not in executor._data_handles:
        return {
            "success": False,
            "error": f"Data handle '{data_handle}' not found",
            "available_handles": list(executor._data_handles.keys()),
        }

    data_info = executor._data_handles[data_handle]
    y = data_info["y"]
    X = data_info.get("X")

    try:
        from sktime.datatypes import check_is_scitype, get_cutoff
        
        valid, _, metadata = check_is_scitype(
            y, scitype=["Series", "Panel", "Hierarchical"], return_metadata=True
        )
        
        mtype = metadata.get("mtype", type(y).__name__)
        scitype = metadata.get("scitype", "Unknown")

        # --- shape ---
        shape = list(y.shape)

        # --- columns ---
        if isinstance(y, pd.DataFrame):
            columns = list(y.columns)
        elif isinstance(y, pd.Series):
            columns = [y.name if y.name else "target"]
        else:
            columns = metadata.get("feature_names", [])

        # Add exogenous columns if present
        if X is not None and isinstance(X, pd.DataFrame):
            columns = columns + [f"X:{c}" for c in X.columns]

        # --- dtypes ---
        if isinstance(y, pd.DataFrame):
            dtypes = {str(col): str(dtype) for col, dtype in y.dtypes.items()}
        elif isinstance(y, pd.Series):
            dtypes = {y.name if y.name else "target": str(y.dtype)}
        else:
            dtypes = {}

        # --- index names ---
        if hasattr(y, "index") and hasattr(y.index, "names"):
            index_names = [str(n) if n is not None else "index" for n in y.index.names]
        else:
            index_names = ["index"]

        # --- frequency ---
        freq = None
        if hasattr(y, "index"):
            if hasattr(y.index, "freq") and y.index.freq is not None:
                freq = str(y.index.freq)
            elif hasattr(y.index, "inferred_freq"):
                freq = y.index.inferred_freq

        # --- cutoff ---
        cutoff = str(get_cutoff(y))

        # --- missing values ---
        if isinstance(y, pd.DataFrame):
            n_missing = int(y.isna().sum().sum())
        elif isinstance(y, pd.Series):
            n_missing = int(y.isna().sum())
        else:
            n_missing = 0

        # --- head (first 5 rows) ---
        head_data = _safe_head(y, n=5)

        # --- summary statistics ---
        summary_stats = _safe_describe(y)

        return {
            "success": True,
            "data_handle": data_handle,
            "mtype": mtype,
            "scitype": scitype,
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            "index_names": index_names,
            "freq": freq,
            "cutoff": cutoff,
            "n_missing": n_missing,
            "head": head_data,
            "summary_stats": summary_stats,
        }

    except Exception as e:
        logger.exception("Error inspecting data handle")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def _safe_head(y: Any, n: int = 5) -> dict:
    """Return the first n rows as a JSON-safe dict."""
    try:
        if isinstance(y, pd.Series):
            head = y.head(n)
            return {str(k): _safe_value(v) for k, v in head.items()}
        if isinstance(y, pd.DataFrame):
            head = y.head(n)
            return {
                str(idx): {str(col): _safe_value(val) for col, val in row.items()}
                for idx, row in head.iterrows()
            }
    except Exception:
        pass
    return {}


def _safe_describe(y: Any) -> dict:
    """Return summary statistics as a JSON-safe dict."""
    try:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            desc = y.describe()
            if isinstance(desc, pd.Series):
                return {str(k): _safe_value(v) for k, v in desc.items()}
            if isinstance(desc, pd.DataFrame):
                return {
                    str(col): {str(stat): _safe_value(val) for stat, val in desc[col].items()}
                    for col in desc.columns
                }
    except Exception:
        pass
    return {}


def _safe_value(val: Any) -> Any:
    """Convert a value to a JSON-safe type."""
    if isinstance(val, float) and (pd.isna(val) or val != val):
        return None
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    return val
