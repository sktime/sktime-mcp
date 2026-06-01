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
    """
    Inspect a loaded data handle and return rich metadata.

    Provides comprehensive information about the data behind a handle,
    including shape, column types, frequency, cutoff point, missing
    value counts, a preview (head), and summary statistics.

    Args:
        data_handle: Handle ID from load_data / load_data_source

    Returns:
        Dictionary with:
        - success: bool
        - data_handle: str
        - mtype: str (e.g. 'pd.Series', 'pd.DataFrame')
        - scitype: str (e.g. 'Series', 'Panel')
        - shape: list[int]
        - columns: list[str]
        - dtypes: dict[str, str]
        - index_names: list[str]
        - freq: str or None
        - cutoff: str or None
        - n_missing: int
        - head: dict (first 5 rows)
        - summary_stats: dict (count, mean, std, min, max per column)
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
        # --- mtype detection ---
        mtype = type(y).__name__
        if isinstance(y, pd.DataFrame):
            mtype = "pd.DataFrame"
        elif isinstance(y, pd.Series):
            mtype = "pd.Series"

        # --- scitype detection ---
        scitype = _detect_scitype(y)

        # --- shape ---
        shape = list(y.shape)

        # --- columns ---
        if isinstance(y, pd.DataFrame):
            columns = list(y.columns)
        elif isinstance(y, pd.Series):
            columns = [y.name if y.name else "target"]
        else:
            columns = []

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
        if hasattr(y.index, "names"):
            index_names = [str(n) if n is not None else "index" for n in y.index.names]
        else:
            index_names = ["index"]

        # --- frequency ---
        freq = None
        if hasattr(y.index, "freq") and y.index.freq is not None:
            freq = str(y.index.freq)
        elif hasattr(y.index, "inferred_freq"):
            freq = y.index.inferred_freq

        # --- cutoff ---
        cutoff = None
        try:
            from sktime.datatypes import get_cutoff as sktime_get_cutoff

            cutoff_val = sktime_get_cutoff(y)
            cutoff = str(cutoff_val)
        except Exception:
            # Fallback: use last index value
            if len(y) > 0:
                cutoff = str(y.index[-1])

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


def _detect_scitype(y: Any) -> str:
    """Detect the sktime scitype of the data."""
    try:
        from sktime.datatypes import scitype as sktime_scitype

        return sktime_scitype(y, candidate_scitypes=["Series", "Panel", "Hierarchical"])
    except Exception:
        pass

    # Fallback heuristic
    if isinstance(y, pd.Series):
        return "Series"
    if isinstance(y, pd.DataFrame):
        if isinstance(y.index, pd.MultiIndex):
            if y.index.nlevels >= 3:
                return "Hierarchical"
            return "Panel"
        return "Series"
    return "Unknown"


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
