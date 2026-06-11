"""
Data splitting tool for sktime MCP.

Provides temporal train/test splitting for time series data,
registering both halves as new data handles.
"""

import logging
import uuid
from typing import Any

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def split_data_tool(
    data_handle: str,
    test_size: float | None = None,
    fh: list[int] | int | None = None,
) -> dict[str, Any]:
    """Split a time series data handle into train and test sets.

    Uses sktime's temporal_train_test_split() when available, falling
    back to a pandas-based implementation. Exactly one of `test_size`
    or `fh` must be provided.

    Parameters
    ----------
    data_handle : str
        Handle ID of the loaded data to split (from load_data_source).
    test_size : float or None, default=None
        Fraction of the data to hold out for testing (0.0–1.0).
        Mutually exclusive with `fh`.
    fh : int, list of int, or None, default=None
        Forecast horizon — the number of final time steps to reserve
        as the test set. Can be a single int or a list of relative
        step indices. Mutually exclusive with `test_size`.

    Returns
    -------
    dict
        Dictionary containing the split train/test handles and metadata:
        - "success" : bool
            True if the split completed successfully, False otherwise.
        - "train_handle" : str
            The new unique data handle ID representing the training set.
        - "test_handle" : str
            The new unique data handle ID representing the test set.
        - "cutoff" : str
            The cutoff timestamp indicating the last training timestamp.
        - "train_size" : int
            Number of observations in the training set.
        - "n_test" : int
            Number of observations in the test set.
        - "error" : str, optional
            Error message if "success" is False.
    """
    executor = get_executor()

    # --- validation --------------------------------------------------------
    if data_handle not in executor._data_handles:
        return {
            "success": False,
            "error": f"Data handle '{data_handle}' not found",
            "available_handles": list(executor._data_handles.keys()),
        }

    if test_size is not None and fh is not None:
        return {
            "success": False,
            "error": "Provide exactly one of 'test_size' or 'fh', not both.",
        }

    if test_size is None and fh is None:
        return {
            "success": False,
            "error": "Provide at least one of 'test_size' or 'fh'.",
        }

    if test_size is not None and (test_size <= 0.0 or test_size >= 1.0):
        return {
            "success": False,
            "error": f"test_size must be between 0.0 and 1.0 (exclusive), got {test_size}",
        }

    if fh is not None:
        if isinstance(fh, int):
            if fh < 1:
                return {
                    "success": False,
                    "error": f"fh must be a positive integer, got {fh}",
                }
        elif isinstance(fh, list):
            if not fh or not all(isinstance(step, int) and step > 0 for step in fh):
                return {
                    "success": False,
                    "error": "fh must be a non-empty list of positive integers.",
                }
        else:
            return {
                "success": False,
                "error": f"fh must be an integer or list of integers, got {type(fh).__name__}",
            }

    data_info = executor._data_handles[data_handle]
    y = data_info["y"]
    X = data_info.get("X")

    try:
        # --- determine split point ----------------------------------------
        n = len(y)

        if test_size is not None:
            n_test = max(1, int(n * test_size))
        elif isinstance(fh, int):
            n_test = fh
        else:
            # fh as list of relative horizon indices — reserve max(fh) final steps
            n_test = max(fh)

        if n_test >= n:
            return {
                "success": False,
                "error": (
                    f"Test set would be {n_test} samples but the series only has {n} observations."
                ),
            }

        split_idx = n - n_test

        # --- try sktime first ---------------------------------------------
        try:
            from sktime.split import temporal_train_test_split

            y_train, y_test = temporal_train_test_split(y, test_size=n_test / n)
            if X is not None:
                X_train, X_test = temporal_train_test_split(X, test_size=n_test / n)
            else:
                X_train, X_test = None, None
        except Exception:
            # Fallback: plain pandas slicing
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            if X is not None:
                X_train = X.iloc[:split_idx]
                X_test = X.iloc[split_idx:]
            else:
                X_train, X_test = None, None

        # --- cutoff -------------------------------------------------------
        cutoff = str(y_train.index[-1])

        # --- register handles --------------------------------------------
        train_handle = f"data_{uuid.uuid4().hex[:8]}"
        test_handle = f"data_{uuid.uuid4().hex[:8]}"

        base_meta = data_info.get("metadata", {}).copy()

        train_meta = {
            **base_meta,
            "split": "train",
            "rows": len(y_train),
            "start_date": str(y_train.index[0]),
            "end_date": str(y_train.index[-1]),
            "parent_handle": data_handle,
        }
        test_meta = {
            **base_meta,
            "split": "test",
            "rows": len(y_test),
            "start_date": str(y_test.index[0]),
            "end_date": str(y_test.index[-1]),
            "parent_handle": data_handle,
        }

        executor._register_data_handle(
            train_handle,
            {
                "y": y_train,
                "X": X_train,
                "metadata": train_meta,
                "validation": data_info.get("validation", {}),
                "config": data_info.get("config", {}),
            },
        )
        executor._register_data_handle(
            test_handle,
            {
                "y": y_test,
                "X": X_test,
                "metadata": test_meta,
                "validation": data_info.get("validation", {}),
                "config": data_info.get("config", {}),
            },
        )

        return {
            "success": True,
            "train_handle": train_handle,
            "test_handle": test_handle,
            "cutoff": cutoff,
            "train_size": len(y_train),
            "n_test": len(y_test),
        }

    except Exception as e:
        logger.exception("Error splitting data")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
