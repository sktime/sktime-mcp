"""
Visualization tool for sktime MCP.

Plots one or more time series using sktime's native plotting utilities,
returning the result as a saved file or a base64-encoded image string.
"""

import base64
import io
import logging
from typing import Any

import pandas as pd

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)

# Default plot dimensions
_DEFAULT_FIGSIZE = (12, 6)
_DEFAULT_DPI = 150
_SUPPORTED_FORMATS = {"png", "svg", "webp"}


def _resolve_series(
    handle: str,
    executor: Any,
) -> pd.Series | pd.DataFrame:
    """Resolve a data handle to a pandas Series/DataFrame.

    Raises KeyError if the handle is not found.
    """
    if handle not in executor._data_handles:
        raise KeyError(handle)
    return executor._data_handles[handle]["y"]


def _coerce_indices(series_list: list) -> list:
    """Coerce mixed PeriodIndex / DatetimeIndex / string Index to DatetimeIndex.

    Modifies the series **in-place** and returns the same list.
    """
    has_period = any(isinstance(s.index, pd.PeriodIndex) for s in series_list)
    has_datetime = any(isinstance(s.index, pd.DatetimeIndex) for s in series_list)
    has_string = any(
        type(s.index).__name__ == "Index" and pd.api.types.is_string_dtype(s.index)
        for s in series_list
    )

    mixed = sum([has_period, has_datetime, has_string]) > 1
    if not mixed:
        return series_list

    logger.info("Coercing mixed index types to DatetimeIndex for plotting")
    for i, s in enumerate(series_list):
        try:
            if isinstance(s.index, pd.PeriodIndex):
                series_list[i] = s.copy()
                series_list[i].index = s.index.to_timestamp()
            elif not isinstance(s.index, pd.DatetimeIndex):
                series_list[i] = s.copy()
                series_list[i].index = pd.to_datetime(s.index)
        except Exception as e:
            logger.warning("Failed to coerce index for series %d: %s", i, e)

    return series_list


def _reconcile_labels(
    labels: list[str] | None,
    n_series: int,
) -> list[str] | None:
    """Return a label list that exactly matches *n_series*.

    - If *labels* is ``None``, returns ``None`` (sktime will auto-label).
    - If the lengths match, returns *labels* unchanged.
    - If they differ, pads with ``"Series N"`` or truncates, and logs a warning.
    """
    if labels is None:
        return None
    if len(labels) == n_series:
        return labels

    logger.warning(
        "Label count (%d) does not match series count (%d); adjusting.",
        len(labels),
        n_series,
    )
    if len(labels) < n_series:
        return labels + [f"Series {i}" for i in range(len(labels), n_series)]
    return labels[:n_series]


def plot_series_tool(
    data_handles: list[str],
    labels: list[str] | None = None,
    title: str | None = None,
    path: str | None = None,
    figsize: list[float] | None = None,
    dpi: int | None = None,
    markers: str | list[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    image_format: str = "png",
) -> dict[str, Any]:
    """Plot one or more time series natively.

    Parameters
    ----------
    data_handles : list of str
        List of data handle IDs to plot (e.g. train, test, forecasts).
    labels : list of str, optional
        Labels for each series. If the count does not match the number
        of data handles, it is silently padded or truncated.
    title : str, optional
        Title of the plot.
    path : str, optional
        Path to save the plot. If not provided, returns base64 encoded image.
    figsize : list of float, optional
        Figure size as ``[width, height]`` in inches. Default ``[12, 6]``.
    dpi : int, optional
        Resolution in dots per inch. Default ``150``.
    markers : str or list of str, optional
        Marker style(s) for data points (e.g. ``"o"``, ``[".", "x"]``).
        Passed through to ``sktime.utils.plotting.plot_series``.
    x_label : str, optional
        Custom label for the x-axis.
    y_label : str, optional
        Custom label for the y-axis.
    image_format : str, optional
        Image output format: ``"png"`` (default), ``"svg"``, or ``"webp"``.

    Returns
    -------
    dict
        Dictionary containing success status, path or base64 string,
        and plot metadata (``n_series``, ``labels_used``, ``figsize``,
        ``dpi``).
    """
    # --- validate image format ---------------------------------------------
    fmt = image_format.lower()
    if fmt not in _SUPPORTED_FORMATS:
        return {
            "success": False,
            "error": (
                f"Unsupported image format '{image_format}'. "
                f"Choose from: {sorted(_SUPPORTED_FORMATS)}"
            ),
        }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sktime.utils.plotting import plot_series
    except ImportError:
        return {
            "success": False,
            "error": "matplotlib and sktime plotting utils are required.",
        }

    executor = get_executor()

    # --- resolve data handles -----------------------------------------------
    series_to_plot: list = []
    missing: list[str] = []

    for handle in data_handles:
        try:
            series_to_plot.append(_resolve_series(handle, executor))
        except KeyError:
            missing.append(handle)

    if missing:
        return {
            "success": False,
            "error": f"Data handle(s) not found: {missing}",
            "available_handles": list(executor._data_handles.keys()),
        }

    # --- prepare plotting args ---------------------------------------------
    try:
        series_to_plot = _coerce_indices(series_to_plot)
        labels = _reconcile_labels(labels, len(series_to_plot))

        effective_figsize = tuple(figsize) if figsize else _DEFAULT_FIGSIZE
        effective_dpi = dpi if dpi else _DEFAULT_DPI

        # Build kwargs for plot_series
        plot_kwargs: dict[str, Any] = {}
        if markers is not None:
            plot_kwargs["markers"] = markers

        fig, ax = plot_series(
            *series_to_plot,
            labels=labels,
            **plot_kwargs,
        )

        # Resize figure after creation (plot_series creates its own figure)
        fig.set_size_inches(effective_figsize)
        fig.set_dpi(effective_dpi)

        if title:
            ax.set_title(title)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        # --- output ---------------------------------------------------------
        result: dict[str, Any] = {
            "success": True,
            "n_series": len(series_to_plot),
            "labels_used": labels or [f"Series {i}" for i in range(len(series_to_plot))],
            "figsize": list(effective_figsize),
            "dpi": effective_dpi,
            "image_format": fmt,
        }

        if path:
            fig.savefig(path, format=fmt, bbox_inches="tight", dpi=effective_dpi)
            result["path"] = path
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=effective_dpi)
            buf.seek(0)
            result["image_base64"] = base64.b64encode(buf.read()).decode("utf-8")

        plt.close(fig)
        return result

    except Exception as e:
        logger.exception("Error plotting series")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
