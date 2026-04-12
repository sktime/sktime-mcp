"""
Plot tools for sktime MCP.

Returns base64-encoded PNG images via MCP ImageContent,
allowing MCP clients (e.g. Claude Desktop) to render
time series charts directly in the chat interface.
"""

import base64
import io
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _fig_to_base64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return data


def _load_series(data_handle: Optional[str], dataset: Optional[str]):
    """
    Return (y, label) from either a data handle or a demo dataset name.

    Raises ValueError if neither is provided or if the handle/dataset is unknown.
    """
    from sktime_mcp.runtime.executor import get_executor

    executor = get_executor()

    if data_handle is not None:
        if data_handle not in executor._data_handles:
            raise ValueError(f"Data handle '{data_handle}' not found")
        info = executor._data_handles[data_handle]
        y = info["y"]
        label = y.name or data_handle
        return y, label

    if dataset is not None:
        result = executor.load_dataset(dataset)
        if not result["success"]:
            raise ValueError(result.get("error", f"Unknown dataset: {dataset}"))
        y = result["data"]
        return y, dataset

    raise ValueError("Provide either 'data_handle' or 'dataset'")


def plot_data_tool(
    data_handle: Optional[str] = None,
    dataset: Optional[str] = None,
) -> dict[str, Any]:
    """
    Plot a time series as a line chart.

    Accepts either a loaded data handle (from load_data_source) or a
    demo dataset name (airline, sunspots, lynx, …).  Returns a
    base64-encoded PNG image that MCP clients can render inline.

    Args:
        data_handle: Handle from load_data_source (optional)
        dataset:     Demo dataset name (optional)

    Returns:
        Dict with:
        - success:   bool
        - image:     base64-encoded PNG string
        - mime_type: "image/png"
        - rows:      number of time steps plotted
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        y, label = _load_series(data_handle, dataset)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        y.index.astype(str),
        y.values,
        color="#2196F3",
        linewidth=1.5,
        label=label,
    )
    ax.set_title(f"Time Series: {label}")
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show at most 12 x-tick labels to avoid crowding
    tick_step = max(1, len(y) // 12)
    ax.set_xticks(range(0, len(y), tick_step))
    ax.set_xticklabels(
        [str(y.index[i]) for i in range(0, len(y), tick_step)],
        rotation=45,
        ha="right",
    )

    plt.tight_layout()
    image_data = _fig_to_base64(fig)
    plt.close(fig)

    return {
        "success": True,
        "image": image_data,
        "mime_type": "image/png",
        "rows": len(y),
    }


def plot_forecast_tool(
    predictions: dict,
    data_handle: Optional[str] = None,
    dataset: Optional[str] = None,
) -> dict[str, Any]:
    """
    Plot forecast predictions overlaid on historical time series.

    The historical series is provided via a data handle or demo dataset name.
    Predictions should be the dict returned by fit_predict or predict
    (keys are timestamp strings, values are forecast values).

    Args:
        predictions: Dict of {timestamp_str: value} from fit_predict / predict
        data_handle: Handle from load_data_source for historical overlay (optional)
        dataset:     Demo dataset name for historical overlay (optional)

    Returns:
        Dict with:
        - success:   bool
        - image:     base64-encoded PNG string
        - mime_type: "image/png"
        - horizon:   number of forecast steps plotted
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if not predictions:
        return {"success": False, "error": "'predictions' must be a non-empty dict"}

    pred_series = pd.Series(
        list(predictions.values()),
        index=list(predictions.keys()),
        name="forecast",
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    # Historical overlay (last 60 points for readability)
    if data_handle is not None or dataset is not None:
        try:
            y, label = _load_series(data_handle, dataset)
            y_tail = y.tail(60)
            ax.plot(
                y_tail.index.astype(str),
                y_tail.values,
                color="#2196F3",
                linewidth=1.5,
                label="Historical",
            )
        except ValueError as exc:
            plt.close(fig)
            return {"success": False, "error": str(exc)}
    else:
        label = "series"

    # Forecast line
    ax.plot(
        pred_series.index.astype(str),
        pred_series.values,
        color="#FF5722",
        linewidth=1.5,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Forecast",
    )

    title = f"Forecast: {label}" if (data_handle or dataset) else "Forecast"
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    image_data = _fig_to_base64(fig)
    plt.close(fig)

    return {
        "success": True,
        "image": image_data,
        "mime_type": "image/png",
        "horizon": len(pred_series),
    }
