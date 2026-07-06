import base64
import io
import logging
import pandas as pd
from typing import Any, List, Optional
import logging
from typing import Any, List, Optional

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)

def plot_series_tool(
    data_handles: List[str], 
    labels: Optional[List[str]] = None, 
    title: Optional[str] = None, 
    path: Optional[str] = None
) -> dict[str, Any]:
    """Plot one or more time series natively.

    Parameters
    ----------
    data_handles : list of str
        List of data handle IDs to plot (e.g. train, test, forecasts).
    labels : list of str, optional
        Labels for each series.
    title : str, optional
        Title of the plot.
    path : str, optional
        Path to save the plot. If not provided, returns base64 encoded image.

    Returns
    -------
    dict
        Dictionary containing success status and path/base64 string.
    """
    try:
        from sktime.utils.plotting import plot_series
        import matplotlib.pyplot as plt
    except ImportError:
        return {"success": False, "error": "matplotlib and sktime plotting utils are required."}

    executor = get_executor()
    
    series_to_plot = []
    
    for handle in data_handles:
        if handle not in executor._data_handles:
            return {"success": False, "error": f"Data handle '{handle}' not found."}
        series = executor._data_handles[handle]["y"]
        series_to_plot.append(series)

    try:
        # sktime requires all series to have the same index type
        # Check if we have a mix of PeriodIndex and DatetimeIndex
        has_period = any(isinstance(s.index, pd.PeriodIndex) for s in series_to_plot)
        has_datetime = any(isinstance(s.index, pd.DatetimeIndex) for s in series_to_plot)
        has_string = any(type(s.index).__name__ == "Index" and pd.api.types.is_string_dtype(s.index) for s in series_to_plot)
        
        if (has_period and has_datetime) or (has_period and has_string) or (has_datetime and has_string):
            # Coerce everything to DatetimeIndex for plotting
            logger.info("Coercing mixed index types to DatetimeIndex for plotting")
            for i in range(len(series_to_plot)):
                try:
                    if isinstance(series_to_plot[i].index, pd.PeriodIndex):
                        series_to_plot[i].index = series_to_plot[i].index.to_timestamp()
                    else:
                        series_to_plot[i].index = pd.to_datetime(series_to_plot[i].index)
                except Exception as e:
                    logger.warning(f"Failed to coerce index for series {i}: {e}")

        if labels and len(labels) != len(series_to_plot):
            logger.warning("Length of labels does not match number of series.")
            # Adjust labels or just pass it and let sktime handle, but it might crash
            # Let's pass it anyway or truncate/extend it if we want to be safe.
            # actually plot_series takes labels but it should match.

        fig, ax = plot_series(*series_to_plot, labels=labels)
        if title:
            ax.set_title(title)
        
        result = {"success": True}
        if path:
            fig.savefig(path, bbox_inches='tight')
            result["path"] = path
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            result["image_base64"] = img_base64
            
        plt.close(fig)
        return result
    except Exception as e:
        logger.exception("Error plotting series")
        return {"success": False, "error": str(e)}
