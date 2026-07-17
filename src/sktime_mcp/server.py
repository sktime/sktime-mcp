"""
MCP Server for sktime.

Main entry point for the Model Context Protocol server
that exposes sktime's registry and execution capabilities to LLMs.
"""

import asyncio
import json
import logging
import os
import sys
from io import TextIOWrapper
from typing import Any

import anyio

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from sktime_mcp.config import settings
from sktime_mcp.tools.codegen import export_code_tool
from sktime_mcp.tools.data_tools import (
    load_data_source_tool,
    release_data_handle_tool,
)
from sktime_mcp.tools.describe_component import (
    describe_component_tool,
)
from sktime_mcp.tools.evaluate import evaluate_tool
from sktime_mcp.tools.fit_predict import (
    fit_tool,
    get_fitted_params_tool,
    predict_tool,
    update_tool,
)
from sktime_mcp.tools.inspect_data import inspect_data_tool
from sktime_mcp.tools.instantiate import (
    instantiate_estimator_tool,
    list_handles_tool,
    load_model_tool,
    release_handle_tool,
)
from sktime_mcp.tools.job_tools import (
    cancel_job_tool,
    check_job_status_tool,
    list_jobs_tool,
)
from sktime_mcp.tools.list_available_data import list_available_data_tool
from sktime_mcp.tools.plotting import plot_series_tool
from sktime_mcp.tools.query_registry import (
    query_registry_tool,
)
from sktime_mcp.tools.run_command import run_command_tool
from sktime_mcp.tools.save_data import save_data_tool
from sktime_mcp.tools.save_model import save_model_tool
from sktime_mcp.tools.split_data import split_data_tool
from sktime_mcp.tools.transform_data import transform_data_tool


# ---------------------------------------------------------------------------
# Server configuration via environment variables
# ---------------------------------------------------------------------------
def _get_int_env(name: str, default: int) -> int:
    """Return an integer env var value, falling back to default on parse errors."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Invalid %s=%r; using default value %d instead.",
            name,
            raw_value,
            default,
        )
        return default


JOB_MAX_AGE_HOURS = _get_int_env("SKTIME_MCP_JOB_MAX_AGE_HOURS", 24)
JOB_CLEANUP_INTERVAL_SECS = _get_int_env("SKTIME_MCP_JOB_CLEANUP_INTERVAL", 3600)

# Configure logging to stderr with detailed format
_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
if settings.log_path:
    _handlers.append(logging.FileHandler(settings.log_path))

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger(__name__)
# Create MCP server instance
server = Server("sktime-mcp")

_CHARS_PER_TOKEN = 4


def _apply_response_token_limit(tool_name: str, text: str) -> str:
    """Truncate *text* to the configured token budget and append a notice.

    Reads ``SKTIME_MCP_MAX_RESPONSE_TOKENS`` from environment variable at call time so
    that live config changes are respected.
    Returns *text* unchanged when the limit is 0 (unlimited) or not set.
    """
    from sktime_mcp.config import settings

    max_tokens = settings.max_response_tokens
    if max_tokens <= 0:
        return text  # unlimited

    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text

    notice = (
        f"\n\n[sktime-mcp] Response truncated: output exceeded the SKTIME_MCP_MAX_RESPONSE_TOKENS "
        f"limit of {max_tokens} tokens (tool: {tool_name}). "
        "Increase SKTIME_MCP_MAX_RESPONSE_TOKENS or narrow your query for full results."
    )
    # Reserve space for the notice inside the budget
    budget = max_chars - len(notice)
    if budget < 0:
        budget = 0
    return text[:budget] + notice


def sanitize_for_json(obj):
    """Recursively convert objects to JSON-serializable format.

    Handles:
    - Standard Python scalars and containers (dict, list, tuple)
    - NumPy integer/float scalars and ndarrays
    - Pandas Timestamp, NaT, NA, and Series/DataFrame
    - Arbitrary objects (fallback to str repr)
    """
    # --- NumPy types ---
    if _NUMPY_AVAILABLE:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [sanitize_for_json(item) for item in obj.tolist()]
        if isinstance(obj, np.complexfloating):
            return str(obj)

    # --- Pandas types ---
    if _PANDAS_AVAILABLE:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if obj is pd.NaT:
            return None
        # pd.NA
        try:
            if obj is pd.NA:
                return None
        except AttributeError:
            pass
        if isinstance(obj, pd.Series):
            return sanitize_for_json(obj.tolist())
        if isinstance(obj, pd.DataFrame):
            return sanitize_for_json(obj.to_dict(orient="records"))

    # --- Standard Python containers ---
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]

    # --- Already JSON-safe scalars ---
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # --- Fallback: objects with __dict__ or anything else ---
    if hasattr(obj, "__dict__"):
        return str(obj)

    # Last resort
    return str(obj)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        # -- Discovery -------------------------------------------------------
        Tool(
            name="query_registry",
            description=(
                "Discover sktime estimators, metrics, or capability tags. "
                "Common tags you can filter estimators by: "
                "'capability:pred_int' (bool) - prediction intervals, "
                "'capability:multivariate' (bool) - multivariate support, "
                "'handles-missing-data' (bool) - NaN handling, "
                "'scitype:y' (str) - target type ('univariate'/'multivariate'/'both'), "
                "'requires-fh-in-fit' (bool) - needs forecast horizon at fit time. "
                "Set task='tag' (or 'tags') to query the full list of capability tags."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Filter by scitype: forecaster, classifier, regressor, "
                            "transformer, clusterer, detector, splitter, metric, "
                            "param_est, aligner, network. "
                            "Set to 'tag' or 'tags' to retrieve capability tags."
                        ),
                    },
                    "tags": {
                        "type": "object",
                        "description": "Filter by capability tags, e.g. {'capability:pred_int': true}. Ignored if task='tag'.",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Search by name or description (substring, case-insensitive). "
                            "Can be combined with task and tags filters."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 50). Ignored if task='tag'.",
                        "default": 50,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip this many results for pagination (default: 0). Ignored if task='tag'.",
                        "default": 0,
                    },
                },
            },
        ),
        Tool(
            name="describe_component",
            description="Get detailed information about ANY class or component in the sktime ecosystem (estimators, splitters, metrics, transformers)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the component class (e.g., 'ARIMA', 'SlidingWindowSplitter', 'MeanAbsolutePercentageError')",
                    },
                },
                "required": ["name"],
            },
        ),
        # -- Instantiation ---------------------------------------------------
        Tool(
            name="instantiate_estimator",
            description=(
                "Create an estimator or pipeline instance using a sktime craft specification. "
                "The spec is a string that evaluates to an estimator, e.g., 'ARIMA(order=(1, 1, 1))' "
                "or 'Detrender() * ARIMA()'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "string",
                        "description": "Craft specification string.",
                    },
                },
                "required": ["spec"],
            },
        ),
        Tool(
            name="list_handles",
            description="List all active estimator handles in memory",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="release_handle",
            description="Release an estimator handle and free it from memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "Handle ID to release",
                    },
                },
                "required": ["handle"],
            },
        ),
        # -- Execution -------------------------------------------------------
        Tool(
            name="fit",
            description=(
                "Fit an estimator on data. "
                "Provide explicit X_handle and/y_handle (or datasets) depending on the estimator's scitype."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle from instantiate_estimator",
                    },
                    "X_handle": {
                        "type": "string",
                        "description": "Optional: Handle from load_data_source for X data (features, panel, etc.)",
                    },
                    "y_handle": {
                        "type": "string",
                        "description": "Optional: Handle from load_data_source for y data (target, labels, etc.)",
                    },
                    "X_dataset": {
                        "type": "string",
                        "description": "Optional: Demo dataset name for X data",
                    },
                    "y_dataset": {
                        "type": "string",
                        "description": "Optional: Demo dataset name for y data",
                    },
                    "fh": {
                        "description": "Optional: Forecast horizon (e.g. 12 or [1,2,3]) to pass to fit",
                    },
                    "run_async": {
                        "type": "boolean",
                        "description": "If True, runs the fit asynchronously in the background and returns a job_id.",
                    },
                },
                "required": ["estimator_handle"],
            },
        ),
        Tool(
            name="predict",
            description=(
                "Generate predictions from a fitted estimator. "
                "Supports different modes like predict, predict_interval, predict_quantiles."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle of a fitted estimator",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Forecast horizon (default: 12)",
                        "default": 12,
                    },
                    "mode": {
                        "type": "string",
                        "description": "Prediction mode",
                        "enum": [
                            "predict",
                            "predict_interval",
                            "predict_quantiles",
                            "predict_proba",
                            "predict_var",
                        ],
                        "default": "predict",
                    },
                    "coverage": {
                        "description": "Coverage level for intervals (float or list of floats)",
                        "default": 0.9,
                    },
                    "alpha": {
                        "description": "Alpha values for quantiles (float or list of floats)",
                    },
                    "X_handle": {
                        "type": "string",
                        "description": "Optional: Handle from load_data_source for X data",
                    },
                    "y_handle": {
                        "type": "string",
                        "description": "Optional: Handle from load_data_source for y data (needed for annotators)",
                    },
                    "X_dataset": {
                        "type": "string",
                        "description": "Optional: Demo dataset name for X data",
                    },
                    "y_dataset": {
                        "type": "string",
                        "description": "Optional: Demo dataset name for y data",
                    },
                },
                "required": ["estimator_handle"],
            },
        ),
        Tool(
            name="update",
            description=("Update a fitted estimator with new data."),
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle of a fitted estimator",
                    },
                    "X_handle": {
                        "type": "string",
                        "description": "Optional: Handle for X data",
                    },
                    "y_handle": {
                        "type": "string",
                        "description": "Optional: Handle for y data",
                    },
                    "X_dataset": {
                        "type": "string",
                        "description": "Optional: Demo dataset for X data",
                    },
                    "y_dataset": {
                        "type": "string",
                        "description": "Optional: Demo dataset for y data",
                    },
                },
                "required": ["estimator_handle"],
            },
        ),
        Tool(
            name="get_fitted_params",
            description="Get fitted parameters from an estimator",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle of a fitted estimator",
                    },
                },
                "required": ["estimator_handle"],
            },
        ),
        Tool(
            name="call_method",
            description=(
                "Dynamically call any native method on an instantiated sktime component (e.g. 'split', 'get_alignment', '__call__'). "
                "Use this tool to interact with non-standard scitypes like Splitters, Metrics, or Aligners that do not support "
                "the generic 'fit' or 'predict' endpoints. Pass 'kwargs' as a dictionary of arguments."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "handle_id": {
                        "type": "string",
                        "description": "Memory handle ID of the instantiated component",
                    },
                    "method_name": {
                        "type": "string",
                        "description": "Name of the method to call (e.g. 'split')",
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Dictionary of keyword arguments to pass to the method. "
                        "Pass '_dataset' or '_data_handle' as suffixes in keys to inject memory data (e.g., {'y_dataset': 'airline'}).",
                    },
                },
                "required": ["handle_id", "method_name"],
            },
        ),
        # -- Execution (Macros) ----------------------------------------------
        Tool(
            name="evaluate",
            description=(
                "Cross-validate an estimator on a dataset. "
                "Dataset and data handle inputs supported for y and X."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle from instantiate_estimator",
                    },
                    "y": {
                        "type": "string",
                        "description": "Target time series: data_handle id or demo dataset name (e.g. 'airline').",
                    },
                    "X": {
                        "type": "string",
                        "description": "Optional: exogenous time series data_handle id or demo dataset name.",
                    },
                    "cv_folds": {
                        "type": "integer",
                        "description": "Number of cross-validation folds (default: 3). Ignored if initial_window is set.",
                        "default": 3,
                    },
                    "metric": {
                        "type": "string",
                        "description": "Optional: performance metric name (e.g. 'MeanAbsolutePercentageError').",
                    },
                    "initial_window": {
                        "type": "integer",
                        "description": "Optional: initial training window size for expanding-window CV.",
                    },
                    "run_async": {
                        "type": "boolean",
                        "description": "If true, runs evaluation as a background job and returns a job_id.",
                    },
                },
                "required": ["estimator_handle", "y"],
            },
        ),
        # -- Data ------------------------------------------------------------
        Tool(
            name="list_available_data",
            description=(
                "List all data available for use — system demo datasets and active "
                "user-loaded data handles — in a single unified response. "
                "Use is_demo=true for demos only, is_demo=false for handles only, "
                "or omit is_demo to get both."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "is_demo": {
                        "type": "boolean",
                        "description": (
                            "Optional filter: true = only system demos, "
                            "false = only active data handles, omit = both."
                        ),
                    },
                },
            },
        ),
        Tool(
            name="load_data_source",
            description=(
                "Load data from various sources into a data handle for forecasting. "
                "Can run synchronously (blocking) or asynchronously in the background. "
                "Supported source types: "
                "'pandas' - from a dict or inline data (keys: data, time_column, target_column). "
                "'file' - from CSV, Excel (.xlsx), or Parquet (keys: path, time_column, target_column). "
                "'sql' - from a SQL database (keys: connection_string, query, time_column, target_column). "
                "'url' - from a web URL pointing to CSV/Excel/Parquet (keys: url, time_column, target_column). "
                "GUIDELINES: "
                "1. NEVER assume a column is a time index unless the user says so. "
                "2. ALWAYS specify 'target_column' if the user mentions a specific variable. "
                "3. The first column is used as target by default — if that's a date column, "
                "specify target_column explicitly. "
                "4. For non-standard date formats, omit 'time_column' to use an integer index."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "description": (
                            "Data source configuration. Must include 'type' "
                            "(pandas, sql, file, url)."
                        ),
                    },
                    "run_async": {
                        "type": "boolean",
                        "description": (
                            "If True, loads data in the background (non-blocking) and "
                            "returns a job_id. If False (default), blocks and returns the "
                            "data_handle directly."
                        ),
                        "default": False,
                    },
                },
                "required": ["config"],
            },
        ),
        Tool(
            name="release_data_handle",
            description="Release a data handle and free memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_handle": {
                        "type": "string",
                        "description": "Data handle to release",
                    },
                },
                "required": ["data_handle"],
            },
        ),
        Tool(
            name="inspect_data",
            description=(
                "Inspect a loaded data handle and return rich metadata for understanding "
                "the series before modelling. Returns mtype, scitype, shape, column names, "
                "dtypes, index level names, inferred frequency, cutoff (last training "
                "timestamp), total missing-value count, a 5-row head preview, and "
                "per-column summary statistics. Works on handles from load_data_source, "
                "split_data, or transform_data. Does not modify the data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_handle": {
                        "type": "string",
                        "description": (
                            "Data handle ID to inspect (from load_data_source, split_data, "
                            "or transform_data)."
                        ),
                    },
                },
                "required": ["data_handle"],
            },
        ),
        Tool(
            name="split_data",
            description=(
                "Split a time series data handle into temporal train and test sets, "
                "registering both halves as new data handles. Provide exactly one of "
                "test_size (fraction in (0, 1)) or fh (forecast horizon). fh may be an "
                "integer (hold out that many final steps) or a list of relative horizon "
                "indices (hold out max(fh) final steps). Returns train_handle, "
                "test_handle, cutoff timestamp, train_size, and n_test."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_handle": {
                        "type": "string",
                        "description": "Data handle ID to split (from load_data_source or transform_data).",
                    },
                    "test_size": {
                        "type": "number",
                        "description": (
                            "Fraction of observations to hold out for the test set, "
                            "exclusive range (0.0, 1.0). Mutually exclusive with fh."
                        ),
                    },
                    "fh": {
                        "description": (
                            "Forecast horizon for the test window. Integer: hold out that "
                            "many final time steps. List of ints: hold out max(fh) final "
                            "steps (e.g. fh=[1,5,10] reserves 10 steps). "
                            "Mutually exclusive with test_size."
                        ),
                    },
                },
                "required": ["data_handle"],
            },
        ),
        Tool(
            name="transform_data",
            description=(
                "Transform a loaded data handle and return a new handle. "
                "action='format' (default): auto-fix common time series issues — "
                "infer/set frequency, remove duplicate timestamps, fill index gaps, "
                "and forward/backward-fill missing values; returns changes_applied. "
                "action='convert': convert y to a different sktime mtype via convert_to() "
                "(requires to_mtype, e.g. 'pd.DataFrame', 'pd.Series', 'np.ndarray'). "
                "Replaces the legacy format_time_series tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_handle": {
                        "type": "string",
                        "description": "Data handle ID to transform.",
                    },
                    "action": {
                        "type": "string",
                        "description": (
                            "Transformation to apply: 'format' (default) or 'convert'."
                        ),
                        "enum": ["format", "convert"],
                        "default": "format",
                    },
                    "auto_infer_freq": {
                        "type": "boolean",
                        "description": "(format only) Infer and set DatetimeIndex frequency (default: true).",
                        "default": True,
                    },
                    "fill_missing": {
                        "type": "boolean",
                        "description": "(format only) Forward/backward fill missing values (default: true).",
                        "default": True,
                    },
                    "remove_duplicates": {
                        "type": "boolean",
                        "description": "(format only) Drop duplicate timestamps, keeping first (default: true).",
                        "default": True,
                    },
                    "to_mtype": {
                        "type": "string",
                        "description": (
                            "(convert only, required) Target sktime mtype string, "
                            "e.g. 'pd.DataFrame', 'pd.Series', 'np.ndarray'."
                        ),
                    },
                },
                "required": ["data_handle"],
            },
        ),
        Tool(
            name="save_data",
            description=(
                "Persist the target series (y) and any exogenous features (X) behind a "
                "data handle to a local file. Combines y and X into one table. Creates "
                "parent directories as needed. Supported formats: csv (default, writes "
                "index as first column), parquet, json (records orient, ISO dates)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_handle": {
                        "type": "string",
                        "description": (
                            "Data handle ID to export (from load_data_source, split_data, "
                            "or transform_data)."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Destination file path. Format is controlled by the format "
                            "argument, not the file extension."
                        ),
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: csv (default), parquet, or json.",
                        "enum": ["csv", "parquet", "json"],
                        "default": "csv",
                    },
                },
                "required": ["data_handle", "path"],
            },
        ),
        # -- Visualization ---------------------------------------------------
        Tool(
            name="plot_series",
            description=(
                "Plot one or more time series natively. "
                "Can save the plot to a specified path as a PNG file or return it as a base64 string."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_handles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of data handle IDs to plot (e.g., train, test, forecasts).",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of labels corresponding to each data handle.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the plot.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional local file path to save the plot (e.g., '/tmp/plot.png'). If omitted, returns base64.",
                    },
                    "figsize": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Figure size as [width, height] in inches (default: [12, 6]).",
                    },
                    "dpi": {
                        "type": "integer",
                        "description": "Resolution in dots per inch (default: 150).",
                        "default": 150,
                    },
                    "markers": {
                        "description": "Marker style(s) for data points (e.g., 'o', ['.', 'x']).",
                    },
                    "x_label": {
                        "type": "string",
                        "description": "Custom x-axis label.",
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Custom y-axis label.",
                    },
                    "image_format": {
                        "type": "string",
                        "description": "Image output format: 'png' (default), 'svg', or 'webp'.",
                        "enum": ["png", "svg", "webp"],
                        "default": "png",
                    },
                },
                "required": ["data_handles"],
            },
        ),
        # -- Export / Persistence --------------------------------------------
        Tool(
            name="export_code",
            description="Export an estimator or pipeline as executable Python code",
            inputSchema={
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "Handle ID of the estimator/pipeline to export",
                    },
                    "var_name": {
                        "type": "string",
                        "description": "Variable name to use in generated code (default: 'model')",
                        "default": "model",
                    },
                    "include_fit_example": {
                        "type": "boolean",
                        "description": "Whether to include a fit/predict example (default: false)",
                        "default": False,
                    },
                    "dataset": {
                        "type": "string",
                        "description": (
                            "Optional dataset name for the fit example "
                            "(e.g. 'airline', 'sunspots'). Defaults to 'airline' if omitted."
                        ),
                    },
                },
                "required": ["handle"],
            },
        ),
        Tool(
            name="save_model",
            description="Save an estimator/pipeline handle using sktime MLflow integration",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle ID of the estimator to save",
                    },
                    "path": {
                        "type": "string",
                        "description": "Local directory or URI where the model will be saved",
                    },
                    "mlflow_params": {
                        "type": "object",
                        "description": "Optional extra parameters for sktime.utils.mlflow_sktime.save_model",
                    },
                },
                "required": ["estimator_handle", "path"],
            },
        ),
        Tool(
            name="load_model",
            description="Load a saved sktime model from a local path and register it for use",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the saved model directory",
                    },
                },
                "required": ["path"],
            },
        ),
        # -- Jobs ------------------------------------------------------------
        Tool(
            name="check_job_status",
            description="Check the status and progress of a background job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to check",
                    },
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="list_jobs",
            description="List all background jobs with optional status filter",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status: pending, running, completed, failed, cancelled",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of jobs to return (default: 20)",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="cancel_job",
            description=(
                "Cancel a running or pending background job. "
                "Set delete=true to also remove the job record entirely "
                "(useful for cleaning up completed/failed jobs)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to cancel",
                    },
                    "delete": {
                        "type": "boolean",
                        "description": "Also remove the job record after cancelling (default: false)",
                        "default": False,
                    },
                },
                "required": ["job_id"],
            },
        ),
        # -- System ----------------------------------------------------------
        Tool(
            name="run_command",
            description=(
                "Run an arbitrary CLI/bash command inside the sktime container. "
                "Use this to install missing python packages (e.g., 'pip install mlflow') "
                "or inspect the file system."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run",
                    },
                },
                "required": ["command"],
            },
        ),
    ]


# ===================================================================
# Tool dispatcher
# ===================================================================


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    import importlib
    import site
    import sys

    # Ensure user site-packages is in sys.path (solves Docker non-root dynamic install issue)
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)

    # Invalidate import caches so newly installed packages are immediately visible
    importlib.invalidate_caches()

    logger.info(f"=== Tool Call: {name} ===")
    logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")

    try:
        # -- Discovery -------------------------------------------------------
        if name == "query_registry":
            result = query_registry_tool(
                task=arguments.get("task"),
                tags=arguments.get("tags"),
                query=arguments.get("query"),
                limit=arguments.get("limit", 50),
                offset=arguments.get("offset", 0),
            )

        elif name == "describe_component":
            result = describe_component_tool(name=arguments["name"])

        # -- Instantiation ---------------------------------------------------
        elif name == "instantiate_estimator":
            result = instantiate_estimator_tool(spec=arguments.get("spec"))

        elif name == "list_handles":
            result = list_handles_tool()

        elif name == "release_handle":
            result = release_handle_tool(arguments["handle"])

        # -- Execution -------------------------------------------------------
        elif name == "fit":
            result = fit_tool(
                estimator_handle=arguments["estimator_handle"],
                X_handle=arguments.get("X_handle"),
                y_handle=arguments.get("y_handle"),
                X_dataset=arguments.get("X_dataset"),
                y_dataset=arguments.get("y_dataset"),
                fh=arguments.get("fh"),
                run_async=arguments.get("run_async", False),
            )

        elif name == "predict":
            result = predict_tool(
                estimator_handle=arguments["estimator_handle"],
                horizon=arguments.get("horizon", 12),
                mode=arguments.get("mode", "predict"),
                coverage=arguments.get("coverage", 0.9),
                alpha=arguments.get("alpha"),
                X_handle=arguments.get("X_handle"),
                y_handle=arguments.get("y_handle"),
                X_dataset=arguments.get("X_dataset"),
                y_dataset=arguments.get("y_dataset"),
            )

        elif name == "update":
            result = update_tool(
                estimator_handle=arguments["estimator_handle"],
                X_handle=arguments.get("X_handle"),
                y_handle=arguments.get("y_handle"),
                X_dataset=arguments.get("X_dataset"),
                y_dataset=arguments.get("y_dataset"),
            )

        elif name == "get_fitted_params":
            result = get_fitted_params_tool(arguments["estimator_handle"])

        elif name == "call_method":
            from sktime_mcp.runtime.executor import get_executor

            result = get_executor().call_method(
                handle_id=arguments["handle_id"],
                method_name=arguments["method_name"],
                kwargs=arguments.get("kwargs", {}),
            )

        elif name == "evaluate":
            result = evaluate_tool(
                estimator_handle=arguments["estimator_handle"],
                y=arguments["y"],
                X=arguments.get("X"),
                cv_folds=arguments.get("cv_folds", 3),
                metric=arguments.get("metric"),
                initial_window=arguments.get("initial_window"),
                run_async=arguments.get("run_async", False),
            )

        # -- Data ------------------------------------------------------------
        elif name == "list_available_data":
            result = list_available_data_tool(arguments.get("is_demo"))
        elif name == "load_data_source":
            result = load_data_source_tool(arguments["config"], arguments.get("run_async", False))

        elif name == "list_data_sources":
            # Deprecated — info is now in load_data_source description
            logger.warning(
                "list_data_sources is deprecated; info is in load_data_source description"
            )
            from sktime_mcp.tools.data_tools import list_data_sources_tool

            result = list_data_sources_tool()

        elif name == "release_data_handle":
            result = release_data_handle_tool(arguments["data_handle"])

        elif name == "inspect_data":
            result = inspect_data_tool(arguments["data_handle"])

        elif name == "split_data":
            result = split_data_tool(
                data_handle=arguments["data_handle"],
                test_size=arguments.get("test_size"),
                fh=arguments.get("fh"),
            )

        elif name == "transform_data":
            result = transform_data_tool(
                data_handle=arguments["data_handle"],
                action=arguments.get("action", "format"),
                auto_infer_freq=arguments.get("auto_infer_freq", True),
                fill_missing=arguments.get("fill_missing", True),
                remove_duplicates=arguments.get("remove_duplicates", True),
                to_mtype=arguments.get("to_mtype"),
            )

        elif name == "save_data":
            result = save_data_tool(
                data_handle=arguments["data_handle"],
                path=arguments["path"],
                format=arguments.get("format", "csv"),
            )

        elif name == "auto_format_on_load":
            # Deprecated — now controlled via SKTIME_MCP_AUTO_FORMAT env var
            logger.warning(
                "auto_format_on_load is deprecated; use env var SKTIME_MCP_AUTO_FORMAT=true/false"
            )
            from sktime_mcp.tools.format_tools import auto_format_on_load_tool

            result = auto_format_on_load_tool(arguments.get("enabled", True))

        # -- Visualization ---------------------------------------------------
        elif name == "plot_series":
            result = plot_series_tool(
                data_handles=arguments["data_handles"],
                labels=arguments.get("labels"),
                title=arguments.get("title"),
                path=arguments.get("path"),
                figsize=arguments.get("figsize"),
                dpi=arguments.get("dpi"),
                markers=arguments.get("markers"),
                x_label=arguments.get("x_label"),
                y_label=arguments.get("y_label"),
                image_format=arguments.get("image_format", "png"),
            )

        # -- Export / Persistence --------------------------------------------
        elif name == "export_code":
            result = export_code_tool(
                arguments["handle"],
                arguments.get("var_name", "model"),
                arguments.get("include_fit_example", False),
                arguments.get("dataset"),
            )

        elif name == "save_model":
            result = save_model_tool(
                arguments["estimator_handle"],
                arguments["path"],
                arguments.get("mlflow_params"),
            )

        elif name == "load_model":
            result = load_model_tool(arguments["path"])

        # -- Jobs ------------------------------------------------------------
        elif name == "check_job_status":
            result = check_job_status_tool(arguments["job_id"])

        elif name == "list_jobs":
            result = list_jobs_tool(
                arguments.get("status"),
                arguments.get("limit", 20),
            )

        elif name == "cancel_job":
            result = cancel_job_tool(
                arguments["job_id"],
                delete=arguments.get("delete", False),
            )

        elif name == "delete_job":
            # Deprecated — kept for backward compatibility
            logger.warning("delete_job is deprecated; use cancel_job(delete=true)")
            result = cancel_job_tool(arguments["job_id"], delete=True)

        elif name == "cleanup_old_jobs":
            # Deprecated — now runs automatically on a periodic timer
            logger.warning("cleanup_old_jobs is deprecated; jobs are cleaned up automatically")
            from sktime_mcp.tools.job_tools import cleanup_old_jobs_tool

            result = cleanup_old_jobs_tool(arguments.get("max_age_hours", 24))

        # -- System ----------------------------------------------------------
        elif name == "run_command":
            result = run_command_tool(arguments["command"])
        else:
            result = {"error": f"Unknown tool: {name}"}

        logger.info(f"=== Result for {name} ===")

        sanitized_result = sanitize_for_json(result)
        logger.info(f"{json.dumps(sanitized_result, indent=2, default=str)}")

        response_text = json.dumps(sanitized_result, indent=2, default=str)
        truncated_text = _apply_response_token_limit(name, response_text)

        return [TextContent(type="text", text=truncated_text)]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]


# ===================================================================
# Server lifecycle
# ===================================================================


async def _periodic_job_cleanup():
    """Automatically clean up old jobs on a timer."""
    from sktime_mcp.runtime.jobs import get_job_manager

    while True:
        await asyncio.sleep(settings.job_cleanup_interval_secs)
        try:
            job_manager = get_job_manager()
            removed = job_manager.cleanup_old_jobs(settings.job_max_age_hours)
            if removed:
                logger.info(f"Periodic cleanup: removed {removed} old job(s)")
        except Exception:
            logger.exception("Error during periodic job cleanup")


async def run_server():
    """Run the MCP server."""
    # Stdio safety: redirect stdout to stderr to protect MCP JSON-RPC
    # streams from being corrupted by stray prints in third-party libraries.
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    # Explicitly wrap the original stdout buffer for the MCP server output
    mcp_stdout = anyio.wrap_file(TextIOWrapper(original_stdout.buffer, encoding="utf-8"))

    asyncio.create_task(_periodic_job_cleanup())

    async with stdio_server(stdout=mcp_stdout) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Main entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nINFO: sktime-mcp server shut down gracefully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
