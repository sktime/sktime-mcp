"""
MCP Server for sktime.

Main entry point for the Model Context Protocol server
that exposes sktime's registry and execution capabilities to LLMs.
"""

import asyncio
import json
import logging
import os
from typing import Any

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

from sktime_mcp.composition.validator import get_composition_validator
from sktime_mcp.tools.codegen import export_code_tool
from sktime_mcp.tools.data_tools import (
    load_data_source_async_tool,
    load_data_source_tool,
    release_data_handle_tool,
)
from sktime_mcp.tools.describe_estimator import describe_estimator_tool
from sktime_mcp.tools.evaluate import evaluate_estimator_tool
from sktime_mcp.tools.fit_predict import (
    fit_predict_async_tool,
    fit_predict_tool,
)
from sktime_mcp.tools.format_tools import format_time_series_tool
from sktime_mcp.tools.instantiate import (
    instantiate_estimator_tool,
    instantiate_pipeline_tool,
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
from sktime_mcp.tools.list_estimators import (
    get_available_tags,
    list_estimators_tool,
)
from sktime_mcp.tools.save_model import save_model_tool

# ---------------------------------------------------------------------------
# Server configuration via environment variables
# ---------------------------------------------------------------------------
JOB_MAX_AGE_HOURS = int(os.environ.get("SKTIME_MCP_JOB_MAX_AGE_HOURS", "24"))
JOB_CLEANUP_INTERVAL_SECS = int(os.environ.get("SKTIME_MCP_JOB_CLEANUP_INTERVAL", "3600"))

# Configure logging to stderr with detailed format
_LOG_LEVEL = os.environ.get("SKTIME_MCP_LOG_LEVEL", "WARNING").upper()

logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
# Create MCP server instance
server = Server("sktime-mcp")


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
            name="list_estimators",
            description=(
                "Discover sktime estimators by task, capability tags, or name search. "
                "Common tags you can filter by: "
                "'capability:pred_int' (bool) - prediction intervals, "
                "'capability:multivariate' (bool) - multivariate support, "
                "'handles-missing-data' (bool) - NaN handling, "
                "'scitype:y' (str) - target type ('univariate'/'multivariate'/'both'), "
                "'requires-fh-in-fit' (bool) - needs forecast horizon at fit time. "
                "Use get_available_tags for the full catalog."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Task type filter: forecasting, classification, "
                            "regression, transformation, clustering"
                        ),
                    },
                    "tags": {
                        "type": "object",
                        "description": "Filter by capability tags, e.g. {'capability:pred_int': true}",
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
                        "description": "Maximum results (default: 50)",
                        "default": 50,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip this many results for pagination (default: 0)",
                        "default": 0,
                    },
                },
            },
        ),
        Tool(
            name="describe_estimator",
            description="Get detailed information about a specific sktime estimator",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator": {
                        "type": "string",
                        "description": "Name of the estimator (e.g., 'ARIMA', 'RandomForest')",
                    },
                },
                "required": ["estimator"],
            },
        ),
        Tool(
            name="get_available_tags",
            description=(
                "List all queryable capability tags with rich metadata. "
                "Returns tag name, description, expected value type, and which "
                "estimator types the tag applies to. Call this before "
                "using tags in list_estimators to ensure correct tag names and values."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        # -- Instantiation ---------------------------------------------------
        Tool(
            name="instantiate_estimator",
            description="Create an estimator instance with given parameters",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator": {
                        "type": "string",
                        "description": "Name of the estimator to instantiate",
                    },
                    "params": {
                        "type": "object",
                        "description": "Hyperparameters for the estimator",
                    },
                },
                "required": ["estimator"],
            },
        ),
        Tool(
            name="instantiate_pipeline",
            description="Create a pipeline instance from a list of components",
            inputSchema={
                "type": "object",
                "properties": {
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of estimator names in pipeline order (e.g., ['Detrender', 'ARIMA'])",
                    },
                    "params_list": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional list of hyperparameter dicts for each component",
                    },
                },
                "required": ["components"],
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
        Tool(
            name="validate_pipeline",
            description="Check if a pipeline composition is valid",
            inputSchema={
                "type": "object",
                "properties": {
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of estimator names in pipeline order",
                    },
                },
                "required": ["components"],
            },
        ),
        # -- Execution -------------------------------------------------------
        Tool(
            name="fit_predict",
            description=(
                "Fit an estimator and generate predictions. "
                "Provide exactly one of: dataset (demo name such as airline, sunspots) "
                "or data_handle (from load_data_source for custom data)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle from instantiate_estimator",
                    },
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name: airline, sunspots, lynx, etc.",
                    },
                    "data_handle": {
                        "type": "string",
                        "description": (
                            "Handle from load_data_source (use this instead "
                            "of dataset for custom data)"
                        ),
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Forecast horizon (default: 12)",
                        "default": 12,
                    },
                },
                "required": ["estimator_handle"],
            },
        ),
        Tool(
            name="fit_predict_async",
            description=(
                "Fit an estimator and generate predictions in the background. "
                "Provide exactly ONE of 'dataset' (built-in demo name) "
                "or 'data_handle' (from load_data_source)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle from instantiate_estimator",
                    },
                    "dataset": {
                        "type": "string",
                        "description": "Demo dataset name: airline, sunspots, lynx, etc.",
                    },
                    "data_handle": {
                        "type": "string",
                        "description": "Data handle from load_data_source (e.g. 'data_abc123')",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Forecast horizon (default: 12)",
                        "default": 12,
                    },
                },
                "required": ["estimator_handle"],
            },
        ),
        Tool(
            name="evaluate_estimator",
            description="Evaluate an estimator using cross-validation on a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle from instantiate_estimator",
                    },
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name: airline, sunspots, lynx, etc.",
                    },
                    "cv_folds": {
                        "type": "integer",
                        "description": "Number of cross-validation folds (default: 3)",
                        "default": 3,
                    },
                },
                "required": ["estimator_handle", "dataset"],
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
                },
                "required": ["config"],
            },
        ),
        Tool(
            name="load_data_source_async",
            description=(
                "Load data from any source in the background "
                "(non-blocking). Returns a job_id to track "
                "progress. The data_handle is available in "
                "the job result when completed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "description": "Data source configuration. Same format as load_data_source.",
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
            name="format_time_series",
            description="Automatically format time series data (frequency, duplicates, missing values)",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_handle": {
                        "type": "string",
                        "description": "Handle from load_data_source",
                    },
                    "auto_infer_freq": {
                        "type": "boolean",
                        "description": "Automatically infer and set frequency (default: True)",
                        "default": True,
                    },
                    "fill_missing": {
                        "type": "boolean",
                        "description": "Fill missing values with forward/backward fill (default: True)",
                        "default": True,
                    },
                    "remove_duplicates": {
                        "type": "boolean",
                        "description": "Remove duplicate timestamps (default: True)",
                        "default": True,
                    },
                },
                "required": ["data_handle"],
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
    ]


# ===================================================================
# Tool dispatcher
# ===================================================================


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"=== Tool Call: {name} ===")
    logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")

    try:
        # -- Discovery -------------------------------------------------------
        if name == "list_estimators":
            result = list_estimators_tool(
                task=arguments.get("task"),
                tags=arguments.get("tags"),
                query=arguments.get("query"),
                limit=arguments.get("limit", 50),
                offset=arguments.get("offset", 0),
            )

        elif name == "search_estimators":
            # Deprecated — kept for backward compatibility, routes to unified list_estimators
            logger.warning("search_estimators is deprecated; use list_estimators(query=...)")
            result = list_estimators_tool(
                query=arguments["query"],
                limit=arguments.get("limit", 20),
            )

        elif name == "describe_estimator":
            result = describe_estimator_tool(arguments["estimator"])

        elif name == "get_available_tags":
            result = get_available_tags()

        # -- Instantiation ---------------------------------------------------
        elif name == "instantiate_estimator":
            result = instantiate_estimator_tool(
                arguments["estimator"],
                arguments.get("params"),
            )

        elif name == "instantiate_pipeline":
            result = instantiate_pipeline_tool(
                arguments["components"],
                arguments.get("params_list"),
            )

        elif name == "list_handles":
            result = list_handles_tool()

        elif name == "release_handle":
            result = release_handle_tool(arguments["handle"])

        # -- Execution -------------------------------------------------------
        elif name == "fit_predict":
            result = fit_predict_tool(
                arguments["estimator_handle"],
                arguments.get("dataset", ""),
                arguments.get("horizon", 12),
                data_handle=arguments.get("data_handle"),
            )

        elif name == "fit_predict_async":
            result = fit_predict_async_tool(
                estimator_handle=arguments["estimator_handle"],
                dataset=arguments.get("dataset"),
                data_handle=arguments.get("data_handle"),
                horizon=arguments.get("horizon", 12),
            )

        elif name == "evaluate_estimator":
            result = evaluate_estimator_tool(
                arguments["estimator_handle"],
                arguments["dataset"],
                arguments.get("cv_folds", 3),
            )

        elif name == "validate_pipeline":
            validator = get_composition_validator()
            validation = validator.validate_pipeline(arguments["components"])
            result = validation.to_dict()
            result["success"] = result["valid"]

        # -- Data ------------------------------------------------------------
        elif name == "list_available_data":
            result = list_available_data_tool(arguments.get("is_demo"))
        elif name == "load_data_source":
            result = load_data_source_tool(arguments["config"])

        elif name == "load_data_source_async":
            result = load_data_source_async_tool(arguments["config"])

        elif name == "list_data_sources":
            # Deprecated — info is now in load_data_source description
            logger.warning(
                "list_data_sources is deprecated; info is in load_data_source description"
            )
            from sktime_mcp.tools.data_tools import list_data_sources_tool

            result = list_data_sources_tool()

        elif name == "release_data_handle":
            result = release_data_handle_tool(arguments["data_handle"])

        elif name == "format_time_series":
            result = format_time_series_tool(
                arguments["data_handle"],
                arguments.get("auto_infer_freq", True),
                arguments.get("fill_missing", True),
                arguments.get("remove_duplicates", True),
            )

        elif name == "auto_format_on_load":
            # Deprecated — now controlled via SKTIME_MCP_AUTO_FORMAT env var
            logger.warning(
                "auto_format_on_load is deprecated; use env var SKTIME_MCP_AUTO_FORMAT=true/false"
            )
            from sktime_mcp.tools.format_tools import auto_format_on_load_tool

            result = auto_format_on_load_tool(arguments.get("enabled", True))

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

        else:
            result = {"error": f"Unknown tool: {name}"}

        logger.info(f"=== Result for {name} ===")

        sanitized_result = sanitize_for_json(result)
        logger.info(f"{json.dumps(sanitized_result, indent=2, default=str)}")

        return [TextContent(type="text", text=json.dumps(sanitized_result, indent=2, default=str))]
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
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SECS)
        try:
            job_manager = get_job_manager()
            removed = job_manager.cleanup_old_jobs(JOB_MAX_AGE_HOURS)
            if removed:
                logger.info(f"Periodic cleanup: removed {removed} old job(s)")
        except Exception:
            logger.exception("Error during periodic job cleanup")


async def run_server():
    """Run the MCP server."""
    asyncio.create_task(_periodic_job_cleanup())

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Main entry point."""
    logger.info("Starting sktime-mcp server...")
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
