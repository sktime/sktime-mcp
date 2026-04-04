"""
MCP Server for sktime.

Main entry point for the Model Context Protocol server
that exposes sktime's registry and execution capabilities to LLMs.
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from sktime_mcp.composition.validator import get_composition_validator
from sktime_mcp.tools.codegen import export_code_tool
from sktime_mcp.tools.data_tools import (
    fit_predict_with_data_tool,
    list_data_sources_tool,
    load_data_source_async_tool,
    load_data_source_tool,
    release_data_handle_tool,
)
from sktime_mcp.tools.describe_estimator import (
    describe_estimator_tool,
    search_estimators_tool,
)
from sktime_mcp.tools.fit_predict import (
    fit_predict_async_tool,
    fit_predict_tool,
)
from sktime_mcp.tools.format_tools import (
    auto_format_on_load_tool,
    format_time_series_tool,
)
from sktime_mcp.tools.instantiate import (
    instantiate_estimator_tool,
    instantiate_pipeline_tool,
    load_model_tool,
)
from sktime_mcp.tools.job_tools import (
    cancel_job_tool,
    check_job_status_tool,
    cleanup_old_jobs_tool,
    delete_job_tool,
    list_jobs_tool,
)
from sktime_mcp.tools.list_available_data import list_available_data_tool
from sktime_mcp.tools.list_estimators import (
    get_available_tags,
    list_estimators_tool,
)
from sktime_mcp.tools.save_model import save_model_tool

# Configure logging to stderr with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create MCP server instance
server = Server("sktime-mcp")


def sanitize_for_json(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    else:
        return obj


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        Tool(
            name="list_estimators",
            description="Discover sktime estimators by task type and capability tags",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task type filter: forecasting, classification, regression, transformation, clustering",
                    },
                    "tags": {
                        "type": "object",
                        "description": "Filter by capability tags, e.g. {'capability:pred_int': true}",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 50)",
                        "default": 50,
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
            name="fit_predict",
            description="Fit an estimator on a dataset and generate predictions",
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
                    "horizon": {
                        "type": "integer",
                        "description": "Forecast horizon (default: 12)",
                        "default": 12,
                    },
                },
                "required": ["estimator_handle", "dataset"],
            },
        ),
        Tool(
            name="fit_predict_async",
            description="Fit an estimator on a dataset and generate predictions (non-blocking background job)",
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
                    "horizon": {
                        "type": "integer",
                        "description": "Forecast horizon (default: 12)",
                        "default": 12,
                    },
                },
                "required": ["estimator_handle", "dataset"],
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
        Tool(
            name="list_available_data",
            description=(
                "list all data available for use - system demo datasets and active "
                "user-loaded data handles - in a single unified response. "
                "replaces list_datasets and list_data_handles. "
                "Use is_demo=true for demos only, is_demo=false for handles only, "
                "or omit is_demo to get both."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "is_demo": {
                        "type": "boolean",
                        "description": (
                            "optional filter: true returns only system demo datasets, "
                            "false returns only active data handles, "
                            "omit to return both."
                        ),
                    },
                },
            },
        ),
        Tool(
            name="get_available_tags",
            description=(
                "List all queryable capability tags with rich metadata. "
                "Returns tag name, description, expected value type, and which "
                "estimator types the tag applies to. ALWAYS call this before "
                "using tags in list_estimators to ensure correct tag names and values."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="search_estimators",
            description="Search estimators by name or description",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
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
                },
                "required": ["handle"],
            },
        ),
        Tool(
            name="load_data_source",
            description=(
                "Load data from various sources. GUIDELINES: "
                "1. NEVER assume a column is a time index (like 'Month' or 'Date') unless the user explicitly tells you to use it as the index. "
                "2. ALWAYS specify 'target_column' if the user mentions a specific variable to forecast (e.g., 'Use Sales as target'). "
                "3. By default, sktime-mcp treats the first column as target; if the first column is 'Month', this is likely WRONG. "
                "4. For non-standard date formats (e.g., '1-01'), omit 'time_column' to use an integer index unless you can handle the format."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "description": "Data source configuration. Must include 'type' (pandas, sql, file, url).",
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
                        "description": (
                            "Data source configuration. Same format as load_data_source."
                        ),
                    },
                },
                "required": ["config"],
            },
        ),
        Tool(
            name="list_data_sources",
            description="List all available data source types and their descriptions",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="fit_predict_with_data",
            description=(
                "Fit an estimator and generate predictions using custom data. GUIDELINES: "
                "1. BEFORE calling this, check 'list_data_handles' or 'load_data_source' output. "
                "2. If the metadata contains warnings about default target columns or column ambiguity, "
                "STOP and re-load the data with explicit 'target_column' and 'time_column' mapping."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle from instantiate_estimator",
                    },
                    "data_handle": {
                        "type": "string",
                        "description": "Handle from load_data_source",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Forecast horizon (default: 12)",
                        "default": 12,
                    },
                },
                "required": ["estimator_handle", "data_handle"],
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
        Tool(
            name="auto_format_on_load",
            description="Enable/disable automatic formatting when loading data",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether to enable auto-formatting (default: True)",
                        "default": True,
                    },
                },
                "required": ["enabled"],
            },
        ),
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
            description="Cancel a running or pending background job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to cancel",
                    },
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="delete_job",
            description="Delete a job from the job manager",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to delete",
                    },
                },
                "required": ["job_id"],
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
        Tool(
            name="cleanup_old_jobs",
            description="Remove jobs older than specified hours",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_age_hours": {
                        "type": "integer",
                        "description": "Maximum age in hours (default: 24)",
                        "default": 24,
                    },
                },
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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"=== Tool Call: {name} ===")
    logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")

    try:
        if name == "list_estimators":
            result = list_estimators_tool(
                task=arguments.get("task"),
                tags=arguments.get("tags"),
                limit=arguments.get("limit", 50),
            )
        elif name == "describe_estimator":
            result = describe_estimator_tool(arguments["estimator"])
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
        elif name == "fit_predict":
            result = fit_predict_tool(
                arguments["estimator_handle"],
                arguments["dataset"],
                arguments.get("horizon", 12),
            )
            # Sanitize immediately to handle Period objects
            result = sanitize_for_json(result)
        elif name == "validate_pipeline":
            validator = get_composition_validator()
            validation = validator.validate_pipeline(arguments["components"])
            result = validation.to_dict()
        elif name == "list_available_data":
            result = list_available_data_tool(arguments.get("is_demo"))
        elif name == "get_available_tags":
            result = get_available_tags()
        elif name == "search_estimators":
            result = search_estimators_tool(
                arguments["query"],
                arguments.get("limit", 20),
            )
        elif name == "export_code":
            result = export_code_tool(
                arguments["handle"],
                arguments.get("var_name", "model"),
                arguments.get("include_fit_example", False),
            )
        elif name == "load_data_source":
            result = load_data_source_tool(arguments["config"])
        elif name == "load_data_source_async":
            result = load_data_source_async_tool(arguments["config"])
        elif name == "list_data_sources":
            result = list_data_sources_tool()
        elif name == "fit_predict_with_data":
            result = fit_predict_with_data_tool(
                arguments["estimator_handle"],
                arguments["data_handle"],
                arguments.get("horizon", 12),
            )
            # Sanitize immediately to handle Period objects
            result = sanitize_for_json(result)
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
            result = auto_format_on_load_tool(arguments["enabled"])
        elif name == "fit_predict_async":
            result = fit_predict_async_tool(
                arguments["estimator_handle"],
                arguments["dataset"],
                arguments.get("horizon", 12),
            )
        elif name == "check_job_status":
            result = check_job_status_tool(arguments["job_id"])
        elif name == "list_jobs":
            result = list_jobs_tool(
                arguments.get("status"),
                arguments.get("limit", 20),
            )
        elif name == "cancel_job":
            result = cancel_job_tool(arguments["job_id"])
        elif name == "delete_job":
            result = delete_job_tool(arguments["job_id"])
        elif name == "cleanup_old_jobs":
            result = cleanup_old_jobs_tool(arguments.get("max_age_hours", 24))
        elif name == "load_model":
            result = load_model_tool(arguments["path"])
        elif name == "save_model":
            result = save_model_tool(
                arguments["estimator_handle"],
                arguments["path"],
                arguments.get("mlflow_params"),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        logger.info(f"=== Result for {name} ===")

        # Sanitize result for JSON serialization
        sanitized_result = sanitize_for_json(result)
        logger.info(f"{json.dumps(sanitized_result, indent=2, default=str)}")

        return [TextContent(type="text", text=json.dumps(sanitized_result, indent=2, default=str))]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Main entry point."""
    print("Starting sktime-mcp server...")
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
