"""
Code generation tool for sktime MCP.

Generates Python code to recreate estimators and pipelines.
"""

import keyword
from typing import Any

from sktime_mcp.runtime.executor import _get_demo_datasets
from sktime_mcp.runtime.handles import get_handle_manager


def _format_value(value: Any) -> str:
    """Format a parameter value for Python code generation."""
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (list, tuple)):
        if isinstance(value, tuple):
            items = ", ".join(_format_value(v) for v in value)
            return f"({items})" if len(value) != 1 else f"({items},)"
        else:
            items = ", ".join(_format_value(v) for v in value)
            return f"[{items}]"
    elif isinstance(value, dict):
        items = ", ".join(f'"{k}": {_format_value(v)}' for k, v in value.items())
        return f"{{{items}}}"
    elif isinstance(value, bool):
        return str(value)
    elif value is None:
        return "None"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        # For complex objects, try to represent as str
        return repr(value)


def _is_valid_var_name(var_name: str) -> bool:
    """Return True when var_name is a valid non-keyword Python identifier."""
    return isinstance(var_name, str) and var_name.isidentifier() and not keyword.iskeyword(var_name)


def export_code_tool(
    handle: str,
    var_name: str = "model",
    include_fit_example: bool = False,
    dataset: str | None = None,
) -> dict[str, Any]:
    """
    Export an estimator or pipeline as executable Python code.

    Args:
        handle: The handle ID of the estimator/pipeline to export
        var_name: Variable name to use in generated code (default: "model")
        include_fit_example: Whether to include a fit/predict example (default: False)
        dataset: Optional dataset name for the fit example (default: None, falls back to airline)

    Returns:
        Dictionary with:
        - success: bool
        - code: Generated Python code string
        - estimator_name: Name of the estimator/pipeline
        - is_pipeline: Whether this is a pipeline

    Example:
        >>> # First create an estimator
        >>> result = instantiate_tool("ARIMA", {"order": [1, 1, 1]})
        >>> handle = result["handle"]
        >>>
        >>> # Export as code
        >>> export_code_tool(handle, var_name="arima_model")
        {
            "success": True,
            "code": "from sktime.forecasting.arima import ARIMA\\n\\narima_model = ARIMA(order=[1, 1, 1])",
            "estimator_name": "ARIMA",
            "is_pipeline": False
        }
    """
    handle_manager = get_handle_manager()

    # Get handle info
    try:
        handle_info = handle_manager.get_info(handle)
    except KeyError:
        return {"success": False, "error": f"Handle not found: {handle}"}

    if not _is_valid_var_name(var_name):
        return {
            "success": False,
            "error": "var_name must be a valid Python identifier and not a keyword.",
        }

    estimator_name = handle_info.estimator_name
    params = handle_info.params

    spec = params.get("spec")
    if not spec:
        return {"success": False, "error": "No craft spec found in handle parameters."}

    is_pipeline = "*" in spec or "Pipeline" in spec or "[" in spec
    code = f"from sktime.registry import craft\n\n{var_name} = craft({_format_value(spec)})"

    # Optionally add fit/predict example
    if include_fit_example:
        # Priority: explicit argument > dataset used during fit > "airline" fallback
        effective_dataset = dataset or handle_info.metadata.get("training_dataset") or "airline"
        # Resolve the dataset loader from discovered demo datasets
        demo_datasets = _get_demo_datasets()
        if effective_dataset in demo_datasets:
            module_path = demo_datasets[effective_dataset]
            module_parts = module_path.rsplit(".", 1)
            loader_module = module_parts[0]
            loader_func = module_parts[1]
        else:
            loader_module = "sktime.datasets"
            loader_func = "load_airline"

        example_code = f"""

# Example usage:
# Load data
from {loader_module} import {loader_func}
y = {loader_func}()

# Fit the model
{var_name}.fit(y)

# Make predictions
fh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 12-step ahead forecast
predictions = {var_name}.predict(fh=fh)
print(predictions)
"""
        code += example_code

    return {
        "success": True,
        "code": code,
        "estimator_name": estimator_name,
        "is_pipeline": is_pipeline,
        "handle": handle,
    }
