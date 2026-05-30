"""
Code generation tool for sktime MCP.

Generates Python code to recreate estimators and pipelines.
"""

import keyword
from typing import Any

from sktime_mcp.registry.interface import get_registry
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


def _get_estimator_module(estimator_name: str) -> str | None:
    """Get the module path for an estimator."""
    registry = get_registry()
    node = registry.get_estimator_by_name(estimator_name)
    if node and node.class_ref:
        return node.class_ref.__module__
    return None


def _is_valid_var_name(var_name: str) -> bool:
    """Return True when var_name is a valid non-keyword Python identifier."""
    return isinstance(var_name, str) and var_name.isidentifier() and not keyword.iskeyword(var_name)


def _generate_single_estimator_code(
    estimator_name: str, params: dict[str, Any], var_name: str = "model"
) -> dict[str, Any]:
    """Generate Python code for a single estimator."""
    module = _get_estimator_module(estimator_name)

    if not module:
        return {"success": False, "error": f"Could not find module for estimator: {estimator_name}"}

    # Build import statement
    imports = [f"from {module} import {estimator_name}"]

    # Build instantiation code
    if params:
        param_strs = []
        for key, value in params.items():
            param_strs.append(f"{key}={_format_value(value)}")
        params_str = ", ".join(param_strs)
        instantiation = f"{var_name} = {estimator_name}({params_str})"
    else:
        instantiation = f"{var_name} = {estimator_name}()"

    # Combine into full code
    code_lines = imports + ["", instantiation]
    code = "\n".join(code_lines)

    return {
        "success": True,
        "code": code,
        "imports": imports,
        "instantiation": instantiation,
    }


def _generate_pipeline_code(
    components: list[str], params_list: list[dict[str, Any]], var_name: str = "pipeline"
) -> dict[str, Any]:
    """Generate Python code for a pipeline."""
    registry = get_registry()

    # Collect all imports needed
    imports = set()

    # Get task types for components
    component_tasks = []
    for comp_name in components:
        node = registry.get_estimator_by_name(comp_name)
        if not node:
            return {"success": False, "error": f"Unknown estimator in pipeline: {comp_name}"}
        component_tasks.append(node.task)
        module = node.class_ref.__module__
        imports.add(f"from {module} import {comp_name}")

    # Determine pipeline type
    all_transformers_except_last = all(task == "transformation" for task in component_tasks[:-1])
    final_task = component_tasks[-1]

    # Build component instantiations
    component_code_lines = []
    for i, (comp_name, params) in enumerate(zip(components, params_list, strict=False)):
        var = f"step_{i}"
        if params:
            param_strs = []
            for key, value in params.items():
                param_strs.append(f"{key}={_format_value(value)}")
            params_str = ", ".join(param_strs)
            component_code_lines.append(f"{var} = {comp_name}({params_str})")
        else:
            component_code_lines.append(f"{var} = {comp_name}()")

    # Build pipeline instantiation based on composition type
    if len(components) == 1:
        # Single component, no pipeline needed
        pipeline_code = f"{var_name} = step_0"
    elif all_transformers_except_last and final_task == "forecasting":
        # Use TransformedTargetForecaster
        imports.add("from sktime.forecasting.compose import TransformedTargetForecaster")

        if len(components) == 2:
            pipeline_code = f"""{var_name} = TransformedTargetForecaster([
    ("transformer", step_0),
    ("forecaster", step_1),
])"""
        else:
            # Multiple transformers - chain them
            imports.add("from sktime.transformations.compose import TransformerPipeline")
            transformer_steps = ", ".join(
                f'("step_{i}", step_{i})' for i in range(len(components) - 1)
            )
            pipeline_code = f"""transformer_chain = TransformerPipeline([
    {transformer_steps}
])
{var_name} = TransformedTargetForecaster([
    ("transformers", transformer_chain),
    ("forecaster", step_{len(components) - 1}),
])"""

    elif all_transformers_except_last and final_task in ("classification", "regression"):
        # Use sklearn-style Pipeline
        imports.add("from sktime.pipeline import Pipeline")
        steps = ", ".join(f'("step_{i}", step_{i})' for i in range(len(components)))
        pipeline_code = f"""{var_name} = Pipeline([
    {steps}
])"""

    elif all(task == "transformation" for task in component_tasks):
        # All transformers - use TransformerPipeline
        imports.add("from sktime.transformations.compose import TransformerPipeline")
        steps = ", ".join(f'("step_{i}", step_{i})' for i in range(len(components)))
        pipeline_code = f"""{var_name} = TransformerPipeline([
    {steps}
])"""

    else:
        return {"success": False, "error": "Unsupported pipeline composition type"}

    # Combine all code
    code_lines = sorted(imports) + [""] + component_code_lines + [""] + [pipeline_code]
    code = "\n".join(code_lines)

    return {
        "success": True,
        "code": code,
        "imports": sorted(imports),
        "pipeline_type": (
            "TransformedTargetForecaster"
            if "TransformedTargetForecaster" in str(imports)
            else "Pipeline"
        ),
    }


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
        >>> result = instantiate_estimator_tool("ARIMA", {"order": [1, 1, 1]})
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

    # Check if this is a pipeline (has components in metadata)
    is_pipeline = "components" in params

    if is_pipeline:
        components = params["components"]
        params_list = params.get("params_list", [{}] * len(components))
        result = _generate_pipeline_code(components, params_list, var_name)
    else:
        result = _generate_single_estimator_code(estimator_name, params, var_name)

    if not result["success"]:
        return result

    code = result["code"]

    # Optionally add fit/predict example
    if include_fit_example:
        # Resolve the dataset loader from demo datasets
        _demo_datasets = _get_demo_datasets()
        if dataset and dataset in _demo_datasets:
            module_path = _demo_datasets[dataset]
            module_parts = module_path.rsplit(".", 1)
            loader_module = module_parts[0]
            loader_func = module_parts[1]
        else:
            # Default to airline for backward compatibility
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
