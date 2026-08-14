"""
Code generation tool for sktime MCP.

Generates Python code to recreate estimators and pipelines.
"""

import json
import keyword
from typing import Any

from sktime_mcp.runtime.executor import _get_demo_datasets
from sktime_mcp.runtime.handles import get_handle_manager


def _format_value(value: Any) -> str:
    """Format a parameter value for Python code generation."""
    if isinstance(value, str):
        # json.dumps escapes embedded quotes/backslashes; its output is also
        # a valid Python string literal
        return json.dumps(value)
    elif isinstance(value, (list, tuple)):
        if isinstance(value, tuple):
            items = ", ".join(_format_value(v) for v in value)
            return f"({items})" if len(value) != 1 else f"({items},)"
        else:
            items = ", ".join(_format_value(v) for v in value)
            return f"[{items}]"
    elif isinstance(value, dict):
        items = ", ".join(f"{_format_value(k)}: {_format_value(v)}" for k, v in value.items())
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


def _loader_for(dataset: str, demo_datasets: dict) -> tuple[str, str]:
    """Return (module, func) for a demo dataset name, defaulting to load_airline."""
    if dataset in demo_datasets:
        module_path = demo_datasets[dataset]
        module, func = module_path.rsplit(".", 1)
        return module, func
    return "sktime.datasets", "load_airline"


def _fit_example(
    var_name: str,
    obj_type: str,
    dataset: str | None,
    handle_info: Any,
    demo_datasets: dict,
) -> str:
    """Build a runnable fit/predict example matching the estimator's scitype.

    A forecaster-shaped example (`fit(y)` / `predict(fh)`) is wrong for
    transformers, splitters, and classifiers, which raise AttributeError when
    the generated code runs (BUG-03).
    """
    if obj_type in ("classifier", "regressor"):
        # Panel X + label/target y — use a classification demo dataset.
        ds = dataset or "arrow_head"
        module, func = _loader_for(ds, demo_datasets)
        verb = "class" if obj_type == "classifier" else "value"
        return f"""

# Example usage ({obj_type}):
from {module} import {func}
X, y = {func}(return_X_y=True)

{var_name}.fit(X, y)
predictions = {var_name}.predict(X)  # predicted {verb} per instance
print(predictions)
"""

    if obj_type == "transformer":
        ds = dataset or handle_info.metadata.get("training_dataset") or "airline"
        module, func = _loader_for(ds, demo_datasets)
        return f"""

# Example usage (transformer):
from {module} import {func}
y = {func}()

y_transformed = {var_name}.fit_transform(y)
print(y_transformed)
"""

    if obj_type == "splitter":
        ds = dataset or "airline"
        module, func = _loader_for(ds, demo_datasets)
        return f"""

# Example usage (splitter):
from {module} import {func}
y = {func}()

for train_idx, test_idx in {var_name}.split(y):
    print("train:", train_idx, "test:", test_idx)
"""

    # Default: forecaster.
    ds = dataset or handle_info.metadata.get("training_dataset") or "airline"
    module, func = _loader_for(ds, demo_datasets)
    return f"""

# Example usage (forecaster):
from {module} import {func}
y = {func}()

{var_name}.fit(y)
fh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 12-step ahead forecast
predictions = {var_name}.predict(fh=fh)
print(predictions)
"""


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
        return {"success": False, "error": handle_manager.describe_missing(handle)}

    if not _is_valid_var_name(var_name):
        return {
            "success": False,
            "error": "var_name must be a valid Python identifier and not a keyword.",
        }

    estimator_name = handle_info.estimator_name
    params = handle_info.params
    spec = params.get("spec")

    instance = handle_manager.get_instance(handle)
    get_tag = getattr(instance, "get_class_tag", None)
    obj_type = get_tag("object_type", "") if callable(get_tag) else ""

    # is_pipeline from the instance, not a spec substring — "[" in spec
    # false-positived on any list argument (BUG-04).
    is_pipeline = bool(spec and "*" in spec) or hasattr(instance, "steps")

    if spec:
        code = f"from sktime.registry import craft\n\n{var_name} = craft({_format_value(spec)})"
    elif handle_info.metadata.get("source") == "loaded" and handle_info.metadata.get("path"):
        # Loaded models carry no craft spec; emit a load_model snippet instead of
        # failing with "No craft spec found" (NB-17).
        model_path = handle_info.metadata["path"]
        code = (
            "from sktime.utils.mlflow_sktime import load_model\n\n"
            f"{var_name} = load_model({_format_value(model_path)})"
        )
    else:
        return {"success": False, "error": "No craft spec found in handle parameters."}

    # Optionally add a scitype-appropriate fit example (BUG-03).
    if include_fit_example:
        demo_datasets = _get_demo_datasets()
        if dataset is not None and dataset not in demo_datasets:
            return {
                "success": False,
                "error": (
                    f"Unknown dataset '{dataset}' for the fit example. Use a demo dataset "
                    "name (see list_available_data) or omit dataset to use a default."
                ),
            }
        example = _fit_example(var_name, obj_type, dataset, handle_info, demo_datasets)
        code += example

    return {
        "success": True,
        "code": code,
        "estimator_name": estimator_name,
        "is_pipeline": is_pipeline,
        "handle": handle,
    }
