"""
Code generation tool for sktime MCP.

Generates Python code to recreate estimators and pipelines.
"""

import ast
import inspect
import json
import keyword
import re
import textwrap
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


def _loader_returns_pair(module: str, func: str) -> bool:
    """Return True when ``module.func()`` returns a ``(y, X)`` 2-tuple.

    Decided statically from the loader's ``return`` statements so that
    export_code never loads a dataset (some demo loaders download data).
    A loader that cannot be inspected is assumed to return a single object.
    """
    try:
        loader = getattr(__import__(module, fromlist=[func]), func)
        tree = ast.parse(textwrap.dedent(inspect.getsource(loader)))
    except Exception:
        return False
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef):
        return False

    arities: set[int] = set()

    def _visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return) and child.value is not None:
                value = child.value
                arities.add(len(value.elts) if isinstance(value, ast.Tuple) else 1)
            _visit(child)

    _visit(fn)
    return arities == {2}


def _load_series_lines(module: str, func: str) -> str:
    """Import + load lines for a demo dataset, unpacking ``(y, X)`` loaders."""
    if _loader_returns_pair(module, func):
        return f"from {module} import {func}\ny, X = {func}()  # X: exogenous features"
    return f"from {module} import {func}\ny = {func}()"


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
{_load_series_lines(module, func)}

y_transformed = {var_name}.fit_transform(y)
print(y_transformed)
"""

    if obj_type == "splitter":
        ds = dataset or "airline"
        module, func = _loader_for(ds, demo_datasets)
        return f"""

# Example usage (splitter):
{_load_series_lines(module, func)}

for train_idx, test_idx in {var_name}.split(y):
    print("train:", train_idx, "test:", test_idx)
"""

    # Default: forecaster.
    ds = dataset or handle_info.metadata.get("training_dataset") or "airline"
    module, func = _loader_for(ds, demo_datasets)
    if _loader_returns_pair(module, func):
        # Exogenous X must also cover the forecast horizon, so hold out the
        # tail of the dataset instead of forecasting past its end (F-28).
        return f"""

# Example usage (forecaster with exogenous X):
from {module} import {func}
from sktime.split import temporal_train_test_split
y, X = {func}()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)

{var_name}.fit(y_train, X=X_train, fh=y_test.index)
predictions = {var_name}.predict(X=X_test)
print(predictions)
"""
    return f"""

# Example usage (forecaster):
from {module} import {func}
y = {func}()

{var_name}.fit(y)
fh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 12-step ahead forecast
predictions = {var_name}.predict(fh=fh)
print(predictions)
"""


# Names the server injects into craft's namespace (executor.instantiate) and
# the import line each one needs in standalone code.
_NAMESPACE_IMPORTS = {
    "np": "import numpy as np",
    "numpy": "import numpy",
    "pd": "import pandas as pd",
    "pandas": "import pandas",
}
_NAMESPACE_USE = re.compile(r"\b(np|numpy|pd|pandas)\.")


def _construction_code(var_name: str, spec: str) -> tuple[str, list[str]]:
    """Return ``(code, warnings)`` that rebuilds ``spec`` outside the server.

    ``craft(spec)`` only works for specs that mention ``np.``/``pd.`` because
    the server injects those modules into craft's registry (F-28). For such
    specs, when the spec is a single expression whose class names sktime can
    resolve, emit the module and class imports plus the spec itself as plain
    Python. Otherwise fall back to craft and warn that the spec needs the
    namespace.
    """
    craft_code = f"from sktime.registry import craft\n\n{var_name} = craft({_format_value(spec)})"
    names = sorted(set(_NAMESPACE_USE.findall(spec)))
    if not names:
        return craft_code, []

    def _fallback(reason: str) -> tuple[str, list[str]]:
        return craft_code, [
            f"The spec references {', '.join(names)}, which craft() resolves only inside "
            f"the server; {reason}, so the generated craft() call needs those names "
            "registered in craft's namespace before it runs."
        ]

    try:
        ast.parse(spec, mode="eval")
    except SyntaxError:
        return _fallback("the spec is a multi-statement block")

    try:
        class_imports = _class_imports(spec)
    except Exception as exc:
        return _fallback(f"its class imports could not be resolved ({exc})")

    import_lines = [_NAMESPACE_IMPORTS[name] for name in names] + class_imports
    return "\n".join(import_lines) + f"\n\n{var_name} = {spec}", []


def _class_imports(spec: str) -> list[str]:
    """Import lines for every bare name in ``spec`` that craft would resolve.

    Only ``ast.Name`` nodes are looked up (``pd.Timedelta`` is an attribute of
    ``pd``, not a class to import), which is what ``eval`` inside craft
    resolves against its registry. Unknown names raise ``KeyError``.
    """
    import builtins

    from sktime.registry import all_estimators
    from sktime.registry._craft import _get_public_import

    register = dict(all_estimators())
    try:
        from sktime.registry._craft import _all_sklearn_estimators

        register = {**dict(_all_sklearn_estimators()), **register}
    except Exception:  # pragma: no cover - sklearn helper is private
        pass

    used = {
        node.id
        for node in ast.walk(ast.parse(spec, mode="eval"))
        if isinstance(node, ast.Name)
        and node.id not in _NAMESPACE_IMPORTS
        and not hasattr(builtins, node.id)
    }
    lines = []
    for name in sorted(used):
        if name not in register:
            raise KeyError(f"{name!r} is not an sktime/sklearn estimator")
        lines.append(f"from {_get_public_import(register[name].__module__)} import {name}")
    return lines


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

    warnings: list[str] = []
    if spec:
        code, warnings = _construction_code(var_name, spec)
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

    result = {
        "success": True,
        "code": code,
        "estimator_name": estimator_name,
        "is_pipeline": is_pipeline,
        "handle": handle,
    }
    if warnings:
        result["warnings"] = warnings
    return result
