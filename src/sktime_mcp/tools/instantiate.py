"""
Instantiation tools for sktime MCP.

Builds estimators and pipelines via sktime's craft registry, with sandboxed
evaluation, dry-run validation, and a unified instantiate_estimator API.
"""

from __future__ import annotations

from typing import Any

from sktime_mcp.composition.validator import get_composition_validator
from sktime_mcp.registry.interface import get_registry
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.craft_utils import (
    SpecValidationError,
    extract_class_names,
    get_spec_deps,
    get_spec_imports,
    safe_craft,
    validate_spec_ast,
)


def _format_value(value: Any) -> str:
    """Format a Python literal for inclusion in a craft spec."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, tuple):
        inner = ", ".join(_format_value(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{repr(k)}: {_format_value(v)}" for k, v in value.items())
        return "{" + items + "}"
    raise TypeError(f"Unsupported value type for craft spec: {type(value).__name__}")


def _format_kwargs(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_format_value(value)}" for key, value in params.items())


def _is_safe_value(value: Any) -> bool:
    """Recursively check if a value is a safe, serializable type."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_safe_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_safe_value(v) for k, v in value.items())
    return False


def _validate_params(
    params: Any,
    estimator_name: str | None = None,
) -> dict[str, Any]:
    """Validate params for type safety and optionally check keys."""
    warnings: list[str] = []

    if params is None:
        return {"valid": True, "warnings": warnings}

    if not isinstance(params, dict):
        return {
            "valid": False,
            "error": (
                f"'params' must be a dictionary, got {type(params).__name__}. "
                f'Example: {{"order": [1, 1, 1], "suppress_warnings": true}}'
            ),
            "warnings": warnings,
        }

    for key, value in params.items():
        if not isinstance(key, str):
            return {
                "valid": False,
                "error": (
                    f"Parameter keys must be strings, got {type(key).__name__} for key: {key!r}"
                ),
                "warnings": warnings,
            }
        if not _is_safe_value(value):
            return {
                "valid": False,
                "error": (
                    f"Unsupported type for parameter '{key}': "
                    f"{type(value).__name__}. "
                    f"Only primitive types (str, int, float, bool, list, "
                    f"tuple, dict, None) are allowed."
                ),
                "warnings": warnings,
            }

    if estimator_name and params:
        registry = get_registry()
        node = registry.get_estimator_by_name(estimator_name)
        if node is not None and node.hyperparameters:
            known_keys = set(node.hyperparameters.keys())
            unknown_keys = set(params.keys()) - known_keys
            if unknown_keys:
                warnings.append(
                    f"Unknown parameter(s) for {estimator_name}: "
                    f"{sorted(unknown_keys)}. "
                    f"Known parameters: {sorted(known_keys)}"
                )

    return {"valid": True, "warnings": warnings}


def _composition_warnings(spec: str) -> list[str]:
    """Return composition warnings for a linear component chain in ``spec``."""
    class_names = extract_class_names(spec)
    if len(class_names) < 2:
        return []

    validator = get_composition_validator()
    result = validator.validate_pipeline(class_names)
    return list(result.warnings)


def _build_estimator_spec(estimator: str, params: dict[str, Any] | None = None) -> str:
    kwargs = _format_kwargs(params or {})
    return f"{estimator}({kwargs})" if kwargs else f"{estimator}()"


def _build_pipeline_spec(
    components: list[str],
    params_list: list[dict[str, Any]] | None = None,
) -> str:
    """Build a craft spec string for a component list."""
    if not components:
        raise ValueError("Pipeline cannot be empty")

    params_list = params_list or [{}] * len(components)
    if len(components) == 1:
        return _build_estimator_spec(components[0], params_list[0])

    registry = get_registry()
    lines: list[str] = []
    for index, (name, params) in enumerate(zip(components, params_list, strict=False)):
        kwargs = _format_kwargs(params) if params else ""
        lines.append(f"c{index} = {name}({kwargs})" if kwargs else f"c{index} = {name}()")

    tasks = [registry.get_estimator_by_name(name).task for name in components]
    if tasks is None or any(task is None for task in tasks):
        missing = [name for name in components if registry.get_estimator_by_name(name) is None]
        raise ValueError(f"Unknown estimator(s): {missing}")

    all_transformers_except_last = all(task == "transformation" for task in tasks[:-1])
    final_task = tasks[-1]

    if all_transformers_except_last and final_task == "forecasting":
        if len(components) == 2:
            lines.append(
                'return TransformedTargetForecaster(steps=[("transformer", c0), ("forecaster", c1)])'
            )
        else:
            transformer_steps = ", ".join(f'("step_{i}", c{i})' for i in range(len(components) - 1))
            lines.append(f"tp = TransformerPipeline(steps=[{transformer_steps}])")
            lines.append(
                f'return TransformedTargetForecaster(steps=[("transformers", tp), ("forecaster", c{len(components) - 1})])'
            )
    elif all_transformers_except_last and final_task in ("classification", "regression"):
        steps = ", ".join(f'("step_{i}", c{i})' for i in range(len(components)))
        lines.append(f"return Pipeline(steps=[{steps}])")
    elif all(task == "transformation" for task in tasks):
        steps = ", ".join(f'("step_{i}", c{i})' for i in range(len(components)))
        lines.append(f"return TransformerPipeline(steps=[{steps}])")
    else:
        raise ValueError(
            "Unsupported pipeline composition. "
            "Supported: transformers→forecaster, transformers→classifier/regressor, "
            "or transformer-only chains."
        )

    return "\n".join(lines)


def instantiate_tool(
    spec: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Instantiate an estimator or pipeline from a sktime craft spec."""
    warnings: list[str] = []

    try:
        validate_spec_ast(spec)
    except SpecValidationError as exc:
        return {"success": False, "error": str(exc)}

    try:
        deps = get_spec_deps(spec)
        imports = get_spec_imports(spec)
    except Exception as exc:
        deps = []
        imports = ""
        warnings.append(f"Could not resolve dependencies: {exc}")

    warnings.extend(_composition_warnings(spec))

    if dry_run:
        return {
            "success": True,
            "valid": True,
            "spec": spec,
            "deps": deps,
            "imports": imports,
            "warnings": warnings,
        }

    try:
        instance = safe_craft(spec)
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "hint": (
                "Check class names against query_registry, parameter types, "
                "and pipeline structure. Use dry_run=true to validate first."
            ),
        }

    canonical_spec = str(instance)
    estimator_name = type(instance).__name__
    handle_manager = get_handle_manager()
    handle_id = handle_manager.create_handle(
        estimator_name=estimator_name,
        instance=instance,
        params={"spec": canonical_spec},
        metadata={"spec": canonical_spec},
    )

    result: dict[str, Any] = {
        "success": True,
        "handle": handle_id,
        "estimator": estimator_name,
        "spec": canonical_spec,
        "deps": deps,
        "imports": imports,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def set_params_tool(handle: str, params: dict[str, Any]) -> dict[str, Any]:
    """Update hyperparameters on an existing unfitted handle."""
    handle_manager = get_handle_manager()

    if not handle_manager.exists(handle):
        return {"success": False, "error": f"Handle not found: {handle}"}

    if handle_manager.is_fitted(handle):
        return {
            "success": False,
            "error": (
                f"Cannot set_params on fitted handle '{handle}'. "
                "Instantiate a new estimator instead."
            ),
        }

    validation = _validate_params(params)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"]}

    try:
        instance = handle_manager.get_instance(handle)
        instance.set_params(**params)
        canonical_spec = str(instance)
        info = handle_manager.get_info(handle)
        info.params["spec"] = canonical_spec
        info.metadata["spec"] = canonical_spec
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    result: dict[str, Any] = {
        "success": True,
        "handle": handle,
        "spec": canonical_spec,
        "estimator": type(instance).__name__,
    }
    if validation["warnings"]:
        result["warnings"] = validation["warnings"]
    return result


def _instantiate_single_craft(
    estimator: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = _validate_params(params, estimator_name=estimator)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"]}

    try:
        spec = _build_estimator_spec(estimator, params)
    except TypeError as exc:
        return {"success": False, "error": str(exc)}

    result = instantiate_tool(spec, dry_run=False)
    if validation["warnings"] and result.get("success"):
        result.setdefault("warnings", []).extend(validation["warnings"])
    if result.get("success"):
        result["params"] = params or {}
    return result


def _instantiate_pipeline_craft(
    components: list[str],
    params_list: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    all_warnings: list[str] = []

    if params_list is not None:
        if not isinstance(params_list, list):
            return {
                "success": False,
                "error": (
                    f"'params_list' must be a list of dictionaries, "
                    f"got {type(params_list).__name__}"
                ),
            }
        for index, comp_params in enumerate(params_list):
            comp_name = components[index] if index < len(components) else None
            validation = _validate_params(comp_params, estimator_name=comp_name)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": (
                        f"Validation failed for component {index} "
                        f"({comp_name or 'unknown'}): {validation['error']}"
                    ),
                }
            all_warnings.extend(validation["warnings"])
    elif len(components) == 1:
        validation = _validate_params(None, estimator_name=components[0])
        all_warnings.extend(validation["warnings"])

    validator = get_composition_validator()
    validation = validator.validate_pipeline(components)
    if not validation.valid:
        return {
            "success": False,
            "error": "Invalid pipeline composition",
            "validation_errors": validation.errors,
            "suggestions": validation.suggestions,
        }

    try:
        spec = _build_pipeline_spec(components, params_list)
    except (TypeError, ValueError) as exc:
        return {"success": False, "error": str(exc)}

    dry_result = instantiate_tool(spec, dry_run=True)
    if not dry_result.get("success"):
        return dry_result

    result = instantiate_tool(spec, dry_run=False)
    all_warnings.extend(dry_result.get("warnings", []))
    if all_warnings and result.get("success"):
        result["warnings"] = all_warnings
    if result.get("success"):
        result["pipeline"] = " → ".join(components)
        result["components"] = components
        result["params_list"] = params_list or [{}] * len(components)
    return result


def instantiate_estimator_tool(
    estimator: str | None = None,
    params: dict[str, Any] | None = None,
    components: list[str] | None = None,
    params_list: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create an estimator or pipeline via craft and return a handle."""
    if estimator is not None and components is not None:
        return {
            "success": False,
            "error": ("Provide either 'estimator' (single) or 'components' (pipeline), not both."),
        }

    if estimator is None and components is None:
        return {
            "success": False,
            "error": (
                "Either 'estimator' (single estimator name) or 'components' "
                "(list of estimator names for a pipeline) is required."
            ),
        }

    if estimator is not None:
        return _instantiate_single_craft(estimator, params)

    assert components is not None

    if params_list is not None and not isinstance(params_list, list):
        return {
            "success": False,
            "error": (
                f"'params_list' must be a list of dictionaries, got {type(params_list).__name__}"
            ),
        }

    if len(components) == 1:
        single_params = params_list[0] if params_list else None
        return _instantiate_single_craft(components[0], single_params)

    return _instantiate_pipeline_craft(components, params_list)


def instantiate_pipeline_tool(
    components: list[str],
    params_list: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a pipeline via craft (legacy positional API)."""
    return instantiate_estimator_tool(components=components, params_list=params_list)


def release_handle_tool(handle: str) -> dict[str, Any]:
    """Release an estimator handle and free resources."""
    handle_manager = get_handle_manager()
    released = handle_manager.release_handle(handle)
    return {
        "success": released,
        "handle": handle,
        "message": "Handle released" if released else "Handle not found",
    }


def list_handles_tool() -> dict[str, Any]:
    """List all active estimator handles."""
    handle_manager = get_handle_manager()
    handles = handle_manager.list_handles()
    return {
        "success": True,
        "handles": handles,
        "count": len(handles),
    }


def load_model_tool(path: str) -> dict[str, Any]:
    """Load a saved model from disk and register a handle."""
    try:
        from sktime.utils.mlflow_sktime import load_model
    except ImportError:
        return {
            "success": False,
            "error": (
                "The 'mlflow' package is required to load saved models. "
                "Please install it with: pip install sktime-mcp[mlflow]"
            ),
        }

    try:
        instance = load_model(path)
        estimator_name = type(instance).__name__
        canonical_spec = str(instance)
        handle_manager = get_handle_manager()
        handle_id = handle_manager.create_handle(
            estimator_name=estimator_name,
            instance=instance,
            params={"spec": canonical_spec},
            metadata={"source": "loaded", "path": path, "spec": canonical_spec},
        )
        handle_manager.mark_fitted(handle_id)
        return {
            "success": True,
            "handle": handle_id,
            "estimator": estimator_name,
            "spec": canonical_spec,
            "path": path,
            "message": f"Successfully loaded {estimator_name}",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to load model: {exc}",
            "path": path,
        }
