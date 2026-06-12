"""
Safe crafting utilities for sktime MCP.

Wraps sktime.registry.craft with AST validation so LLM-supplied specs
cannot execute arbitrary Python code.
"""

from __future__ import annotations

import ast
from typing import Any

from sktime.registry import all_estimators
from sktime.registry._craft import _extract_class_names


class SpecValidationError(ValueError):
    """Raised when a craft spec fails security or structure checks."""


_ALLOWED_NAMES = frozenset({"True", "False", "None"})
_ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Return,
    ast.Assign,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.keyword,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
)


def _registry_names() -> frozenset[str]:
    return frozenset(dict(all_estimators()).keys())


def validate_spec_ast(spec: str) -> None:
    """
    Validate that ``spec`` only uses registry classes and safe literals.

    Raises
    ------
    SpecValidationError
        If the spec contains disallowed syntax or names.
    """
    if not spec or not spec.strip():
        raise SpecValidationError("spec must be a non-empty string")

    try:
        tree = ast.parse(spec, mode="exec")
    except SyntaxError as exc:
        raise SpecValidationError(f"Invalid Python syntax in spec: {exc}") from exc

    registry = _registry_names()
    assigned: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise SpecValidationError(
                    "Only simple assignments to a single variable are allowed"
                )
            assigned.add(node.targets[0].id)
            _validate_expr(node.value, registry, assigned)
        elif isinstance(node, (ast.Return, ast.Expr)):
            _validate_expr(node.value, registry, assigned)
        else:
            raise SpecValidationError(
                f"Disallowed statement type: {type(node).__name__}. "
                "Use a single expression or assignments ending with 'return'."
            )


def _validate_expr(
    node: ast.AST,
    registry: frozenset[str],
    assigned: set[str],
) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise SpecValidationError(f"Disallowed expression type: {type(node).__name__}")

    if isinstance(node, ast.Name):
        if node.id not in registry and node.id not in assigned and node.id not in _ALLOWED_NAMES:
            raise SpecValidationError(
                f"Disallowed name '{node.id}'. "
                "Only sktime registry class names, local variables, and "
                "True/False/None are permitted."
            )
        return

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id not in registry:
                raise SpecValidationError(
                    f"Call target '{node.func.id}' is not a registered sktime class"
                )
        else:
            raise SpecValidationError("Only direct class constructor calls are allowed")
        for arg in node.args:
            _validate_expr(arg, registry, assigned)
        for kw in node.keywords:
            if kw.arg is None:
                raise SpecValidationError("**kwargs unpacking is not allowed in specs")
            _validate_expr(kw.value, registry, assigned)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise SpecValidationError("Only unary +/- on numeric literals are allowed")
        _validate_expr(node.operand, registry, assigned)
        return

    if isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _validate_expr(elt, registry, assigned)
        return

    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=False):
            if key is not None:
                _validate_expr(key, registry, assigned)
            _validate_expr(value, registry, assigned)
        return

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool, type(None))):
            return
        raise SpecValidationError(f"Disallowed constant type: {type(node.value).__name__}")


def safe_craft(spec: str) -> Any:
    """
    Build a sktime object from a validated craft spec.

      Parameters
      ----------
      spec : str
          sktime craft specification (expression or assignments + return).

      Returns
      -------
      object
          Constructed sktime estimator, transformer, or pipeline.
    """
    validate_spec_ast(spec)
    register = dict(all_estimators())
    namespace = dict(register)
    namespace["__builtins__"] = {}

    try:
        return eval(spec, namespace, namespace)  # noqa: S307
    except Exception:
        from textwrap import indent

        spec_fun = indent(spec, "    ")
        spec_fun = "def build_obj():\n" + spec_fun
        exec(spec_fun, namespace, namespace)  # noqa: S102
        return eval("build_obj()", namespace, namespace)  # noqa: S307


def extract_class_names(spec: str) -> list[str]:
    """Return registry class names referenced in a spec string."""
    return _extract_class_names(spec)


def get_spec_deps(spec: str) -> list[str]:
    """Return PEP 440 dependency strings required to craft ``spec``."""
    from sktime.registry import deps

    validate_spec_ast(spec)
    return deps(spec)


def get_spec_imports(spec: str) -> str:
    """Return import statements required to craft ``spec``."""
    from sktime.registry import imports

    validate_spec_ast(spec)
    return imports(spec)
