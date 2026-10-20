"""
Registry Interface for sktime MCP.

Thin adapter over sktime's estimator registry.
Delegates to ``sktime.registry`` functions directly.
"""

import difflib
import importlib
import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ``package.module.ClassName``: identifiers joined by dots, nothing else. Used
# to decide whether a name may be resolved by *importing* it; anything that
# does not match (calls, brackets, operators, ...) is never evaluated.
_DOTTED_PATH = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


@dataclass
class EstimatorNode:
    """
    Represents a single estimator in the sktime registry.

    This is the semantic representation of an estimator that gets
    exposed to the LLM through the MCP.

    Attributes:
        name: The class name of the estimator (e.g., "ARIMA")
        task: The scitype (e.g., "forecaster", "transformer", "classifier")
        class_ref: Reference to the actual Python class
        module: Full module path to the estimator
        tags: Dictionary of capability tags
        parameters: Parameter names with their defaults
        docstring: The estimator's docstring
    """

    name: str
    task: str
    class_ref: type
    module: str
    tags: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    docstring: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "task": self.task,
            "module": self.module,
            "tags": self.tags,
            "parameters": self.parameters,
            "hyperparameters": self.parameters,  # keep for backward compatibility
            "docstring": (self.docstring[:500] if self.docstring else None),
        }

    def to_summary(self) -> dict[str, Any]:
        """Return a minimal summary for list operations."""
        return {
            "name": self.name,
            "task": self.task,
            "module": self.module,
            "tags": self.tags,
        }


class RegistryInterface:
    """
    Adapter over sktime's estimator registry.

    Delegates to ``sktime.registry.all_estimators`` for discovery and
    filtering, ``sktime.registry.craft`` for name-based lookups, and
    uses public class methods (``get_class_tags``,
    ``get_param_names``, ``get_param_defaults``) for metadata extraction.
    """

    def __init__(self):
        """Initialize the registry interface."""
        self._index: dict[str, type] | None = None
        try:
            from sktime.registry import all_estimators  # noqa: F401
        except ImportError as e:
            logger.error(f"Failed to import sktime registry: {e}")
            raise RuntimeError("sktime must be installed to use sktime-mcp") from e

    @staticmethod
    def _create_node(name: str, cls: type) -> EstimatorNode:
        """Create an EstimatorNode from sktime estimator."""
        # Scitype from the class tag (may be a str or list of str)
        scitype = "estimator"
        try:
            raw = cls.get_class_tag("object_type", "estimator")
            if isinstance(raw, list):  # noqa: SIM108
                # Prefer the shortest / most general type, e.g. "metric" over
                # "metric_forecasting".  Fallback to first element.
                scitype = min(raw, key=len) if raw else "estimator"
            else:
                scitype = raw
        except Exception:
            pass

        tags: dict[str, Any] = {}
        try:
            if hasattr(cls, "get_class_tags"):
                tags = cls.get_class_tags()
        except Exception as e:
            logger.debug(f"Failed to get tags for {cls.__name__}: {e}")

        parameters: dict[str, Any] = {}
        try:
            param_names = cls.get_param_names()
            param_defaults = cls.get_param_defaults()
            for p in param_names:
                default = param_defaults.get(p)
                required = p not in param_defaults
                if default is not None and not isinstance(
                    default, (int, float, str, bool, list, dict, type(None))
                ):
                    default = str(default)
                parameters[p] = {"default": default, "required": required}
        except Exception as e:
            logger.debug(f"Failed to get parameters for {cls.__name__}: {e}")

        return EstimatorNode(
            name=name,
            task=scitype,
            class_ref=cls,
            module=f"{cls.__module__}.{cls.__name__}",
            tags=tags,
            parameters=parameters,
            docstring=inspect.getdoc(cls),
        )

    # ---- query methods (delegate to sktime.registry) --------------------

    def get_all_estimators(
        self,
        task: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> list[EstimatorNode]:
        """
        Get all estimators, optionally filtered by scitype and tags.

        Delegates filtering to ``sktime.registry.all_estimators``.

        Args:
            task: Filter by scitype (e.g., "forecaster", "classifier").
            tags: Filter by capability tags.
        """
        from sktime.registry import all_estimators

        estimators = all_estimators(
            estimator_types=task,
            filter_tags=tags,
            return_names=True,
            as_dataframe=False,
        )

        results = []
        for name, cls in estimators:
            try:
                results.append(self._create_node(name, cls))
            except Exception as e:
                logger.debug(f"Failed to create node for {name}: {e}")
        return results

    def _name_index(self) -> dict[str, type]:
        """Map every registry class name to its class (built once, then cached)."""
        if self._index is None:
            from sktime.registry import all_estimators

            self._index = dict(all_estimators(return_names=True, as_dataframe=False))
        return self._index

    @staticmethod
    def _import_base_object(dotted_path: str) -> type | None:
        """Import ``package.module.ClassName`` and return it if it is a ``BaseObject`` subclass.

        Plain attribute lookup on an imported module; no expression is ever
        evaluated. Returns ``None`` for import errors, missing attributes and
        objects that are not sktime/skbase classes.
        """
        from skbase.base import BaseObject

        module_name, _, attr = dotted_path.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except Exception as e:  # ImportError, or anything the module raises on import
            logger.debug(f"Could not import {module_name!r}: {e}")
            return None
        obj = getattr(module, attr, None)
        if isinstance(obj, type) and issubclass(obj, BaseObject):
            return obj
        return None

    def get_estimator_by_name(self, name: str) -> EstimatorNode | None:
        """
        Get a specific estimator by its class name or dotted import path.

        Resolution order: exact registry name, case-insensitive registry name,
        then ``package.module.ClassName`` import restricted to ``BaseObject``
        subclasses. Names are never evaluated as expressions; anything that
        does not resolve returns ``None``.

        Args:
            name: The class name of the estimator (e.g., "ARIMA") or its
                dotted import path (e.g., "sktime.forecasting.arima.ARIMA").
        """
        if not isinstance(name, str) or not name.strip():
            return None
        name = name.strip()

        index = self._name_index()
        cls = index.get(name)
        if cls is None:
            lowered = name.lower()
            cls = next((c for n, c in index.items() if n.lower() == lowered), None)
        if cls is None and _DOTTED_PATH.match(name):
            cls = self._import_base_object(name)
        if cls is None:
            return None
        return self._create_node(cls.__name__, cls)

    def suggest_names(self, name: str, n: int = 3) -> list[str]:
        """Return up to *n* registry class names that look like *name* (did-you-mean)."""
        if not isinstance(name, str) or not name.strip():
            return []
        return difflib.get_close_matches(name.strip(), list(self._name_index()), n=n, cutoff=0.6)

    def get_available_tasks(self) -> list[str]:
        """Get list of available scitypes from sktime's base class register."""
        from sktime.registry import get_base_class_register

        return [row[0] for row in get_base_class_register()]

    def get_available_tags(self) -> list[dict[str, Any]]:
        """Get rich metadata for all available tags using sktime's registry.

        Returns a list of dicts with: tag, description, value_type, applies_to.
        """
        from sktime.registry import all_tags

        try:
            tags_df = all_tags(as_dataframe=True)
        except Exception:
            return []

        result = []
        for _, row in tags_df.iterrows():
            scitype = row.get("scitype", [])
            if isinstance(scitype, str):
                scitype = [scitype]
            elif not isinstance(scitype, list):
                scitype = list(scitype) if hasattr(scitype, "__iter__") else [str(scitype)]

            value_type = row.get("type", "")
            if not isinstance(value_type, str):
                value_type = str(value_type)

            result.append(
                {
                    "tag": row["name"],
                    "description": row.get("description", ""),
                    "value_type": value_type,
                    "applies_to": scitype,
                }
            )

        result.sort(key=lambda x: x["tag"])
        return result

    def search_estimators(self, query: str) -> list[EstimatorNode]:
        """
        Search estimators by name, module, or docstring.

        Args:
            query: Search string (case-insensitive).
        """
        all_ests = self.get_all_estimators()
        query_lower = query.strip().lower()

        results = []
        for node in all_ests:
            name_lower = node.name.lower()
            module_lower = node.module.lower()
            docstring_lower = node.docstring.lower() if node.docstring else ""

            if name_lower == query_lower:
                score = 0
            elif name_lower.startswith(query_lower):
                score = 1
            elif query_lower in name_lower:
                score = 2
            elif query_lower in module_lower:
                score = 3
            elif query_lower in docstring_lower:
                score = 4
            else:
                continue

            results.append((score, node.name.lower(), node))

        results.sort(key=lambda item: (item[0], item[1]))
        return [node for _, _, node in results]


# Singleton instance for shared use
_registry_instance: RegistryInterface | None = None


def get_registry() -> RegistryInterface:
    """Get the singleton registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = RegistryInterface()
    return _registry_instance
