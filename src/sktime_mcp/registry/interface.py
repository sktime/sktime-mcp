"""
Registry Interface for sktime MCP.

Thin adapter over sktime's estimator registry.
Delegates to ``sktime.registry`` functions directly.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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
        try:
            from sktime.registry import all_estimators  # noqa: F401
        except ImportError as e:
            logger.error(f"Failed to import sktime registry: {e}")
            raise RuntimeError("sktime must be installed to use sktime-mcp") from e
        self._scitype_bases: dict[str, type] | None = None
        self._tag_value_types: dict[str, Any] | None = None

    def _get_scitype_bases(self) -> dict[str, type]:
        """Map each valid scitype to its sktime base class (cached)."""
        if self._scitype_bases is None:
            from sktime.registry import get_base_class_register

            self._scitype_bases = {row[0]: row[1] for row in get_base_class_register()}
        return self._scitype_bases

    def _resolve_task(self, cls: type) -> str:
        """Pick the most specific *valid* scitype for ``cls``.

        Candidates are the class's ``object_type`` tag values (str or list)
        that are registered scitypes; if none is valid (e.g. skchange's
        ``"interval_scorer"``), the registered base classes ``cls`` inherits
        from are used instead.  Among candidates the most specific one wins
        (deepest base-class MRO), e.g. ``"metric_forecasting"`` over
        ``"metric"``.  The raw ``object_type`` stays available in ``tags``.
        """
        bases = self._get_scitype_bases()
        candidates = self._valid_object_types(cls)
        if not candidates:
            candidates = [s for s, base in bases.items() if issubclass(cls, base)]
        if not candidates:
            return "object"
        return max(candidates, key=lambda s: len(bases[s].__mro__))

    def _valid_object_types(self, cls: type) -> list[str]:
        """Values of the ``object_type`` tag that are registered scitypes."""
        bases = self._get_scitype_bases()
        try:
            raw = cls.get_class_tag("object_type", None)
        except Exception:
            raw = None
        raw_list = raw if isinstance(raw, list) else [raw]
        return [c for c in raw_list if isinstance(c, str) and c in bases]

    def _create_node(self, name: str, cls: type) -> EstimatorNode:
        """Create an EstimatorNode from sktime estimator."""
        scitype = self._resolve_task(cls)

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

        Scitype filtering is delegated to ``sktime.registry.all_estimators``
        (scitype hierarchy: ``task="estimator"`` includes forecasters).  Objects
        whose ``object_type`` tag is not a registered scitype (e.g. skchange's
        ``"interval_scorer"``) are invisible to sktime's filter, so they are
        added by base-class inheritance under the task they resolve to.
        Tag filtering uses the same semantics as sktime's ``filter_tags`` --
        a scalar value must equal the estimator's tag, a list value matches
        any of its elements -- but is applied here so that list-valued
        estimator tags (``python_dependencies``, ``y_inner_mtype``, ...) are
        matched by membership instead of raising ``unhashable type: 'list'``.

        Args:
            task: Filter by scitype (e.g., "forecaster", "classifier").
            tags: Filter by capability tags.
        """
        from sktime.registry import all_estimators

        estimators = all_estimators(
            estimator_types=task,
            return_names=True,
            as_dataframe=False,
        )
        if task is not None:
            bases = self._get_scitype_bases()
            seen = {name for name, _ in estimators}
            for name, cls in all_estimators(return_names=True, as_dataframe=False):
                if name in seen or self._valid_object_types(cls):
                    continue
                if issubclass(bases[self._resolve_task(cls)], bases[task]):
                    estimators.append((name, cls))
            estimators.sort(key=lambda item: item[0])
        if tags:
            estimators = [
                (name, cls)
                for name, cls in estimators
                if all(self._tag_matches(cls, key, wanted) for key, wanted in tags.items())
            ]

        results = []
        for name, cls in estimators:
            try:
                results.append(self._create_node(name, cls))
            except Exception as e:
                logger.debug(f"Failed to create node for {name}: {e}")
        return results

    def get_estimator_by_name(self, name: str) -> EstimatorNode | None:
        """
        Get a specific estimator by its class name.

        Uses ``sktime.registry.craft`` for direct class resolution
        instead of scanning all estimators.

        Args:
            name: The class name of the estimator (e.g., "ARIMA").
        """
        from sktime.registry import craft

        try:
            cls = craft(name)
            if isinstance(cls, type):
                return self._create_node(name, cls)
        except Exception:
            pass
        return None

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

    @staticmethod
    def _tag_matches(cls: type, key: str, wanted: Any) -> bool:
        """True if the class tag ``key`` equals / contains any of ``wanted``."""
        try:
            actual = cls.get_class_tag(key, None)
        except Exception:
            return False
        wanted_list = wanted if isinstance(wanted, list) else [wanted]
        actual_list = actual if isinstance(actual, list) else [actual]
        return any(a == w for a in actual_list for w in wanted_list)

    def _get_tag_value_types(self) -> dict[str, Any]:
        """Map tag name -> raw sktime value type spec (cached).

        The spec is ``"bool"``, ``"int"``, ``"str"``, ``"list"``, ``"type"``,
        ``("list", "str")``, ``("str", [allowed...])`` or ``("list", [allowed...])``.
        """
        if self._tag_value_types is None:
            from sktime.registry import all_tags

            try:
                self._tag_value_types = {row[0]: row[2] for row in all_tags(as_dataframe=False)}
            except Exception:
                self._tag_value_types = {}
        return self._tag_value_types

    def coerce_tag_filters(
        self, tags: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Coerce tag filter values to the tag's declared value type.

        ``"true"``/``"false"``/``"1"``/``"0"`` (and 0/1) become bools for bool
        tags, numeric strings become ints for int tags, and values for tags
        with an allowed-values list are checked against it.  A list of values
        is coerced element-wise (any-of semantics).  Tags with an unknown or
        unconstrained type are passed through unchanged.

        Returns:
            ``(coerced_tags, errors)`` where each error is a dict with
            ``tag``, ``value`` and ``expected`` keys.
        """
        types = self._get_tag_value_types()
        coerced: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []
        for key, value in tags.items():
            spec = types.get(key)
            values = value if isinstance(value, list) else [value]
            out = []
            for v in values:
                ok, cv, expected = self._coerce_tag_value(v, spec)
                if not ok:
                    errors.append({"tag": key, "value": v, "expected": expected})
                    break
                out.append(cv)
            else:
                coerced[key] = out if isinstance(value, list) else out[0]
        return coerced, errors

    @staticmethod
    def _coerce_tag_value(value: Any, spec: Any) -> tuple[bool, Any, str]:
        """Coerce one value against a sktime tag type spec.

        Returns ``(ok, coerced_value, expected_description)``.
        """
        base, allowed = spec, None
        if isinstance(spec, tuple) and len(spec) == 2:
            base, allowed = spec
        if allowed == "str":
            allowed = None

        if base == "bool":
            if isinstance(value, bool):
                return True, value, "bool"
            if isinstance(value, int) and value in (0, 1):
                return True, bool(value), "bool"
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1"):
                    return True, True, "bool"
                if lowered in ("false", "0"):
                    return True, False, "bool"
            return False, value, "bool (true/false)"
        if base == "int":
            if isinstance(value, bool):
                return False, value, "int"
            if isinstance(value, int):
                return True, value, "int"
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                return True, int(value.strip()), "int"
            return False, value, "int"
        if base in ("str", "list"):
            if isinstance(allowed, list):
                if value in allowed:
                    return True, value, f"one of {allowed}"
                return False, value, f"one of {allowed}"
            if base == "str" and not isinstance(value, str):
                return False, value, "str"
            return True, value, "str"
        return True, value, str(spec)

    def search_estimators(
        self,
        query: str,
        task: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> list[EstimatorNode]:
        """
        Search estimators by name, module, or docstring.

        ``task`` and ``tags`` are applied first through
        :meth:`get_all_estimators` (the same filter path used without a
        query); the substring match then narrows that list and ranks it.

        Args:
            query: Search string (case-insensitive).
            task: Optional scitype filter.
            tags: Optional capability tag filter.
        """
        all_ests = self.get_all_estimators(task=task, tags=tags)
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
