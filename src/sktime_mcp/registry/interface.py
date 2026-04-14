"""Registry interface to sktime's estimator registry."""

import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EstimatorNode:
    """Semantic representation of an sktime estimator exposed through the MCP.

    Attributes
    ----------
    name : str
        The class name of the estimator as registered in sktime (e.g. ``"NaiveForecaster"``).
    task : str
        Human-readable task category derived from ``TASK_MAP`` (e.g. ``"forecasting"``).
    class_ref : type
        Direct reference to the estimator class object; used for instantiation.
    module : str
        Fully-qualified module path including the class name (``"<module>.<ClassName>"``).
    tags : dict[str, Any]
        Shallow copy of the estimator's class-level tags returned by ``get_class_tags()``.
    hyperparameters : dict[str, Any]
        Mapping of ``__init__`` parameter names to ``{"default": ..., "required": bool}``
        dicts extracted from the constructor signature.
    docstring : Optional[str]
        Cleaned docstring produced by ``inspect.getdoc``; ``None`` if the class has no
        docstring.  Truncated to 500 characters in ``to_dict()`` for MCP payload safety.
    """

    name: str
    task: str
    class_ref: type
    module: str
    tags: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    docstring: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "task": self.task,
            "module": self.module,
            "tags": self.tags,
            "hyperparameters": self.hyperparameters,
            "docstring": (
                self.docstring[:500] if self.docstring else None
            ),  # L-1: truncate to 500 chars to keep MCP responses JSON-safe
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
    """Single source of truth for sktime estimator metadata, tags, and documentation."""

    # Map of sktime estimator types to task names
    TASK_MAP = {
        "forecaster": "forecasting",
        "transformer": "transformation",
        "classifier": "classification",
        "regressor": "regression",
        "clusterer": "clustering",
        "param_est": "parameter_estimation",
        "splitter": "splitting",
        # "alignment": "alignment",  # L-2: excluded — known failure, see issue #91
        "network": "network",
    }

    def __init__(self):
        """Initialise with an empty cache. Use :func:`get_registry` for the shared singleton.

        Direct instantiation creates an independent instance whose thread-safety is
        unrelated to the module-level singleton; prefer ``get_registry()`` in production.
        """
        self._cache: dict[str, EstimatorNode] = {}
        self._all_tags: set = set()
        self._loaded = False
        # RLock guards against re-entrant acquisition if a custom estimator hook
        # (e.g. get_class_tags override) calls back into _ensure_loaded on the same thread.
        self._lock = threading.RLock()

    def _ensure_loaded(self):
        """Lazy-load the registry on first access (thread-safe via double-checked locking).

        _loaded is set only on success.  If _load_registry raises (e.g. sktime not
        installed → RuntimeError), _loaded stays False so the next call retries and
        re-raises — intentional: the error should surface on every access until the
        environment is fixed, not be silently swallowed into an empty cache.
        """
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._load_registry()
                    self._loaded = True

    def _load_registry(self):
        """Load all estimators from sktime's registry.

        Called while ``_lock`` is held by ``_ensure_loaded``.  Must not call any
        public method that invokes ``_ensure_loaded``; doing so would re-acquire
        ``_lock`` on the same thread (safe with RLock, but logically incorrect).
        """
        try:
            from sktime.registry import all_estimators
        except ImportError as e:
            logger.debug(f"Failed to import sktime registry: {e}")
            raise RuntimeError("sktime must be installed to use sktime-mcp") from e

        new_cache: dict[str, "EstimatorNode"] = {}
        new_all_tags: set = set()

        for estimator_type in self.TASK_MAP:
            try:
                estimators = all_estimators(
                    estimator_types=estimator_type,
                    return_names=True,
                    as_dataframe=False,
                )
            except Exception as e:
                logger.debug(f"Failed to load estimators for type {estimator_type!r}: {e}")
                continue

            for name, cls in estimators:
                try:
                    node = self._create_node(name, cls, estimator_type)
                    if name in new_cache:
                        logger.warning(
                            f"Name collision: {name!r} already in cache; "
                            f"overwriting with {cls.__module__}.{cls.__name__}"
                        )
                    new_cache[name] = node
                    new_all_tags.update(node.tags.keys())
                except Exception as e:
                    logger.debug(f"Failed to load estimator {name!r}: {e}")
                    continue

        # Two separate reference stores — written in sequence, not as a unit.
        self._cache = new_cache
        self._all_tags = new_all_tags
        logger.info(f"Loaded {len(self._cache)} estimators from sktime registry")

    def _create_node(self, name: str, cls: type, estimator_type: str) -> EstimatorNode:
        """Create an EstimatorNode from a class."""
        tags = self._get_tags(cls)
        hyperparameters = self._get_hyperparameters(cls)
        # inspect.getdoc strips indentation and follows __doc__ inheritance, unlike cls.__doc__
        docstring = inspect.getdoc(cls)

        return EstimatorNode(
            name=name,
            task=self.TASK_MAP.get(estimator_type, estimator_type),
            class_ref=cls,
            module=f"{cls.__module__}.{cls.__name__}",
            tags=tags,
            hyperparameters=hyperparameters,
            docstring=docstring,
        )

    def _get_tags(self, cls: type) -> dict[str, Any]:
        """Extract tags from an estimator class."""
        tags = {}

        try:
            if hasattr(cls, "get_class_tags"):
                tags = dict(cls.get_class_tags())
            elif hasattr(cls, "_tags"):
                tags = dict(cls._tags) if cls._tags else {}
        except Exception as e:
            logger.debug(f"Failed to get tags for {cls.__name__}: {e}")

        return tags

    def _get_hyperparameters(self, cls: type) -> dict[str, Any]:
        """Extract hyperparameters from __init__ signature."""
        params = {}

        try:
            sig = inspect.signature(cls.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == "self" or param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                default = None
                if param.default is not inspect.Parameter.empty:
                    default = param.default
                    # Convert non-serializable defaults to string representation
                    if not isinstance(default, (int, float, str, bool, list, dict, type(None))):
                        default = str(default)

                params[param_name] = {
                    "default": default,
                    "required": param.default is inspect.Parameter.empty,
                }
        except Exception as e:
            logger.debug(f"Failed to get hyperparameters for {cls.__name__}: {e}")

        return params

    def get_all_estimators(
        self,
        task: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
    ) -> list[EstimatorNode]:
        """Return all estimators, optionally filtered by task and tags."""
        self._ensure_loaded()

        results = list(self._cache.values())

        if task:
            results = [e for e in results if e.task == task]

        if tags:
            results = self._filter_by_tags(results, tags)

        return results

    def _filter_by_tags(
        self,
        estimators: list[EstimatorNode],
        required_tags: dict[str, Any],
    ) -> list[EstimatorNode]:
        """Filter estimators by required tag values."""
        filtered = []

        for estimator in estimators:
            matches = True
            for tag_name, tag_value in required_tags.items():
                est_tag_value = estimator.tags.get(tag_name)
                if est_tag_value != tag_value:
                    matches = False
                    break

            if matches:
                filtered.append(estimator)

        return filtered

    def get_estimator_by_name(self, name: str) -> Optional[EstimatorNode]:
        """Return the EstimatorNode for the given class name, or None if not found."""
        self._ensure_loaded()
        return self._cache.get(name)

    def get_available_tasks(self) -> list[str]:
        """Get list of available task types."""
        return list(self.TASK_MAP.values())

    def get_available_tags(self) -> list[dict[str, Any]]:
        """Return rich metadata for all available tags from sktime's registry."""
        self._ensure_loaded()

        try:
            from sktime.registry import all_tags

            tags_df = all_tags(as_dataframe=True)
        except ImportError:
            # Fallback when all_tags is not available. Re-acquire the lock so the
            # read of _all_tags is not racing with a concurrent _load_registry write.
            with self._lock:
                return [{"tag": t} for t in sorted(self._all_tags, key=str)]

        result = []
        for _, row in tags_df.iterrows():
            tag_name = row.get("name")
            if not tag_name or not isinstance(tag_name, str):
                continue
            # Normalize scitype to a list for consistency
            scitype = row.get("scitype", [])
            if isinstance(scitype, str):
                scitype = [scitype]
            elif not isinstance(scitype, list):
                scitype = list(scitype) if hasattr(scitype, "__iter__") else [str(scitype)]

            # Convert value_type to a JSON-safe string representation
            value_type = row.get("type", "")
            if not isinstance(value_type, str):
                value_type = str(value_type)

            result.append(
                {
                    "tag": tag_name,
                    "description": row.get("description", ""),
                    "value_type": value_type,
                    "applies_to": scitype,
                }
            )

        result.sort(key=lambda x: x["tag"])
        return result

    def search_estimators(self, query: str) -> list[EstimatorNode]:
        """Return estimators whose name or docstring contains the query string (case-insensitive)."""
        if not isinstance(query, str):
            return []
        self._ensure_loaded()
        query_lower = query.lower()

        results = []
        for node in self._cache.values():
            if query_lower in node.name.lower():
                results.append(node)
                continue

            if node.docstring and query_lower in node.docstring.lower():
                results.append(node)

        return results


_registry_instance: Optional[RegistryInterface] = None
_registry_lock = threading.Lock()


def get_registry() -> RegistryInterface:
    """Get the singleton registry instance."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = RegistryInterface()
    return _registry_instance
