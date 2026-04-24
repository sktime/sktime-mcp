"""
Registry Interface for sktime MCP.

This module provides the core interface to sktime's estimator registry,
exposing structured semantic information about all available estimators.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EstimatorNode:
    """
    Represents a single estimator in the sktime registry.

    This is the semantic representation of an estimator that gets
    exposed to the LLM through the MCP.

    Attributes:
        name: The class name of the estimator (e.g., "ARIMA")
        task: The task type (e.g., "forecaster", "transformer", "classifier")
        class_ref: Reference to the actual Python class
        module: Full module path to the estimator
        tags: Dictionary of capability tags
        hyperparameters: List of hyperparameter names with their defaults
        docstring: The estimator's docstring for understanding usage
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
            ),  # L-1: Truncate docstring to 500 characters, we can also try summarization
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
    Interface to sktime's estimator registry.

    This class wraps sktime's `all_estimators` function and provides
    structured access to estimator metadata, tags, and documentation.

    The registry is the single source of truth for all estimator information.
    """

    # Map of sktime estimator types to task names
    TASK_MAP = {
        "forecaster": "forecasting",
        "transformer": "transformation",
        "classifier": "classification",
        "regressor": "regression",
        "clusterer": "clustering",
        "param_est": "parameter_estimation",
        "splitter": "splitting",
        # "alignment": "alignment", L-2: It is failing, but I will investigate it later
        "network": "network",
        "detector": "detection",
    }

    def __init__(self):
        """Initialize the registry interface."""
        self._cache: dict[str, EstimatorNode] = {}
        self._all_tags: set = set()
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load the registry on first access."""
        if not self._loaded:
            self._load_registry()
            self._loaded = True

    def _load_registry(self):
        """Load all estimators from sktime's registry."""
        # L-3: Sometimes, We need to import other packages as well to load the estimators
        try:
            from sktime.registry import all_estimators
        except ImportError as e:
            logger.error(f"Failed to import sktime registry: {e}")
            raise RuntimeError("sktime must be installed to use sktime-mcp") from e

        # Load each type of estimator
        for estimator_type in self.TASK_MAP:
            try:
                estimators = all_estimators(
                    estimator_types=estimator_type,
                    return_names=True,
                    as_dataframe=False,
                )

                for name, cls in estimators:
                    try:
                        node = self._create_node(name, cls, estimator_type)
                        self._cache[name] = node
                        self._all_tags.update(node.tags.keys())
                    except Exception as e:
                        logger.debug(f"Failed to load estimator {name}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Failed to load estimator type {estimator_type}: {e}")
                continue

        logger.info(f"Loaded {len(self._cache)} estimators from sktime registry")

    def _create_node(self, name: str, cls: type, estimator_type: str) -> EstimatorNode:
        """Create an EstimatorNode from a class."""
        # Get tags
        tags = self._get_tags(cls)

        # Get hyperparameters from __init__ signature
        hyperparameters = self._get_hyperparameters(cls)

        # Get docstring
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
            # sktime estimators have a get_class_tags() method
            if hasattr(cls, "get_class_tags"):
                tags = cls.get_class_tags()
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
                if param_name in ("self", "args", "kwargs"):
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
        """
        Get all estimators, optionally filtered by task and tags.

        Args:
            task: Filter by task type (e.g., "forecasting", "classification")
            tags: Filter by capability tags (e.g., {"capability:pred_int": True})

        Returns:
            List of matching EstimatorNode objects
        """
        self._ensure_loaded()

        results = list(self._cache.values())

        # Filter by task
        if task:
            results = [e for e in results if e.task == task]

        # Filter by tags
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
        """
        Get a specific estimator by its class name.

        Args:
            name: The class name of the estimator (e.g., "ARIMA")

        Returns:
            EstimatorNode if found, None otherwise
        """
        self._ensure_loaded()
        return self._cache.get(name)

    def get_available_tasks(self) -> list[str]:
        """Get list of available task types."""
        return list(self.TASK_MAP.values())

    def get_available_tags(self) -> list[dict[str, Any]]:
        """Get rich metadata for all available tags using sktime's registry.

        Returns a list of dicts, each containing:
        - tag: the tag name (e.g., "scitype:y")
        - description: human-readable explanation of what the tag means
        - value_type: the expected value type (e.g., "bool", "str")
        - applies_to: list of estimator types this tag applies to
        """
        self._ensure_loaded()

        try:
            from sktime.registry import all_tags

            tags_df = all_tags(as_dataframe=True)
        except ImportError:
            # Fallback to old behaviour if all_tags is not available
            return [{"tag": t} for t in sorted(self._all_tags)]

        result = []
        for _, row in tags_df.iterrows():
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
        Search estimators by name or docstring.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching EstimatorNode objects
        """
        self._ensure_loaded()
        query_lower = query.lower()

        results = []
        for node in self._cache.values():
            # Search in name
            if query_lower in node.name.lower():
                results.append(node)
                continue

            # Search in docstring
            if node.docstring and query_lower in node.docstring.lower():
                results.append(node)

        return results


# Singleton instance for shared use
_registry_instance: Optional[RegistryInterface] = None


def get_registry() -> RegistryInterface:
    """Get the singleton registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = RegistryInterface()
    return _registry_instance
