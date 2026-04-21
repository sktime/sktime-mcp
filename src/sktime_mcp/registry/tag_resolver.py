"""
Tag Resolver for sktime MCP.

Tags encode estimator capabilities and constraints. This module provides
utilities for working with tags and understanding their meanings.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from sktime_mcp.registry.interface import EstimatorNode, get_registry

logger = logging.getLogger(__name__)


@dataclass
class TagInfo:
    """Information about a specific tag."""

    name: str
    description: str
    value_type: str  # "bool", "str", "list", etc.
    possible_values: Optional[list[Any]] = None
    category: str = "general"


class TagResolver:
    """
    Resolver for sktime estimator tags.

    Tags encode important semantic information about estimators:
    - Supported data types
    - Probabilistic vs deterministic predictions
    - Composability rules
    - Missing value handling
    - And many more...

    This class provides utilities for understanding and querying tags.
    """

    # Cache for tag definitions loaded from sktime
    _tag_definitions_cache: Optional[dict[str, TagInfo]] = None

    def __init__(self):
        """Initialize the tag resolver."""
        self._registry = get_registry()
        self._load_tag_definitions()

    def _load_tag_definitions(self):
        """Load tag definitions from sktime.registry.all_tags()."""
        if TagResolver._tag_definitions_cache is not None:
            return

        try:
            from sktime.registry import all_tags

            all_tags_list = all_tags(as_dataframe=False)

            tag_definitions = {}
            for tag_tuple in all_tags_list:
                tag_name = tag_tuple[0]
                scitype = tag_tuple[1]  # Can be a string or list of strings
                tag_type = tag_tuple[2]  # Can be 'bool', 'str', 'int', or tuple like ('str', [...])
                description = tag_tuple[3]

                # Determine value_type and possible_values
                value_type = "str"
                possible_values = None

                if isinstance(tag_type, tuple):
                    value_type = tag_type[0]
                    if len(tag_type) > 1 and isinstance(tag_type[1], list):
                        possible_values = tag_type[1]
                elif tag_type in ["bool", "str", "int", "type", "dict"]:
                    value_type = tag_type

                # Determine category based on tag name prefix or scitype
                category = "general"
                if tag_name.startswith("capability:"):
                    category = "capability"
                elif tag_name.startswith("scitype:"):
                    category = "data"
                elif tag_name.startswith("requires"):
                    category = "behavior"
                elif tag_name.startswith("transform"):
                    category = "transformation"
                elif tag_name.startswith("python_"):
                    category = "requirements"
                elif "mtype" in tag_name:
                    category = "data"
                elif isinstance(scitype, str):
                    category = scitype
                elif isinstance(scitype, list) and len(scitype) > 0:
                    category = scitype[0]

                tag_definitions[tag_name] = TagInfo(
                    name=tag_name,
                    description=description,
                    value_type=value_type,
                    possible_values=possible_values,
                    category=category,
                )

            TagResolver._tag_definitions_cache = tag_definitions
            logger.info(f"Loaded {len(tag_definitions)} tag definitions from sktime")

        except ImportError as e:
            logger.warning(f"Could not import sktime.registry.all_tags: {e}")
            TagResolver._tag_definitions_cache = {}
        except Exception as e:
            logger.error(f"Error loading tag definitions: {e}")
            TagResolver._tag_definitions_cache = {}

    @property
    def tag_definitions(self) -> dict[str, TagInfo]:
        """Get tag definitions, loading them if necessary."""
        if TagResolver._tag_definitions_cache is None:
            self._load_tag_definitions()
        return TagResolver._tag_definitions_cache or {}

    def get_tag_info(self, tag_name: str) -> Optional[TagInfo]:
        """
        Get information about a specific tag.

        Args:
            tag_name: The tag name to look up

        Returns:
            TagInfo if known, None otherwise
        """
        return self.tag_definitions.get(tag_name)

    def get_tag_description(self, tag_name: str) -> str:
        """
        Get human-readable description of a tag.

        Args:
            tag_name: The tag name

        Returns:
            Description string, or generic message if unknown
        """
        info = self.get_tag_info(tag_name)
        if info:
            return info.description
        return f"Tag '{tag_name}' (no description available)"

    def get_tags_by_category(self, category: str) -> list[TagInfo]:
        """
        Get all known tags in a specific category.

        Args:
            category: Category name (e.g., "capability", "data", "behavior")

        Returns:
            List of TagInfo objects in that category
        """
        return [tag for tag in self.tag_definitions.values() if tag.category == category]

    def get_all_categories(self) -> list[str]:
        """Get list of all tag categories."""
        categories = {tag.category for tag in self.tag_definitions.values()}
        return sorted(categories)

    def explain_tags(self, tags: dict[str, Any]) -> dict[str, str]:
        """
        Get human-readable explanations for a set of tags.

        Args:
            tags: Dictionary of tag names to values

        Returns:
            Dictionary of tag names to explanation strings
        """
        explanations = {}

        for tag_name, tag_value in tags.items():
            info = self.get_tag_info(tag_name)
            if info:
                if info.value_type == "bool":
                    status = "Yes" if tag_value else "No"
                    explanations[tag_name] = f"{info.description}: {status}"
                else:
                    explanations[tag_name] = f"{info.description}: {tag_value}"
            else:
                explanations[tag_name] = f"{tag_name}: {tag_value}"

        return explanations

    def filter_estimators_by_capability(
        self,
        task: Optional[str] = None,
        probabilistic: Optional[bool] = None,
        handles_missing: Optional[bool] = None,
        multivariate: Optional[bool] = None,
    ) -> list[EstimatorNode]:
        """
        Filter estimators by common capability requirements.

        This is a convenience method that translates human-friendly
        requirements into the appropriate tag queries.

        Args:
            task: Task type filter
            probabilistic: Require probabilistic predictions
            handles_missing: Require missing data handling
            multivariate: Require multivariate support

        Returns:
            List of matching EstimatorNode objects
        """
        tags = {}

        if probabilistic is not None:
            tags["capability:pred_int"] = probabilistic

        if handles_missing is not None:
            tags["handles-missing-data"] = handles_missing

        if multivariate is not None:
            tags["capability:multivariate"] = multivariate

        return self._registry.get_all_estimators(task=task, tags=tags if tags else None)

    def check_compatibility(
        self,
        estimator: EstimatorNode,
        requirements: dict[str, Any],
    ) -> dict[str, bool]:
        """
        Check if an estimator meets specific requirements.

        Args:
            estimator: The estimator to check
            requirements: Dictionary of required tag values

        Returns:
            Dictionary mapping requirement names to whether they are met
        """
        results = {}

        for req_name, req_value in requirements.items():
            actual_value = estimator.tags.get(req_name)
            results[req_name] = actual_value == req_value

        return results

    def suggest_similar_estimators(
        self,
        estimator: EstimatorNode,
        max_results: int = 5,
    ) -> list[EstimatorNode]:
        """
        Find estimators with similar capabilities.

        Args:
            estimator: Reference estimator
            max_results: Maximum number of results

        Returns:
            List of similar estimators (same task, similar tags)
        """
        # Get all estimators of the same task
        same_task = self._registry.get_all_estimators(task=estimator.task)

        # Score by tag similarity
        scored = []
        for candidate in same_task:
            if candidate.name == estimator.name:
                continue

            # Count matching tags
            score = 0
            for tag_name, tag_value in estimator.tags.items():
                if candidate.tags.get(tag_name) == tag_value:
                    score += 1

            scored.append((candidate, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in scored[:max_results]]


# Singleton instance
_resolver_instance: Optional[TagResolver] = None


def get_tag_resolver() -> TagResolver:
    """Get the singleton tag resolver instance."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = TagResolver()
    return _resolver_instance
