"""
Composition Validator for sktime MCP.

sktime estimators are composable:
- transformers → forecasters
- pipelines
- reduction strategies

This module enforces:
- Compatible task types
- Valid composition order
- Tag compatibility

This prevents invalid pipelines at planning time.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sktime_mcp.registry.interface import EstimatorNode, get_registry

logger = logging.getLogger(__name__)


class CompositionType(Enum):
    """Types of composition in sktime."""

    PIPELINE = "pipeline"
    TRANSFORMER_PIPELINE = "transformer_pipeline"
    FORECASTING_PIPELINE = "forecasting_pipeline"
    MULTIPLEXER = "multiplexer"
    ENSEMBLE = "ensemble"
    REDUCTION = "reduction"


@dataclass
class CompositionRule:
    """
    A rule describing valid compositions for an estimator type.

    Attributes:
        source_task: The task type that can be composed
        target_task: The task type it can compose with
        composition_type: Type of composition
        position: Where in the pipeline (before, after, any)
        description: Human-readable description
    """

    source_task: str
    target_task: str
    composition_type: CompositionType
    position: str  # "before", "after", "any"
    description: str


@dataclass
class ValidationResult:
    """
    Result of a composition validation.

    Attributes:
        valid: Whether the composition is valid
        errors: List of validation errors
        warnings: List of warnings (valid but potentially problematic)
        suggestions: Suggested fixes for invalid compositions
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


class CompositionValidator:
    """
    Validator for sktime estimator compositions.

    This class encodes the rules for valid estimator compositions
    in sktime, allowing validation at planning time rather than
    runtime.
    """

    # Valid composition rules
    COMPOSITION_RULES: list[CompositionRule] = [
        # Transformers can precede forecasters
        CompositionRule(
            source_task="transformation",
            target_task="forecasting",
            composition_type=CompositionType.FORECASTING_PIPELINE,
            position="before",
            description="Transformers can be applied before forecasters in a pipeline",
        ),
        # Transformers can be chained
        CompositionRule(
            source_task="transformation",
            target_task="transformation",
            composition_type=CompositionType.TRANSFORMER_PIPELINE,
            position="before",
            description="Transformers can be chained together",
        ),
        # Forecasters can be ensembled
        CompositionRule(
            source_task="forecasting",
            target_task="forecasting",
            composition_type=CompositionType.ENSEMBLE,
            position="any",
            description="Forecasters can be combined in an ensemble",
        ),
        # Classifiers can be ensembled
        CompositionRule(
            source_task="classification",
            target_task="classification",
            composition_type=CompositionType.ENSEMBLE,
            position="any",
            description="Classifiers can be combined in an ensemble",
        ),
        # Transformers can precede classifiers
        CompositionRule(
            source_task="transformation",
            target_task="classification",
            composition_type=CompositionType.PIPELINE,
            position="before",
            description="Transformers can be applied before classifiers",
        ),
        # Transformers can precede regressors
        CompositionRule(
            source_task="transformation",
            target_task="regression",
            composition_type=CompositionType.PIPELINE,
            position="before",
            description="Transformers can be applied before regressors",
        ),
    ]

    # Known transformer categories
    TRANSFORMER_CATEGORIES = {
        "Imputer": "missing_value_handler",
        "Detrend": "trend_remover",
        "Deseasonalize": "seasonality_remover",
        "Differencer": "differencer",
        "BoxCoxTransformer": "power_transform",
        "LogTransformer": "power_transform",
        "ScaledLogitTransformer": "power_transform",
        "Lag": "lag_creator",
        "WindowSummarizer": "feature_creator",
        "DateTimeFeatures": "feature_creator",
    }

    def __init__(self):
        """Initialize the validator."""
        self._registry = get_registry()

    def validate_pipeline(
        self,
        components: list[str],
    ) -> ValidationResult:
        """
        Validate a proposed pipeline composition.

        Args:
            components: List of estimator names in pipeline order

        Returns:
            ValidationResult with validity status and any issues
        """
        if not components:
            return ValidationResult(
                valid=False,
                errors=["Pipeline cannot be empty"],
            )

        if len(components) == 1:
            # Single component is always valid if it exists
            estimator = self._registry.get_estimator_by_name(components[0])
            if estimator is None:
                return ValidationResult(
                    valid=False,
                    errors=[f"Unknown estimator: {components[0]}"],
                )
            return ValidationResult(valid=True)

        errors = []
        warnings = []
        suggestions = []

        # Get all estimator nodes
        nodes: list[tuple[str, EstimatorNode | None]] = []
        for name in components:
            node = self._registry.get_estimator_by_name(name)
            nodes.append((name, node))
            if node is None:
                errors.append(f"Unknown estimator: {name}")

        if errors:
            return ValidationResult(
                valid=False,
                errors=errors,
            )

        # Check pairwise compatibility
        for i in range(len(nodes) - 1):
            current_name, current_node = nodes[i]
            next_name, next_node = nodes[i + 1]

            # Check if this composition is valid
            valid_pair, pair_errors, pair_warnings = self._check_pair_compatibility(
                current_node, next_node
            )

            if not valid_pair:
                errors.extend(pair_errors)
            warnings.extend(pair_warnings)

        # Check final component is executable (forecaster, classifier, etc.)
        final_name, final_node = nodes[-1]
        if final_node.task == "transformation":
            errors.append(
                f"Pipeline ends with transformer '{final_name}'. "
                "The final component should be a forecaster, classifier, or regressor."
            )
            suggestions.append("Add a forecaster like 'ARIMA' or 'ExponentialSmoothing' at the end")

        # Check for duplicate consecutive components
        for i in range(len(components) - 1):
            if components[i] == components[i + 1]:
                warnings.append(
                    f"Duplicate consecutive component: '{components[i]}' at positions {i + 1} and {i + 2}"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

    def _check_pair_compatibility(
        self,
        first: EstimatorNode,
        second: EstimatorNode,
    ) -> tuple[bool, list[str], list[str]]:
        """
        Check if two estimators can be composed in sequence.

        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Find applicable rule
        applicable_rule = None
        for rule in self.COMPOSITION_RULES:
            if (
                rule.source_task == first.task
                and rule.target_task == second.task
                and rule.position in ("before", "any")
            ):
                applicable_rule = rule
                break

        if applicable_rule is None:
            # No rule found - check if it's an obvious error
            if first.task == second.task == "forecasting":
                errors.append(
                    f"Cannot chain forecasters '{first.name}' → '{second.name}' directly. "
                    "Use an ensemble or multiplexer instead."
                )
            elif first.task in ("classification", "regression") and second.task != first.task:
                errors.append(
                    f"Invalid composition: {first.task} '{first.name}' → {second.task} '{second.name}'"
                )
            else:
                warnings.append(
                    f"Unusual composition: {first.task} '{first.name}' → {second.task} '{second.name}'"
                )

        # Check tag compatibility
        tag_errors, tag_warnings = self._check_tag_compatibility(first, second)
        errors.extend(tag_errors)
        warnings.extend(tag_warnings)

        return len(errors) == 0, errors, warnings

    def _check_tag_compatibility(
        self,
        first: EstimatorNode,
        second: EstimatorNode,
    ) -> tuple[list[str], list[str]]:
        """Check tag-based compatibility between estimators."""
        errors = []
        warnings = []

        # Check univariate vs multivariate
        first_univariate = first.tags.get("univariate-only", False)
        second_multivariate = second.tags.get("capability:multivariate", False)

        if first_univariate and second_multivariate:
            warnings.append(
                f"'{first.name}' is univariate-only but placed before "
                f"multivariate-capable '{second.name}'"
            )

        # Check if transformer output is compatible with next component's input
        # This is a simplified check - full check would need mtype resolution

        return errors, warnings

    def get_valid_compositions(
        self,
        estimator_name: str,
    ) -> dict[str, list[str]]:
        """
        Get valid compositions for an estimator.

        Args:
            estimator_name: Name of the estimator

        Returns:
            Dictionary with "can_precede" and "can_follow" lists
        """
        estimator = self._registry.get_estimator_by_name(estimator_name)
        if estimator is None:
            return {
                "can_precede": [],
                "can_follow": [],
                "error": f"Unknown estimator: {estimator_name}",
            }

        can_precede = []
        can_follow = []

        for rule in self.COMPOSITION_RULES:
            if rule.source_task == estimator.task and rule.position in ("before", "any"):
                # This estimator can precede things of target_task
                can_precede.append(rule.target_task)

            if rule.target_task == estimator.task and rule.position in ("before", "any"):
                # Things of source_task can precede this estimator
                can_follow.append(rule.source_task)

        return {
            "can_precede": list(set(can_precede)),
            "can_follow": list(set(can_follow)),
        }

    def suggest_pipeline(
        self,
        task: str,
        requirements: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Suggest a valid pipeline for a given task.

        Args:
            task: Target task (e.g., "forecasting")
            requirements: Optional requirements (e.g., {"handles_missing": True})

        Returns:
            List of suggested estimator names forming a valid pipeline
        """
        suggestions = []

        if task == "forecasting":
            # Suggest common preprocessing → forecaster pipeline
            if requirements and requirements.get("handles_missing"):
                suggestions.append("Imputer")

            # Get a suitable forecaster
            forecasters = self._registry.get_all_estimators(
                task="forecasting",
                tags=requirements if requirements else None,
            )

            if forecasters:
                # Pick first match
                suggestions.append(forecasters[0].name)
            else:
                suggestions.append("NaiveForecaster")  # Fallback

        elif task == "classification":
            # Suggest transformer → classifier
            classifiers = self._registry.get_all_estimators(task="classification")
            if classifiers:
                suggestions.append(classifiers[0].name)

        return suggestions


# Singleton instance
_validator_instance: CompositionValidator | None = None


def get_composition_validator() -> CompositionValidator:
    """Get the singleton composition validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = CompositionValidator()
    return _validator_instance
