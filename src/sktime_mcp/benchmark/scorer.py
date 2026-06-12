"""
Scorer for sktime-mcp benchmark results.

Scores TaskResults on three dimensions:
1. pipeline_validity_score - is the composition valid?
2. performance_score       - how good is the forecast (MAPE based)?
3. task_match_score        - did the estimator match the expected task?

Final score is a weighted combination of all three.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sktime_mcp.benchmark.runner import TaskResult
from sktime_mcp.benchmark.tasks import BenchmarkTask
from sktime_mcp.registry.interface import get_registry

logger = logging.getLogger(__name__)

# Weights for final score
WEIGHT_PIPELINE_VALIDITY = 0.3
WEIGHT_PERFORMANCE = 0.5
WEIGHT_TASK_MATCH = 0.2


@dataclass
class Score:
    """
    Scores for a single TaskResult.

    Attributes
    ----------
    task_name : str
    estimator_name : str
    pipeline_validity_score : float
        1.0 if pipeline is valid, 0.0 otherwise
    performance_score : float
        Normalized score based on MAPE (lower MAPE = higher score)
    task_match_score : float
        1.0 if estimator matches expected task type, 0.0 otherwise
    overall_score : float
        Weighted combination of all three scores
    mape : float or None
        Raw MAPE value if available
    notes : list[str]
        Human readable notes about the score
    """

    task_name: str
    estimator_name: str
    pipeline_validity_score: float = 0.0
    performance_score: float = 0.0
    task_match_score: float = 0.0
    overall_score: float = 0.0
    mape: float | None = None
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "estimator_name": self.estimator_name,
            "pipeline_validity_score": round(self.pipeline_validity_score, 3),
            "performance_score": round(self.performance_score, 3),
            "task_match_score": round(self.task_match_score, 3),
            "overall_score": round(self.overall_score, 3),
            "mape": round(self.mape, 4) if self.mape is not None else None,
            "notes": self.notes,
        }


class BenchmarkScorer:
    """
    Scores benchmark results from BenchmarkRunner.

    Usage
    -----
    >>> scorer = BenchmarkScorer()
    >>> scores = scorer.score_results(results, task)
    >>> report = scorer.summary(scores)
    """

    def __init__(self):
        self._registry = get_registry()

    def score_result(
        self,
        result: TaskResult,
        task: BenchmarkTask,
    ) -> Score:
        """
        Score a single TaskResult against its task definition.

        Parameters
        ----------
        result : TaskResult
        task : BenchmarkTask

        Returns
        -------
        Score
        """
        score = Score(
            task_name=result.task_name,
            estimator_name=result.estimator_name,
        )

        # Score 1 - pipeline validity
        score.pipeline_validity_score = 1.0 if result.pipeline_valid else 0.0
        if not result.pipeline_valid:
            score.notes.append(
                f"Invalid pipeline: {result.errors}"
            )

        # Score 2 - task match
        score.task_match_score = self._score_task_match(
            result.estimator_name, task.expected_task
        )

        # Score 3 - performance via MAPE
        score.mape, score.performance_score = self._score_performance(
            result.cv_results
        )

        # Final weighted score
        score.overall_score = (
            WEIGHT_PIPELINE_VALIDITY * score.pipeline_validity_score
            + WEIGHT_PERFORMANCE * score.performance_score
            + WEIGHT_TASK_MATCH * score.task_match_score
        )

        return score

    def score_results(
        self,
        results: list[TaskResult],
        task: BenchmarkTask,
    ) -> list[Score]:
        """Score a list of TaskResults for a given task."""
        return [self.score_result(r, task) for r in results]

    def summary(self, scores: list[Score]) -> dict:
        """
        Generate a summary report from a list of scores.

        Parameters
        ----------
        scores : list[Score]

        Returns
        -------
        dict with ranked results and best estimator
        """
        if not scores:
            return {"error": "No scores to summarize"}

        ranked = sorted(
            scores, key=lambda s: s.overall_score, reverse=True
        )

        return {
            "best_estimator": ranked[0].estimator_name,
            "best_score": round(ranked[0].overall_score, 3),
            "ranking": [s.to_dict() for s in ranked],
            "n_estimators_evaluated": len(scores),
        }

    def _score_task_match(
        self,
        estimator_name: str,
        expected_task: str,
    ) -> float:
        """Check if estimator matches expected task type."""
        try:
            node = self._registry.get_estimator_by_name(estimator_name)
            if node is None:
                return 0.0
            return 1.0 if node.task == expected_task else 0.0
        except Exception as e:
            logger.warning(f"Could not check task match for {estimator_name}: {e}")
            return 0.0

    def _score_performance(
        self,
        cv_results: list[dict],
    ) -> tuple[float | None, float]:
        """
        Extract MAPE from cv_results and normalize to 0-1 score.

        Lower MAPE = higher score.
        MAPE of 0 = score 1.0
        MAPE of 100% or more = score 0.0
        """
        if not cv_results:
            return None, 0.0

        mape_values = []
        for fold in cv_results:
            for key, val in fold.items():
                if "MAPE" in key or "mape" in key:
                    try:
                        mape_values.append(float(val))
                    except (TypeError, ValueError):
                        pass

        if not mape_values:
            return None, 0.0

        avg_mape = sum(mape_values) / len(mape_values)
        # Normalize: MAPE=0 -> score=1.0, MAPE>=1.0 (100%) -> score=0.0
        performance_score = max(0.0, 1.0 - avg_mape)

        return avg_mape, performance_score