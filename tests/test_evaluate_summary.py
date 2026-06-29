"""Tests for evaluate_estimator_tool summary statistics."""

import sys

import pytest

sys.path.insert(0, "src")


class TestEvaluateSummary:
    """Tests for the summary statistics added to evaluate_estimator_tool."""

    def test_evaluate_returns_summary_key(self):
        """evaluate_estimator_tool result must include a 'summary' key."""
        from sktime_mcp.tools.evaluate import evaluate_estimator_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        result = evaluate_estimator_tool(handle, "airline", cv_folds=2)
        assert result["success"], result
        assert "summary" in result, "Expected 'summary' key in evaluate result"

    def test_summary_contains_expected_stat_keys(self):
        """Each metric in summary must have mean, std, min, max."""
        from sktime_mcp.tools.evaluate import evaluate_estimator_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        result = evaluate_estimator_tool(handle, "airline", cv_folds=2)
        assert result["success"], result

        summary = result["summary"]
        assert isinstance(summary, dict)
        assert len(summary) > 0, "Summary should contain at least one metric"

        for metric_name, stats in summary.items():
            for key in ("mean", "std", "min", "max"):
                assert key in stats, f"Expected '{key}' in summary['{metric_name}'], got {stats}"
                assert isinstance(stats[key], float), (
                    f"summary['{metric_name}']['{key}'] should be float, got {type(stats[key])}"
                )

    def test_summary_mean_between_min_and_max(self):
        """Sanity check: mean must be between min and max for each metric."""
        from sktime_mcp.tools.evaluate import evaluate_estimator_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        result = evaluate_estimator_tool(handle, "airline", cv_folds=3)
        assert result["success"], result

        for metric_name, stats in result["summary"].items():
            assert stats["min"] <= stats["mean"] <= stats["max"], (
                f"Invariant violated for {metric_name}: "
                f"min={stats['min']}, mean={stats['mean']}, max={stats['max']}"
            )

    def test_raw_results_still_present(self):
        """The existing 'results' key must still be returned alongside 'summary'."""
        from sktime_mcp.tools.evaluate import evaluate_estimator_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        result = evaluate_estimator_tool(handle, "airline", cv_folds=2)
        assert result["success"], result
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0
