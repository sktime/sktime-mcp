"""Tests for evaluate_tool summary statistics."""

import sys

import pytest

sys.path.insert(0, "src")


class TestEvaluateSummary:
    """Tests for the summary statistics returned by evaluate_tool."""

    def test_evaluate_returns_summary_key(self):
        """evaluate_tool result must include a 'summary' key."""
        from sktime_mcp.tools.evaluate import evaluate_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool(spec="NaiveForecaster(strategy='last')")
        assert inst["success"], inst
        handle = inst["handle"]

        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=2)
        assert result["success"], result
        assert "summary" in result, "Expected 'summary' key in evaluate result"

    def test_summary_contains_expected_stat_keys(self):
        """Each metric in summary must have mean, std, min, max."""
        from sktime_mcp.tools.evaluate import evaluate_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool(spec="NaiveForecaster(strategy='last')")
        assert inst["success"], inst
        handle = inst["handle"]

        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=2)
        assert result["success"], result

        summary = result["summary"]
        assert isinstance(summary, dict)
        assert len(summary) > 0, "Summary should contain at least one metric"

        for metric_name, stats in summary.items():
            for key in ("mean", "std", "min", "max"):
                assert key in stats, f"Expected '{key}' in summary['{metric_name}'], got {stats}"
                assert isinstance(stats[key], float), f"{metric_name}.{key} not float"

    def test_summary_mean_between_min_and_max(self):
        """Sanity check: mean must be between min and max for each metric."""
        from sktime_mcp.tools.evaluate import evaluate_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool(spec="NaiveForecaster(strategy='last')")
        assert inst["success"], inst
        handle = inst["handle"]

        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=3)
        assert result["success"], result

        for metric_name, stats in result["summary"].items():
            assert stats["min"] <= stats["mean"] <= stats["max"], (
                f"Invariant violated for {metric_name}: "
                f"min={stats['min']}, mean={stats['mean']}, max={stats['max']}"
            )

    def test_fold_results_present(self):
        """The 'fold_results' list must contain per-fold metric dicts."""
        from sktime_mcp.tools.evaluate import evaluate_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool(spec="NaiveForecaster(strategy='last')")
        assert inst["success"], inst
        handle = inst["handle"]

        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=2)
        assert result["success"], result
        assert "fold_results" in result
        assert isinstance(result["fold_results"], list)
        assert len(result["fold_results"]) > 0
