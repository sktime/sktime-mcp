"""
Tests for list_metrics_tool and compute_metric_tool.

Covers:
- list_metrics_tool returns all expected metrics with correct structure
- compute_metric_tool basic correctness for common metrics
- compute_metric_tool error handling (unknown metric, length mismatch, missing y_train)
- compute_metric_tool with y_train for MASE / RMSSE
"""

import importlib
import sys

import pytest

sys.path.insert(0, "src")

# compute_metric_tool calls sktime's metric classes at runtime — skip those
# tests when sktime is not installed (same behaviour as test_core.py).
_SKTIME_AVAILABLE = importlib.util.find_spec("sktime") is not None
_SKIP_NO_SKTIME = pytest.mark.skipif(
    not _SKTIME_AVAILABLE,
    reason="sktime is not installed",
)


class TestListMetricsTool:
    """Tests for list_metrics_tool."""

    def test_returns_success(self):
        """list_metrics should always succeed."""
        from sktime_mcp.tools.metrics import list_metrics_tool

        result = list_metrics_tool()

        assert result["success"] is True

    def test_returns_metrics_list(self):
        """Result must contain a non-empty 'metrics' list."""
        from sktime_mcp.tools.metrics import list_metrics_tool

        result = list_metrics_tool()

        assert "metrics" in result
        assert isinstance(result["metrics"], list)
        assert len(result["metrics"]) > 0

    def test_count_matches_metrics_length(self):
        """'count' must match len(metrics)."""
        from sktime_mcp.tools.metrics import list_metrics_tool

        result = list_metrics_tool()

        assert result["count"] == len(result["metrics"])

    def test_each_metric_has_required_keys(self):
        """Every metric descriptor must have name, description, lower_is_better, scale_dependent."""
        from sktime_mcp.tools.metrics import list_metrics_tool

        result = list_metrics_tool()

        required_keys = {"name", "description", "lower_is_better", "scale_dependent", "requires_y_train"}
        for metric in result["metrics"]:
            assert required_keys.issubset(metric.keys()), (
                f"Metric {metric.get('name')} missing keys: "
                f"{required_keys - set(metric.keys())}"
            )

    def test_standard_metrics_present(self):
        """Key metrics (MAE, RMSE, MAPE, MASE) must be discoverable."""
        from sktime_mcp.tools.metrics import list_metrics_tool

        result = list_metrics_tool()
        names = {m["name"] for m in result["metrics"]}

        for expected in ["MAE", "RMSE", "MAPE", "MASE"]:
            assert expected in names, f"Expected metric {expected} not in list_metrics output"

    def test_mase_requires_y_train(self):
        """MASE must flag requires_y_train=True."""
        from sktime_mcp.tools.metrics import list_metrics_tool

        result = list_metrics_tool()
        mase = next(m for m in result["metrics"] if m["name"] == "MASE")

        assert mase["requires_y_train"] is True

    def test_mae_not_scale_normalized(self):
        """MAE is scale-dependent and does not require y_train."""
        from sktime_mcp.tools.metrics import list_metrics_tool

        result = list_metrics_tool()
        mae = next(m for m in result["metrics"] if m["name"] == "MAE")

        assert mae["scale_dependent"] is True
        assert mae["requires_y_train"] is False

    def test_usage_hint_present(self):
        """A usage_hint string must be included."""
        from sktime_mcp.tools.metrics import list_metrics_tool

        result = list_metrics_tool()

        assert "usage_hint" in result
        assert isinstance(result["usage_hint"], str)
        assert len(result["usage_hint"]) > 0


@_SKIP_NO_SKTIME
class TestComputeMetricTool:
    """Tests for compute_metric_tool."""

    # ------------------------------------------------------------------
    # Correct computation tests
    # ------------------------------------------------------------------

    def test_mae_exact_value(self):
        """MAE([100, 110], [105, 108]) == 3.5."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("MAE", y_true=[100.0, 110.0], y_pred=[105.0, 108.0])

        assert result["success"] is True
        assert result["metric"] == "MAE"
        assert abs(result["value"] - 3.5) < 1e-9

    def test_rmse_exact_value(self):
        """RMSE([0, 4], [0, 0]) == 2.0."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("RMSE", y_true=[0.0, 4.0], y_pred=[0.0, 0.0])

        assert result["success"] is True
        assert abs(result["value"] - 2.0) < 1e-6

    def test_mse_exact_value(self):
        """MSE([0, 4], [0, 0]) == 8.0."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("MSE", y_true=[0.0, 4.0], y_pred=[0.0, 0.0])

        assert result["success"] is True
        assert abs(result["value"] - 8.0) < 1e-6

    def test_perfect_prediction_mae_zero(self):
        """Perfect predictions → MAE == 0."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("MAE", y_true=[1.0, 2.0, 3.0], y_pred=[1.0, 2.0, 3.0])

        assert result["success"] is True
        assert abs(result["value"]) < 1e-9

    def test_smape_zero_for_perfect(self):
        """SMAPE of a perfect forecast should be 0."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("SMAPE", y_true=[100.0, 200.0], y_pred=[100.0, 200.0])

        assert result["success"] is True
        assert abs(result["value"]) < 1e-6

    def test_case_insensitive_metric_name(self):
        """Metric name lookup must be case-insensitive ('mae' == 'MAE')."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("mae", y_true=[1.0, 2.0], y_pred=[1.0, 2.0])

        assert result["success"] is True
        assert result["metric"] == "MAE"

    def test_lower_is_better_flag_returned(self):
        """lower_is_better must be present in every successful result."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("RMSE", y_true=[1.0, 2.0], y_pred=[1.0, 2.0])

        assert "lower_is_better" in result
        assert result["lower_is_better"] is True

    def test_mase_with_y_train(self):
        """MASE returns success when y_train is provided."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        y_train = list(range(1, 13))  # 12 points
        y_true = [13.0, 14.0, 15.0]
        y_pred = [12.5, 13.5, 14.5]

        result = compute_metric_tool("MASE", y_true=y_true, y_pred=y_pred, y_train=y_train)

        assert result["success"] is True
        assert isinstance(result["value"], float)

    def test_rmsse_with_y_train(self):
        """RMSSE returns success when y_train is provided."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        y_train = list(range(1, 13))
        y_true = [13.0, 14.0]
        y_pred = [12.0, 13.0]

        result = compute_metric_tool("RMSSE", y_true=y_true, y_pred=y_pred, y_train=y_train)

        assert result["success"] is True

    # ------------------------------------------------------------------
    # Error-handling tests
    # ------------------------------------------------------------------

    def test_unknown_metric_returns_error(self):
        """An unknown metric name must return success=False with an informative error."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("NOTAMETRIC", y_true=[1.0], y_pred=[1.0])

        assert result["success"] is False
        assert "error" in result
        assert "available_metrics" in result
        assert len(result["available_metrics"]) > 0

    def test_length_mismatch_returns_error(self):
        """y_true and y_pred of different lengths must return success=False."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("MAE", y_true=[1.0, 2.0, 3.0], y_pred=[1.0, 2.0])

        assert result["success"] is False
        assert "error" in result
        assert "length" in result["error"].lower()

    def test_mase_without_y_train_returns_error(self):
        """MASE without y_train must return success=False."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("MASE", y_true=[1.0, 2.0], y_pred=[1.0, 2.0])

        assert result["success"] is False
        assert "y_train" in result["error"]

    def test_rmsse_without_y_train_returns_error(self):
        """RMSSE without y_train must return success=False."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("RMSSE", y_true=[1.0, 2.0], y_pred=[1.0, 2.0])

        assert result["success"] is False
        assert "y_train" in result["error"]

    def test_unknown_metric_lists_available_options(self):
        """Error response for unknown metric must include available metric names."""
        from sktime_mcp.tools.metrics import compute_metric_tool

        result = compute_metric_tool("ZZZNOPE", y_true=[1.0], y_pred=[1.0])

        assert not result["success"]
        assert "MAE" in result["available_metrics"]
        assert "RMSE" in result["available_metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
