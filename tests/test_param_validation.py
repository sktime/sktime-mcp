"""Tests for instantiate parameter validation."""

import sys

import pytest

sys.path.insert(0, "src")

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.tools.instantiate import instantiate_estimator_tool


def _requires_naive_forecaster() -> None:
    """Skip test when NaiveForecaster is unavailable in registry."""
    node = get_registry().get_estimator_by_name("NaiveForecaster")
    if node is None:
        pytest.skip("NaiveForecaster not available in current sktime registry")


def test_rejects_non_dict_params() -> None:
    """params must be dict or None."""
    with pytest.raises(TypeError, match="params must be a dict or None"):
        instantiate_estimator_tool("NaiveForecaster", params="bad")


def test_rejects_unsafe_param_value() -> None:
    """Unsafe/unsupported param value types should fail validation."""
    _requires_naive_forecaster()

    with pytest.raises(TypeError, match="unsupported parameter type"):
        instantiate_estimator_tool("NaiveForecaster", params={"strategy": object()})


def test_rejects_unknown_param_key() -> None:
    """Unknown parameter names should fail validation."""
    _requires_naive_forecaster()

    with pytest.raises(ValueError, match=r"Unknown parameter\(s\)"):
        instantiate_estimator_tool("NaiveForecaster", params={"not_a_real_param": 1})
