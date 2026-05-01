"""
Tests that fit_predict returns JSON-serializable native Python types.

Fixes: numpy.float64 values caused json.dumps() to fail in MCP responses.
"""

import json

import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager


class TestNumpySerialization:

    def test_predictions_are_json_serializable(self):
        executor = get_executor()
        handle_manager = get_handle_manager()

        inst_result = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        assert inst_result["success"], f"Failed to instantiate: {inst_result}"
        handle = inst_result["handle"]

        try:
            result = executor.fit_predict(handle, "airline", horizon=6)
            assert result["success"], f"fit_predict failed: {result.get('error')}"

            predictions = result["predictions"]
            assert len(predictions) > 0

            # Must not raise TypeError (before fix this failed)
            try:
                json_output = json.dumps(predictions)
                assert isinstance(json_output, str)
            except TypeError as e:
                pytest.fail(f"Predictions are not JSON serializable: {e}")

            # Values must be native Python types, not numpy scalars
            for key, value in predictions.items():
                assert type(value) in (int, float), (
                    f"Value {key}={value} is {type(value).__name__}, "
                    f"expected int or float"
                )
                assert not hasattr(value, "dtype"), (
                    f"Value {key}={value} is a numpy type, "
                    f"should be native Python"
                )
        finally:
            handle_manager.release_handle(handle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
