"""
Tests for MCP tool input schema constraints.

Verifies:
- task parameter in list_estimators has an enum
- cv_folds in evaluate_estimator has minimum and maximum bounds
"""

import pytest
from sktime_mcp.server import list_tools

# requires: pytest-asyncio


class TestSchemaConstraints:

    @pytest.fixture
    async def tools(self):
        return {t.name: t for t in await list_tools()}

    @pytest.mark.asyncio
    async def test_list_estimators_task_has_enum(self, tools):
        assert "list_estimators" in tools

        task_param = tools["list_estimators"].inputSchema["properties"]["task"]
        assert "enum" in task_param, (
            "task parameter must have enum to guide LLM clients"
        )
        assert len(task_param["enum"]) > 0, "enum must not be empty"
        assert "forecasting" in task_param["enum"]

    @pytest.mark.asyncio
    async def test_evaluate_estimator_cv_folds_has_bounds(self, tools):
        assert "evaluate_estimator" in tools

        cv_folds = (
            tools["evaluate_estimator"].inputSchema["properties"]["cv_folds"]
        )
        assert "minimum" in cv_folds, "cv_folds must have a minimum"
        assert "maximum" in cv_folds, "cv_folds must have a maximum"
        assert cv_folds["minimum"] >= 1, "minimum must be at least 1"
        assert cv_folds["maximum"] > cv_folds["minimum"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
