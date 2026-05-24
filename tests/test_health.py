"""Tests for the get_server_health MCP tool."""

import platform

from sktime_mcp.tools.health import get_server_health_tool


class TestGetServerHealthTool:
    """Test suite for get_server_health_tool."""

    def test_returns_success(self):
        """Health check should always succeed."""
        result = get_server_health_tool()
        assert result["success"] is True

    def test_contains_version_info(self):
        """Should include server, Python, and sktime versions."""
        result = get_server_health_tool()
        assert "server_version" in result
        assert "python_version" in result
        assert result["python_version"] == platform.python_version()
        assert "sktime_version" in result

    def test_contains_platform(self):
        """Should include platform info."""
        result = get_server_health_tool()
        assert "platform" in result

    def test_contains_uptime(self):
        """Should include uptime_seconds as a non-negative number."""
        result = get_server_health_tool()
        assert "uptime_seconds" in result
        assert isinstance(result["uptime_seconds"], (int, float))
        assert result["uptime_seconds"] >= 0

    def test_contains_handle_counts(self):
        """Should report estimator handle counts."""
        result = get_server_health_tool()
        assert "active_estimator_handles" in result
        assert "max_estimator_handles" in result

    def test_contains_data_handle_count(self):
        """Should report data handle count."""
        result = get_server_health_tool()
        assert "active_data_handles" in result

    def test_contains_job_summary(self):
        """Should report job summary statistics."""
        result = get_server_health_tool()
        assert "job_summary" in result
        assert "total_jobs" in result

    def test_contains_optional_dependencies(self):
        """Should report availability of optional packages."""
        result = get_server_health_tool()
        assert "optional_dependencies" in result
        deps = result["optional_dependencies"]
        assert isinstance(deps, dict)
        assert "numpy" in deps
        assert "pandas" in deps
        assert "scipy" in deps
        assert "matplotlib" in deps
        assert "mlflow" in deps
        # numpy and pandas should be available since sktime depends on them
        assert deps["numpy"] is True
        assert deps["pandas"] is True

    def test_handle_count_increases_after_instantiation(self):
        """Creating a handle should increment the count."""
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        before = get_server_health_tool()
        before_count = before["active_estimator_handles"]

        inst = instantiate_estimator_tool("NaiveForecaster")
        assert inst["success"]

        after = get_server_health_tool()
        after_count = after["active_estimator_handles"]
        assert after_count >= before_count + 1

    def test_idempotent(self):
        """Calling health check multiple times should not change state."""
        r1 = get_server_health_tool()
        r2 = get_server_health_tool()
        assert r1["success"] == r2["success"]
        assert r1["python_version"] == r2["python_version"]
        assert r1["sktime_version"] == r2["sktime_version"]
