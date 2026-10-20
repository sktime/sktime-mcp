"""Tests for the describe_component tool."""

from types import SimpleNamespace

import pytest
from sktime.utils.dependencies import _check_soft_dependencies

import sktime_mcp.tools.describe_component as describe_component_module


class _MissingDependencyEstimator:
    """Minimal estimator class with known-missing soft dependencies."""

    @classmethod
    def get_class_tag(cls, tag_name, tag_value_default=None):
        tags = {
            "python_dependencies": [
                "sktime-mcp-missing-package>=1.0",
                ["sktime-mcp-missing-option-a", "sktime-mcp-missing-option-b"],
            ],
            "python_version": None,
            "env_marker": None,
        }
        return tags.get(tag_name, tag_value_default)


def test_describe_component_reports_missing_soft_dependencies(monkeypatch):
    """Known-missing requirements are reported with a usable install hint."""
    node = SimpleNamespace(
        name="MissingDependencyEstimator",
        task="forecaster",
        class_ref=_MissingDependencyEstimator,
        module="tests.MissingDependencyEstimator",
        parameters={},
        tags={},
        docstring="Test estimator.",
    )
    registry = SimpleNamespace(get_estimator_by_name=lambda name: node)
    tag_resolver = SimpleNamespace(explain_tags=lambda tags: {})
    monkeypatch.setattr(describe_component_module, "get_registry", lambda: registry)
    monkeypatch.setattr(describe_component_module, "get_tag_resolver", lambda: tag_resolver)

    result = describe_component_module.describe_component_tool(node.name)

    assert result["success"]
    assert not result["dependencies_available"]
    assert result["missing_dependencies"] == [
        "sktime-mcp-missing-package>=1.0",
        "sktime-mcp-missing-option-a or sktime-mcp-missing-option-b",
    ]
    assert result["install_hint"] == (
        'pip install "sktime-mcp-missing-package>=1.0" sktime-mcp-missing-option-a'
    )


def test_dependency_info_reports_available_component():
    """Components without soft dependencies require no install hint."""

    class NoDependencyEstimator(_MissingDependencyEstimator):
        @classmethod
        def get_class_tag(cls, tag_name, tag_value_default=None):
            return tag_value_default

    assert describe_component_module._dependency_info(NoDependencyEstimator) == {
        "dependencies_available": True,
        "missing_dependencies": [],
        "install_hint": None,
    }


def test_describe_prophet_reports_known_missing_soft_dependency():
    """A real registry component exposes its missing optional dependency."""
    if _check_soft_dependencies("prophet", severity="none"):
        pytest.skip("prophet is installed in this environment")

    result = describe_component_module.describe_component_tool("Prophet")

    assert result["success"]
    assert not result["dependencies_available"]
    assert result["missing_dependencies"] == ["prophet"]
    assert result["install_hint"] == "pip install prophet"
