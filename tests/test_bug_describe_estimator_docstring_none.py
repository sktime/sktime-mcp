"""Regression test for describe_estimator with missing docstring."""

from types import SimpleNamespace


def test_describe_estimator_handles_none_docstring(monkeypatch):
    """describe_estimator should not fail when node.docstring is None."""
    from sktime_mcp.tools import describe_estimator as module

    fake_node = SimpleNamespace(
        name="NoDocEstimator",
        task="forecasting",
        module="tests.fake",
        hyperparameters={},
        tags={},
        docstring=None,
    )

    class FakeRegistry:
        def get_estimator_by_name(self, estimator):
            if estimator == "NoDocEstimator":
                return fake_node
            return None

        def get_all_estimators(self):
            return [fake_node]

    class FakeTagResolver:
        def explain_tags(self, tags):
            return {}

    monkeypatch.setattr(module, "get_registry", lambda: FakeRegistry())
    monkeypatch.setattr(module, "get_tag_resolver", lambda: FakeTagResolver())

    result = module.describe_estimator_tool("NoDocEstimator")

    assert result["success"] is True
    assert result["docstring"] == "No description available."
