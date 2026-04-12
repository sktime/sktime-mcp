"""Demo script to validate BUG: describe_estimator with __doc__ = None."""

from types import SimpleNamespace

from sktime_mcp.tools import describe_estimator as module


class _FakeRegistry:
    """Minimal registry used to reproduce the bug scenario."""

    def __init__(self):
        self._node = SimpleNamespace(
            name="NoDocEstimator",
            task="forecasting",
            module="demo.no_doc_estimator",
            hyperparameters={"window_length": {"default": 3, "required": False}},
            tags={"capability:pred_int": False},
            docstring=None,
        )

    def get_estimator_by_name(self, estimator):
        if estimator == "NoDocEstimator":
            return self._node
        return None

    def get_all_estimators(self):
        return [self._node]


class _FakeTagResolver:
    """Minimal tag resolver for demo execution."""

    def explain_tags(self, tags):
        return {k: "demo tag" for k in tags}


def run_demo():
    """Execute the bug-check demo."""
    module.get_registry = lambda: _FakeRegistry()
    module.get_tag_resolver = lambda: _FakeTagResolver()

    result = module.describe_estimator_tool("NoDocEstimator")

    print("Demo result:")
    print(result)

    expected = "No description available."
    if result.get("docstring") != expected:
        raise AssertionError(
            f"Bug persists: expected docstring={expected!r}, got {result.get('docstring')!r}"
        )

    print("\nPASS: describe_estimator handles missing docstring gracefully.")


if __name__ == "__main__":
    run_demo()
