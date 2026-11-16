"""Coroutines must use ``asyncio.get_running_loop``, not ``get_event_loop`` (#65).

The maintainer triage (2026-09-06) narrowed #65 to three coroutines that
still fetched the loop with ``asyncio.get_event_loop()``. Inside a coroutine
a loop is always running, so ``get_running_loop()`` is the supported call;
``get_event_loop()`` is deprecated and fragile across Python versions.
"""

import ast
import warnings
from pathlib import Path

import pandas as pd

from sktime_mcp.data.base import DataSourceAdapter

SRC = Path(__file__).resolve().parent.parent / "src" / "sktime_mcp"
KNOWN_SITES = [
    SRC / "data" / "base.py",
    SRC / "data" / "adapters" / "url_adapter.py",
    SRC / "runtime" / "executor.py",
]


def _loop_calls(path: Path) -> list:
    """Return the ``asyncio.get_*_loop`` attribute calls made in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
            if (
                isinstance(target, ast.Name)
                and target.id == "asyncio"
                and node.func.attr in ("get_event_loop", "get_running_loop")
            ):
                calls.append(node.func.attr)
    return calls


def test_no_get_event_loop_in_known_coroutines():
    """Each known coroutine must fetch the loop with get_running_loop."""
    for path in KNOWN_SITES:
        calls = _loop_calls(path)
        assert calls, f"expected asyncio loop access in {path.name}"
        assert "get_event_loop" not in calls, f"{path.name} still uses get_event_loop"


def test_no_get_event_loop_anywhere_in_src():
    """Guard against the deprecated call creeping back into the package."""
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "get_event_loop" in _loop_calls(path)
    ]
    assert not offenders, f"get_event_loop used in: {offenders}"


class _StubAdapter(DataSourceAdapter):
    """Minimal adapter exercising the default async load implementation."""

    def load(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"value": [1.0, 2.0, 3.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

    def validate(self, data: pd.DataFrame):
        return True, {"valid": True, "errors": [], "warnings": []}


async def test_load_async_runs_on_running_loop():
    """Default load_async resolves through the running loop, warning-free."""
    adapter = _StubAdapter({"type": "stub"})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = await adapter.load_async()
    pd.testing.assert_frame_equal(result, adapter.load())
