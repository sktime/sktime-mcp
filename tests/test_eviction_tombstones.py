"""Evicted handles must report "evicted", not an indistinguishable "not found" (#532).

Both handle stores bulk-evict the oldest entries at their cap. Previously the
caller only saw a generic "not found" — identical to a typo. Now evicted ids
are tombstoned and the error says so.
"""

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import Executor
from sktime_mcp.runtime.handles import HandleManager
from sktime_mcp.tools.inspect_data import inspect_data_tool


class _Est:
    """Minimal stand-in with the attributes create_handle stores."""


def test_estimator_eviction_message():
    hm = HandleManager(max_handles=5)
    ids = [hm.create_handle("Dummy", _Est(), {}) for _ in range(5)]
    # next create triggers _cleanup_oldest (removes 10, i.e. all present)
    hm.create_handle("Dummy", _Est(), {})
    first = ids[0]
    assert hm.was_evicted(first)
    msg = hm.describe_missing(first)
    assert "evicted" in msg.lower()
    assert "limit 5" in msg
    # a never-seen id is still a plain not-found
    assert "not found" in hm.describe_missing("est_never").lower()
    assert "evicted" not in hm.describe_missing("est_never").lower()


def test_get_instance_raises_eviction_message():
    hm = HandleManager(max_handles=3)
    ids = [hm.create_handle("Dummy", _Est(), {}) for _ in range(3)]
    hm.create_handle("Dummy", _Est(), {})
    with pytest.raises(KeyError) as exc:
        hm.get_instance(ids[0])
    assert "evicted" in str(exc.value).lower()


def test_data_handle_eviction_message(monkeypatch):
    ex = Executor()
    ex._max_data_handles = 5

    def _mk(i):
        idx = pd.period_range("2020-01", periods=6, freq="M")
        hid = f"data_test_{i:02d}"
        ex._register_data_handle(hid, {"y": pd.Series(range(6), index=idx), "X": None,
                                       "metadata": {}, "validation": {}, "config": {}})
        return hid

    ids = [_mk(i) for i in range(12)]  # far past the cap -> oldest evicted
    evicted = [h for h in ids if h in ex._evicted_data]
    assert evicted, "expected some handles to be evicted past the cap"

    body = ex.data_handle_missing(evicted[0])
    assert "evicted" in body["error"].lower()
    assert "n_available_handles" in body

    unknown = ex.data_handle_missing("data_never")
    assert "not found" in unknown["error"].lower()
    assert "evicted" not in unknown["error"].lower()


def test_inspect_data_surfaces_eviction():
    from sktime_mcp.runtime.executor import get_executor

    ex = get_executor()
    ex._max_data_handles = 5
    idx = pd.period_range("2020-01", periods=6, freq="M")
    ids = []
    for i in range(12):
        hid = f"data_evict_{i:02d}"
        ex._register_data_handle(hid, {"y": pd.Series(range(6), index=idx), "X": None,
                                       "metadata": {}, "validation": {}, "config": {}})
        ids.append(hid)
    evicted = next(h for h in ids if h in ex._evicted_data)
    res = inspect_data_tool(data_handle=evicted)
    assert res["success"] is False
    assert "evicted" in res["error"].lower()
    for h in ids:
        ex._data_handles.pop(h, None)
