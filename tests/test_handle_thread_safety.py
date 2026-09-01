"""``HandleManager`` must serialise access to its handle store (#554).

The store is reachable from worker threads — ``fit_async`` runs ``fit`` in a
``ThreadPoolExecutor`` and ``fit`` writes via ``mark_fitted`` — while the
asyncio thread keeps serving other tool calls. Without a lock, two
``create_handle`` calls at the cap evict the same ids (``KeyError``) and an
eviction during ``list_handles`` breaks the iteration (``RuntimeError``).
"""

import contextlib
import itertools
import sys
import threading
import time

from sktime_mcp.runtime import handles as handles_mod
from sktime_mcp.runtime.handles import HandleManager


class _Est:
    """Minimal stand-in for a stored estimator instance."""


def test_cleanup_oldest_is_atomic():
    """Two threads evicting at the cap must not delete the same id twice."""
    mgr = HandleManager(max_handles=10)
    for _ in range(10):
        mgr.create_handle("Est", _Est())

    real_sorted = sorted
    barrier = threading.Barrier(2)

    def instrumented(*args, **kwargs):
        out = real_sorted(*args, **kwargs)
        # Both threads hold the same victim list before either deletes. Once
        # the section is locked this cannot be satisfied, and the barrier
        # timing out is itself the pass condition.
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=0.5)
        return out

    errors = []

    def worker():
        try:
            mgr.create_handle("Est", _Est())
        except Exception as exc:  # recorded, then asserted on below
            errors.append(exc)

    handles_mod.sorted = instrumented
    try:
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
    finally:
        del handles_mod.sorted

    assert not errors, f"concurrent eviction raised {errors[0]!r}"


def test_list_handles_survives_concurrent_eviction():
    """An eviction while ``list_handles`` iterates must not break the walk."""
    mgr = HandleManager(max_handles=10)
    for _ in range(10):
        mgr.create_handle("Est", _Est())

    started = threading.Event()
    calls = itertools.count()
    real_to_dict = handles_mod.HandleInfo.to_dict

    def slow_to_dict(self):
        if next(calls) == 0:
            started.set()
            time.sleep(0.3)  # hold the iteration open
        return real_to_dict(self)

    errors = []

    def lister():
        try:
            mgr.list_handles()
        except Exception as exc:  # recorded, then asserted on below
            errors.append(exc)

    def evictor():
        started.wait(timeout=2)
        try:
            mgr.create_handle("Est", _Est())  # trips _cleanup_oldest
        except Exception as exc:  # recorded, then asserted on below
            errors.append(exc)

    handles_mod.HandleInfo.to_dict = slow_to_dict
    try:
        threads = [threading.Thread(target=lister), threading.Thread(target=evictor)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
    finally:
        handles_mod.HandleInfo.to_dict = real_to_dict

    assert not errors, f"eviction during list_handles raised {errors[0]!r}"


def test_concurrent_handle_churn_is_clean():
    """Uninstrumented churn under a short switch interval must stay quiet."""
    mgr = HandleManager(max_handles=30)
    errors = []

    def worker():
        for _ in range(60):
            try:
                handle = mgr.create_handle("Est", _Est())
                mgr.mark_fitted(handle)
                mgr.list_handles()
                mgr.is_fitted(handle)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # expose interleavings the 5 ms default hides
    try:
        threads = [threading.Thread(target=worker) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        sys.setswitchinterval(previous)

    assert not errors, f"{len(errors)} errors, first was {errors[0]!r}"
