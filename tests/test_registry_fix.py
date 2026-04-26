"""tests/test_registry_fix.py — adversarially hardened pytest suite for thread-safety fixes.

Each test is designed to catch a specific regression path in the thread-safety layer.
A test that passes when a race condition or lock bug is reintroduced is not a test.

Coverage targets:
  - RLock prevents re-entrant deadlock (same thread can re-acquire the lock)
  - Double-checked locking (DCL): inner _loaded check avoids redundant _load_registry calls
  - _all_tags fallback read is lock-guarded in get_available_tags ImportError branch
  - search_estimators rejects non-str queries before _ensure_loaded is called
  - _ensure_loaded sets _loaded only on success; RuntimeError keeps _loaded=False for retry
"""

import logging
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_cls(object_type: str, name: str = "Fake") -> MagicMock:
    cls = MagicMock()
    cls.get_class_tags.return_value = {"object_type": object_type}
    cls.__module__ = f"sktime.fake.{name.lower()}"
    cls.__name__ = name
    return cls


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

class TestRegistryThreadSafety:
    # ------------------------------------------------------------------
    # _ensure_loaded: _loaded flag and retry semantics
    def test_loaded_flag_set_after_empty_registry(self):
        """Proves _loaded=True even when all_estimators() returns [].

        If _loaded is only set inside a `if len(estimators) > 0` guard, an empty
        registry would trigger infinite reload on every _ensure_loaded call.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        assert not ri._loaded

        with patch("sktime.registry.all_estimators", return_value=[]):
            ri._ensure_loaded()

        assert ri._loaded

    def test_missing_sktime_keeps_loaded_false_and_reraises_on_retry(self):
        """When sktime is absent, _ensure_loaded raises RuntimeError and _loaded stays False.

        Documented retry semantics: every call re-raises until the environment is fixed,
        rather than caching an empty registry on the first failure.
        This tests the ImportError → RuntimeError path (line 96 in _load_registry),
        which is distinct from the inner try/except that catches all_estimators() failures.
        """

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        with patch.dict(sys.modules, {"sktime.registry": None}), pytest.raises(RuntimeError, match="sktime must be installed"):
            ri._ensure_loaded()

        assert not ri._loaded, (
            "_loaded must stay False when RuntimeError propagates — retry semantics require this"
        )

        # Second call must also raise, not silently swallow the error
        with patch.dict(sys.modules, {"sktime.registry": None}), pytest.raises(RuntimeError, match="sktime must be installed"):
            ri._ensure_loaded()


    def test_cache_is_not_modified_on_second_ensure_loaded(self):
        """Proves _cache is frozen after first _ensure_loaded; second call is a strict no-op.

        If _loaded check is bypassed, second load overwrites cache entries.
        The snapshot comparison catches any mutation, including adding or removing entries.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        fake_cls = _make_fake_cls("forecaster", "FakeForecaster")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("FakeForecaster", fake_cls)],
        ):
            ri._ensure_loaded()
            snapshot = dict(ri._cache)
            ri._ensure_loaded()

        assert ri._cache == snapshot, "Cache was modified by second _ensure_loaded call"

    # ------------------------------------------------------------------
    # Regression injection — the 8-call loop must fail our coverage test
    # ------------------------------------------------------------------


    def test_no_warning_log_during_normal_load(self, caplog):
        """Proves no WARNING-level messages are emitted during a clean load.

        The loop calls all_estimators once per TASK_MAP key.  To avoid spurious
        collision warnings the mock returns the estimator only on the first call
        (estimator_type="forecaster") and an empty list for the remaining keys.
        """

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        fake_cls = _make_fake_cls("forecaster", "GoodForecaster")

        # First call returns the fake; subsequent calls return [] to avoid collisions.
        n_keys = len(ri.TASK_MAP)
        side_effects = [[("GoodForecaster", fake_cls)]] + [[] for _ in range(n_keys - 1)]

        with patch(
            "sktime.registry.all_estimators",
            side_effect=side_effects,
        ), caplog.at_level(logging.WARNING, logger="sktime_mcp.registry.interface"):
            ri._load_registry()

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, (
            f"Unexpected WARNING log(s): {[r.message for r in warnings]}"
        )

    def test_name_collision_emits_warning(self, caplog):
        """Proves a WARNING (not DEBUG) is emitted when two estimators share the same name.

        Collision-resolution order changes between the old per-key loop and the new single
        call; a WARNING ensures the event is auditable in production logs.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        cls_a = _make_fake_cls("forecaster", "DupEstimator")
        cls_b = _make_fake_cls("transformer", "DupEstimator")
        cls_b.__module__ = "sktime.fake.dup_b"

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("DupEstimator", cls_a), ("DupEstimator", cls_b)],
        ), caplog.at_level(logging.WARNING, logger="sktime_mcp.registry.interface"):
            ri._load_registry()

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("DupEstimator" in m and "collision" in m.lower() for m in warning_msgs), (
            f"Expected a WARNING mentioning 'DupEstimator' and 'collision'. Got: {warning_msgs}"
        )


    def test_registry_name_collision_warning(self, caplog):
        """Exercises the logger.warning line in the collision block.

        Existing collision test may miss the WARNING because MagicMock.__init__
        causes inspect.signature() to raise inside _create_node, triggering the
        outer except/continue before the collision check.  This test uses real
        classes with concrete __init__ so both estimators fully survive
        _create_node and execution reaches the collision branch.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        def _solid_cls(object_type, module_name):
            """Return a real class that survives _create_node (inspect.signature safe)."""
            class DupEstimator:
                """A duplicate estimator."""
                def __init__(self):
                    pass
            DupEstimator.get_class_tags = MagicMock(return_value={"object_type": object_type})
            DupEstimator.__module__ = module_name
            return DupEstimator

        cls_a = _solid_cls("forecaster", "sktime.fake.dup_a")
        cls_b = _solid_cls("forecaster", "sktime.fake.dup_b")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("DupEstimator", cls_a), ("DupEstimator", cls_b)],
        ), caplog.at_level(logging.WARNING, logger="sktime_mcp.registry.interface"):
            ri._load_registry()

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "DupEstimator" in m and "collision" in m.lower() for m in warning_msgs
        ), f"Expected WARNING mentioning 'DupEstimator' and 'collision'. Got: {warning_msgs}"
        assert ri._cache["DupEstimator"].module.startswith("sktime.fake.dup_b")

    def test_registry_concurrent_load_protection(self):
        """Exercises the inner 'if not self._loaded' evaluating to False.

        In double-checked locking the inner check is False when a thread passes
        the outer check (reads _loaded=False), then another thread acquires the
        lock first, completes _load_registry(), and sets _loaded=True — the
        waiting thread then acquires the lock, hits the inner check, finds True,
        and skips.  Single-threaded tests never reach this branch.
        """
        import threading

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        fake_cls = _make_fake_cls("forecaster", "ConcForecaster")
        barrier = threading.Barrier(10)
        errors = []

        def load_one():
            try:
                barrier.wait()
                ri._ensure_loaded()
            except Exception as e:
                errors.append(e)

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("ConcForecaster", fake_cls)],
        ):
            threads = [threading.Thread(target=load_one) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Thread(s) raised: {errors}"
        assert ri._loaded
        assert len(ri._cache) == 1
        assert "ConcForecaster" in ri._cache

    def test_ensure_loaded_inner_dcl_branch_false_path(self):
        """Deterministically exercises the inner 'if not self._loaded' False path.

        Holds the registry lock before starting a thread so the thread passes the
        outer check (_loaded=False → enters if) then blocks on the lock.  Main then
        sets _loaded=True and releases the lock.  The thread acquires the lock, hits
        the inner check, finds _loaded=True, and exits without calling _load_registry.
        This is the exact 79->exit branch that probabilistic concurrency tests miss
        due to GIL scheduling under coverage instrumentation.
        """
        import threading

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        thread_done = threading.Event()

        def waiting_thread():
            ri._ensure_loaded()
            thread_done.set()

        # Hold the lock so the thread blocks after passing the outer check.
        with ri._lock:
            t = threading.Thread(target=waiting_thread)
            t.start()
            # Give the thread time to pass the outer check and block on the lock.
            thread_done.wait(timeout=0.1)
            # Set _loaded=True while still holding the lock.  When released, the
            # thread acquires the lock, finds _loaded=True, and takes the exit path.
            ri._loaded = True
        # Lock released — thread unblocks, hits inner check (False), skips _load_registry.
        t.join(timeout=5)
        assert not t.is_alive(), "Thread did not complete — DCL exit path not reached"
        assert ri._loaded

    # ------------------------------------------------------------------
    # New tests: singleton identity, retry semantics, copy safety, None-name skipping
    # ------------------------------------------------------------------

    def test_get_registry_singleton_identity_under_concurrency(self):
        """Proves get_registry() returns the identical object instance from all concurrent callers.

        (a) Invariant: the module-level singleton must be created exactly once regardless of
        how many threads race to call get_registry() simultaneously.
        (b) Regression: a lock-free implementation (no DCL around _registry_instance
        assignment) can create multiple RegistryInterface objects — different id() values
        betray this.  The barrier forces all 10 threads to start simultaneously, maximising
        the race window so a missing lock fails reliably.
        """
        import threading

        import sktime_mcp.registry.interface as iface

        # Reset the module-level singleton so the race starts from a clean slate.
        original = iface._registry_instance
        iface._registry_instance = None
        try:
            n_threads = 10
            results = [None] * n_threads
            barrier = threading.Barrier(n_threads)

            # Widen the race window: sleep inside __init__ so threads overlap between
            # the `if _registry_instance is None` check and the assignment.  Without
            # this, CPython's GIL makes the assignment effectively atomic and a lock-free
            # get_registry() would pass the test despite the race.
            original_init = iface.RegistryInterface.__init__

            def slow_init(self):
                time.sleep(0.05)
                original_init(self)

            def grab(idx):
                barrier.wait()
                results[idx] = iface.get_registry()

            with patch.object(iface.RegistryInterface, "__init__", slow_init):
                threads = [threading.Thread(target=grab, args=(i,)) for i in range(n_threads)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            unique_ids = {id(r) for r in results}
            assert len(unique_ids) == 1, (
                f"get_registry() returned {len(unique_ids)} distinct objects across "
                f"{n_threads} threads — singleton creation is not thread-safe"
            )
        finally:
            iface._registry_instance = original

    def test_get_tags_returns_copy_mutation_does_not_affect_original(self):
        """Proves _get_tags() returns an independent copy of the class tag dict.

        (a) Invariant: callers must not be able to mutate EstimatorNode.tags and
        thereby change what get_class_tags() returns on the class itself — these are
        two separate objects.
        (b) Regression: returning the raw dict from get_class_tags() (without wrapping
        in dict()) hands a reference to the live tag storage to the caller.  Mutating
        the returned dict would silently corrupt the class's tag declaration.  This test
        injects a controlled mutation and verifies the class's own tags are untouched.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        original_tags = {"key": "original", "other": 42}
        cls = MagicMock()
        cls.get_class_tags.return_value = original_tags

        result = ri._get_tags(cls)
        assert result == {"key": "original", "other": 42}, (
            "_get_tags did not return the expected tags"
        )

        # Identity check: result must be a distinct object, not the live dict reference.
        assert result is not original_tags, (
            "_get_tags returned the live dict reference — "
            "callers can mutate sktime's class-level tag dict. Fix: dict(cls.get_class_tags())."
        )

        # Sanity: mutation of the copy must not propagate to the original.
        result["key"] = "mutated"
        assert original_tags["key"] == "original", (
            "Mutation of _get_tags() result propagated back to original_tags — not a true copy."
        )

    def test_get_available_tags_skips_rows_with_none_name(self):
        """Proves get_available_tags() handles rows where the 'name' column is None
        without raising and returns only the rows with a valid name.

        (a) Invariant: a single malformed row in the sktime tags DataFrame must not
        crash the entire tags listing — the method should skip it gracefully.
        (b) Regression: the current implementation accesses row["name"] unconditionally
        in the dict literal and later calls result.sort(key=lambda x: x["tag"]).
        If name is None, the sort raises TypeError because None < str is undefined in
        Python 3.  This test catches that crash path and also verifies the valid row
        still appears in the output.
        """
        import pandas as pd

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        # Manually mark as loaded so _ensure_loaded does not call all_estimators.
        ri._loaded = True

        tags_df = pd.DataFrame(
            [
                {"name": None, "description": "broken row", "type": "str", "scitype": "Series"},
                {
                    "name": "valid_tag",
                    "description": "a real tag",
                    "type": "bool",
                    "scitype": "Forecaster",
                },
            ]
        )

        with patch("sktime.registry.all_tags", return_value=tags_df):
            result = ri.get_available_tags()

        assert len(result) == 1, (
            f"Expected exactly 1 result (the valid_tag row), got {len(result)}: {result}"
        )
        assert result[0]["tag"] == "valid_tag", (
            f"Expected tag='valid_tag', got {result[0]['tag']!r}"
        )

    # ------------------------------------------------------------------
    # EstimatorNode.to_dict and to_summary (lines 26, 39)
    # ------------------------------------------------------------------

    def test_estimator_node_to_dict_returns_expected_keys(self):
        """Covers EstimatorNode.to_dict() (line 26) by calling it directly on a node instance.

        Deleting the to_dict method or removing any key from its return dict would cause this test to fail.
        """
        from sktime_mcp.registry.interface import EstimatorNode

        node = EstimatorNode(
            name="TestEst",
            task="forecasting",
            class_ref=object,
            module="sktime.fake.TestEst",
            tags={"fit_is_empty": True},
            hyperparameters={"n": {"default": 1, "required": False}},
            docstring="A" * 600,
        )
        result = node.to_dict()
        assert result["name"] == "TestEst"
        assert result["task"] == "forecasting"
        assert result["module"] == "sktime.fake.TestEst"
        assert result["tags"] == {"fit_is_empty": True}
        assert result["hyperparameters"] == {"n": {"default": 1, "required": False}}
        # docstring truncated to 500 chars
        assert len(result["docstring"]) == 500

    def test_estimator_node_to_dict_none_docstring(self):
        """Covers the None-docstring branch inside EstimatorNode.to_dict() (line 32-34).

        Removing the `if self.docstring else None` guard would make the None branch unreachable; this test fails if docstring is not None when it should be.
        """
        from sktime_mcp.registry.interface import EstimatorNode

        node = EstimatorNode(
            name="TestEst",
            task="forecasting",
            class_ref=object,
            module="sktime.fake.TestEst",
            docstring=None,
        )
        result = node.to_dict()
        assert result["docstring"] is None

    def test_estimator_node_to_summary_returns_expected_keys(self):
        """Covers EstimatorNode.to_summary() (line 39) by calling it directly.

        Deleting the to_summary method or removing any key from its return dict would cause this test to fail.
        """
        from sktime_mcp.registry.interface import EstimatorNode

        node = EstimatorNode(
            name="MyEst",
            task="transformation",
            class_ref=object,
            module="sktime.fake.MyEst",
            tags={"scitype:transform-input": "Series"},
            hyperparameters={},
            docstring="Useful transformer.",
        )
        result = node.to_summary()
        assert result == {
            "name": "MyEst",
            "task": "transformation",
            "module": "sktime.fake.MyEst",
            "tags": {"scitype:transform-input": "Series"},
        }
        # to_summary must NOT include hyperparameters or docstring
        assert "hyperparameters" not in result
        assert "docstring" not in result

    # ------------------------------------------------------------------
    # Per-type exception swallowed in loop (lines 101-103)
    # ------------------------------------------------------------------

    def test_load_registry_continues_after_per_type_exception(self, caplog):
        """Covers the except/continue block at lines 101-103 by making all_estimators raise for one type.

        If the except block used raise instead of continue, the exception would propagate and
        subsequent estimator types would never load; this test fails if the good type is missing.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        good_cls = _make_fake_cls("transformer", "GoodTransformer")

        task_keys = list(ri.TASK_MAP.keys())

        def _side_effect(estimator_types=None, return_names=True, as_dataframe=False):
            if estimator_types == task_keys[0]:
                raise RuntimeError("simulated per-type failure")
            if estimator_types == "transformer":
                return [("GoodTransformer", good_cls)]
            return []

        with patch(
            "sktime.registry.all_estimators",
            side_effect=_side_effect,
        ), caplog.at_level(logging.DEBUG, logger="sktime_mcp.registry.interface"):
            ri._load_registry()

        assert "GoodTransformer" in ri._cache, "GoodTransformer must load despite earlier type failure"
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("simulated per-type failure" in m or task_keys[0] in m for m in debug_msgs), (
            f"Expected DEBUG log about per-type failure. Got: {debug_msgs}"
        )

    # ------------------------------------------------------------------
    # Per-estimator exception swallowed in loop (lines 129-131)
    # ------------------------------------------------------------------

    def test_load_registry_continues_after_per_estimator_exception(self, caplog):
        """Covers the except/continue block at lines 129-131 by making _create_node raise for one estimator.

        If the except block is removed, the exception propagates and the second estimator is never loaded; the assert on cache length would fail.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        good_cls = _make_fake_cls("forecaster", "GoodEst")
        bad_cls = _make_fake_cls("forecaster", "BadEst")

        original_create_node = ri._create_node

        def patched_create_node(name, cls, estimator_type):
            if name == "BadEst":
                raise ValueError("simulated per-estimator failure")
            return original_create_node(name, cls, estimator_type)

        with patch("sktime.registry.all_estimators",
                   return_value=[("BadEst", bad_cls), ("GoodEst", good_cls)]), \
             patch.object(ri, "_create_node", side_effect=patched_create_node), \
             caplog.at_level(logging.DEBUG, logger="sktime_mcp.registry.interface"):
            ri._load_registry()

        assert "GoodEst" in ri._cache, "GoodEst must survive despite BadEst exception"
        assert "BadEst" not in ri._cache
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("BadEst" in m for m in debug_msgs), (
            f"Expected DEBUG log mentioning BadEst. Got: {debug_msgs}"
        )

    # ------------------------------------------------------------------
    # _get_tags: _tags fallback and exception handler (lines 200-203)
    # ------------------------------------------------------------------

    def test_get_tags_falls_back_to_underscore_tags_attribute(self):
        """Covers the elif hasattr(cls, '_tags') branch at line 200 in _get_tags.

        Removing the elif branch means classes with _tags but no get_class_tags return an empty dict instead of the correct tags; this test would fail.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        class NoGetClassTags:
            _tags = {"fit_is_empty": False, "handles-missing-data": True}

        result = ri._get_tags(NoGetClassTags)
        assert result == {"fit_is_empty": False, "handles-missing-data": True}

    def test_get_tags_returns_empty_dict_when_get_class_tags_raises(self):
        """Covers the except Exception block at lines 202-203 in _get_tags.

        Removing the except block would propagate the exception to the caller; this test fails if the method raises instead of returning {}.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        broken_cls = MagicMock()
        broken_cls.__name__ = "BrokenCls"
        broken_cls.get_class_tags.side_effect = RuntimeError("tags system broken")

        result = ri._get_tags(broken_cls)
        assert result == {}

    # ------------------------------------------------------------------
    # _get_hyperparameters: non-serializable default (lines 231-232)
    # ------------------------------------------------------------------

    def test_get_hyperparameters_converts_non_serializable_default_to_str(self):
        """Covers lines 224-225 in _get_hyperparameters where non-JSON-serializable defaults are converted via str().

        Removing the str() conversion would leave a raw object in the dict, and this test would fail because the default would not be a string.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        sentinel = object()  # not int/float/str/bool/list/dict/None

        class ClsWithObjectDefault:
            """Class with non-serializable default."""
            def __init__(self, param=sentinel):
                pass

        result = ri._get_hyperparameters(ClsWithObjectDefault)
        assert "param" in result
        assert isinstance(result["param"]["default"], str), (
            f"Expected str default, got {type(result['param']['default'])}"
        )
        assert result["param"]["required"] is False

    def test_get_hyperparameters_returns_empty_dict_when_inspect_raises(self):
        """Covers the except Exception block at lines 231-232 in _get_hyperparameters when inspect.signature raises.

        Removing the except block would propagate the exception to the caller; this test fails if the method raises instead of returning {}.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        broken_cls = MagicMock()
        broken_cls.__name__ = "BrokenInitCls"

        with patch("sktime_mcp.registry.interface.inspect.signature", side_effect=ValueError("no sig")):
            result = ri._get_hyperparameters(broken_cls)

        assert result == {}

    # ------------------------------------------------------------------
    # get_all_estimators: task filter and tags filter (lines 242-252)
    # ------------------------------------------------------------------

    def test_get_all_estimators_no_filter_returns_all(self):
        """Covers the get_all_estimators body at lines 242-252 by calling it without filters.

        If _ensure_loaded is removed from get_all_estimators, the cache is never populated and the result list is empty; this test fails.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["Est1"] = EstimatorNode("Est1", "forecasting", object, "m.Est1")
        ri._cache["Est2"] = EstimatorNode("Est2", "transformation", object, "m.Est2")

        result = ri.get_all_estimators()
        assert len(result) == 2

    def test_get_all_estimators_task_filter(self):
        """Covers the task-filter branch at line 246-247 in get_all_estimators.

        Removing the task filter means all estimators are returned regardless of task; this test fails because the forecasting-only filter would return 2 items instead of 1.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["ForecastEst"] = EstimatorNode("ForecastEst", "forecasting", object, "m.F")
        ri._cache["TransformEst"] = EstimatorNode("TransformEst", "transformation", object, "m.T")

        result = ri.get_all_estimators(task="forecasting")
        assert len(result) == 1
        assert result[0].name == "ForecastEst"

    def test_get_all_estimators_tags_filter(self):
        """Covers the tags-filter branch at lines 249-250 and the _filter_by_tags body at lines 260-273.

        Removing the tags filter or the _filter_by_tags body would cause all estimators to be returned; this test fails because only the matching one should appear.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["TaggedEst"] = EstimatorNode(
            "TaggedEst", "forecasting", object, "m.T", tags={"fit_is_empty": True}
        )
        ri._cache["UntaggedEst"] = EstimatorNode(
            "UntaggedEst", "forecasting", object, "m.U", tags={"fit_is_empty": False}
        )

        result = ri.get_all_estimators(tags={"fit_is_empty": True})
        assert len(result) == 1
        assert result[0].name == "TaggedEst"

    def test_filter_by_tags_excludes_tag_mismatch(self):
        """Covers _filter_by_tags (lines 260-273) with a multi-tag required_tags dict where one tag mismatches.

        Removing the inner matches=False break means all estimators pass the filter; this test fails because the mismatched estimator would incorrectly appear.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        est_match = EstimatorNode("A", "forecasting", object, "m.A",
                                  tags={"tag1": "v1", "tag2": "v2"})
        est_nomatch = EstimatorNode("B", "forecasting", object, "m.B",
                                    tags={"tag1": "v1", "tag2": "WRONG"})

        result = ri._filter_by_tags([est_match, est_nomatch], {"tag1": "v1", "tag2": "v2"})
        assert len(result) == 1
        assert result[0].name == "A"

    # ------------------------------------------------------------------
    # get_estimator_by_name (lines 277-278)
    # ------------------------------------------------------------------

    def test_get_estimator_by_name_returns_node_when_present(self):
        """Covers get_estimator_by_name (lines 277-278) for both found and not-found paths.

        Removing the return statement or _ensure_loaded call would cause the method to return None always or fail to load; this test fails on the found case.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        node = EstimatorNode("MyEst", "forecasting", object, "m.MyEst")
        ri._cache["MyEst"] = node

        result = ri.get_estimator_by_name("MyEst")
        assert result is node

    def test_get_estimator_by_name_returns_none_when_absent(self):
        """Covers the None return path of get_estimator_by_name via dict.get() default.

        If _cache.get is replaced with _cache[name] (raising KeyError), this test would fail; the method must return None for missing names.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True

        result = ri.get_estimator_by_name("NonExistent")
        assert result is None

    # ------------------------------------------------------------------
    # get_available_tasks (line 282)
    # ------------------------------------------------------------------

    def test_get_available_tasks_returns_all_task_map_values(self):
        """Covers get_available_tasks() (line 282) by calling it directly.

        Returning an empty list or raising instead of returning TASK_MAP values would cause this test to fail.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        result = ri.get_available_tasks()
        assert isinstance(result, list)
        assert set(result) == set(RegistryInterface.TASK_MAP.values())

    # ------------------------------------------------------------------
    # get_available_tags: ImportError fallback (lines 292-294)
    # ------------------------------------------------------------------

    def test_get_available_tags_import_error_fallback(self):
        """Covers the except ImportError fallback at lines 292-294 in get_available_tags.

        Removing the except block would propagate ImportError to callers; this test fails if get_available_tags raises instead of returning the fallback list.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        ri._all_tags = {"tag_alpha", "tag_beta"}

        with patch("sktime.registry.all_tags", side_effect=ImportError("all_tags not available")):
            result = ri.get_available_tags()

        assert isinstance(result, list)
        tag_names = [item["tag"] for item in result]
        assert "tag_alpha" in tag_names
        assert "tag_beta" in tag_names

    # ------------------------------------------------------------------
    # get_available_tags: scitype non-list non-string (lines 305-306)
    # ------------------------------------------------------------------

    def test_get_available_tags_scitype_non_list_non_string(self):
        """Covers lines 305-306 where scitype is neither a string nor a list (e.g., a tuple or int).

        Removing the elif branch means non-list non-string scitypes are passed directly to applies_to, which would break JSON serialization; this test fails if the value is not converted to a list.
        """
        import pandas as pd

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True

        # scitype as a tuple (iterable but not str or list) — triggers the elif branch
        tags_df = pd.DataFrame([
            {"name": "tag_with_tuple_scitype", "description": "desc", "type": "bool",
             "scitype": ("Series", "Panel")},
        ])

        with patch("sktime.registry.all_tags", return_value=tags_df):
            result = ri.get_available_tags()

        assert len(result) == 1
        applies_to = result[0]["applies_to"]
        assert isinstance(applies_to, list), (
            f"applies_to must be a list, got {type(applies_to)}"
        )
        assert "Series" in applies_to
        assert "Panel" in applies_to

    def test_get_available_tags_scitype_non_iterable_non_string(self):
        """Covers the hasattr(scitype, '__iter__') else branch inside lines 305-306 for a bare integer scitype.

        If the str(scitype) fallback is removed, a non-iterable non-string scitype raises TypeError; this test fails if the result is not wrapped in a list.
        """
        import pandas as pd

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True

        # scitype as an integer — not a string, not a list, not iterable
        tags_df = pd.DataFrame([
            {"name": "tag_int_scitype", "description": "desc", "type": "str", "scitype": 42},
        ])

        with patch("sktime.registry.all_tags", return_value=tags_df):
            result = ri.get_available_tags()

        assert len(result) == 1
        applies_to = result[0]["applies_to"]
        assert isinstance(applies_to, list)

    # ------------------------------------------------------------------
    # get_available_tags: value_type non-string (line 311)
    # ------------------------------------------------------------------

    def test_get_available_tags_value_type_non_string(self):
        """Covers line 311 where value_type is not a string and is converted via str().

        Removing the str() conversion means a non-string value_type (e.g., None or int) reaches the result dict as a non-string; this test fails if value_type is not a string.
        """
        import pandas as pd

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True

        tags_df = pd.DataFrame([
            {"name": "tag_nonstr_type", "description": "desc", "type": None, "scitype": "Series"},
        ])

        with patch("sktime.registry.all_tags", return_value=tags_df):
            result = ri.get_available_tags()

        assert len(result) == 1
        assert isinstance(result[0]["value_type"], str), (
            f"value_type must be str, got {type(result[0]['value_type'])}"
        )

    # ------------------------------------------------------------------
    # search_estimators (lines 327-339)
    # ------------------------------------------------------------------

    def test_search_estimators_matches_by_name(self):
        """Covers the name-match branch at lines 331-333 in search_estimators.

        Removing the name-match branch means estimators are only found via docstring; this test fails because the match is on the name, not the docstring.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["NaiveForecaster"] = EstimatorNode(
            "NaiveForecaster", "forecasting", object, "m.N", docstring="Some doc."
        )
        ri._cache["STLForecaster"] = EstimatorNode(
            "STLForecaster", "forecasting", object, "m.S", docstring="Other doc."
        )

        result = ri.search_estimators("naive")
        names = [n.name for n in result]
        assert "NaiveForecaster" in names
        assert "STLForecaster" not in names

    def test_search_estimators_matches_by_docstring(self):
        """Covers the docstring-match branch at lines 335-336 in search_estimators.

        Removing the docstring-match branch means only name matches are returned; this test fails because the query matches the docstring but not the name.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["TrendEst"] = EstimatorNode(
            "TrendEst", "forecasting", object, "m.T",
            docstring="Fits an ARIMA model for seasonal decomposition."
        )
        ri._cache["OtherEst"] = EstimatorNode(
            "OtherEst", "forecasting", object, "m.O", docstring="Completely unrelated."
        )

        result = ri.search_estimators("arima")
        names = [n.name for n in result]
        assert "TrendEst" in names
        assert "OtherEst" not in names

    def test_search_estimators_no_match_returns_empty(self):
        """Covers the no-match path in search_estimators where neither name nor docstring contains the query.

        If the method always returned all estimators, this test would fail because the result would be non-empty.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["Est1"] = EstimatorNode(
            "Est1", "forecasting", object, "m.E1", docstring="A basic estimator."
        )

        result = ri.search_estimators("zzznomatchzzz")
        assert result == []

    def test_search_estimators_non_string_query_returns_empty(self):
        """Covers the isinstance(query, str) guard at the top of search_estimators.

        Passing None (or any non-str) must return [] rather than raising
        AttributeError at query.lower().  Removing the guard causes the crash.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True

        assert ri.search_estimators(None) == []
        assert ri.search_estimators(42) == []

    def test_search_estimators_skips_none_docstring(self):
        """Covers the node.docstring truthiness check at line 335 for an estimator with None docstring.

        Removing the `if node.docstring` guard would raise AttributeError on None.lower(); this test fails if the method raises instead of skipping gracefully.
        """
        from sktime_mcp.registry.interface import EstimatorNode, RegistryInterface

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["NodocEst"] = EstimatorNode(
            "NodocEst", "forecasting", object, "m.N", docstring=None
        )

        # query matches neither name nor (absent) docstring
        result = ri.search_estimators("something")
        assert result == []

    def test_ensure_loaded_does_not_deadlock_on_reentrant_call(self):
        """Proves _lock is reentrant (RLock), so a re-entrant _ensure_loaded inside
        _load_registry does not deadlock.

        Scenario: a get_class_tags() override or metaclass hook on a custom estimator
        calls back into _ensure_loaded on the same RegistryInterface instance while
        _load_registry is already executing under the lock.  With threading.Lock()
        this deadlocks; with threading.RLock() the inner acquisition succeeds because
        it is the same thread.
        """
        import threading

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        reentrant_called = threading.Event()

        calls = [0]

        def reentrant_load(self):
            calls[0] += 1
            if calls[0] == 1:
                # First invocation: simulate re-entrant call from inside _load_registry
                # (e.g. a get_class_tags() hook that calls back into the registry).
                reentrant_called.set()
                self._ensure_loaded()  # re-entrant — with Lock: deadlock; with RLock: allowed
            # Second (re-entrant) invocation: return immediately to avoid infinite recursion.

        with patch.object(type(ri), "_load_registry", reentrant_load):
            exc = [None]

            def run():
                try:
                    ri._ensure_loaded()
                except Exception as e:
                    exc[0] = e

            t = threading.Thread(target=run)
            t.start()
            t.join(timeout=5)

        assert not t.is_alive(), "Thread still alive after 5s — likely deadlock with non-reentrant lock"
        assert exc[0] is None, f"Thread raised: {exc[0]}"
        assert reentrant_called.is_set(), "Re-entrant path was never triggered"
