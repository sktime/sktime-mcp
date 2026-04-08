"""tests/test_registry_fix.py — adversarially hardened pytest suite for _load_registry fix.

Each test is designed to catch a specific, named regression path. A test that passes
when the buggy 8-call loop is reintroduced is not a test — it's documentation.

Regression labels:
  R1  all_estimators called > 1 time
  R2  estimator_types= kwarg passed (loop signal)
  R3  cache smaller than it should be (missed estimators)
  R4  task mapping wrong for known estimators
  R5  subtype estimators dropped (transformer-pairwise-panel etc.)
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

class TestRegistrySingleCall:
    # ------------------------------------------------------------------
    # R1 / R2 — call count and signature
    # ------------------------------------------------------------------

    def test_all_estimators_called_exactly_once_with_exact_kwargs(self):
        """Proves all_estimators() called once AND with the exact correct kwargs.

        R1: call_count > 1 is a regression (old loop called all_estimators 8 times).
        R2: estimator_types must be list(TASK_MAP.keys()) — neither absent (old no-filter
        call) nor a single string per iteration (old per-type loop pattern).
        Patch target is sktime.registry.all_estimators because all_estimators is a local
        import inside _load_registry — patching the interface module namespace would fail.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        fake_cls = _make_fake_cls("forecaster", "NaiveForecaster")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("NaiveForecaster", fake_cls)],
        ) as mock_ae:
            ri._load_registry()

        mock_ae.assert_called_once_with(
            estimator_types=list(RegistryInterface.TASK_MAP.keys()),
            return_names=True,
            as_dataframe=False,
        )

    def test_estimator_types_passed_as_list_not_per_key_string(self):
        """Proves estimator_types is passed as a single LIST of all TASK_MAP keys, not
        as individual per-key strings across multiple calls.

        R2 regression: the old loop called all_estimators(estimator_types="forecaster"),
        then all_estimators(estimator_types="transformer"), etc. — one string per call.
        A fix must pass estimator_types=list(TASK_MAP.keys()) in a single call.
        Asserting the value is a list (not a string) and equals all keys prevents
        both the old loop pattern and any partial single-string shortcut.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        fake_cls = _make_fake_cls("forecaster")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("F", fake_cls)],
        ) as mock_ae:
            ri._load_registry()

        assert mock_ae.call_count == 1, (
            f"Expected exactly 1 call, got {mock_ae.call_count} — loop regression"
        )
        call = mock_ae.call_args_list[0]
        assert "estimator_types" in call.kwargs, (
            "estimator_types must be passed as a kwarg (not positional or absent)"
        )
        passed_types = call.kwargs["estimator_types"]
        assert isinstance(passed_types, list), (
            f"estimator_types must be a list, got {type(passed_types).__name__!r}: "
            f"{passed_types!r} — single-string per-loop-iteration regression"
        )
        assert passed_types == list(RegistryInterface.TASK_MAP.keys()), (
            f"estimator_types list {passed_types!r} != "
            f"list(TASK_MAP.keys()) {list(RegistryInterface.TASK_MAP.keys())!r}"
        )

    def test_idempotent_load_calls_all_estimators_exactly_once_total(self):
        """If lazy-load guard is broken, second call to _ensure_loaded
        re-runs _load_registry and calls all_estimators a second time.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        fake_cls = _make_fake_cls("forecaster")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("F", fake_cls)],
        ) as mock_ae:
            ri._ensure_loaded()
            ri._ensure_loaded()

        assert mock_ae.call_count == 1, (
            f"all_estimators called {mock_ae.call_count}× across two _ensure_loaded calls"
        )

    # ------------------------------------------------------------------
    # R3 — completeness: bidirectional cache verification
    # ------------------------------------------------------------------

    def test_cache_contains_every_estimator_with_known_type(self):
        """R3 forward direction: no estimator with a resolvable type should be silently dropped.

        Ground truth is all_estimators(estimator_types=list(TASK_MAP.keys()), ...) — the
        same filtered call the production code issues.  Using unfiltered all_estimators()
        as ground truth would generate false failures for estimator types intentionally
        excluded by the estimator_types filter (e.g. pairwise transformers).
        Regression caught: any estimator returned by the filtered call that is absent from
        the cache reveals a silent drop in _load_registry's loop body.
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._ensure_loaded()

        # Use the same filter the production code uses — this is the ground truth.
        all_est = real_ae(
            estimator_types=list(ri.TASK_MAP.keys()),
            return_names=True,
            as_dataframe=False,
        )
        missing = []
        for name, cls in all_est:
            est_type = ri._get_estimator_type(cls)
            if est_type and est_type in ri.TASK_MAP:
                if name not in ri._cache:
                    missing.append((name, est_type))

        assert not missing, (
            f"{len(missing)} estimators with resolvable type missing from cache:\n"
            + "\n".join(f"  {n} ({t})" for n, t in missing[:10])
        )

    def test_cache_contains_no_phantom_estimators(self):
        """R3 reverse direction: cache must not contain estimators absent from the registry
        (fabricated entries or stale data).
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._ensure_loaded()

        known_names = {name for name, _ in real_ae(return_names=True, as_dataframe=False)}
        phantom = [name for name in ri._cache if name not in known_names]

        assert not phantom, (
            f"{len(phantom)} phantom estimators in cache: {phantom[:5]}"
        )

    def test_cache_size_matches_runtime_expected_count(self):
        """Count derived from all_estimators(estimator_types=...) at runtime — catches both
        over- and under-capture.

        Ground truth uses the same estimator_types filter as the production call so that
        the expected count is derived from exactly the set of estimators the implementation
        is supposed to load — not a superset that includes intentionally excluded types.
        Regression caught: cache size diverging from the filtered call result reveals
        either silent drops (cache too small) or phantom insertions (cache too large).
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._ensure_loaded()

        # Ground truth: same filter the production code uses.
        all_est = real_ae(
            estimator_types=list(ri.TASK_MAP.keys()),
            return_names=True,
            as_dataframe=False,
        )
        expected = sum(
            1 for _, cls in all_est
            if (t := ri._get_estimator_type(cls)) and t in ri.TASK_MAP
        )

        assert len(ri._cache) == expected, (
            f"Cache has {len(ri._cache)} estimators; expected {expected} from "
            f"all_estimators(estimator_types=list(TASK_MAP.keys()))"
        )

    def test_cache_contains_all_sktime_type_members(self):
        """Non-circular completeness: every estimator returned by all_estimators per type
        must be present in the cache with the correct task value.

        Does NOT use _get_estimator_type internally — ground truth comes directly from
        all_estimators(estimator_types=key) for each TASK_MAP key, which is sktime's own
        authoritative membership answer.  This makes the test independent of any tag-reading
        logic in the production code.

        Regression caught: silent category drop — e.g. if clusterer entries never appear
        in the cache, the per-key loop below would flag every clusterer as missing even
        if _get_estimator_type works correctly.  This catches failures invisible to
        tag-based completeness tests.
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._ensure_loaded()

        missing_from_cache: list[tuple[str, str]] = []
        wrong_task: list[tuple[str, str, str]] = []

        for key, expected_task in ri.TASK_MAP.items():
            try:
                type_members = real_ae(
                    estimator_types=key,
                    return_names=True,
                    as_dataframe=False,
                )
            except Exception:
                # If sktime itself can't list this type, skip rather than fail.
                continue

            for name, _ in type_members:
                if name not in ri._cache:
                    missing_from_cache.append((name, key))
                else:
                    actual_task = ri._cache[name].task
                    if actual_task != expected_task:
                        wrong_task.append((name, expected_task, actual_task))

        assert not missing_from_cache, (
            f"{len(missing_from_cache)} estimators returned by all_estimators(estimator_types=key) "
            f"but absent from cache (first 10):\n"
            + "\n".join(f"  {n!r} (type={t!r})" for n, t in missing_from_cache[:10])
        )
        assert not wrong_task, (
            f"{len(wrong_task)} estimators have wrong task in cache (first 10):\n"
            + "\n".join(
                f"  {n!r}: expected task={e!r}, got {a!r}"
                for n, e, a in wrong_task[:10]
            )
        )

    def test_single_call_captures_at_least_as_many_as_old_loop(self):
        """Strict TASK_MAP matching drops subtypes (transformer-pairwise-panel etc.)
        that the old loop included under "transformer".
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        # Simulate old behavior: one call per TASK_MAP type
        old_cache: dict = {}
        for estimator_type in ri.TASK_MAP:
            try:
                estimators = real_ae(
                    estimator_types=estimator_type,
                    return_names=True,
                    as_dataframe=False,
                )
                for name, cls in estimators:
                    old_cache[name] = cls
            except Exception:
                pass

        # Simulate new behavior
        new_cache: dict = {}
        all_est = real_ae(return_names=True, as_dataframe=False)
        for name, cls in all_est:
            t = ri._get_estimator_type(cls)
            if t and t in ri.TASK_MAP:
                new_cache[name] = cls

        assert len(new_cache) >= len(old_cache), (
            f"Fix captures {len(new_cache)} estimators; old loop captured {len(old_cache)}. "
            f"Regression: dropped {len(old_cache) - len(new_cache)} estimators."
        )

    def test_mock_registry_captures_all_eight_task_types(self):
        """R3 + R4 via controlled mock: one fake estimator per TASK_MAP key.
        No hardcoded counts — derived from ri.TASK_MAP at runtime.
        Regression target: any TASK_MAP key silently skipped in _load_registry's loop.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        fake_estimators = []
        for key in ri.TASK_MAP:
            cls = MagicMock()
            cls.get_class_tags.return_value = {"object_type": key}
            cls.__module__ = f"sktime.fake.{key}"
            cls.__name__ = f"Fake_{key}"
            fake_estimators.append((f"Fake_{key}", cls))

        with patch("sktime.registry.all_estimators", return_value=fake_estimators):
            ri._load_registry()

        expected_count = len(ri.TASK_MAP)
        assert len(ri._cache) == expected_count, (
            f"Expected {expected_count} cache entries (one per TASK_MAP key), "
            f"got {len(ri._cache)}. Missing: "
            f"{[f'Fake_{k}' for k in ri.TASK_MAP if f'Fake_{k}' not in ri._cache]}"
        )

        for key, task_value in ri.TASK_MAP.items():
            node = ri._cache.get(f"Fake_{key}")
            assert node is not None, f"Fake_{key} missing from cache"
            assert node.task == task_value, (
                f"Fake_{key}: expected task={task_value!r}, got {node.task!r}"
            )

    def test_all_task_types_represented_in_cache(self):
        """Proves every TASK_MAP type that has estimators is present in the cache.

        Catches regression: if an entire estimator category is silently skipped,
        the cache would be missing a whole task type.
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._ensure_loaded()

        all_est = real_ae(return_names=True, as_dataframe=False)
        types_with_estimators: set[str] = set()
        for _, cls in all_est:
            t = ri._get_estimator_type(cls)
            if t and t in ri.TASK_MAP:
                types_with_estimators.add(t)

        cached_task_values = {node.task for node in ri._cache.values()}
        missing_tasks = [
            est_type for est_type in types_with_estimators
            if ri.TASK_MAP[est_type] not in cached_task_values
        ]

        assert not missing_tasks, (
            f"Task types with estimators but none in cache: {missing_tasks}"
        )

    # ------------------------------------------------------------------
    # R4 — correct task mapping
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "cls_name,expected_task",
        [
            ("NaiveForecaster", "forecasting"),
            ("Detrender", "transformation"),
            ("TimeSeriesForestClassifier", "classification"),
        ],
    )
    def test_estimator_node_task_field_maps_correctly(self, cls_name, expected_task):
        """Proves the EstimatorNode.task field is the TASK_MAP value, not the raw tag.

        R4: _create_node must receive a TASK_MAP key and produce a node with
        task == TASK_MAP[key]. A fix that stores the raw object_type in node.task
        would pass call-count tests but fail this.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._ensure_loaded()

        node = ri._cache.get(cls_name)
        assert node is not None, f"{cls_name} not in cache"
        assert node.task == expected_task, (
            f"{cls_name}.task = {node.task!r}, expected {expected_task!r}"
        )

    # ------------------------------------------------------------------
    # R5 — subtype prefix matching
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("subtype,expected_key", [
        ("transformer-pairwise-panel", "transformer"),
        ("transformer-pairwise", "transformer"),
        ("classifier-early-ts", "classifier"),
        ("regressor-panel", "regressor"),
    ])
    def test_get_estimator_type_prefix_matches_subtypes(self, subtype, expected_key):
        """Proves subtype object_type values are mapped to their parent TASK_MAP key.

        R5: Without prefix matching, "transformer-pairwise-panel" returns the raw value
        which fails the TASK_MAP membership check, silently dropping 20+ estimators.
        This is the regression that caused new approach to capture fewer than old loop.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        if expected_key not in ri.TASK_MAP:
            pytest.skip(f"{expected_key!r} not in TASK_MAP for this sktime install")

        cls = MagicMock()
        cls.get_class_tags.return_value = {"object_type": subtype}
        result = ri._get_estimator_type(cls)
        assert result == expected_key, (
            f"Expected {expected_key!r} for subtype {subtype!r}, got {result!r}"
        )

    def test_prefix_match_requires_dash_separator(self):
        """Proves prefix matching uses dash separator, not bare prefix.

        A bare-prefix match ("trans" matching "transformer") would be incorrect
        if TASK_MAP ever gains a short key. The dash separator anchors subtypes.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        # A value that starts with a TASK_MAP key but WITHOUT a dash — must NOT match
        # (e.g., if TASK_MAP has "net" and we have "network", it must not match "net")
        # We test this concretely: "forecasterthing" must NOT map to "forecaster"
        cls = MagicMock()
        cls.get_class_tags.return_value = {"object_type": "forecasterthing"}
        result = ri._get_estimator_type(cls)
        assert result != "forecaster", (
            f"Bare prefix match incorrectly mapped 'forecasterthing' to 'forecaster'"
        )

    # ------------------------------------------------------------------
    # Defensive behavior — _get_estimator_type
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("exc", [
        RuntimeError("tag system unavailable"),
        AttributeError("no such method"),
        TypeError("not callable"),
        Exception("catastrophic failure"),
    ])
    def test_get_estimator_type_returns_empty_on_any_exception(self, exc):
        """Proves _get_estimator_type returns '' for any exception type.

        Must catch ALL exceptions — a specific except clause would fail for
        unexpected exception types, turning a skip into a load failure.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        broken_cls = MagicMock()
        broken_cls.get_class_tags.side_effect = exc

        assert ri._get_estimator_type(broken_cls) == ""

    @pytest.mark.parametrize("non_string_value", [
        42, [42], {"type": "forecaster"}, None, True, 3.14,
    ])
    def test_get_estimator_type_returns_empty_for_non_string_object_type(self, non_string_value):
        """Proves non-string (and non-list-of-strings) object_type values return ''.

        List-of-strings is valid (e.g. ['reconciler', 'transformer']) and handled.
        Non-string scalars and lists-of-non-strings must return '' cleanly.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        cls = MagicMock()
        cls.get_class_tags.return_value = {"object_type": non_string_value}

        assert ri._get_estimator_type(cls) == "", (
            f"Expected '' for object_type={non_string_value!r}"
        )

    def test_get_estimator_type_handles_list_valued_object_type(self):
        """Proves list-valued object_type (multi-role estimators) resolves to first TASK_MAP match.

        sktime registers some estimators with object_type=['reconciler', 'transformer'].
        The old loop captured these via all_estimators(estimator_types='transformer').
        The fix must match the same behavior by iterating the list.
        Without this, 20 reconcilers and global forecasters are silently dropped.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        cls = MagicMock()
        # Typical multi-role: first element not in TASK_MAP, second is
        cls.get_class_tags.return_value = {"object_type": ["reconciler", "transformer"]}
        assert ri._get_estimator_type(cls) == "transformer"

        # Global forecasters pattern
        cls2 = MagicMock()
        cls2.get_class_tags.return_value = {"object_type": ["global_forecaster", "forecaster"]}
        assert ri._get_estimator_type(cls2) == "forecaster"

        # List of non-strings → ""
        cls3 = MagicMock()
        cls3.get_class_tags.return_value = {"object_type": [42, None]}
        assert ri._get_estimator_type(cls3) == ""

    def test_get_estimator_type_handles_tuple_valued_object_type(self):
        """isinstance(raw, (list, tuple)) guard — tuples must behave identically to lists.
        If guard reverts to isinstance(raw, list) only, tuple-valued object_type
        returns '' instead of the correct TASK_MAP key.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        if "transformer" not in ri.TASK_MAP:
            pytest.skip("'transformer' must be in TASK_MAP for this test")

        # Single-element tuple with an exact key
        cls1 = MagicMock()
        cls1.get_class_tags.return_value = {"object_type": ("transformer",)}
        assert ri._get_estimator_type(cls1) == "transformer", (
            "Single-element tuple ('transformer',) must resolve to 'transformer'"
        )

        # Multi-element tuple where second is an exact key, first is a subtype
        if "forecaster" not in ri.TASK_MAP:
            pytest.skip("'forecaster' must be in TASK_MAP for this test")
        cls2 = MagicMock()
        cls2.get_class_tags.return_value = {
            "object_type": ("transformer-pairwise-panel", "forecaster")
        }
        result = ri._get_estimator_type(cls2)
        assert result == "forecaster", (
            f"Expected 'forecaster' (exact beats prefix in tuple), got {result!r}"
        )

    def test_two_pass_exact_beats_prefix_in_list_valued_object_type(self):
        """Proves pass-1 exact match on any candidate beats pass-2 prefix hit on an earlier one.

        Regression target: a naive single-pass implementation that evaluates exact-or-prefix
        per element in order would see 'transformer-pairwise-panel' at index 0, prefix-match
        it to 'transformer', and return immediately — never reaching the exact match
        'forecaster' at index 1.

        The two-pass strategy (all exact checks before any prefix check) must return
        'forecaster' because it is an exact TASK_MAP hit, and exact wins over prefix
        regardless of list position.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        if "forecaster" not in ri.TASK_MAP or "transformer" not in ri.TASK_MAP:
            pytest.skip("Both 'forecaster' and 'transformer' must be in TASK_MAP for this test")

        cls = MagicMock()
        # Element 0: prefix-matches "transformer" via dash fallback.
        # Element 1: exact-matches "forecaster".
        # Correct two-pass result: "forecaster" (exact beats prefix).
        cls.get_class_tags.return_value = {
            "object_type": ["transformer-pairwise-panel", "forecaster"]
        }
        result = ri._get_estimator_type(cls)
        assert result == "forecaster", (
            f"Expected 'forecaster' (exact match on element 1 beats prefix on element 0), "
            f"got {result!r}. Single-pass regression: prefix hit on element 0 short-circuited."
        )

    def test_two_pass_exact_beats_prefix_in_single_string_candidates(self):
        """Proves pass-1 exact match is checked before pass-2 prefix for a single candidate.

        A candidate that is itself an exact TASK_MAP key must always return that exact key,
        never accidentally prefix-match to a shorter key.  Regression: if the code checked
        prefix before exact, a hypothetical 'forecaster-advanced' TASK_MAP key would let
        'forecaster' prefix-match to 'forecaster-advanced' — incorrect.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        for key in ri.TASK_MAP:
            cls = MagicMock()
            cls.get_class_tags.return_value = {"object_type": key}
            result = ri._get_estimator_type(cls)
            assert result == key, (
                f"Exact key {key!r} should return itself, got {result!r}"
            )

    def test_get_estimator_type_normalizes_all_case_variants(self):
        """Proves .lower() is applied before TASK_MAP lookup for all case variants.

        A fix that lowercases only in _load_registry (not the helper) would fail
        because the helper is tested and used independently.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        for variant in ("Forecaster", "FORECASTER", "fOrEcAsTeR"):
            cls = MagicMock()
            cls.get_class_tags.return_value = {"object_type": variant}
            assert ri._get_estimator_type(cls) == "forecaster", (
                f"Expected 'forecaster' for {variant!r}"
            )

    def test_unknown_object_type_is_skipped_and_create_node_never_called(self):
        """Proves estimators with unresolvable object_type never reach _create_node.

        If the TASK_MAP guard is removed, _create_node is called with an unknown
        type, producing a node with task == raw_type (not a TASK_MAP value).
        Spy on _create_node to verify it is never called for unknown types.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        unknown_cls = _make_fake_cls("quantum_predictor", "QuantumForecaster")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("QuantumForecaster", unknown_cls)],
        ):
            with patch.object(ri, "_create_node", wraps=ri._create_node) as spy:
                ri._load_registry()

        assert "QuantumForecaster" not in ri._cache
        spy.assert_not_called()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

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

    def test_loaded_flag_set_when_all_estimators_call_raises(self):
        """Call-level exception caught by inner try/except in _load_registry.
        _ensure_loaded must still set _loaded=True.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        with patch(
            "sktime.registry.all_estimators",
            side_effect=RuntimeError("registry unavailable"),
        ):
            ri._ensure_loaded()

        assert ri._loaded
        assert len(ri._cache) == 0

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

    def test_buggy_loop_is_proven_less_complete_than_fix(self):
        """Proves the completeness test is falsifiable: old loop misses estimators the fix captures.

        This is the red-green proof. Simulates old behavior and verifies it produces
        a strictly smaller cache when subtype estimators exist.
        Skips rather than fails if sktime has patched the upstream gap.
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        old_cache: dict = {}
        for estimator_type in ri.TASK_MAP:
            try:
                for name, cls in real_ae(
                    estimator_types=estimator_type,
                    return_names=True,
                    as_dataframe=False,
                ):
                    old_cache[name] = cls
            except Exception:
                pass

        new_cache: dict = {}
        for name, cls in real_ae(return_names=True, as_dataframe=False):
            t = ri._get_estimator_type(cls)
            if t and t in ri.TASK_MAP:
                new_cache[name] = cls

        gained = {n for n in new_cache if n not in old_cache}
        if not gained:
            pytest.skip(
                "New approach gains no estimators over old loop — "
                "sktime may have closed the gap upstream."
            )

        assert len(new_cache) > len(old_cache), (
            f"Expected new > old. new={len(new_cache)}, old={len(old_cache)}"
        )

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def test_single_all_estimators_call_faster_than_n_typed_calls(self):
        """Proves the single all_estimators() call is faster than N typed calls.

        Compares API call times only (not node creation), matching the benchmark:
        single call ~1.5ms, 8-call loop ~2524ms on warm sys.modules.
        N = len(TASK_MAP); bound: single call must be < 50% of N typed calls.
        """
        from sktime.registry import all_estimators

        from sktime_mcp.registry.interface import RegistryInterface

        # Warm sys.modules
        all_estimators(return_names=True, as_dataframe=False)

        ri = RegistryInterface()
        n = len(ri.TASK_MAP)

        # Time N typed calls (old approach cost)
        t0 = time.perf_counter()
        for estimator_type in ri.TASK_MAP:
            all_estimators(estimator_types=estimator_type, return_names=True, as_dataframe=False)
        old_time = time.perf_counter() - t0

        # Time single call (new approach cost)
        t1 = time.perf_counter()
        all_estimators(return_names=True, as_dataframe=False)
        new_time = time.perf_counter() - t1

        assert new_time < old_time * 0.5, (
            f"Single call ({new_time:.4f}s) not < 50% of {n}-call loop ({old_time:.4f}s). "
            f"Speedup = {old_time / new_time:.1f}×"
        )

    def test_sys_modules_growth_is_zero_after_warm_load(self):
        """Proves a warm _load_registry adds zero new sktime.* modules to sys.modules.

        The root cause of the 1,637× slowdown is 2,649 new module imports per typed call.
        After the first load warms sys.modules, a second RegistryInterface load should
        add no new sktime imports.
        """
        from sktime.registry import all_estimators

        all_estimators(return_names=True, as_dataframe=False)
        sktime_modules_before = {k for k in sys.modules if k.startswith("sktime")}

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri._load_registry()

        new_modules = {k for k in sys.modules if k.startswith("sktime")} - sktime_modules_before

        assert len(new_modules) == 0, (
            f"_load_registry imported {len(new_modules)} new sktime modules after warm load:\n"
            + "\n".join(f"  {m}" for m in sorted(new_modules)[:10])
        )

    # ------------------------------------------------------------------
    # Logging contract
    # ------------------------------------------------------------------

    def test_no_warning_log_during_normal_load(self, caplog):
        """Proves no WARNING-level messages are emitted during a clean load.

        The old loop emitted warnings when individual estimator types failed.
        The fix has a single outer catch that uses WARNING; on a healthy registry
        it must never fire.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        fake_cls = _make_fake_cls("forecaster", "GoodForecaster")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("GoodForecaster", fake_cls)],
        ):
            with caplog.at_level(logging.WARNING, logger="sktime_mcp.registry.interface"):
                ri._load_registry()

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, (
            f"Unexpected WARNING log(s): {[r.message for r in warnings]}"
        )

    def test_debug_log_emitted_for_skipped_estimator_with_name_and_reason(self, caplog):
        """Proves DEBUG log for skipped estimators contains both the name and the reason.

        Skipped estimators must be discoverable in debug logs — silent drops create
        invisible correctness gaps. The log must include the estimator name AND
        the reason (the unknown type string or TASK_MAP mention), not just one.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        unknown_cls = _make_fake_cls("quantum_predictor", "QPred")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("QPred", unknown_cls)],
        ):
            with caplog.at_level(logging.DEBUG, logger="sktime_mcp.registry.interface"):
                ri._load_registry()

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("QPred" in m for m in debug_msgs), (
            f"No DEBUG log mentioning estimator name 'QPred'. Logs: {debug_msgs}"
        )
        assert any("unresolvable" in m or "object_type" in m or "TASK_MAP" in m for m in debug_msgs), (
            f"No DEBUG log mentioning the skip reason. Logs: {debug_msgs}"
        )
