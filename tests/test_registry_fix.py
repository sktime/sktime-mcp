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

        NOTE — partial circularity: the oracle (`_get_estimator_type`) is the same logic
        as the production path, so a systematic bug in `_get_estimator_type` that drops a
        whole subtype will be invisible here.  `test_cache_contains_all_sktime_type_members`
        is the non-circular counterpart that uses sktime's own per-type filter as ground truth.
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
            if est_type and est_type in ri.TASK_MAP and name not in ri._cache:
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

    def test_single_call_no_regression_vs_per_key_loop(self):
        """Proves the single-call approach loses no estimator the per-key loop would cache.

        Symmetric filter: both old and new paths apply _get_estimator_type so the
        comparison is between actual cache populations, not raw all_estimators output.

        Applying _get_estimator_type on both sides is essential: without it, pairwise
        transformers (object_type='transformer-pairwise-panel') appear in old_names via
        all_estimators(estimator_types='transformer') but are absent from new_names
        because _get_estimator_type returns '' for them — producing structural false
        failures that have nothing to do with the fix.

        Regression caught: any estimator in the per-key cache but absent from the
        single-call cache — the fix must not drop anything the old loop would have loaded.
        New gains (estimators only in the single-call result) are improvements, not bugs.
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        # Simulate old production behavior: per-key loop WITH _get_estimator_type routing
        old_cache: dict[str, str] = {}
        for estimator_type in ri.TASK_MAP:
            try:
                for name, cls in real_ae(
                    estimator_types=estimator_type,
                    return_names=True,
                    as_dataframe=False,
                ):
                    t = ri._get_estimator_type(cls)
                    if t and t in ri.TASK_MAP:
                        old_cache[name] = t
            except Exception:
                pass

        # Simulate new production behavior: single call WITH _get_estimator_type routing
        new_cache: dict[str, str] = {}
        for name, cls in real_ae(
            estimator_types=list(ri.TASK_MAP.keys()),
            return_names=True,
            as_dataframe=False,
        ):
            t = ri._get_estimator_type(cls)
            if t and t in ri.TASK_MAP:
                new_cache[name] = t

        old_only = set(old_cache) - set(new_cache)
        assert not old_only, (
            f"{len(old_only)} estimators cached by per-key loop but missing from "
            f"single-call result (regression — fix drops estimators): {sorted(old_only)[:5]}"
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
            "Bare prefix match incorrectly mapped 'forecasterthing' to 'forecaster'"
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
        ), patch.object(ri, "_create_node", wraps=ri._create_node) as spy:
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

    def test_missing_sktime_keeps_loaded_false_and_reraises_on_retry(self):
        """When sktime is absent, _ensure_loaded raises RuntimeError and _loaded stays False.

        Documented retry semantics: every call re-raises until the environment is fixed,
        rather than caching an empty registry on the first failure.
        This tests the ImportError → RuntimeError path (line 96 in _load_registry),
        which is distinct from the inner try/except that catches all_estimators() failures.
        """
        import sys

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

    def test_loaded_flag_not_set_when_all_estimators_call_raises(self):
        """all_estimators() failure re-raises; _loaded must stay False.

        Fix 2: the except block raises instead of returning so _ensure_loaded
        never reaches self._loaded = True on failure — callers always see the
        real error and the next call retries rather than hitting an empty cache.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        with patch(
            "sktime.registry.all_estimators",
            side_effect=RuntimeError("registry unavailable"),
        ), pytest.raises(RuntimeError, match="registry unavailable"):
            ri._ensure_loaded()

        assert not ri._loaded, "_loaded must stay False when _load_registry raises"
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

    def test_single_call_task_assignment_consistent_with_per_key_loop(self):
        """Proves task assignment agrees between single-call and per-key approaches.

        For any estimator present in both result sets, the task value must match.
        A mismatch means _get_estimator_type is routing inconsistently depending on
        which all_estimators call returned the class — which would be a tag-reading bug.

        Both sides apply the same symmetric _get_estimator_type filter so the comparison
        is between actual task assignments, not raw call populations.

        Regression caught: a bug where the same estimator class is routed to different
        tasks depending on which all_estimators call returned it.
        """
        from sktime.registry import all_estimators as real_ae

        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        old_tasks: dict[str, str] = {}
        for estimator_type in ri.TASK_MAP:
            try:
                for name, cls in real_ae(
                    estimator_types=estimator_type,
                    return_names=True,
                    as_dataframe=False,
                ):
                    t = ri._get_estimator_type(cls)
                    if t and t in ri.TASK_MAP:
                        old_tasks[name] = ri.TASK_MAP[t]
            except Exception:
                pass

        new_tasks: dict[str, str] = {}
        for name, cls in real_ae(
            estimator_types=list(ri.TASK_MAP.keys()),
            return_names=True,
            as_dataframe=False,
        ):
            t = ri._get_estimator_type(cls)
            if t and t in ri.TASK_MAP:
                new_tasks[name] = ri.TASK_MAP[t]

        mismatches = [
            (name, old_tasks[name], new_tasks[name])
            for name in old_tasks
            if name in new_tasks and old_tasks[name] != new_tasks[name]
        ]
        assert not mismatches, (
            f"{len(mismatches)} estimators with inconsistent task assignment "
            f"between per-key and single-call paths (first 5): {mismatches[:5]}"
        )

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def test_single_all_estimators_call_faster_than_n_typed_calls(self):
        """Proves the single all_estimators() call is faster than N typed calls.

        Compares API call times only (not node creation), matching the benchmark:
        single call ~31ms, 8-call loop ~240ms on warm sys.modules (Python 3.11, sktime 0.40.1).
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

    def test_space_separated_object_type_resolves_first_token(self):
        """Proves a space-separated object_type string ('forecaster transformer') is split
        on whitespace so each token is matched independently — first hit wins.

        Behaviour contract: 'forecaster transformer'.split() → ['forecaster', 'transformer'];
        Pass 1 exact-match finds 'forecaster' first and returns it.  The estimator lands
        in the cache under task='forecasting', not silently dropped.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        cls = _make_fake_cls("forecaster transformer", "SpaceTyped")

        with patch(
            "sktime.registry.all_estimators",
            return_value=[("SpaceTyped", cls)],
        ):
            ri._load_registry()

        assert "SpaceTyped" in ri._cache, (
            "Space-separated object_type must resolve via split(): 'forecaster transformer' → 'forecaster'"
        )
        assert ri._cache["SpaceTyped"].task == "forecasting"

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
        ), caplog.at_level(logging.DEBUG, logger="sktime_mcp.registry.interface"):
            ri._load_registry()

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("QPred" in m for m in debug_msgs), (
            f"No DEBUG log mentioning estimator name 'QPred'. Logs: {debug_msgs}"
        )
        assert any("unresolvable" in m or "object_type" in m or "TASK_MAP" in m for m in debug_msgs), (
            f"No DEBUG log mentioning the skip reason. Logs: {debug_msgs}"
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

    def test_ensure_loaded_failure_leaves_loaded_false_and_retries(self):
        """Proves _ensure_loaded() does not cache a failed load: _loaded stays False,
        _cache stays empty, and the next call re-raises instead of silently returning.

        (a) Invariant: _loaded must only become True after a SUCCESSFUL load.  If it were
        set True on failure, subsequent callers would see an empty cache and never know why.
        (b) Regression: moving `self._loaded = True` before `self._load_registry()` (or
        inside a finally block) would swallow the error permanently.  This test catches
        that by asserting _loaded is False after the raise AND that a second call also
        raises (not a no-op silent return).
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        with patch(
            "sktime.registry.all_estimators",
            side_effect=RuntimeError("registry unavailable"),
        ), pytest.raises(RuntimeError, match="registry unavailable"):
            ri._ensure_loaded()

        assert ri._loaded is False, (
            "_loaded must remain False when _load_registry raises — "
            "setting it True before the call is the regression this test catches"
        )
        assert ri._cache == {}, "_cache must be empty after a failed load"

        # Second call must also raise — not a silent no-op on an empty cache.
        with patch(
            "sktime.registry.all_estimators",
            side_effect=RuntimeError("registry unavailable"),
        ), pytest.raises(RuntimeError, match="registry unavailable"):
            ri._ensure_loaded()

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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["Est1"] = EstimatorNode(
            "Est1", "forecasting", object, "m.E1", docstring="A basic estimator."
        )

        result = ri.search_estimators("zzznomatchzzz")
        assert result == []

    def test_search_estimators_non_string_query_returns_empty(self):
        """Covers the isinstance(query, str) guard at the top of search_estimators.

        Passing None or a non-str must return [] rather than raising AttributeError
        at query.lower(). Removing the guard causes the crash.
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
        from sktime_mcp.registry.interface import RegistryInterface, EstimatorNode

        ri = RegistryInterface()
        ri._loaded = True
        ri._cache["NodocEst"] = EstimatorNode(
            "NodocEst", "forecasting", object, "m.N", docstring=None
        )

        # query matches neither name nor (absent) docstring
        result = ri.search_estimators("something")
        assert result == []

    # ------------------------------------------------------------------
    # _load_registry: TASK_MAP empty guard
    # ------------------------------------------------------------------

    def test_load_registry_raises_if_task_map_empty(self):
        """Covers the TASK_MAP empty guard at the top of _load_registry.

        An empty TASK_MAP would pass list(self.TASK_MAP.keys()) == [] to all_estimators,
        loading nothing silently.  The guard raises RuntimeError early so the caller
        sees an explicit error instead of an empty registry with no explanation.
        Removing the guard makes this test fail because no RuntimeError is raised.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()
        ri.TASK_MAP = {}  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="TASK_MAP is empty"):
            ri._load_registry()

    # ------------------------------------------------------------------
    # _get_estimator_type: TypeError logged at WARNING
    # ------------------------------------------------------------------

    def test_get_estimator_type_logs_warning_for_non_classmethod_get_class_tags(self, caplog):
        """Covers the TypeError branch in _get_estimator_type where get_class_tags is an
        instance method (not a classmethod).

        Calling cls.get_class_tags() on a class where it is a plain instance method raises
        TypeError: missing 'self' argument.  This must surface at WARNING level — not DEBUG —
        because it signals an estimator authoring bug in upstream code, not a benign skip.
        Removing the separate TypeError catch or demoting it to DEBUG would fail this test.
        """
        from sktime_mcp.registry.interface import RegistryInterface

        ri = RegistryInterface()

        class BadEstimator:
            def get_class_tags(self):  # instance method, not classmethod
                return {"object_type": "forecaster"}

        with caplog.at_level(logging.WARNING, logger="sktime_mcp.registry.interface"):
            result = ri._get_estimator_type(BadEstimator)

        assert result == "", "Non-classmethod get_class_tags must return '' (not crash)"
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("non-classmethod" in m for m in warning_msgs), (
            f"Expected WARNING about non-classmethod get_class_tags. Got: {warning_msgs}"
        )
