"""export_code must generate correct, runnable code per scitype (#534).

- BUG-03: fit example was forecaster-shaped for every scitype (AttributeError
  when run for transformers/splitters/classifiers).
- BUG-04: is_pipeline false-positived on any "[" in the spec.
- NB-15: the dataset argument was ignored; unknown datasets silently fell back
  to airline.
- NB-17: loaded models (empty params) failed with "No craft spec found".
"""

import contextlib

import pytest

from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.codegen import export_code_tool
from sktime_mcp.tools.instantiate import instantiate_tool


def _handle(spec):
    res = instantiate_tool(spec=spec)
    assert res["success"], res
    return res["handle"]


def _release(handle):
    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(handle)


class TestIsPipeline:
    def test_list_arg_is_not_a_pipeline(self):
        h = _handle("SlidingWindowSplitter(window_length=24, fh=[1, 2, 3], step_length=12)")
        try:
            res = export_code_tool(h)
            assert res["success"]
            assert res["is_pipeline"] is False
        finally:
            _release(h)

    def test_star_spec_is_a_pipeline(self):
        h = _handle("Deseasonalizer() * NaiveForecaster()")
        try:
            res = export_code_tool(h)
            assert res["success"]
            assert res["is_pipeline"] is True
        finally:
            _release(h)


class TestScitypeExampleRuns:
    def _export_and_exec(self, spec):
        h = _handle(spec)
        try:
            res = export_code_tool(h, include_fit_example=True)
            assert res["success"], res
            compile(res["code"], "<export>", "exec")
            exec(res["code"], {})  # must run without AttributeError
        finally:
            _release(h)

    def test_forecaster_example_runs(self):
        self._export_and_exec("NaiveForecaster(sp=12)")

    def test_transformer_example_runs(self):
        self._export_and_exec("Deseasonalizer()")

    def test_splitter_example_runs(self):
        self._export_and_exec("SlidingWindowSplitter(window_length=24, step_length=12)")

    def test_classifier_example_runs(self):
        self._export_and_exec("KNeighborsTimeSeriesClassifier()")


class TestLoadedModelExport:
    def test_loaded_model_emits_load_model_snippet(self):
        from sktime.forecasting.naive import NaiveForecaster

        hm = get_handle_manager()
        # Simulate a load_model handle: no craft spec, metadata carries the path.
        handle = hm.create_handle(
            estimator_name="NaiveForecaster",
            instance=NaiveForecaster(),
            params={},
            metadata={"source": "loaded", "path": "/tmp/some_model_dir"},
        )
        try:
            res = export_code_tool(handle)
            assert res["success"], res
            assert "load_model" in res["code"]
            assert "/tmp/some_model_dir" in res["code"]
            compile(res["code"], "<export>", "exec")
        finally:
            _release(handle)


class TestDatasetValidation:
    def test_unknown_dataset_rejected(self):
        h = _handle("NaiveForecaster()")
        try:
            res = export_code_tool(h, include_fit_example=True, dataset="not_a_dataset_zzz")
            assert not res["success"]
            assert "not_a_dataset_zzz" in res["error"]
        finally:
            _release(h)

    def test_known_dataset_used(self):
        h = _handle("NaiveForecaster()")
        try:
            res = export_code_tool(h, include_fit_example=True, dataset="lynx")
            assert res["success"]
            assert "load_lynx" in res["code"]
            assert "load_airline" not in res["code"]
        finally:
            _release(h)
