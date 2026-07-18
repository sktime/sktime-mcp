"""
Executor for sktime MCP.

Responsible for instantiating estimators, loading datasets,
and running fit/predict operations.
"""

import asyncio
import inspect
import logging
import uuid
from typing import Any

import pandas as pd

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager

logger = logging.getLogger(__name__)


# Dynamically discover all available sktime demo datasets at import time.
# This replaces the old hardcoded dictionary and automatically exposes every
# load_* function in sktime.datasets to the MCP server.
def _discover_demo_datasets() -> dict:
    """Return a mapping of dataset name -> dotted module path for every
    ``load_*`` function exported by ``sktime.datasets``."""
    try:
        import sktime.datasets as _ds_module

        return {
            name.removeprefix("load_"): f"sktime.datasets.{name}"
            for name, obj in inspect.getmembers(_ds_module, inspect.isfunction)
            if name.startswith("load_")
        }
    except Exception:  # pragma: no cover
        return {}  # fallback: empty dict if sktime not installed


_DEMO_DATASETS: dict | None = None


def _get_demo_datasets() -> dict:
    """Lazy singleton — discovers datasets only on first call."""
    global _DEMO_DATASETS
    if _DEMO_DATASETS is None:
        _DEMO_DATASETS = _discover_demo_datasets()
    return _DEMO_DATASETS


def _get_index_frequency_metadata(
    index: pd.Index,
    fallback: str | None = None,
) -> str | None:
    """Return a stable frequency label for metadata without assuming datetime-only indexes."""
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        freq = getattr(index, "freq", None)
        if freq is not None:
            return str(freq)
        inferred = pd.infer_freq(index)
        if inferred is not None:
            return inferred

    return fallback


def _resolve_metric_scoring(metric_name: str) -> Any | None:
    """Return an instantiated sktime forecasting metric by name, or None if not found."""
    try:
        from sktime.registry import all_estimators
    except ImportError:  # pragma: no cover
        return None
    try:
        metrics_df = all_estimators("metric", as_dataframe=True)
        row = metrics_df[metrics_df["name"] == metric_name]
        if row.empty:
            return None
        return row.iloc[0]["object"]()
    except Exception as e:
        logger.warning(f"Failed to resolve metric '{metric_name}': {e}")
        return None


def _run_evaluate(
    instance: Any,
    y: Any,
    X: Any,
    cv_folds: int,
    scoring: Any | None,
    initial_window: int | None,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, dict[str, float]]]:
    """
    Run sktime.evaluate with an expanding-window splitter and summarize results.

    Returns
    -------
    fold_results : list of dict
        Per-fold rows from sktime.evaluate.
    metrics : dict
        Mean value per ``test_*`` metric column.
    summary : dict
        Mean, std, min, max per ``test_*`` metric column.
    """
    from sktime.forecasting.model_evaluation import evaluate

    try:
        from sktime.split import ExpandingWindowSplitter
    except ImportError:  # pragma: no cover - sktime < 0.29
        from sktime.forecasting.model_selection import ExpandingWindowSplitter

    n = len(y)
    if initial_window is not None:
        win = initial_window
    else:
        folds = max(1, min(int(cv_folds), max(1, n - 1)))
        win = max(1, n - folds)
    cv = ExpandingWindowSplitter(initial_window=win, step_length=1, fh=[1])

    results = evaluate(forecaster=instance, y=y, X=X, cv=cv, scoring=scoring)
    if "estimator" in results.columns:
        results = results.drop(columns=["estimator"])

    fold_results = results.to_dict(orient="records")
    metric_cols = [
        c for c in results.select_dtypes(include="number").columns if c.startswith("test_")
    ]
    metrics = {c: float(results[c].mean()) for c in metric_cols}
    summary = {
        c: {
            "mean": float(results[c].mean()),
            "std": float(results[c].std()),
            "min": float(results[c].min()),
            "max": float(results[c].max()),
        }
        for c in metric_cols
    }
    return fold_results, metrics, summary


class Executor:
    """
    Execution runtime for sktime estimators.

    Handles instantiation, fitting, and prediction.
    """

    def __init__(self):
        self._registry = get_registry()
        self._handle_manager = get_handle_manager()
        self._job_manager = get_job_manager()
        self._data_handles: dict[str, Any] = {}
        from sktime_mcp.config import settings

        self._max_data_handles = settings.max_data_handles
        self._auto_format_enabled = settings.auto_format

    def _cleanup_oldest_data(self, count: int = 10) -> None:
        to_remove = list(self._data_handles.keys())[:count]
        for handle_id in to_remove:
            del self._data_handles[handle_id]
            logger.debug("Evicted data handle %s (limit=%d)", handle_id, self._max_data_handles)

    def _register_data_handle(self, handle_id: str, data: dict[str, Any]) -> None:
        if len(self._data_handles) >= self._max_data_handles:
            self._cleanup_oldest_data(count=max(1, self._max_data_handles // 5))
        self._data_handles[handle_id] = data

    def _resolve_source(self, source: str) -> dict[str, Any]:
        """Resolve a source id to a series, trying data_handle then demo dataset."""
        if source in self._data_handles:
            return {"success": True, "data": self._data_handles[source]["y"]}
        res = self.load_dataset(source)
        if res["success"]:
            return {"success": True, "data": res["data"]}
        return res

    def instantiate(
        self,
        spec: str,
    ) -> dict[str, Any]:
        """Instantiate an estimator or pipeline from a spec and return a handle."""
        import importlib

        importlib.invalidate_caches()

        try:
            from sktime.utils.dependencies._dependencies import _get_installed_packages_private

            _get_installed_packages_private.cache_clear()
        except ImportError:
            pass

        import numpy as np
        import pandas as pd
        import sktime.registry._craft as _craft_module
        from sktime.registry import craft

        # Temporarily patch all_estimators to inject standard libraries into craft's registry.
        # This allows users to pass callables like `numpy.exp` into estimators
        # like CurveFitForecaster via the craft spec.
        original_all = _craft_module.all_estimators

        def mock_all_estimators(*args, **kwargs):
            results = original_all(*args, **kwargs)
            # results is a list of tuples: [(name, class), ...]
            # We append numpy and pandas so they enter the register dict!
            results.append(("np", np))
            results.append(("numpy", np))
            results.append(("pd", pd))
            results.append(("pandas", pd))
            return results

        _craft_module.all_estimators = mock_all_estimators
        try:
            try:
                instance = craft(spec)
            finally:
                _craft_module.all_estimators = original_all

            estimator_name = type(instance).__name__
            handle_id = self._handle_manager.create_handle(
                estimator_name=estimator_name,
                instance=instance,
                params={"spec": spec},
            )
            return {
                "success": True,
                "handle": handle_id,
                "estimator": estimator_name,
                "spec": spec,
            }
        except Exception as e:
            import sys
            import traceback

            error_msg = str(e)
            if (
                "requires package" in error_msg
                or "pip install" in error_msg
                or "ModuleNotFoundError" in type(e).__name__
            ):
                error_msg += f"\n\n(Hint for AI: To install missing dependencies, use the server's exact python environment by running: `{sys.executable} -m pip install <package_name>`)"

            return {
                "success": False,
                "error": error_msg,
                "traceback": traceback.format_exc(),
            }

    # L-7: We can also add custom load_dataset functions here
    def load_dataset(self, name: str) -> dict[str, Any]:
        """Load a demo dataset."""
        demo_datasets = _get_demo_datasets()
        if name not in demo_datasets:
            return {
                "success": False,
                "error": f"Unknown dataset: {name}",
                "available": list(demo_datasets.keys()),
            }

        try:
            module_path = demo_datasets[name]
            parts = module_path.rsplit(".", 1)
            module = __import__(parts[0], fromlist=[parts[1]])
            loader = getattr(module, parts[1])
            data = loader()

            if isinstance(data, tuple):
                # sktime classifier/clusterer datasets typically return (X, y)
                # whereas forecaster datasets typically return (y) or (y, X)
                # Let's check the shape/type to be safe, or just hardcode known ones
                if name in (
                    "arrow_head",
                    "italy_power_demand",
                    "basic_motions",
                    "gunpoint",
                    "osuleaf",
                    "plaid",
                ):
                    X, y = data[0], data[1] if len(data) > 1 else None
                    # swap them back for our internal representation where 'data' is the primary object requested
                    return {
                        "success": True,
                        "name": name,
                        "data": X,
                        "exog": y,
                        "type": str(type(X).__name__),
                    }
                else:
                    y, X = data[0], data[1] if len(data) > 1 else None
            else:
                y, X = data, None

            return {
                "success": True,
                "name": name,
                "shape": y.shape if hasattr(y, "shape") else len(y),
                "type": str(type(y).__name__),
                "data": y,
                "exog": X,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fit(
        self,
        handle_id: str,
        y: Any,
        X: Any | None = None,
        fh: Any | None = None,
    ) -> dict[str, Any]:
        """Fit an estimator."""
        try:
            handle_info = self._handle_manager.get_info(handle_id)
            instance = handle_info.instance
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}

        obj_type = getattr(instance, "get_class_tag", lambda x, y: "")("object_type", "")
        if not hasattr(instance, "fit"):
            return {
                "success": False,
                "error": f"The {obj_type or 'estimator'} scitype does not support fit(). Please use the 'call_method' tool to interact with its native methods.",
            }

        # Check scitype to determine how to call fit
        # By default in sktime:
        # - Forecasters: fit(y, X=None, fh=None)
        # - Classifiers/Regressors: fit(X, y)
        # - Transformers/Clusterers: fit(X, y=None)

        is_classifier_or_regressor = False
        is_transformer = False
        if hasattr(instance, "get_class_tag"):
            obj_type = instance.get_class_tag("object_type", "")
            if obj_type in ("classifier", "regressor"):
                is_classifier_or_regressor = True
            elif obj_type == "transformer":
                is_transformer = True

        try:
            if is_classifier_or_regressor:
                # With decoupled X and y handles, X is features and y is labels
                instance.fit(X, y)
            elif is_transformer:
                if X is not None:
                    instance.fit(y, X)
                else:
                    instance.fit(y)
            elif obj_type == "clusterer":
                if y is not None:
                    instance.fit(X, y)
                else:
                    instance.fit(X)
            else:
                # Assume forecaster or similar default
                if fh is not None:
                    instance.fit(y, X=X, fh=fh)
                elif X is not None:
                    instance.fit(y, X=X)
                else:
                    instance.fit(y)

            self._handle_manager.mark_fitted(handle_id)
            return {"success": True, "handle": handle_id, "fitted": True}
        except Exception as e:
            import traceback

            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def predict(
        self,
        handle_id: str,
        fh: int | list[int] | None = None,
        X: Any | None = None,
        y: Any | None = None,
        mode: str = "predict",
        coverage: float | list[float] = 0.9,
        alpha: float | list[float] | None = None,
    ) -> dict[str, Any]:
        """Generate predictions."""
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}

        obj_type = getattr(instance, "get_class_tag", lambda x, y: "")("object_type", "")
        if (
            not hasattr(instance, "predict")
            and mode == "predict"
            and not (hasattr(instance, "transform") and obj_type == "transformer")
        ):
            return {
                "success": False,
                "error": f"The {obj_type or 'estimator'} scitype does not support predict(). Please use the 'call_method' tool to interact with its native methods.",
            }

        if not self._handle_manager.is_fitted(handle_id):
            return {"success": False, "error": "Estimator not fitted"}

        is_classifier_or_regressor = False
        is_transformer = False
        if hasattr(instance, "get_class_tag"):
            obj_type = instance.get_class_tag("object_type", "")
            if obj_type in ("classifier", "regressor"):
                is_classifier_or_regressor = True
            elif obj_type in ("transformer", "clusterer"):
                is_transformer = True

        try:
            if fh is None and not (is_classifier_or_regressor or is_transformer):
                fh = list(range(1, 13))

            kwargs = {}
            if X is not None:
                kwargs["X"] = X
            if y is not None:
                kwargs["y"] = y

            if is_classifier_or_regressor:
                # Classifiers take X in predict (X is the feature matrix)
                # But instance.predict(X) is the signature.
                # Since kwargs["X"] has it, we can just pass X positionally
                if mode == "predict":
                    predictions = instance.predict(X)
                elif mode == "predict_proba":
                    predictions = instance.predict_proba(X)
                else:
                    return {"success": False, "error": f"Mode {mode} not supported for {obj_type}"}
            elif is_transformer:
                if mode == "predict":
                    if obj_type == "clusterer":
                        predictions = (
                            instance.predict(X) if X is not None else instance.predict(fh=fh)
                        )  # some clusterers might use predict(X)
                    else:
                        # For transformer, transform is basically the predict equivalent if X is passed
                        if X is not None:
                            predictions = instance.transform(X)
                        else:
                            return {"success": False, "error": "Transform requires X"}
                else:
                    return {"success": False, "error": f"Mode {mode} not supported for {obj_type}"}
            else:
                if mode == "predict":
                    predictions = instance.predict(fh=fh, **kwargs)
                elif mode == "predict_interval":
                    predictions = instance.predict_interval(fh=fh, coverage=coverage, **kwargs)
                elif mode == "predict_quantiles":
                    predictions = instance.predict_quantiles(fh=fh, alpha=alpha, **kwargs)
                elif mode == "predict_proba":
                    predictions = instance.predict_proba(fh=fh, **kwargs)
                elif mode == "predict_var":
                    predictions = instance.predict_var(fh=fh, **kwargs)
                else:
                    return {"success": False, "error": f"Unknown prediction mode: {mode}"}

            from sktime_mcp.server import sanitize_for_json

            if isinstance(predictions, pd.Series):
                predictions_copy = predictions.copy()
                predictions_copy.index = predictions_copy.index.astype(str)
                result = predictions_copy.to_dict()
            elif isinstance(predictions, pd.DataFrame):
                predictions_copy = predictions.copy()
                predictions_copy.index = predictions_copy.index.astype(str)
                # Need to handle multiindex columns if they exist (like in predict_interval)
                if isinstance(predictions_copy.columns, pd.MultiIndex):
                    # Flatten multiindex for JSON serialization
                    predictions_copy.columns = [
                        "_".join(map(str, col)) for col in predictions_copy.columns.values
                    ]
                result = predictions_copy.to_dict(orient="list")
            else:
                result = sanitize_for_json(predictions)

            out = {
                "success": True,
                "horizon": len(fh) if hasattr(fh, "__len__") else fh,
                "mode": mode,
            }
            if mode == "predict":
                out["predictions"] = result
            elif mode == "predict_interval":
                out["intervals"] = result
                out["coverage"] = coverage
            elif mode == "predict_quantiles":
                out["quantiles"] = result
                out["alpha"] = alpha
            else:
                out["predictions"] = result
            return out
        except Exception as e:
            return {"success": False, "error": str(e)}

    def call_method(
        self,
        handle_id: str,
        method_name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dynamically call a method on the underlying estimator."""
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}

        if not hasattr(instance, method_name):
            obj_type = getattr(instance, "get_class_tag", lambda x, y: "")("object_type", "")
            return {
                "success": False,
                "error": f"The {obj_type or 'estimator'} does not have a method '{method_name}'.",
            }

        kwargs = kwargs or {}

        try:
            method = getattr(instance, method_name)

            # Map data_handle and dataset from kwargs if they exist
            # This allows the LLM to pass 'dataset': 'airline' and we inject the actual data
            for k, v in list(kwargs.items()):
                if k.endswith("_dataset") and isinstance(v, str):
                    data_res = self.load_dataset(v)
                    if data_res.get("success"):
                        # Replace the kwarg with the actual data (e.g. y_dataset -> y)
                        actual_key = k.replace("_dataset", "")
                        kwargs[actual_key] = data_res["data"]
                        del kwargs[k]
                elif k.endswith("_data_handle") and isinstance(v, str):
                    if v in self._data_handles:
                        actual_key = k.replace("_data_handle", "")
                        kwargs[actual_key] = self._data_handles[v]["y"]
                        del kwargs[k]
                    else:
                        return {"success": False, "error": f"Unknown data handle: {v}"}

            result = method(**kwargs)

            from sktime_mcp.server import sanitize_for_json

            if hasattr(result, "to_dict"):
                if isinstance(result, __import__("pandas").DataFrame) and isinstance(
                    result.columns, __import__("pandas").MultiIndex
                ):
                    result.columns = ["_".join(map(str, col)) for col in result.columns.values]
                    sanitized = result.to_dict(orient="list")
                else:
                    sanitized = result.to_dict()
            else:
                sanitized = sanitize_for_json(result)

            return {"success": True, "result": sanitized}
        except Exception as e:
            import traceback

            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def update(
        self,
        handle_id: str,
        y: Any,
        X: Any | None = None,
        update_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a fitted estimator with new data."""
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}

        if not self._handle_manager.is_fitted(handle_id):
            return {"success": False, "error": "Estimator not fitted"}

        try:
            kwargs = update_params or {}
            if X is not None:
                instance.update(y, X=X, **kwargs)
            else:
                instance.update(y, **kwargs)
            return {
                "success": True,
                "handle": handle_id,
                "message": "Estimator updated successfully",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_fitted_params(self, handle_id: str) -> dict[str, Any]:
        """Get fitted parameters from an estimator."""
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}

        if not self._handle_manager.is_fitted(handle_id):
            return {"success": False, "error": "Estimator not fitted"}

        try:
            from sktime_mcp.server import sanitize_for_json

            params = instance.get_fitted_params()
            return {"success": True, "fitted_params": sanitize_for_json(params)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def fit_async(
        self,
        handle_id: str,
        X_dataset: str | None = None,
        y_dataset: str | None = None,
        X_handle: str | None = None,
        y_handle: str | None = None,
        fh: Any | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of fit with job tracking."""
        try:
            import asyncio

            from sktime_mcp.runtime.jobs import JobStatus

            # Update status to RUNNING
            self._job_manager.update_job(job_id, status=JobStatus.RUNNING)

            # Step 1: Load data
            self._job_manager.update_job(
                job_id,
                completed_steps=0,
                current_step="Loading data...",
            )
            await asyncio.sleep(0.01)

            X = None
            y = None

            if X_handle:
                if X_handle not in self._data_handles:
                    raise ValueError(f"Unknown X data handle: {X_handle}")
                X = self._data_handles[X_handle]["y"]

            if y_handle:
                if y_handle not in self._data_handles:
                    raise ValueError(f"Unknown y data handle: {y_handle}")
                y = self._data_handles[y_handle]["y"]

            if X_dataset and X_dataset == y_dataset:
                data_res = self.load_dataset(X_dataset)
                if not data_res["success"]:
                    raise ValueError(data_res["error"])
                if data_res.get("exog") is not None:
                    X = data_res["data"]
                    y = data_res["exog"]
                else:
                    y = data_res["data"]
            else:
                if X_dataset:
                    data_res = self.load_dataset(X_dataset)
                    if not data_res["success"]:
                        raise ValueError(data_res["error"])
                    X = data_res["data"]

                if y_dataset:
                    data_res = self.load_dataset(y_dataset)
                    if not data_res["success"]:
                        raise ValueError(data_res["error"])
                    y = data_res["data"]

            # Step 2: Fit model
            self._job_manager.update_job(
                job_id,
                completed_steps=1,
                current_step="Fitting model (this may take a while)...",
            )

            # Run fit in thread pool so it doesn't block async loop
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:

                def run_fit():
                    return self.fit(handle_id, y, X=X, fh=fh)

                fit_result = await loop.run_in_executor(pool, run_fit)

            if not fit_result["success"]:
                raise ValueError(fit_result["error"])

            if X_dataset or y_dataset:
                try:
                    handle_info = self._handle_manager.get_info(handle_id)
                    handle_info.metadata["training_dataset"] = y_dataset or X_dataset
                except Exception:
                    pass

            self._job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                completed_steps=2,
                current_step="Training completed successfully.",
                result={"success": True, "handle": handle_id, "fitted": True},
            )
            return {"success": True, "handle": handle_id}

        except Exception as e:
            import traceback

            from sktime_mcp.runtime.jobs import JobStatus

            self._job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                current_step="Training failed.",
                errors=[str(e), traceback.format_exc()],
            )
            return {"success": False, "error": str(e)}

    async def evaluate_async(
        self,
        handle_id: str,
        y: str,
        *,
        X: str | None = None,
        cv_folds: int = 3,
        metric: str | None = None,
        initial_window: int | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of evaluate with job tracking."""
        try:
            self._job_manager.update_job(job_id, status=JobStatus.RUNNING)

            # Step 1: Load data
            self._job_manager.update_job(job_id, completed_steps=0, current_step="Loading data...")
            await asyncio.sleep(0.01)

            try:
                instance = self._handle_manager.get_instance(handle_id)
            except KeyError as err:
                raise ValueError(f"Handle not found: {handle_id}") from err

            y_res = self._resolve_source(y)
            if not y_res["success"]:
                raise ValueError(y_res["error"])
            _y = y_res["data"]

            _X = None
            if X:
                x_res = self._resolve_source(X)
                if not x_res["success"]:
                    raise ValueError(x_res["error"])
                _X = x_res["data"]

            scoring = None
            if metric:
                scoring = _resolve_metric_scoring(metric)
                if scoring is None:
                    raise ValueError(f"Unknown metric: {metric}")

            # Step 2: Run cross-validation
            self._job_manager.update_job(
                job_id, completed_steps=1, current_step="Running cross-validation..."
            )
            await asyncio.sleep(0.01)

            loop = asyncio.get_running_loop()
            fold_results, metrics, summary = await loop.run_in_executor(
                None,
                lambda: _run_evaluate(instance, _y, _X, cv_folds, scoring, initial_window),
            )

            # Step 3: Summarize results
            self._job_manager.update_job(
                job_id, completed_steps=2, current_step="Summarizing results..."
            )
            await asyncio.sleep(0.01)

            result = {
                "success": True,
                "metrics": metrics,
                "fold_results": fold_results,
                "summary": summary,
                "cv_folds_run": len(fold_results),
                "cv_folds_requested": cv_folds,
            }
            self._job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                completed_steps=3,
                current_step="Evaluation completed.",
                result=result,
            )
            return result

        except Exception as e:
            import traceback

            self._job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                current_step="Evaluation failed.",
                errors=[str(e), traceback.format_exc()],
            )
            return {"success": False, "error": str(e)}

    # L-9: We can add more methods here to handle diverse use cases and their pipelines

    def list_datasets(self) -> list[str]:
        """List available demo datasets."""
        return list(_get_demo_datasets().keys())

    def load_data_source(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Load data from any source (pandas, SQL, file, etc.).

        Args:
            config: Data source configuration with 'type' key
                Examples:
                - {"type": "pandas", "data": df, "time_column": "date", "target_column": "value"}
                - {"type": "sql", "connection_string": "...", "query": "...", "time_column": "date"}
                - {"type": "file", "path": "/path/to/data.csv", "time_column": "date"}

        Returns:
            Dictionary with:
            - success: bool
            - data_handle: str (handle ID for the loaded data)
            - metadata: dict (information about the data)
            - validation: dict (validation results)
        """
        try:
            from sktime_mcp.data import DataSourceRegistry

            # Create adapter
            adapter = DataSourceRegistry.create_adapter(config)

            # Load data
            data = adapter.load()

            # Validate
            is_valid, validation_report = adapter.validate(data)
            if not is_valid:
                return {
                    "success": False,
                    "error": "Data validation failed",
                    "validation": validation_report,
                }

            # Convert to sktime format
            y, X = adapter.to_sktime_format(data)

            # Update metadata to reflect the target and used columns
            metadata = adapter.get_metadata().copy()
            metadata["columns"] = [y.name if hasattr(y, "name") and y.name else "target"]
            if X is not None:
                metadata["exog_columns"] = list(X.columns)
            # Inject column dtypes so LLMs can distinguish time index vs target
            metadata["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
            # Generate handle
            data_handle = f"data_{uuid.uuid4().hex[:8]}"

            # Store (enforces max_data_handles limit)
            self._register_data_handle(
                data_handle,
                {
                    "y": y,
                    "X": X,
                    "metadata": metadata,
                    "validation": validation_report,
                    "config": config,
                },
            )

            # Apply auto-formatting if enabled
            if getattr(self, "_auto_format_enabled", True):
                try:
                    format_result = self.format_data_handle(
                        data_handle, auto_infer_freq=True, fill_missing=True, remove_duplicates=True
                    )
                    if format_result["success"]:
                        # Free the raw handle — the formatted copy supersedes it
                        if data_handle in self._data_handles:
                            del self._data_handles[data_handle]
                        return {
                            "success": True,
                            "data_handle": format_result["data_handle"],
                            "metadata": format_result["metadata"],
                            "validation": validation_report,
                            "formatted": True,
                            "changes_made": format_result["changes_made"],
                        }
                except Exception as e:
                    logger.warning(f"Auto-formatting failed: {e}")
                    # Continue with unformatted data if formatting fails
            _final_meta = adapter.get_metadata().copy()
            _final_meta["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
            return {
                "success": True,
                "data_handle": data_handle,
                "metadata": _final_meta,
                "validation": validation_report,
            }

        except Exception as e:
            logger.exception("Error loading data source")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def load_data_source_async(
        self,
        config: dict[str, Any],
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Async version of load_data_source with job tracking.

        Runs data loading in the background without blocking the
        MCP server. Progress is tracked via the JobManager.

        Args:
            config: Data source configuration
            job_id: Optional job ID (created if not provided)

        Returns:
            Dictionary with data_handle and metadata
        """
        source_type = config.get("type", "unknown")

        if job_id is None:
            job_id = self._job_manager.create_job(
                job_type="data_loading",
                estimator_handle="",
                dataset_name=source_type,
                total_steps=3,
            )

        try:
            self._job_manager.update_job(job_id, status=JobStatus.RUNNING)

            # Step 1: Load raw data
            self._job_manager.update_job(
                job_id, completed_steps=0, current_step=f"Loading data from '{source_type}'..."
            )
            await asyncio.sleep(0.01)

            from sktime_mcp.data import DataSourceRegistry

            loop = asyncio.get_event_loop()
            adapter = DataSourceRegistry.create_adapter(config)
            data = await loop.run_in_executor(None, adapter.load)

            # Step 2: Validate
            self._job_manager.update_job(
                job_id, completed_steps=1, current_step="Validating data..."
            )
            await asyncio.sleep(0.01)

            is_valid, validation_report = adapter.validate(data)
            if not is_valid:
                self._job_manager.update_job(
                    job_id, status=JobStatus.FAILED, errors=["Data validation failed"]
                )
                return {
                    "success": False,
                    "error": "Data validation failed",
                    "validation": validation_report,
                }

            # Step 3: Convert, store, and format
            self._job_manager.update_job(
                job_id, completed_steps=2, current_step="Converting to sktime format..."
            )
            await asyncio.sleep(0.01)

            y, X = adapter.to_sktime_format(data)

            metadata = adapter.get_metadata().copy()
            metadata["columns"] = [y.name if hasattr(y, "name") and y.name else "target"]
            if X is not None:
                metadata["exog_columns"] = list(X.columns)
            # Inject column dtypes so LLMs can distinguish time index vs target
            metadata["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
            data_handle = f"data_{uuid.uuid4().hex[:8]}"

            self._register_data_handle(
                data_handle,
                {
                    "y": y,
                    "X": X,
                    "metadata": metadata,
                    "validation": validation_report,
                    "config": config,
                },
            )

            # auto-format if enabled
            if getattr(self, "_auto_format_enabled", True):
                try:
                    format_result = self.format_data_handle(
                        data_handle, auto_infer_freq=True, fill_missing=True, remove_duplicates=True
                    )
                    if format_result["success"]:
                        data_handle = format_result["data_handle"]
                        metadata = format_result["metadata"]
                except Exception as e:
                    logger.warning(f"Auto-formatting failed: {e}")

            result = {
                "success": True,
                "data_handle": data_handle,
                "metadata": metadata,
                "validation": validation_report,
            }

            # mark completed with the data_handle in the result
            self._job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                completed_steps=3,
                current_step="Completed",
                result=result,
            )

            return result

        except Exception as e:
            logger.exception(f"Error in async data loading for job {job_id}")
            self._job_manager.update_job(job_id, status=JobStatus.FAILED, errors=[str(e)])
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id,
            }

    def format_data_handle(
        self,
        data_handle: str,
        auto_infer_freq: bool = True,
        fill_missing: bool = True,
        remove_duplicates: bool = True,
    ) -> dict[str, Any]:
        """
        Format data associated with a handle.
        """
        if data_handle not in self._data_handles:
            return {"success": False, "error": f"Data handle '{data_handle}' not found"}

        data_info = self._data_handles[data_handle]
        y = data_info["y"].copy()
        X = data_info["X"].copy() if data_info["X"] is not None else None

        changes_made = {
            "frequency_set": False,
            "duplicates_removed": 0,
            "missing_filled": 0,
            "gaps_filled": 0,
        }
        original_frequency = data_info["metadata"].get("frequency")

        # 1. Remove duplicates
        if remove_duplicates and y.index.duplicated().any():
            n_duplicates = y.index.duplicated().sum()
            y = y[~y.index.duplicated(keep="first")]
            if X is not None:
                X = X[~X.index.duplicated(keep="first")]
            changes_made["duplicates_removed"] = n_duplicates

        # 2. Sort by index
        y = y.sort_index()
        if X is not None:
            X = X.sort_index()

        # 3. Infer and set frequency
        if auto_infer_freq:
            freq = getattr(y.index, "freq", None)

            if freq is None and isinstance(y.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                # Try to infer
                freq = pd.infer_freq(y.index)

                if freq is None:
                    # Manual inference
                    time_diffs = y.index.to_series().diff().dropna()
                    if len(time_diffs) > 0:
                        most_common_diff = time_diffs.mode()[0]

                        if most_common_diff == pd.Timedelta(days=1):
                            freq = "D"
                        elif most_common_diff == pd.Timedelta(hours=1):
                            freq = "h"
                        elif most_common_diff == pd.Timedelta(minutes=1):
                            freq = "min"
                        elif most_common_diff == pd.Timedelta(seconds=1):
                            freq = "s"
                        elif most_common_diff == pd.Timedelta(days=7):
                            freq = "W"
                        elif most_common_diff.days >= 28 and most_common_diff.days <= 31:
                            freq = "MS"
                        else:
                            freq = "D"

                # Create complete date range
                if freq:
                    full_range = pd.date_range(start=y.index.min(), end=y.index.max(), freq=freq)

                    n_gaps = len(full_range) - len(y)

                    y = y.reindex(full_range)
                    if X is not None:
                        X = X.reindex(full_range)

                    changes_made["gaps_filled"] = n_gaps
                    changes_made["frequency_set"] = True
                    changes_made["frequency"] = freq

        # 4. Fill missing values
        if fill_missing and y.isna().any():
            n_missing = y.isna().sum()
            y = y.ffill().bfill()
            if X is not None:
                X = X.ffill().bfill()
            changes_made["missing_filled"] = n_missing

        # 5. Set frequency explicitly on index
        if hasattr(y.index, "freq") and changes_made.get("frequency"):
            y.index.freq = changes_made["frequency"]
            if X is not None:
                X.index.freq = changes_made["frequency"]

        # Generate new handle
        new_handle = f"data_{uuid.uuid4().hex[:8]}"

        new_data = {
            "y": y,
            "X": X,
            "metadata": {
                **data_info["metadata"],
                "formatted": True,
                "frequency": _get_index_frequency_metadata(
                    y.index,
                    fallback=changes_made.get("frequency") or original_frequency,
                ),
                "rows": len(y),
                "start_date": str(y.index.min()),
                "end_date": str(y.index.max()),
            },
            "validation": data_info.get("validation", {}),
            "config": data_info.get("config", {}),
            "original_handle": data_handle,
        }
        self._register_data_handle(new_handle, new_data)

        # Release the original to prevent intermediate handles from accumulating
        if data_handle in self._data_handles:
            del self._data_handles[data_handle]

        return {
            "success": True,
            "data_handle": new_handle,
            "metadata": new_data["metadata"],
            "changes_made": changes_made,
        }

    def list_data_handles(self) -> dict[str, Any]:
        """
        List all loaded data handles.

        Returns:
            Dictionary with list of data handles and their metadata
        """
        handles = []
        for handle_id, data_info in self._data_handles.items():
            handles.append(
                {
                    "handle": handle_id,
                    "metadata": data_info["metadata"],
                    "validation": data_info["validation"],
                }
            )

        return {
            "success": True,
            "count": len(handles),
            "handles": handles,
        }

    def release_data_handle(self, data_handle: str) -> dict[str, Any]:
        """
        Release a data handle and free memory.

        Args:
            data_handle: Data handle to release

        Returns:
            Dictionary with success status
        """
        if data_handle in self._data_handles:
            del self._data_handles[data_handle]
            return {
                "success": True,
                "message": f"Data handle '{data_handle}' released",
            }
        else:
            return {
                "success": False,
                "error": f"Data handle '{data_handle}' not found",
            }


_executor_instance: Executor | None = None


def get_executor() -> Executor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = Executor()
    return _executor_instance
