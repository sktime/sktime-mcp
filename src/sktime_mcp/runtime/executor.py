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


def _validate_horizon(horizon: Any) -> str | None:
    """Return error message if horizon is invalid, else None."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        return f"Invalid fh={horizon!r}. fh must be a positive integer."
    return None


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


# Classification demo datasets (return X_train, y_train with split parameter)
CLASSIFICATION_DATASETS = {
    "arrow_head": "sktime.datasets.load_arrow_head",
    "gunpoint": "sktime.datasets.load_gunpoint",
    "basic_motions": "sktime.datasets.load_basic_motions",
    "italy_power_demand": "sktime.datasets.load_italy_power_demand",
}

# Regression demo datasets (return X_train, y_train with split parameter)
REGRESSION_DATASETS = {
    "covid_3month": "sktime.datasets.load_covid_3month",
    "cardano_sentiment": "sktime.datasets.load_cardano_sentiment",
}


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

    def instantiate(
        self,
        spec: str,
    ) -> dict[str, Any]:
        """Instantiate an estimator or pipeline from a spec and return a handle."""
        from sktime.registry import craft
        import sktime.registry._craft as _craft_module
        import numpy as np
        import pandas as pd
        
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
            import traceback
            return {
                "success": False,
                "error": str(e),
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
                "error": f"The {obj_type or 'estimator'} scitype does not support fit(). Please use the 'call_method' tool to interact with its native methods."
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
            elif obj_type in ("transformer", "clusterer"):
                is_transformer = True

        try:
            if is_classifier_or_regressor:
                # For classifiers/regressors, y from load_dataset might actually be the feature (X_data) 
                # because the loader returns X, y. But fit_predict.py extracts y=data, X=exog.
                # So if X is None, 'y' contains X, and we can't fit because labels are missing.
                # Actually, our executor.load_dataset maps (X, y) -> y=X, exog=y for classifiers.
                # So the argument 'y' here is actually X (features), and 'X' here is y (labels).
                # Therefore we call instance.fit(y, X).
                instance.fit(y, X)
            elif is_transformer:
                if X is not None:
                    instance.fit(y, X)
                else:
                    instance.fit(y)
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
        if not hasattr(instance, "predict") and mode == "predict" and not (hasattr(instance, "transform") and obj_type == "transformer"):
            return {
                "success": False, 
                "error": f"The {obj_type or 'estimator'} scitype does not support predict(). Please use the 'call_method' tool to interact with its native methods."
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

            if is_classifier_or_regressor:
                # Classifiers take X in predict (and we mapped X in fit_predict to X)
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
                        predictions = instance.predict(X) if X is not None else instance.predict(fh=fh) # some clusterers might use predict(X)
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
                    predictions_copy.columns = ["_".join(map(str, col)) for col in predictions_copy.columns.values]
                result = predictions_copy.to_dict(orient="list")
            else:
                result = sanitize_for_json(predictions)

            out = {
                "success": True,
                "horizon": len(fh) if hasattr(fh, "__len__") else fh,
                "mode": mode
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
            return {"success": False, "error": f"The {obj_type or 'estimator'} does not have a method '{method_name}'."}

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
                if isinstance(result, __import__("pandas").DataFrame) and isinstance(result.columns, __import__("pandas").MultiIndex):
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
            return {"success": True, "handle": handle_id, "message": "Estimator updated successfully"}
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


    def fit_predict(
        self,
        handle_id: str,
        dataset: str,
        horizon: int = 12,
        data_handle: str | None = None,
    ) -> dict[str, Any]:
        """Convenience method: load data, fit, and predict."""
        horizon_error = _validate_horizon(horizon)
        if horizon_error:
            return {"success": False, "error": horizon_error}

        if dataset and data_handle:
            return {
                "success": False,
                "error": "Provide either 'dataset' or 'data_handle', not both.",
            }

        if data_handle is None and (not dataset or not str(dataset).strip()):
            return {
                "success": False,
                "error": (
                    "Either 'dataset' (e.g. 'airline') or "
                    "'data_handle' (from load_data_source) is required."
                ),
            }
        if data_handle is not None:
            # Use custom loaded data
            if data_handle not in self._data_handles:
                return {
                    "success": False,
                    "error": f"Unknown data handle: {data_handle}",
                    "available_handles": list(self._data_handles.keys()),
                }
            data_info = self._data_handles[data_handle]
            y = data_info["y"]
            X = data_info.get("X")
        else:
            # Use demo dataset
            data_result = self.load_dataset(dataset)
            if not data_result["success"]:
                return data_result
            y = data_result["data"]
            X = data_result.get("exog")

        fh = list(range(1, horizon + 1))

        fit_result = self.fit(handle_id, y, X=X, fh=fh)
        if not fit_result["success"]:
            return fit_result

        # Record which dataset was used so export_code can reference it
        if dataset:
            handle_info = self._handle_manager.get_info(handle_id)
            handle_info.metadata["training_dataset"] = dataset

        return self.predict(handle_id, fh=fh, X=X)

    async def fit_async(
        self,
        handle_id: str,
        dataset: str | None = None,
        data_handle: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of fit with job tracking."""
        try:
            import asyncio
            from sktime_mcp.runtime.jobs import JobStatus
            
            # Update status to RUNNING
            self._job_manager.update_job(job_id, status=JobStatus.RUNNING)

            # Step 1: Load data
            if data_handle:
                self._job_manager.update_job(
                    job_id,
                    completed_steps=0,
                    current_step=f"Loading data from handle '{data_handle}'...",
                )
                await asyncio.sleep(0.01)
                
                if data_handle not in self._data_handles:
                    raise ValueError(f"Unknown data handle: {data_handle}")
                data_info = self._data_handles[data_handle]
                y = data_info["y"]
                X = data_info.get("X")
            else:
                self._job_manager.update_job(
                    job_id,
                    completed_steps=0,
                    current_step=f"Loading dataset '{dataset}'...",
                )
                await asyncio.sleep(0.01)
                
                data_result = self.load_dataset(dataset)
                if not data_result["success"]:
                    raise ValueError(data_result["error"])
                y = data_result["data"]
                X = data_result.get("exog")
                
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
                    return self.fit(handle_id, y, X=X)
                fit_result = await loop.run_in_executor(pool, run_fit)
            
            if not fit_result["success"]:
                raise ValueError(fit_result["error"])
                
            if dataset:
                try:
                    handle_info = self._handle_manager.get_info(handle_id)
                    handle_info.metadata["training_dataset"] = dataset
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

    async def fit_predict_async(
        self,
        handle_id: str,
        dataset: str | None = None,
        data_handle: str | None = None,
        horizon: int = 12,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Async version of fit_predict with job tracking.

        Runs the training in the background without blocking the MCP server.
        Accepts either a demo dataset name or a data handle from
        load_data_source.

        Args:
            handle_id: Estimator handle
            dataset: Demo dataset name
            data_handle: Data handle from load_data_source
            horizon: Forecast horizon
            job_id: Optional job ID for tracking (created if not provided)

        Returns:
            Dictionary with success status and job_id
        """
        horizon_error = _validate_horizon(horizon)
        if horizon_error:
            return {"success": False, "error": horizon_error}

        # Get estimator info for job tracking
        try:
            handle_info = self._handle_manager.get_info(handle_id)
            estimator_name = handle_info.estimator_name
        except Exception as e:
            logger.warning(f"Could not get estimator name: {e}")
            estimator_name = "Unknown"

        source_name = dataset if dataset else data_handle

        # Create job if not provided
        if job_id is None:
            job_id = self._job_manager.create_job(
                job_type="fit_predict",
                estimator_handle=handle_id,
                estimator_name=estimator_name,
                dataset_name=source_name,
                horizon=horizon,
                total_steps=3,
            )

        try:
            # Update status to RUNNING
            self._job_manager.update_job(job_id, status=JobStatus.RUNNING)

            # Step 1: Load data
            if data_handle:
                # Use custom data from a loaded handle
                self._job_manager.update_job(
                    job_id,
                    completed_steps=0,
                    current_step=f"Loading data from handle '{data_handle}'...",
                )
                await asyncio.sleep(0.01)

                if data_handle not in self._data_handles:
                    self._job_manager.update_job(
                        job_id,
                        status=JobStatus.FAILED,
                        errors=[f"Unknown data handle: {data_handle}"],
                    )
                    return {
                        "success": False,
                        "error": f"Unknown data handle: {data_handle}",
                        "available_handles": list(self._data_handles.keys()),
                    }

                data_info = self._data_handles[data_handle]
                y = data_info["y"]
                X = data_info.get("X")
            else:
                # Use built-in demo dataset
                self._job_manager.update_job(
                    job_id,
                    completed_steps=0,
                    current_step=f"Loading dataset '{dataset}'...",
                )
                await asyncio.sleep(0.01)

                data_result = self.load_dataset(dataset)
                if not data_result["success"]:
                    self._job_manager.update_job(
                        job_id,
                        status=JobStatus.FAILED,
                        errors=[f"Failed to load dataset: {data_result.get('error')}"],
                    )
                    return data_result

                y = data_result["data"]
                X = data_result.get("exog")

            fh = list(range(1, horizon + 1))

            # Step 2: Fit model
            self._job_manager.update_job(
                job_id,
                completed_steps=1,
                current_step=f"Fitting {estimator_name} on {source_name}...",
            )
            await asyncio.sleep(0.01)

            loop = asyncio.get_event_loop()
            fit_result = await loop.run_in_executor(
                None, lambda: self.fit(handle_id, y, X=X, fh=fh)
            )

            if not fit_result["success"]:
                self._job_manager.update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    errors=[f"Fit failed: {fit_result.get('error')}"],
                )
                return fit_result

            # Step 3: Generate predictions
            self._job_manager.update_job(
                job_id,
                completed_steps=2,
                current_step=f"Generating predictions (horizon={horizon})...",
            )
            await asyncio.sleep(0.01)

            predict_result = await loop.run_in_executor(
                None, lambda: self.predict(handle_id, fh=fh, X=X)
            )

            if not predict_result["success"]:
                self._job_manager.update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    errors=[f"Prediction failed: {predict_result.get('error')}"],
                )
                return predict_result

            # Mark as completed
            self._job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                completed_steps=3,
                current_step="Completed",
                result=predict_result,
            )

            return predict_result

        except Exception as e:
            logger.exception(f"Error in async fit_predict for job {job_id}")
            self._job_manager.update_job(job_id, status=JobStatus.FAILED, errors=[str(e)])
            return {"success": False, "error": str(e), "job_id": job_id}

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

    def fit_predict_with_data(
        self,
        estimator_handle: str,
        data_handle: str,
        horizon: int = 12,
    ) -> dict[str, Any]:
        """
        Fit and predict using a data handle.

        Args:
            estimator_handle: Estimator handle from instantiate_estimator
            data_handle: Data handle from load_data_source
            horizon: Forecast horizon

        Returns:
            Dictionary with predictions
        """
        horizon_error = _validate_horizon(horizon)
        if horizon_error:
            return {"success": False, "error": horizon_error}

        if data_handle not in self._data_handles:
            return {
                "success": False,
                "error": f"Unknown data handle: {data_handle}",
                "available_handles": list(self._data_handles.keys()),
            }

        data = self._data_handles[data_handle]
        y = data["y"]
        X = data.get("X")

        # Fit
        fh = list(range(1, horizon + 1))
        fit_result = self.fit(estimator_handle, y=y, X=X, fh=fh)
        if not fit_result["success"]:
            return fit_result

        # Predict
        return self.predict(estimator_handle, fh=fh, X=X)

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

    def _load_supervised_dataset(
        self,
        name: str,
        dataset_registry: dict[str, str],
    ) -> dict[str, Any]:
        """
        Load a supervised (classification/regression) demo dataset.

        These datasets support split='train' and split='test' parameters.

        Args:
            name: Dataset name
            dataset_registry: The dataset registry dict to look up in

        Returns:
            Dictionary with X_train, y_train, X_test, y_test
        """
        if name not in dataset_registry:
            return {
                "success": False,
                "error": f"Unknown dataset: {name}",
                "available": list(dataset_registry.keys()),
            }

        try:
            module_path = dataset_registry[name]
            parts = module_path.rsplit(".", 1)
            module = __import__(parts[0], fromlist=[parts[1]])
            loader = getattr(module, parts[1])

            X_train, y_train = loader(split="train")
            X_test, y_test = loader(split="test")

            return {
                "success": True,
                "name": name,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "train_shape": X_train.shape if hasattr(X_train, "shape") else len(X_train),
                "test_shape": X_test.shape if hasattr(X_test, "shape") else len(X_test),
                "n_classes": len(set(y_train)) if hasattr(y_train, "__iter__") else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _resolve_supervised_data(
        self,
        dataset: str = None,
        X_train_handle: str = None,
        y_train_handle: str = None,
        X_test_handle: str = None,
        dataset_registry: dict[str, str] = None,
    ) -> dict[str, Any]:
        """
        Resolve data from either a demo dataset name or custom data handles.

        Returns dict with X_train, y_train, X_test (and optionally y_test).
        """
        if dataset:
            return self._load_supervised_dataset(dataset, dataset_registry)

        if X_train_handle and y_train_handle and X_test_handle:
            # Resolve from custom data handles
            errors = []
            for h in [X_train_handle, y_train_handle, X_test_handle]:
                if h not in self._data_handles:
                    errors.append(f"Data handle not found: {h}")
            if errors:
                return {"success": False, "error": "; ".join(errors)}

            X_train_data = self._data_handles[X_train_handle]
            y_train_data = self._data_handles[y_train_handle]
            X_test_data = self._data_handles[X_test_handle]

            # Extract the actual data from handles
            # For X handles, use 'y' field (the main data) or 'X' if available
            X_train = (
                X_train_data.get("X") if X_train_data.get("X") is not None else X_train_data["y"]
            )
            y_train = y_train_data["y"]
            X_test = X_test_data.get("X") if X_test_data.get("X") is not None else X_test_data["y"]

            return {
                "success": True,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": None,
            }

        return {
            "success": False,
            "error": "Provide either 'dataset' for a demo dataset, or all three handles: 'X_train_handle', 'y_train_handle', 'X_test_handle'",
        }

    def fit_predict_classification(
        self,
        estimator_handle: str,
        dataset: str = None,
        X_train_handle: str = None,
        y_train_handle: str = None,
        X_test_handle: str = None,
    ) -> dict[str, Any]:
        """
        Fit a classifier and predict class labels.

        Args:
            estimator_handle: Handle from instantiate_estimator
            dataset: Demo dataset name (arrow_head, gunpoint, etc.)
            X_train_handle: Data handle for training features
            y_train_handle: Data handle for training labels
            X_test_handle: Data handle for test features

        Returns:
            Dictionary with predicted class labels
        """
        # Get the estimator instance
        try:
            instance = self._handle_manager.get_instance(estimator_handle)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {estimator_handle}"}

        # Resolve data
        data = self._resolve_supervised_data(
            dataset=dataset,
            X_train_handle=X_train_handle,
            y_train_handle=y_train_handle,
            X_test_handle=X_test_handle,
            dataset_registry=CLASSIFICATION_DATASETS,
        )
        if not data["success"]:
            return data

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]

        try:
            # Fit
            instance.fit(X_train, y_train)
            self._handle_manager.mark_fitted(estimator_handle)

            # Predict
            predictions = instance.predict(X_test)

            # Convert predictions to serializable format
            if hasattr(predictions, "tolist"):
                pred_list = predictions.tolist()
            else:
                pred_list = list(predictions)

            result = {
                "success": True,
                "predictions": pred_list,
                "n_predictions": len(pred_list),
                "classes": sorted(set(pred_list)),
            }

            # Add accuracy if ground truth is available
            y_test = data.get("y_test")
            if y_test is not None:
                y_test_list = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
                correct = sum(1 for p, t in zip(pred_list, y_test_list, strict=False) if p == t)
                result["accuracy"] = round(correct / len(y_test_list), 4)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def fit_predict_regression(
        self,
        estimator_handle: str,
        dataset: str = None,
        X_train_handle: str = None,
        y_train_handle: str = None,
        X_test_handle: str = None,
    ) -> dict[str, Any]:
        """
        Fit a regressor and predict target values.

        Args:
            estimator_handle: Handle from instantiate_estimator
            dataset: Demo dataset name (covid_3month, etc.)
            X_train_handle: Data handle for training features
            y_train_handle: Data handle for training target values
            X_test_handle: Data handle for test features

        Returns:
            Dictionary with predicted values
        """
        # Get the estimator instance
        try:
            instance = self._handle_manager.get_instance(estimator_handle)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {estimator_handle}"}

        # Resolve data
        data = self._resolve_supervised_data(
            dataset=dataset,
            X_train_handle=X_train_handle,
            y_train_handle=y_train_handle,
            X_test_handle=X_test_handle,
            dataset_registry=REGRESSION_DATASETS,
        )
        if not data["success"]:
            return data

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]

        try:
            # Fit
            instance.fit(X_train, y_train)
            self._handle_manager.mark_fitted(estimator_handle)

            # Predict
            predictions = instance.predict(X_test)

            # Convert predictions to serializable format
            if hasattr(predictions, "tolist"):
                pred_list = predictions.tolist()
            elif isinstance(predictions, pd.Series):
                pred_list = predictions.values.tolist()
            else:
                pred_list = list(predictions)

            result = {
                "success": True,
                "predictions": pred_list,
                "n_predictions": len(pred_list),
            }

            # Add metrics if ground truth is available
            y_test = data.get("y_test")
            if y_test is not None:
                import numpy as np

                y_test_arr = np.array(y_test)
                pred_arr = np.array(pred_list)
                mse = float(np.mean((y_test_arr - pred_arr) ** 2))
                result["mse"] = round(mse, 6)
                result["rmse"] = round(float(np.sqrt(mse)), 6)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}


_executor_instance: Executor | None = None


def get_executor() -> Executor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = Executor()
    return _executor_instance
