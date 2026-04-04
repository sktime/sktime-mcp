"""
Executor for sktime MCP.

Responsible for instantiating estimators, loading datasets,
and running fit/predict operations.
"""

import asyncio
import logging
import uuid
from typing import Any, Optional, Union

import pandas as pd

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager

logger = logging.getLogger(__name__)


# Available demo datasets
# L-5: We can add more datasets here by directly wrapping on top of sktime datasets (https://www.sktime.net/en/latest/api_reference/datasets.html)
# L-6: We can also add custom datasets here
DEMO_DATASETS = {
    "airline": "sktime.datasets.load_airline",
    "longley": "sktime.datasets.load_longley",
    "lynx": "sktime.datasets.load_lynx",
    "shampoo": "sktime.datasets.load_shampoo_sales",
    "sunspots": "sktime.datasets.load_sunspot",
    "uschange": "sktime.datasets.load_uschange",
}


class Executor:
    """
    Execution runtime for sktime estimators.

    Handles instantiation, fitting, and prediction.
    """

    def __init__(self):
        self._registry = get_registry()
        self._handle_manager = get_handle_manager()
        self._job_manager = get_job_manager()
        self._data_handles = {}  # Store data handles

    def instantiate(
        self,
        estimator_name: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Instantiate an estimator and return a handle."""
        node = self._registry.get_estimator_by_name(estimator_name)
        if node is None:
            return {"success": False, "error": f"Unknown estimator: {estimator_name}"}

        try:
            instance = node.class_ref(**(params or {}))
            handle_id = self._handle_manager.create_handle(
                estimator_name=estimator_name,
                instance=instance,
                params=params or {},
            )
            return {
                "success": True,
                "handle": handle_id,
                "estimator": estimator_name,
                "params": params or {},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # L-7: We can also add custom load_dataset functions here
    def load_dataset(self, name: str) -> dict[str, Any]:
        """Load a demo dataset."""
        if name not in DEMO_DATASETS:
            return {
                "success": False,
                "error": f"Unknown dataset: {name}",
                "available": list(DEMO_DATASETS.keys()),
            }

        try:
            module_path = DEMO_DATASETS[name]
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
        X: Optional[Any] = None,
        fh: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Fit an estimator."""
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}

        try:
            if fh is not None:
                instance.fit(y, X=X, fh=fh)
            elif X is not None:
                instance.fit(y, X=X)
            else:
                instance.fit(y)

            self._handle_manager.mark_fitted(handle_id)
            return {"success": True, "handle": handle_id, "fitted": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(
        self,
        handle_id: str,
        fh: Optional[Union[int, list[int]]] = None,
        X: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Generate predictions."""
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}

        if not self._handle_manager.is_fitted(handle_id):
            return {"success": False, "error": "Estimator not fitted"}

        try:
            if fh is None:
                fh = list(range(1, 13))

            predictions = instance.predict(fh=fh, X=X) if X is not None else instance.predict(fh=fh)

            if isinstance(predictions, pd.Series):
                # Convert index to string to avoid JSON serialization issues with Period/DatetimeIndex
                predictions_copy = predictions.copy()
                predictions_copy.index = predictions_copy.index.astype(str)
                result = predictions_copy.to_dict()
            elif isinstance(predictions, pd.DataFrame):
                predictions_copy = predictions.copy()
                predictions_copy.index = predictions_copy.index.astype(str)
                result = predictions_copy.to_dict(orient="list")
            else:
                result = predictions.tolist() if hasattr(predictions, "tolist") else predictions

            return {
                "success": True,
                "predictions": result,
                "horizon": len(fh) if hasattr(fh, "__len__") else fh,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fit_predict(
        self,
        handle_id: str,
        dataset: str,
        horizon: int = 12,
    ) -> dict[str, Any]:
        """Convenience method: load data, fit, and predict."""
        data_result = self.load_dataset(dataset)
        if not data_result["success"]:
            return data_result

        y = data_result["data"]
        X = data_result.get("exog")
        fh = list(range(1, horizon + 1))

        fit_result = self.fit(handle_id, y, X=X, fh=fh)
        if not fit_result["success"]:
            return fit_result

        return self.predict(handle_id, fh=fh, X=X)

    async def fit_predict_async(
        self,
        handle_id: str,
        dataset: str,
        horizon: int = 12,
        job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Async version of fit_predict with job tracking.

        This method runs the training in the background without blocking the MCP server.
        Progress is tracked via the JobManager.

        Args:
            handle_id: Estimator handle
            dataset: Dataset name
            horizon: Forecast horizon
            job_id: Optional job ID for tracking (created if not provided)

        Returns:
            Dictionary with success status and job_id
        """
        # Get estimator info for job tracking
        try:
            handle_info = self._handle_manager.get_info(handle_id)
            estimator_name = handle_info.estimator_name
        except Exception as e:
            logger.warning(f"Could not get estimator name: {e}")
            estimator_name = "Unknown"

        # Create job if not provided
        if job_id is None:
            job_id = self._job_manager.create_job(
                job_type="fit_predict",
                estimator_handle=handle_id,
                estimator_name=estimator_name,
                dataset_name=dataset,
                horizon=horizon,
                total_steps=3,  # load data, fit, predict
            )

        try:
            # Update status to RUNNING
            self._job_manager.update_job(job_id, status=JobStatus.RUNNING)

            # Step 1: Load dataset
            self._job_manager.update_job(
                job_id, completed_steps=0, current_step=f"Loading dataset '{dataset}'..."
            )
            await asyncio.sleep(0.01)  # Yield control to event loop

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
                job_id, completed_steps=1, current_step=f"Fitting {estimator_name} on {dataset}..."
            )
            await asyncio.sleep(0.01)  # Yield control

            # Run fit in executor to avoid blocking
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
            await asyncio.sleep(0.01)  # Yield control

            # Run predict in executor
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
    def instantiate_pipeline(
        self,
        components: list[str],
        params_list: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Instantiate a pipeline from a list of components.

        Args:
            components: List of estimator names in pipeline order
            params_list: Optional list of parameter dicts for each component

        Returns:
            Dictionary with success status and handle
        """
        if not components:
            return {"success": False, "error": "Pipeline cannot be empty"}

        # Validate the pipeline first
        from sktime_mcp.composition.validator import get_composition_validator

        validator = get_composition_validator()
        validation = validator.validate_pipeline(components)

        if not validation.valid:
            return {
                "success": False,
                "error": "Invalid pipeline composition",
                "validation_errors": validation.errors,
                "suggestions": validation.suggestions,
            }

        try:
            # If only one component, just instantiate it directly
            if len(components) == 1:
                params = params_list[0] if params_list else {}
                return self.instantiate(components[0], params)

            # Build the pipeline
            # Get all component nodes
            component_instances = []
            params_list = params_list or [{}] * len(components)

            for i, comp_name in enumerate(components):
                node = self._registry.get_estimator_by_name(comp_name)
                if node is None:
                    return {"success": False, "error": f"Unknown estimator: {comp_name}"}

                params = params_list[i] if i < len(params_list) else {}
                instance = node.class_ref(**params)
                component_instances.append(instance)

            # Determine the type of pipeline to create
            # Check if all but last are transformers
            all_transformers_except_last = all(
                self._registry.get_estimator_by_name(comp).task == "transformation"
                for comp in components[:-1]
            )

            final_task = self._registry.get_estimator_by_name(components[-1]).task

            if all_transformers_except_last and final_task == "forecasting":
                # Use TransformedTargetForecaster
                from sktime.forecasting.compose import TransformedTargetForecaster

                # Chain transformers if multiple
                if len(component_instances) == 2:
                    pipeline = TransformedTargetForecaster(
                        [
                            ("transformer", component_instances[0]),
                            ("forecaster", component_instances[1]),
                        ]
                    )
                else:
                    # Multiple transformers - chain them
                    from sktime.transformations.compose import TransformerPipeline

                    transformer_pipeline = TransformerPipeline(
                        [(f"step_{i}", comp) for i, comp in enumerate(component_instances[:-1])]
                    )
                    pipeline = TransformedTargetForecaster(
                        [
                            ("transformers", transformer_pipeline),
                            ("forecaster", component_instances[-1]),
                        ]
                    )

            elif all_transformers_except_last and final_task in ("classification", "regression"):
                # Use sklearn-style Pipeline
                from sktime.pipeline import Pipeline

                pipeline = Pipeline(
                    [(f"step_{i}", comp) for i, comp in enumerate(component_instances)]
                )

            elif all(
                self._registry.get_estimator_by_name(comp).task == "transformation"
                for comp in components
            ):
                # All transformers - use TransformerPipeline
                from sktime.transformations.compose import TransformerPipeline

                pipeline = TransformerPipeline(
                    [(f"step_{i}", comp) for i, comp in enumerate(component_instances)]
                )

            else:
                return {
                    "success": False,
                    "error": "Unsupported pipeline composition type",
                    "hint": "Currently supports: transformers → forecaster, transformers → classifier/regressor, or transformer chains",
                }

            # Create a handle for the pipeline
            pipeline_name = " → ".join(components)
            handle_id = self._handle_manager.create_handle(
                estimator_name=pipeline_name,
                instance=pipeline,
                params={"components": components, "params_list": params_list},
            )

            return {
                "success": True,
                "handle": handle_id,
                "pipeline": pipeline_name,
                "components": components,
                "params_list": params_list,
            }

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def list_datasets(self) -> list[str]:
        """List available demo datasets."""
        return list(DEMO_DATASETS.keys())

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

            # Generate handle
            data_handle = f"data_{uuid.uuid4().hex[:8]}"

            # Store
            self._data_handles[data_handle] = {
                "y": y,
                "X": X,
                "metadata": metadata,
                "validation": validation_report,
                "config": config,  # Store config for reference
            }

            # Apply auto-formatting if enabled
            if getattr(self, "_auto_format_enabled", True):
                try:
                    format_result = self.format_data_handle(
                        data_handle, auto_infer_freq=True, fill_missing=True, remove_duplicates=True
                    )
                    if format_result["success"]:
                        # Return the NEW handle (formatted)
                        return {
                            "success": True,
                            "data_handle": format_result["data_handle"],
                            "original_handle": data_handle,
                            "metadata": format_result["metadata"],
                            "validation": validation_report,
                            "formatted": True,
                            "changes_made": format_result["changes_made"],
                        }
                except Exception as e:
                    logger.warning(f"Auto-formatting failed: {e}")
                    # Continue with unformatted data if formatting fails

            return {
                "success": True,
                "data_handle": data_handle,
                "metadata": adapter.get_metadata(),
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
        job_id: Optional[str] = None,
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

            data_handle = f"data_{uuid.uuid4().hex[:8]}"

            self._data_handles[data_handle] = {
                "y": y,
                "X": X,
                "metadata": metadata,
                "validation": validation_report,
                "config": config,
            }

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
            freq = y.index.freq

            if freq is None:
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

        # Store formatted data
        self._data_handles[new_handle] = {
            "y": y,
            "X": X,
            "metadata": {
                **data_info["metadata"],
                "formatted": True,
                "frequency": str(y.index.freq) if y.index.freq else changes_made.get("frequency"),
                "rows": len(y),
                "start_date": str(y.index.min()),
                "end_date": str(y.index.max()),
            },
            "validation": data_info.get("validation", {}),
            "config": data_info.get("config", {}),
            "original_handle": data_handle,
        }

        return {
            "success": True,
            "data_handle": new_handle,
            "metadata": self._data_handles[new_handle]["metadata"],
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


_executor_instance: Optional[Executor] = None


def get_executor() -> Executor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = Executor()
    return _executor_instance
