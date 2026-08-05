"""
Benchmark runner for sktime-mcp agentic workflows.

Runs benchmark tasks through the MCP tools and collects results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sktime_mcp.benchmark.tasks import BenchmarkTask, load_all_tasks
from sktime_mcp.composition.validator import get_composition_validator
from sktime_mcp.tools.evaluate import evaluate_estimator_tool
from sktime_mcp.tools.fit_predict import fit_predict_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """
    Result of running a single benchmark task.

    Attributes
    ----------
    task_name : str
        Name of the task
    estimator_name : str
        Estimator used
    pipeline : list[str]
        Pipeline components used
    pipeline_valid : bool
        Whether the pipeline composition is valid
    fit_predict_success : bool
        Whether fit-predict completed successfully
    cv_results : list[dict]
        Cross-validation results from evaluate tool
    errors : list[str]
        Any errors encountered
    raw : dict
        Raw tool outputs
    """

    task_name: str
    estimator_name: str
    pipeline: list[str] = field(default_factory=list)
    pipeline_valid: bool = False
    fit_predict_success: bool = False
    cv_results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """
    Runs benchmark tasks through sktime-mcp tools.

    Usage
    -----
    >>> runner = BenchmarkRunner()
    >>> results = runner.run_all()

    Or run a single task:
    >>> result = runner.run_task(task, estimator_name="NaiveForecaster")
    """

    def __init__(self):
        self._validator = get_composition_validator()

    def run_task(
        self,
        task: BenchmarkTask,
        estimator_name: str,
        pipeline: list[str] | None = None,
    ) -> TaskResult:
        """
        Run a single benchmark task with a given estimator.

        Parameters
        ----------
        task : BenchmarkTask
            The task to run
        estimator_name : str
            Name of the estimator to use
        pipeline : list[str], optional
            Pipeline components. Defaults to [estimator_name]

        Returns
        -------
        TaskResult
        """
        pipeline = pipeline or [estimator_name]
        result = TaskResult(
            task_name=task.name,
            estimator_name=estimator_name,
            pipeline=pipeline,
        )

        # Step 1 - validate pipeline composition
        validation = self._validator.validate_pipeline(pipeline)
        result.pipeline_valid = validation.valid
        if not validation.valid:
            result.errors.extend(validation.errors)
            logger.warning(
                f"Invalid pipeline {pipeline} for task {task.name}: "
                f"{validation.errors}"
            )

        # Step 2 - instantiate estimator
        instantiate_result = instantiate_estimator_tool(estimator_name)
        result.raw["instantiate"] = instantiate_result

        if not instantiate_result.get("success"):
            result.errors.append(
                f"Failed to instantiate {estimator_name}: "
                f"{instantiate_result.get('error')}"
            )
            return result

        handle = instantiate_result["handle"]

        # Step 3 - fit predict
        fp_result = fit_predict_tool(handle, task.dataset, task.horizon)
        result.raw["fit_predict"] = fp_result
        result.fit_predict_success = fp_result.get("success", False)

        if not result.fit_predict_success:
            result.errors.append(
                f"fit_predict failed: {fp_result.get('error')}"
            )

        # Step 4 - evaluate via cross validation
        eval_result = evaluate_estimator_tool(handle, task.dataset)
        result.raw["evaluate"] = eval_result

        if eval_result.get("success"):
            result.cv_results = eval_result.get("results", [])
        else:
            result.errors.append(
                f"evaluate failed: {eval_result.get('error')}"
            )

        return result

    def run_task_all_estimators(
        self,
        task: BenchmarkTask,
    ) -> list[TaskResult]:
        """
        Run a task against all its valid estimators.

        Parameters
        ----------
        task : BenchmarkTask

        Returns
        -------
        list[TaskResult]
        """
        results = []
        for estimator_name in task.valid_estimators:
            logger.info(f"Running {task.name} with {estimator_name}")
            result = self.run_task(task, estimator_name)
            results.append(result)
        return results

    def run_all(
        self,
    ) -> dict[str, list[TaskResult]]:
        """
        Run all benchmark tasks.

        Returns
        -------
        dict mapping task_name -> list of TaskResults
        """
        tasks = load_all_tasks()
        all_results = {}

        for task in tasks:
            logger.info(f"Starting benchmark task: {task.name}")
            results = self.run_task_all_estimators(task)
            all_results[task.name] = results

        return all_results