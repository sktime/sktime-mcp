"""
Tests for sktime-mcp benchmark suite.
"""

from __future__ import annotations

import pytest

from sktime_mcp.benchmark.runner import BenchmarkRunner, TaskResult
from sktime_mcp.benchmark.scorer import BenchmarkScorer
from sktime_mcp.benchmark.tasks import BenchmarkTask, load_all_tasks


class TestBenchmarkTasks:
    """Tests for task loading."""

    def test_load_all_tasks(self):
        tasks = load_all_tasks()
        assert len(tasks) > 0

    def test_task_fields(self):
        tasks = load_all_tasks()
        task = tasks[0]
        assert task.name
        assert task.dataset
        assert task.horizon > 0
        assert task.expected_task
        assert len(task.valid_estimators) > 0

    def test_task_from_dict(self):
        data = {
            "name": "test_task",
            "dataset": "airline",
            "horizon": 12,
            "expected_task": "forecasting",
            "valid_estimators": ["NaiveForecaster"],
        }
        task = BenchmarkTask.from_dict(data)
        assert task.name == "test_task"
        assert task.difficulty == "beginner"  # default


class TestBenchmarkRunner:
    """Tests for the benchmark runner."""

    def test_run_single_task(self):
        runner = BenchmarkRunner()
        task = BenchmarkTask.from_dict({
            "name": "test_airline",
            "dataset": "airline",
            "horizon": 12,
            "expected_task": "forecasting",
            "valid_estimators": ["NaiveForecaster"],
        })
        result = runner.run_task(task, "NaiveForecaster")
        assert isinstance(result, TaskResult)
        assert result.task_name == "test_airline"
        assert result.estimator_name == "NaiveForecaster"
        assert result.fit_predict_success

    def test_run_invalid_estimator(self):
        runner = BenchmarkRunner()
        task = BenchmarkTask.from_dict({
            "name": "test_invalid",
            "dataset": "airline",
            "horizon": 12,
            "expected_task": "forecasting",
            "valid_estimators": ["NonExistentEstimator999"],
        })
        result = runner.run_task(task, "NonExistentEstimator999")
        assert isinstance(result, TaskResult)
        assert len(result.errors) > 0

    def test_pipeline_validation(self):
        runner = BenchmarkRunner()
        task = BenchmarkTask.from_dict({
            "name": "test_pipeline",
            "dataset": "airline",
            "horizon": 12,
            "expected_task": "forecasting",
            "valid_estimators": ["NaiveForecaster"],
        })
        result = runner.run_task(
            task,
            "NaiveForecaster",
            pipeline=["NaiveForecaster"]
        )
        assert result.pipeline_valid


class TestBenchmarkScorer:
    """Tests for the benchmark scorer."""

    def test_score_result(self):
        runner = BenchmarkRunner()
        scorer = BenchmarkScorer()
        task = BenchmarkTask.from_dict({
            "name": "test_score",
            "dataset": "airline",
            "horizon": 12,
            "expected_task": "forecasting",
            "valid_estimators": ["NaiveForecaster"],
        })
        result = runner.run_task(task, "NaiveForecaster")
        score = scorer.score_result(result, task)

        assert score.overall_score >= 0.0
        assert score.overall_score <= 1.0
        assert score.task_match_score == 1.0
        assert score.pipeline_validity_score == 1.0

    def test_summary(self):
        runner = BenchmarkRunner()
        scorer = BenchmarkScorer()
        task = BenchmarkTask.from_dict({
            "name": "test_summary",
            "dataset": "airline",
            "horizon": 12,
            "expected_task": "forecasting",
            "valid_estimators": ["NaiveForecaster", "AutoETS"],
        })
        results = runner.run_task_all_estimators(task)
        scores = scorer.score_results(results, task)
        summary = scorer.summary(scores)

        assert "best_estimator" in summary
        assert "ranking" in summary
        assert summary["n_estimators_evaluated"] == 2