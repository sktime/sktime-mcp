"""
Task definitions for sktime-mcp benchmark suite.

Tasks are loaded from YAML files in the tasks/ directory.
Contributors can add new tasks by adding a new YAML file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).parent / "tasks"


@dataclass
class BenchmarkTask:
    """
    A single benchmark task definition.

    Attributes
    ----------
    name : str
        Unique task identifier
    dataset : str
        Demo dataset name (e.g. "airline", "sunspots")
    horizon : int
        Forecast horizon
    expected_task : str
        Expected sktime task type (e.g. "forecasting")
    valid_estimators : list[str]
        List of estimators considered correct for this task
    valid_pipelines : list[list[str]]
        List of valid pipeline compositions for this task
    metric : str
        Evaluation metric name
    difficulty : str
        Task difficulty: "beginner", "intermediate", "advanced"
    description : str
        Human readable description
    tags : dict
        Optional extra metadata
    """

    name: str
    dataset: str
    horizon: int
    expected_task: str
    valid_estimators: list[str]
    valid_pipelines: list[list[str]] = field(default_factory=list)
    metric: str = "MeanAbsolutePercentageError"
    difficulty: str = "beginner"
    description: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkTask":
        """Load a BenchmarkTask from a dictionary."""
        return cls(
            name=data["name"],
            dataset=data["dataset"],
            horizon=data["horizon"],
            expected_task=data["expected_task"],
            valid_estimators=data["valid_estimators"],
            valid_pipelines=data.get("valid_pipelines", []),
            metric=data.get("metric", "MeanAbsolutePercentageError"),
            difficulty=data.get("difficulty", "beginner"),
            description=data.get("description", ""),
            tags=data.get("tags", {}),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkTask":
        """Load a BenchmarkTask from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def load_all_tasks(tasks_dir: Path | None = None) -> list[BenchmarkTask]:
    """
    Load all benchmark tasks from the tasks directory.

    Parameters
    ----------
    tasks_dir : Path, optional
        Directory to load tasks from. Defaults to built-in tasks dir.

    Returns
    -------
    list[BenchmarkTask]
        List of loaded benchmark tasks
    """
    tasks_dir = tasks_dir or TASKS_DIR
    tasks = []

    for yaml_file in sorted(tasks_dir.glob("*.yaml")):
        try:
            task = BenchmarkTask.from_yaml(yaml_file)
            tasks.append(task)
            logger.info(f"Loaded task: {task.name}")
        except Exception as e:
            logger.warning(f"Failed to load task from {yaml_file}: {e}")

    return tasks


def load_task(name: str, tasks_dir: Path | None = None) -> BenchmarkTask | None:
    """
    Load a single task by name.

    Parameters
    ----------
    name : str
        Task name to load
    tasks_dir : Path, optional
        Directory to search in

    Returns
    -------
    BenchmarkTask or None
    """
    tasks_dir = tasks_dir or TASKS_DIR
    yaml_file = tasks_dir / f"{name}.yaml"

    if not yaml_file.exists():
        logger.warning(f"Task file not found: {yaml_file}")
        return None

    return BenchmarkTask.from_yaml(yaml_file)