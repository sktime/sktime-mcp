"""
Job management for long-running operations in sktime MCP.

Handles background training jobs with progress tracking and status updates.
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class JobStatus(Enum):
    """Status of a background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Information about a background job."""

    job_id: str
    job_type: str  # "fit", "fit_predict", "transform", etc.
    estimator_handle: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Progress tracking
    total_steps: int = 0
    completed_steps: int = 0
    current_step: str = ""

    # Results
    result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    # Metadata
    dataset_name: str | None = None
    horizon: int | None = None
    estimator_name: str | None = None

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100

    @property
    def elapsed_time(self) -> float | None:
        """Calculate elapsed time in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def estimated_time_remaining(self) -> float | None:
        """Estimate remaining time in seconds."""
        if self.status != JobStatus.RUNNING or self.completed_steps == 0:
            return None

        elapsed = self.elapsed_time
        if elapsed is None:
            return None

        avg_time_per_step = elapsed / self.completed_steps
        remaining_steps = self.total_steps - self.completed_steps
        return remaining_steps * avg_time_per_step

    @property
    def estimated_time_remaining_human(self) -> str | None:
        """Human-readable estimated time remaining."""
        remaining = self.estimated_time_remaining
        if remaining is None:
            return None

        if remaining < 60:
            return f"{int(remaining)}s"
        elif remaining < 3600:
            return f"{int(remaining / 60)}m {int(remaining % 60)}s"
        else:
            hours = int(remaining / 3600)
            minutes = int((remaining % 3600) / 60)
            return f"{hours}h {minutes}m"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "estimator_handle": self.estimator_handle,
            "estimator_name": self.estimator_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "progress_percentage": self.progress_percentage,
            "elapsed_time": self.elapsed_time,
            "estimated_time_remaining": self.estimated_time_remaining,
            "estimated_time_remaining_human": self.estimated_time_remaining_human,
            "dataset_name": self.dataset_name,
            "horizon": self.horizon,
            "result": self.result,
            "errors": self.errors,
        }


class JobManager:
    """
    Thread-safe manager for background jobs.

    Handles job creation, status updates, and cleanup.
    """

    def __init__(self):
        self.jobs: dict[str, JobInfo] = {}
        self.lock = threading.Lock()

    def create_job(
        self,
        job_type: str,
        estimator_handle: str,
        estimator_name: str | None = None,
        dataset_name: str | None = None,
        horizon: int | None = None,
        total_steps: int = 3,  # Default: load data, fit, predict
    ) -> str:
        """
        Create a new job and return its ID.

        Args:
            job_type: Type of job (fit, fit_predict, etc.)
            estimator_handle: Handle of the estimator
            estimator_name: Name of the estimator
            dataset_name: Name of the dataset (if applicable)
            horizon: Forecast horizon (if applicable)
            total_steps: Total number of steps in the job

        Returns:
            Job ID (UUID)
        """
        job_id = str(uuid.uuid4())

        with self.lock:
            self.jobs[job_id] = JobInfo(
                job_id=job_id,
                job_type=job_type,
                estimator_handle=estimator_handle,
                estimator_name=estimator_name,
                dataset_name=dataset_name,
                horizon=horizon,
                total_steps=total_steps,
            )

        return job_id

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        completed_steps: int | None = None,
        current_step: str | None = None,
        result: dict[str, Any] | None = None,
        errors: list[str] | None = None,
    ) -> bool:
        """
        Update job status and progress.

        Args:
            job_id: Job ID to update
            status: New status
            completed_steps: Number of completed steps
            current_step: Description of current step
            result: Job result (when completed)
            errors: List of errors (when failed)

        Returns:
            True if job was updated, False if not found
        """
        with self.lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]

            # Update status
            if status is not None:
                old_status = job.status
                job.status = status

                # Set timestamps based on status transitions
                if status == JobStatus.RUNNING and old_status == JobStatus.PENDING:
                    job.start_time = datetime.now()
                elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    job.end_time = datetime.now()

            # Update progress
            if completed_steps is not None:
                job.completed_steps = completed_steps

            if current_step is not None:
                job.current_step = current_step

            # Update result
            if result is not None:
                job.result = result

            # Update errors
            if errors is not None:
                job.errors = errors

            return True

    def get_job(self, job_id: str) -> JobInfo | None:
        """
        Get job information.

        Args:
            job_id: Job ID

        Returns:
            JobInfo if found, None otherwise
        """
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int | None = None,
    ) -> list[JobInfo]:
        """
        List all jobs, optionally filtered by status.

        Args:
            status: Filter by status (None = all jobs)
            limit: Maximum number of jobs to return

        Returns:
            List of JobInfo objects
        """
        with self.lock:
            jobs = list(self.jobs.values())

            # Filter by status
            if status is not None:
                jobs = [j for j in jobs if j.status == status]

            # Sort by creation time (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            # Apply limit
            if limit is not None:
                jobs = jobs[:limit]

            return jobs

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled, False if not found or already completed
        """
        with self.lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]

            # Can only cancel pending or running jobs
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                job.status = JobStatus.CANCELLED
                job.end_time = datetime.now()
                return True

            return False

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Remove jobs older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of jobs removed
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        with self.lock:
            old_job_ids = [job_id for job_id, job in self.jobs.items() if job.created_at < cutoff]

            for job_id in old_job_ids:
                del self.jobs[job_id]

            return len(old_job_ids)

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job.

        Args:
            job_id: Job ID to delete

        Returns:
            True if job was deleted, False if not found
        """
        with self.lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                return True
            return False


# Singleton instance
_job_manager_instance: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get the singleton JobManager instance."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = JobManager()
    return _job_manager_instance
