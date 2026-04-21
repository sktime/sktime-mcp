"""
MCP tools for job management.

Provides tools for checking job status, listing jobs, and cancelling jobs.
"""

import logging
from typing import Any, Optional

from sktime_mcp.runtime.jobs import JobStatus, get_job_manager

logger = logging.getLogger(__name__)


def check_job_status_tool(job_id: str) -> dict[str, Any]:
    """
    Check the status of a background job.

    Args:
        job_id: Job ID to check

    Returns:
        Dictionary with job status and progress information
    """
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if job is None:
        return {
            "success": False,
            "error": f"Job '{job_id}' not found",
        }

    return {
        "success": True,
        **job.to_dict(),
    }


def list_jobs_tool(
    status: Optional[str] = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    List all background jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)
        limit: Maximum number of jobs to return

    Returns:
        Dictionary with list of jobs
    """
    job_manager = get_job_manager()

    # Convert status string to enum
    status_filter = None
    if status is not None:
        try:
            status_filter = JobStatus(status.lower())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid status '{status}'. Valid values: pending, running, completed, failed, cancelled",
            }

    jobs = job_manager.list_jobs(status=status_filter, limit=limit)

    return {
        "success": True,
        "count": len(jobs),
        "jobs": [job.to_dict() for job in jobs],
    }


def cancel_job_tool(job_id: str, delete: bool = False) -> dict[str, Any]:
    """
    Cancel a running/pending job, and optionally remove its record.

    Args:
        job_id: Job ID to cancel
        delete: Also remove the job record after cancelling (default: False).
                For jobs that are already completed/failed, set delete=True
                to remove them.

    Returns:
        Dictionary with success status and message
    """
    job_manager = get_job_manager()

    job = job_manager.get_job(job_id)
    if job is None:
        return {"success": False, "error": f"Job '{job_id}' not found"}

    if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        job_manager.cancel_job(job_id)
        msg = f"Job '{job_id}' cancelled"
        if delete:
            job_manager.delete_job(job_id)
            msg += " and removed"
        return {"success": True, "message": msg}

    if delete:
        job_manager.delete_job(job_id)
        return {"success": True, "message": f"Job '{job_id}' removed"}

    return {
        "success": False,
        "error": (
            f"Job is already '{job.status.value}'. "
            "Use delete=true to remove the record."
        ),
    }


def cleanup_old_jobs_tool(max_age_hours: int = 24) -> dict[str, Any]:
    """
    Clean up old jobs.

    Args:
        max_age_hours: Maximum age in hours (default: 24)

    Returns:
        Dictionary with number of jobs removed
    """
    job_manager = get_job_manager()

    count = job_manager.cleanup_old_jobs(max_age_hours)

    return {
        "success": True,
        "message": f"Removed {count} old job(s)",
        "count": count,
    }
