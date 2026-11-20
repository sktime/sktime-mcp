"""
MCP tools for job management.

Provides tools for checking job status, listing jobs, and cancelling jobs.
"""

import logging
from typing import Any

from sktime_mcp.runtime.jobs import JobStatus, get_job_manager

logger = logging.getLogger(__name__)

# Per-fold rows kept inline in a job-status response. A completed evaluate
# job can hold hundreds of folds (observed: 120 folds ≈ 15k tokens), and
# embedding them all floods the MCP client's context.
_MAX_FOLD_RESULTS_IN_STATUS = 10


def _compact_result(result: Any, max_folds: int) -> Any:
    """Truncate oversized fold_results inside a job result payload.

    The aggregate metrics/summary are always kept intact; only the
    per-fold rows are capped, with an explicit truncation marker.
    """
    if not isinstance(result, dict):
        return result
    folds = result.get("fold_results")
    if not isinstance(folds, list) or len(folds) <= max_folds:
        return result
    return {
        **result,
        "fold_results": folds[:max_folds],
        "fold_results_truncated": {
            "shown": max_folds,
            "total": len(folds),
            "note": "metrics and summary cover all folds; rerun evaluate for full per-fold rows",
        },
    }


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

    job_dict = job.to_dict()
    job_dict["result"] = _compact_result(job_dict.get("result"), _MAX_FOLD_RESULTS_IN_STATUS)
    return {
        "success": True,
        **job_dict,
    }


def list_jobs_tool(
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List background jobs with offset/limit pagination.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)
        limit: Maximum number of jobs to return in this page
        offset: Number of jobs to skip (for paging through results)

    Returns:
        Dictionary with the page of jobs plus pagination metadata
        (total, offset, limit, has_more).
    """
    job_manager = get_job_manager()

    # Convert status string to enum
    status_filter = None
    if status is not None:
        if not isinstance(status, str):
            return {
                "success": False,
                "error": (
                    f"Invalid status type '{type(status).__name__}'. "
                    "Expected one of: pending, running, completed, failed, cancelled"
                ),
            }
        try:
            status_filter = JobStatus(status.lower())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid status '{status}'. Valid values: pending, running, completed, failed, cancelled",
            }

    if limit < 1:
        return {
            "success": False,
            "error": "limit must be a positive integer.",
        }

    if offset < 0:
        return {
            "success": False,
            "error": "offset must be a non-negative integer.",
        }

    total = job_manager.count_jobs(status=status_filter)
    jobs = job_manager.list_jobs(status=status_filter, limit=limit, offset=offset)

    job_dicts = []
    for job in jobs:
        job_dict = job.to_dict()
        # the list view is an overview — drop per-fold rows entirely
        job_dict["result"] = _compact_result(job_dict.get("result"), 0)
        job_dicts.append(job_dict)

    return {
        "success": True,
        "count": len(jobs),
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(jobs) < total,
        "jobs": job_dicts,
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
            msg += "; record retained because active jobs cannot be removed immediately"
        return {"success": True, "message": msg}

    if delete:
        job_manager.delete_job(job_id)
        return {"success": True, "message": f"Job '{job_id}' removed"}

    return {
        "success": False,
        "error": (f"Job is already '{job.status.value}'. Use delete=true to remove the record."),
    }
