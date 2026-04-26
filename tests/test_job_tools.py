"""
Tests for MCP job management tools.

Covers the four tool functions in ``sktime_mcp.tools.job_tools``:

* ``check_job_status_tool``  – retrieve status of a single job
* ``list_jobs_tool``         – list/filter jobs with optional limit
* ``cancel_job_tool``        – cancel or delete a job record
* ``cleanup_old_jobs_tool``  – remove stale jobs by age
"""

import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, "src")

from sktime_mcp.runtime.jobs import JobStatus, get_job_manager
from sktime_mcp.tools.job_tools import (
    cancel_job_tool,
    check_job_status_tool,
    cleanup_old_jobs_tool,
    list_jobs_tool,
)


def _create_job(job_manager, job_type="fit_predict", handle="test_handle",
                name="NaiveForecaster", **kwargs):
    """Create a test job and return its job_id."""
    return job_manager.create_job(
        job_type=job_type,
        estimator_handle=handle,
        estimator_name=name,
        **kwargs,
    )


def _delete_jobs(job_manager, *job_ids):
    """Safely delete one or more jobs."""
    for jid in job_ids:
        if jid:
            job_manager.delete_job(jid)


class TestCheckJobStatusTool:
    """Tests for check_job_status_tool."""

    def test_check_pending_job(self):
        """A freshly created job should report status=pending."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        try:
            result = check_job_status_tool(job_id)

            assert result["success"] is True
            assert result["job_id"] == job_id
            assert result["status"] == "pending"
            assert result["job_type"] == "fit_predict"
            assert result["estimator_name"] == "NaiveForecaster"
        finally:
            _delete_jobs(jm, job_id)

    def test_check_running_job(self):
        """A job transitioned to RUNNING should report start_time."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        try:
            jm.update_job(job_id, status=JobStatus.RUNNING)
            result = check_job_status_tool(job_id)

            assert result["success"] is True
            assert result["status"] == "running"
            assert result["start_time"] is not None
        finally:
            _delete_jobs(jm, job_id)

    def test_check_completed_job_contains_result(self):
        """A completed job should include its result payload."""
        jm = get_job_manager()
        job_id = _create_job(jm, total_steps=3)
        try:
            jm.update_job(job_id, status=JobStatus.RUNNING)
            jm.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                completed_steps=3,
                result={"predictions": {1: 100}},
            )
            result = check_job_status_tool(job_id)

            assert result["success"] is True
            assert result["status"] == "completed"
            assert result["end_time"] is not None
            assert result["result"] == {"predictions": {1: 100}}
            assert result["progress_percentage"] == 100.0
        finally:
            _delete_jobs(jm, job_id)

    def test_check_job_includes_to_dict_keys(self):
        """Result should contain all keys from JobInfo.to_dict()."""
        jm = get_job_manager()
        job_id = _create_job(jm, dataset_name="airline", horizon=12)
        try:
            result = check_job_status_tool(job_id)

            expected_keys = {
                "success", "job_id", "job_type", "estimator_handle",
                "estimator_name", "status", "created_at", "start_time",
                "end_time", "total_steps", "completed_steps", "current_step",
                "progress_percentage", "elapsed_time",
                "estimated_time_remaining", "estimated_time_remaining_human",
                "dataset_name", "horizon", "result", "errors",
            }
            assert expected_keys.issubset(result.keys())
            assert result["dataset_name"] == "airline"
            assert result["horizon"] == 12
        finally:
            _delete_jobs(jm, job_id)

    def test_check_unknown_job_id(self):
        """Non-existent job_id should return success=False."""
        result = check_job_status_tool("job_does_not_exist_xyz")

        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_check_empty_job_id(self):
        """Empty string job_id should return success=False."""
        result = check_job_status_tool("")

        assert result["success"] is False
        assert "error" in result


class TestListJobsTool:
    """Tests for list_jobs_tool."""

    def test_list_all_jobs(self):
        """Listing without filters should return all jobs."""
        jm = get_job_manager()
        j1 = _create_job(jm, name="ARIMA")
        j2 = _create_job(jm, name="ETS")
        try:
            result = list_jobs_tool()

            assert result["success"] is True
            assert result["count"] >= 2
            assert isinstance(result["jobs"], list)
        finally:
            _delete_jobs(jm, j1, j2)

    def test_list_jobs_filter_by_status(self):
        """Filtering by status string should return only matching jobs."""
        jm = get_job_manager()
        j1 = _create_job(jm, name="ARIMA")
        j2 = _create_job(jm, name="ETS")
        try:
            jm.update_job(j1, status=JobStatus.RUNNING)
            # j2 stays PENDING

            result = list_jobs_tool(status="running")

            assert result["success"] is True
            statuses = [j["status"] for j in result["jobs"]]
            assert all(s == "running" for s in statuses)
            assert result["count"] >= 1
        finally:
            _delete_jobs(jm, j1, j2)

    def test_list_jobs_filter_case_insensitive(self):
        """Status filter should be case-insensitive (e.g. 'PENDING')."""
        jm = get_job_manager()
        j1 = _create_job(jm)
        try:
            result = list_jobs_tool(status="PENDING")

            assert result["success"] is True
            assert result["count"] >= 1
        finally:
            _delete_jobs(jm, j1)

    def test_list_jobs_invalid_status(self):
        """An invalid status string should return success=False."""
        result = list_jobs_tool(status="nonexistent_status")

        assert result["success"] is False
        assert "error" in result
        assert "invalid status" in result["error"].lower()

    def test_list_jobs_with_limit(self):
        """Limit parameter should cap the number of returned jobs."""
        jm = get_job_manager()
        jobs = [_create_job(jm, name=f"Model_{i}") for i in range(5)]
        try:
            result = list_jobs_tool(limit=2)

            assert result["success"] is True
            assert result["count"] <= 2
        finally:
            _delete_jobs(jm, *jobs)

    def test_list_jobs_no_match(self):
        """Filtering by a status with no jobs should return empty list."""
        jm = get_job_manager()
        j1 = _create_job(jm)  # PENDING by default
        try:
            result = list_jobs_tool(status="failed")

            assert result["success"] is True
            # There might be leftover jobs from other tests, but our
            # freshly created PENDING job should not appear.
            for job in result["jobs"]:
                assert job["status"] != "pending" or job["job_id"] != j1
        finally:
            _delete_jobs(jm, j1)

    def test_list_jobs_returns_serialized_dicts(self):
        """Each job in the list should be a dict (from to_dict)."""
        jm = get_job_manager()
        j1 = _create_job(jm)
        try:
            result = list_jobs_tool()

            assert result["success"] is True
            for job_dict in result["jobs"]:
                assert isinstance(job_dict, dict)
                assert "job_id" in job_dict
                assert "status" in job_dict
        finally:
            _delete_jobs(jm, j1)


class TestCancelJobTool:
    """Tests for cancel_job_tool."""

    #core functionality

    def test_cancel_pending_job(self):
        """Cancelling a PENDING job should succeed."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        try:
            result = cancel_job_tool(job_id)

            assert result["success"] is True
            assert "cancelled" in result["message"].lower()

            job = jm.get_job(job_id)
            assert job.status == JobStatus.CANCELLED
        finally:
            _delete_jobs(jm, job_id)

    def test_cancel_running_job(self):
        """Cancelling a RUNNING job should succeed."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        try:
            jm.update_job(job_id, status=JobStatus.RUNNING)
            result = cancel_job_tool(job_id)

            assert result["success"] is True
            assert "cancelled" in result["message"].lower()
        finally:
            _delete_jobs(jm, job_id)

    def test_cancel_and_delete_pending_job(self):
        """cancel_job_tool with delete=True should cancel AND remove record."""
        jm = get_job_manager()
        job_id = _create_job(jm)

        result = cancel_job_tool(job_id, delete=True)

        assert result["success"] is True
        assert "cancelled" in result["message"].lower()
        assert "removed" in result["message"].lower()
        # Job record should be gone
        assert jm.get_job(job_id) is None

    def test_delete_completed_job(self):
        """A completed job cannot be cancelled, but delete=True removes it."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        jm.update_job(job_id, status=JobStatus.RUNNING)
        jm.update_job(job_id, status=JobStatus.COMPLETED)

        result = cancel_job_tool(job_id, delete=True)

        assert result["success"] is True
        assert "removed" in result["message"].lower()
        assert jm.get_job(job_id) is None

    def test_delete_failed_job(self):
        """A failed job cannot be cancelled, but delete=True removes it."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        jm.update_job(job_id, status=JobStatus.RUNNING)
        jm.update_job(job_id, status=JobStatus.FAILED, errors=["timeout"])

        result = cancel_job_tool(job_id, delete=True)

        assert result["success"] is True
        assert "removed" in result["message"].lower()
        assert jm.get_job(job_id) is None

    #invalid inputs and edge cases

    def test_cancel_completed_without_delete(self):
        """Cancelling a completed job without delete should fail."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        jm.update_job(job_id, status=JobStatus.RUNNING)
        jm.update_job(job_id, status=JobStatus.COMPLETED)
        try:
            result = cancel_job_tool(job_id, delete=False)

            assert result["success"] is False
            assert "error" in result
            assert "already" in result["error"].lower()
            assert "delete=true" in result["error"].lower()
        finally:
            _delete_jobs(jm, job_id)

    def test_cancel_already_cancelled_job(self):
        """Cancelling an already-cancelled job should fail (not pending/running)."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        jm.cancel_job(job_id)
        try:
            result = cancel_job_tool(job_id, delete=False)

            assert result["success"] is False
            assert "error" in result
            assert "already" in result["error"].lower()
        finally:
            _delete_jobs(jm, job_id)

    def test_cancel_unknown_job_id(self):
        """Cancelling a non-existent job should return not found."""
        result = cancel_job_tool("job_nonexistent_xyz")

        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestCleanupOldJobsTool:
    """Tests for cleanup_old_jobs_tool."""

    def test_cleanup_removes_old_jobs(self):
        """Jobs with backdated timestamps should be removed by cleanup."""
        jm = get_job_manager()
        j1 = _create_job(jm, name="Old1")
        j2 = _create_job(jm, name="Old2")

        # Backdate jobs so they are unambiguously "old"
        one_hour_ago = datetime.now() - timedelta(hours=1)
        jm.get_job(j1).created_at = one_hour_ago
        jm.get_job(j2).created_at = one_hour_ago

        result = cleanup_old_jobs_tool(max_age_hours=0)

        assert result["success"] is True
        assert result["count"] >= 2
        assert "removed" in result["message"].lower()
        # Both jobs should be gone
        assert jm.get_job(j1) is None
        assert jm.get_job(j2) is None

    def test_cleanup_preserves_recent_jobs(self):
        """Jobs created just now should NOT be removed with default age."""
        jm = get_job_manager()
        job_id = _create_job(jm, name="Fresh")
        try:
            result = cleanup_old_jobs_tool(max_age_hours=24)

            assert result["success"] is True
            # The job was created <1s ago, so it should survive 24h cleanup
            assert jm.get_job(job_id) is not None
        finally:
            _delete_jobs(jm, job_id)

    def test_cleanup_returns_zero_when_no_old_jobs(self):
        """If no jobs are old enough, count should be 0."""
        jm = get_job_manager()
        job_id = _create_job(jm)
        try:
            result = cleanup_old_jobs_tool(max_age_hours=9999)

            assert result["success"] is True
            assert result["count"] == 0
        finally:
            _delete_jobs(jm, job_id)

    def test_cleanup_result_structure(self):
        """Result should contain success, message, and count."""
        result = cleanup_old_jobs_tool(max_age_hours=9999)

        assert "success" in result
        assert "message" in result
        assert "count" in result
        assert isinstance(result["count"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
