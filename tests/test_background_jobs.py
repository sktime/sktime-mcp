"""
Test background job management.
"""

from sktime_mcp.runtime.jobs import JobStatus, get_job_manager


def test_job_creation():
    """Test creating a job."""
    job_manager = get_job_manager()

    job_id = job_manager.create_job(
        job_type="fit_predict",
        estimator_handle="test_handle",
        estimator_name="ARIMA",
        dataset_name="airline",
        horizon=12,
        total_steps=3,
    )

    assert job_id is not None
    job = job_manager.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.PENDING
    assert job.estimator_name == "ARIMA"
    assert job.dataset_name == "airline"

    print(f"✓ Job created: {job_id}")
    print(f"  Status: {job.status.value}")
    print(f"  Estimator: {job.estimator_name}")
    print(f"  Dataset: {job.dataset_name}")

    # Cleanup
    job_manager.delete_job(job_id)


def test_job_updates():
    """Test updating job status."""
    job_manager = get_job_manager()

    job_id = job_manager.create_job(
        job_type="fit_predict",
        estimator_handle="test_handle",
        estimator_name="ARIMA",
        total_steps=3,
    )

    # Update to running
    job_manager.update_job(job_id, status=JobStatus.RUNNING)
    job = job_manager.get_job(job_id)
    assert job.status == JobStatus.RUNNING
    assert job.start_time is not None

    # Update progress
    job_manager.update_job(job_id, completed_steps=1, current_step="Fitting model...")
    job = job_manager.get_job(job_id)
    assert job.completed_steps == 1
    assert job.current_step == "Fitting model..."
    assert job.progress_percentage == 33.33333333333333

    # Complete
    job_manager.update_job(
        job_id,
        status=JobStatus.COMPLETED,
        completed_steps=3,
        result={"predictions": {1: 100, 2: 200}},
    )
    job = job_manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    assert job.end_time is not None
    assert job.result is not None

    print("✓ Job updated successfully")
    print(f"  Progress: {job.progress_percentage}%")
    print(f"  Elapsed time: {job.elapsed_time}s")

    # Cleanup
    job_manager.delete_job(job_id)


def test_list_jobs():
    """Test listing jobs."""
    job_manager = get_job_manager()

    # Create multiple jobs
    job1 = job_manager.create_job("fit_predict", "handle1", "ARIMA")
    job2 = job_manager.create_job("fit_predict", "handle2", "Prophet")
    job3 = job_manager.create_job("fit_predict", "handle3", "ETS")

    # Update statuses
    job_manager.update_job(job1, status=JobStatus.RUNNING)
    job_manager.update_job(job2, status=JobStatus.COMPLETED)
    job_manager.update_job(job3, status=JobStatus.FAILED)

    # List all jobs
    all_jobs = job_manager.list_jobs()
    assert len(all_jobs) >= 3

    # List running jobs
    running_jobs = job_manager.list_jobs(status=JobStatus.RUNNING)
    assert len(running_jobs) >= 1

    # List completed jobs
    completed_jobs = job_manager.list_jobs(status=JobStatus.COMPLETED)
    assert len(completed_jobs) >= 1

    print("✓ Listed jobs successfully")
    print(f"  Total jobs: {len(all_jobs)}")
    print(f"  Running: {len(running_jobs)}")
    print(f"  Completed: {len(completed_jobs)}")

    # Cleanup
    job_manager.delete_job(job1)
    job_manager.delete_job(job2)
    job_manager.delete_job(job3)


def test_cancel_job():
    """Test cancelling a job."""
    job_manager = get_job_manager()

    job_id = job_manager.create_job("fit_predict", "handle", "ARIMA")

    # Cancel pending job
    success = job_manager.cancel_job(job_id)
    assert success

    job = job_manager.get_job(job_id)
    assert job.status == JobStatus.CANCELLED

    print("✓ Job cancelled successfully")

    # Cleanup
    job_manager.delete_job(job_id)


def test_list_jobs_tool_rejects_non_string_status():
    """Non-string status values should return validation errors, not crash."""
    from sktime_mcp.tools.job_tools import list_jobs_tool

    for bad_status in (True, 1, ["running"], {"status": "running"}):
        result = list_jobs_tool(status=bad_status)
        assert result["success"] is False
        assert "Invalid status type" in result["error"]


def test_list_jobs_tool_accepts_case_insensitive_status():
    """Valid status strings should still work regardless of case."""
    from sktime_mcp.tools.job_tools import list_jobs_tool

    job_manager = get_job_manager()
    job_id = job_manager.create_job("fit_predict", "handle", "ARIMA")
    job_manager.update_job(job_id, status=JobStatus.RUNNING)

    result = list_jobs_tool(status="RUNNING")

    assert result["success"] is True
    assert any(job["job_id"] == job_id for job in result["jobs"])

    job_manager.delete_job(job_id)


def test_cleanup_old_jobs():
    """Test cleaning up old jobs."""
    job_manager = get_job_manager()

    # Create a job
    job_id = job_manager.create_job("fit_predict", "handle", "ARIMA")

    # Cleanup jobs older than 0 hours (should remove all)
    count = job_manager.cleanup_old_jobs(max_age_hours=0)

    print(f"✓ Cleaned up {count} old job(s)")

    # Job should be gone
    job = job_manager.get_job(job_id)
    assert job is None
