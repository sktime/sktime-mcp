"""
Test background job management.
"""

import asyncio
import time

import pytest

from sktime_mcp.runtime.executor import get_executor
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


@pytest.mark.asyncio
async def test_async_fit_predict():
    """Test async fit_predict."""
    executor = get_executor()
    job_manager = get_job_manager()

    # Instantiate a simple forecaster
    result = executor.instantiate("NaiveForecaster")
    assert result["success"]
    handle = result["handle"]

    print(f"✓ Instantiated NaiveForecaster: {handle}")

    # Create job
    job_id = job_manager.create_job(
        job_type="fit_predict",
        estimator_handle=handle,
        estimator_name="NaiveForecaster",
        dataset_name="airline",
        horizon=12,
        total_steps=3,
    )

    print(f"✓ Created job: {job_id}")

    # Run async fit_predict
    result = await executor.fit_predict_async(handle, "airline", 12, job_id)

    # Check result
    assert result["success"]
    assert "predictions" in result

    # Check job status
    job = job_manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    assert job.result is not None

    print("✓ Async fit_predict completed")
    print(f"  Status: {job.status.value}")
    print(f"  Progress: {job.progress_percentage}%")
    print(f"  Elapsed time: {job.elapsed_time}s")
    print(f"  Predictions: {len(result['predictions'])} steps")

    # Cleanup
    job_manager.delete_job(job_id)


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


def test_cleanup_old_jobs():
    """Test cleaning up old jobs with a valid max_age_hours."""
    from datetime import datetime, timedelta

    job_manager = get_job_manager()

    # Create a job
    job_id = job_manager.create_job("fit_predict", "handle", "ARIMA")

    # Manually backdating creation time to simulate an old job (25 hours ago)
    with job_manager.lock:
        job_manager.jobs[job_id].created_at = datetime.now() - timedelta(hours=25)

    # Cleanup jobs older than 24 hours — should remove our backdated job
    count = job_manager.cleanup_old_jobs(max_age_hours=24)
    assert count >= 1

    print(f"\u2713 Cleaned up {count} old job(s)")

    # Job should be gone
    job = job_manager.get_job(job_id)
    assert job is None


def test_cleanup_old_jobs_tool_zero_age_rejected():
    """Test that cleanup_old_jobs_tool rejects max_age_hours=0.

    Passing 0 would make the cutoff == datetime.now(), causing every job
    to appear 'old' and be silently deleted.  The tool must refuse.
    """
    from sktime_mcp.tools.job_tools import cleanup_old_jobs_tool

    result = cleanup_old_jobs_tool(max_age_hours=0)
    assert result["success"] is False
    assert "error" in result
    assert ">= 1" in result["error"]

    print("\u2713 max_age_hours=0 correctly rejected")


def test_cleanup_old_jobs_tool_negative_age_rejected():
    """Test that cleanup_old_jobs_tool rejects negative max_age_hours.

    A negative value makes timedelta subtract a negative delta, shifting
    the cutoff into the future so ALL jobs match and get deleted.
    """
    from sktime_mcp.tools.job_tools import cleanup_old_jobs_tool

    for bad_value in [-1, -10, -100]:
        result = cleanup_old_jobs_tool(max_age_hours=bad_value)
        assert result["success"] is False, f"Expected failure for max_age_hours={bad_value}"
        assert "error" in result
        assert "max_age_hours" in result["error"]

    print("\u2713 Negative max_age_hours values correctly rejected")


def test_cleanup_old_jobs_tool_valid_age():
    """Test that cleanup_old_jobs_tool succeeds with a valid max_age_hours."""
    from sktime_mcp.tools.job_tools import cleanup_old_jobs_tool

    result = cleanup_old_jobs_tool(max_age_hours=24)
    assert result["success"] is True
    assert "count" in result
    assert "message" in result

    print(f"\u2713 Valid max_age_hours=24 accepted, removed {result['count']} job(s)")



def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Background Job Management")
    print("=" * 60)

    print("\n1. Testing job creation...")
    test_job_creation()

    print("\n2. Testing job updates...")
    test_job_updates()

    print("\n3. Testing list jobs...")
    test_list_jobs()

    print("\n4. Testing async fit_predict...")
    asyncio.run(test_async_fit_predict())

    print("\n5. Testing cancel job...")
    test_cancel_job()

    print("\n6. Testing cleanup old jobs...")
    test_cleanup_old_jobs()

    print("\n7. Testing cleanup tool validation (zero age)...")
    test_cleanup_old_jobs_tool_zero_age_rejected()

    print("\n8. Testing cleanup tool validation (negative age)...")
    test_cleanup_old_jobs_tool_negative_age_rejected()

    print("\n9. Testing cleanup tool validation (valid age)...")
    test_cleanup_old_jobs_tool_valid_age()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
