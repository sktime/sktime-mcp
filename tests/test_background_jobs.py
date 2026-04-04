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

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
