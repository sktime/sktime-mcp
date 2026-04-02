"""
Simple demonstration of the job management system.

This shows how jobs are created, tracked, and managed.
"""

import time

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager


def main():
    print("=" * 70)
    print("Job Management System Demo")
    print("=" * 70)

    executor = get_executor()
    job_manager = get_job_manager()

    # 1. Instantiate a model
    print("\n1️⃣  Instantiating model...")
    result = executor.instantiate("NaiveForecaster")
    handle = result["handle"]
    print(f"   ✅ Model created: {handle}")

    # 2. Create a job manually
    print("\n2️⃣  Creating a background job...")
    job_id = job_manager.create_job(
        job_type="fit_predict",
        estimator_handle=handle,
        estimator_name="NaiveForecaster",
        dataset_name="airline",
        horizon=12,
        total_steps=3,
    )
    print(f"   ✅ Job created: {job_id}")

    # 3. Simulate job progress
    print("\n3️⃣  Simulating job execution...")

    # Start job
    job_manager.update_job(job_id, status=JobStatus.RUNNING)
    job = job_manager.get_job(job_id)
    print(f"   Status: {job.status.value}")

    # Step 1: Load data
    job_manager.update_job(job_id, completed_steps=0, current_step="Loading dataset 'airline'...")
    job = job_manager.get_job(job_id)
    print(f"   Progress: {job.progress_percentage:.1f}% - {job.current_step}")
    time.sleep(0.5)

    # Step 2: Fit model
    job_manager.update_job(
        job_id, completed_steps=1, current_step="Fitting NaiveForecaster on airline..."
    )
    job = job_manager.get_job(job_id)
    print(f"   Progress: {job.progress_percentage:.1f}% - {job.current_step}")
    time.sleep(0.5)

    # Step 3: Predict
    job_manager.update_job(
        job_id, completed_steps=2, current_step="Generating predictions (horizon=12)..."
    )
    job = job_manager.get_job(job_id)
    print(f"   Progress: {job.progress_percentage:.1f}% - {job.current_step}")
    time.sleep(0.5)

    # Complete
    job_manager.update_job(
        job_id,
        status=JobStatus.COMPLETED,
        completed_steps=3,
        current_step="Completed",
        result={"success": True, "predictions": {i: 100 + i for i in range(1, 13)}},
    )
    job = job_manager.get_job(job_id)
    print(f"   Progress: {job.progress_percentage:.1f}% - {job.current_step}")

    # 4. Show job details
    print("\n4️⃣  Job details:")
    job_dict = job.to_dict()
    print(f"   Job ID: {job_dict['job_id']}")
    print(f"   Status: {job_dict['status']}")
    print(f"   Estimator: {job_dict['estimator_name']}")
    print(f"   Dataset: {job_dict['dataset_name']}")
    print(f"   Progress: {job_dict['progress_percentage']}%")
    print(f"   Elapsed time: {job_dict['elapsed_time']:.3f}s")
    print(f"   Result: {job_dict['result']['success']}")

    # 5. List all jobs
    print("\n5️⃣  Listing all jobs...")
    all_jobs = job_manager.list_jobs()
    print(f"   Total jobs: {len(all_jobs)}")
    for j in all_jobs[:3]:
        print(f"   - {j.job_id[:16]}... | {j.status.value} | {j.estimator_name}")

    # 6. Demonstrate cancellation
    print("\n6️⃣  Testing job cancellation...")
    cancel_job_id = job_manager.create_job(
        job_type="fit_predict",
        estimator_handle=handle,
        estimator_name="TestModel",
        total_steps=3,
    )
    job_manager.update_job(cancel_job_id, status=JobStatus.RUNNING)
    print(f"   Created job: {cancel_job_id[:16]}...")

    success = job_manager.cancel_job(cancel_job_id)
    if success:
        cancelled_job = job_manager.get_job(cancel_job_id)
        print(f"   ✅ Job cancelled: {cancelled_job.status.value}")

    # 7. Cleanup
    print("\n7️⃣  Cleaning up...")
    count = job_manager.cleanup_old_jobs(max_age_hours=0)
    print(f"   ✅ Removed {count} job(s)")

    print("\n" + "=" * 70)
    print("✨ Demo completed!")
    print("=" * 70)
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Job creation and tracking")
    print("   ✅ Real-time progress updates")
    print("   ✅ Time estimation")
    print("   ✅ Job listing and filtering")
    print("   ✅ Job cancellation")
    print("   ✅ Automatic cleanup")
    print("=" * 70)


if __name__ == "__main__":
    main()
