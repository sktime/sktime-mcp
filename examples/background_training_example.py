"""
Example: Training a model in the background with progress tracking.

This demonstrates the non-blocking async training capabilities.
"""

import time

from sktime_mcp.tools.fit_predict import fit_predict_async_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool
from sktime_mcp.tools.job_tools import (
    check_job_status_tool,
    cleanup_old_jobs_tool,
    list_jobs_tool,
)


def main():
    print("=" * 70)
    print("Example: Non-Blocking Model Training with Progress Tracking")
    print("=" * 70)

    # Step 1: Instantiate a model
    print("\n📦 Step 1: Instantiating ARIMA model...")
    model_result = instantiate_estimator_tool(
        estimator="ARIMA", params={"order": [1, 1, 1], "suppress_warnings": True}
    )

    if not model_result["success"]:
        print(f"❌ Failed to instantiate model: {model_result.get('error')}")
        return

    handle = model_result["handle"]
    print(f"✅ Model instantiated: {handle}")

    # Step 2: Start training in background (NON-BLOCKING!)
    print("\n🚀 Step 2: Starting training in background...")
    print("   (Server remains responsive during training)")

    job_result = fit_predict_async_tool(
        estimator_handle=handle,
        dataset="airline",
        horizon=24,
    )

    if not job_result["success"]:
        print(f"❌ Failed to start training: {job_result.get('error')}")
        return

    job_id = job_result["job_id"]
    print("✅ Training started!")
    print(f"   Job ID: {job_id}")
    print(f"   Estimator: {job_result['estimator']}")
    print(f"   Dataset: {job_result['dataset']}")
    print(f"   Horizon: {job_result['horizon']}")

    # Step 3: Monitor progress
    print("\n📊 Step 3: Monitoring progress...")
    print("   (Checking status every 0.5 seconds)")
    print()

    last_progress = -1
    while True:
        status = check_job_status_tool(job_id)

        if not status["success"]:
            print(f"❌ Failed to check status: {status.get('error')}")
            break

        current_status = status["status"]
        progress = status.get("progress_percentage", 0)
        current_step = status.get("current_step", "")
        time_remaining = status.get("estimated_time_remaining_human", "")

        # Only print when progress changes
        if progress != last_progress:
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)

            print(f"\r   [{bar}] {progress:.1f}%", end="")
            if time_remaining:
                print(f" | ETA: {time_remaining}", end="")
            if current_step:
                print(f" | {current_step}", end="")
            print(" " * 10, end="")  # Clear any leftover text

            last_progress = progress

        # Check if completed
        if current_status in ["completed", "failed", "cancelled"]:
            print()  # New line after progress bar
            break

        time.sleep(0.5)

    # Step 4: Get results
    print("\n📈 Step 4: Getting results...")

    final_status = check_job_status_tool(job_id)

    if final_status["status"] == "completed":
        result = final_status.get("result", {})
        predictions = result.get("predictions", {})
        elapsed = final_status.get("elapsed_time", 0)

        print("✅ Training completed successfully!")
        print(f"   Elapsed time: {elapsed:.2f}s")
        print(f"   Predictions generated: {len(predictions)} steps")
        print("\n   First 5 predictions:")
        for _i, (key, value) in enumerate(list(predictions.items())[:5]):
            print(f"      Step {key}: {value:.2f}")

    elif final_status["status"] == "failed":
        errors = final_status.get("errors", [])
        print("❌ Training failed!")
        print(f"   Errors: {errors}")

    else:
        print("⚠️  Training was cancelled")

    # Step 5: List all jobs
    print("\n📋 Step 5: Listing all jobs...")
    jobs_result = list_jobs_tool(limit=5)

    if jobs_result["success"]:
        jobs = jobs_result["jobs"]
        print(f"   Total jobs: {jobs_result['count']}")
        for job in jobs[:3]:  # Show first 3
            print(
                f"   - {job['job_id'][:8]}... | {job['status']} | {job.get('estimator_name', 'Unknown')}"
            )

    # Step 6: Cleanup
    print("\n🧹 Step 6: Cleaning up old jobs...")
    cleanup_result = cleanup_old_jobs_tool(max_age_hours=0)

    if cleanup_result["success"]:
        print(f"✅ Cleaned up {cleanup_result['count']} job(s)")

    print("\n" + "=" * 70)
    print("✨ Example completed successfully!")
    print("=" * 70)
    print("\n💡 Key takeaway:")
    print("   The server remained responsive throughout the entire training process!")
    print("   You could have made other requests while training was running.")
    print("=" * 70)


if __name__ == "__main__":
    main()
