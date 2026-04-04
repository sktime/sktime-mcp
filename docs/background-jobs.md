# Background Job Management

## Overview

The sktime-mcp server now supports **non-blocking background jobs** for long-running operations like model training. This architecture prevents the MCP server from becoming unresponsive when training large models.

## Architecture

The system uses a multi-threaded, asynchronous architecture inspired by CodeGraphContext:

### Key Components

1. **JobManager** (`runtime/jobs.py`) - Thread-safe job tracking
2. **Executor** (`runtime/executor.py`) - Async execution with job tracking
3. **MCP Server** (`server.py`) - Event loop coordination

## How It Works

### 1. Non-Blocking Execution

When you call `fit_predict_async`, the server:

1. Creates a job with a unique ID
2. Schedules the training as an async coroutine on the event loop
3. **Returns immediately** with the job ID
4. Continues handling other requests while training runs in the background

```python
# Traditional (blocking) - Server freezes during training
result = fit_predict(handle, "airline", horizon=12)

# Async (non-blocking) - Server stays responsive
job_info = fit_predict_async(handle, "airline", horizon=12)
# Returns: {"success": True, "job_id": "abc-123-def"}

# Check progress
status = check_job_status("abc-123-def")
# Returns: {"status": "running", "progress_percentage": 66.7, ...}
```

### 2. Progress Tracking

Jobs track their progress through multiple steps:

```python
{
  "job_id": "abc-123-def",
  "status": "running",
  "total_steps": 3,
  "completed_steps": 2,
  "current_step": "Generating predictions (horizon=12)...",
  "progress_percentage": 66.7,
  "estimated_time_remaining": 5.2,
  "estimated_time_remaining_human": "5s"
}
```

### 3. Job Lifecycle

```
PENDING → RUNNING → COMPLETED
                  ↘ FAILED
                  ↘ CANCELLED
```

## Available Tools

### Training Tools

#### `fit_predict_async`

Train a model in the background (non-blocking).

**Arguments:**
- `estimator_handle`: Handle from `instantiate_estimator`
- `dataset`: Dataset name (e.g., "airline", "sunspots")
- `horizon`: Forecast horizon (default: 12)

**Returns:**
```json
{
  "success": true,
  "job_id": "abc-123-def-456",
  "message": "Training job started for ARIMA on airline...",
  "estimator": "ARIMA",
  "dataset": "airline",
  "horizon": 12
}
```

**Example:**
```python
# Instantiate model
model = instantiate_estimator("ARIMA", {"order": [1, 1, 1]})
# Returns: {"handle": "est_abc123"}

# Start training (non-blocking!)
job = fit_predict_async("est_abc123", "airline", horizon=24)
# Returns immediately with job_id

# Server is still responsive - you can do other things!
```

### Job Management Tools

#### `check_job_status`

Check the status and progress of a background job.

**Arguments:**
- `job_id`: Job ID to check

**Returns:**
```json
{
  "success": true,
  "job_id": "abc-123-def",
  "status": "running",
  "job_type": "fit_predict",
  "estimator_name": "ARIMA",
  "dataset_name": "airline",
  "total_steps": 3,
  "completed_steps": 2,
  "current_step": "Generating predictions...",
  "progress_percentage": 66.7,
  "elapsed_time": 10.5,
  "estimated_time_remaining": 5.2,
  "estimated_time_remaining_human": "5s",
  "result": null
}
```

When completed:
```json
{
  "status": "completed",
  "result": {
    "success": true,
    "predictions": {...},
    "horizon": 12
  }
}
```

#### `list_jobs`

List all background jobs with optional filtering.

**Arguments:**
- `status` (optional): Filter by status ("pending", "running", "completed", "failed", "cancelled")
- `limit` (optional): Maximum number of jobs to return (default: 20)

**Returns:**
```json
{
  "success": true,
  "count": 3,
  "jobs": [
    {
      "job_id": "abc-123",
      "status": "completed",
      "estimator_name": "ARIMA",
      ...
    },
    {
      "job_id": "def-456",
      "status": "running",
      "progress_percentage": 50.0,
      ...
    }
  ]
}
```

**Example:**
```python
# List all running jobs
running_jobs = list_jobs(status="running")

# List all jobs (any status)
all_jobs = list_jobs(limit=50)
```

#### `cancel_job`

Cancel a running or pending job.

**Arguments:**
- `job_id`: Job ID to cancel

**Returns:**
```json
{
  "success": true,
  "message": "Job 'abc-123-def' cancelled"
}
```

**Note:** You can only cancel jobs in "pending" or "running" status.

#### `delete_job`

Delete a job from the job manager.

**Arguments:**
- `job_id`: Job ID to delete

**Returns:**
```json
{
  "success": true,
  "message": "Job 'abc-123-def' deleted"
}
```

#### `cleanup_old_jobs`

Remove jobs older than specified hours.

**Arguments:**
- `max_age_hours` (optional): Maximum age in hours (default: 24)

**Returns:**
```json
{
  "success": true,
  "message": "Removed 5 old job(s)",
  "count": 5
}
```

## Complete Workflow Example

```python
# 1. Instantiate a model
model_result = instantiate_estimator("AutoARIMA")
handle = model_result["handle"]

# 2. Start training in background
job_result = fit_predict_async(handle, "airline", horizon=24)
job_id = job_result["job_id"]

# 3. Server is responsive - you can do other things!
# Check available demo datasets
datasets = list_available_data(is_demo=True)

# Search for other estimators
estimators = search_estimators("prophet")

# 4. Check training progress
status = check_job_status(job_id)
print(f"Progress: {status['progress_percentage']}%")
print(f"Current step: {status['current_step']}")
print(f"Time remaining: {status['estimated_time_remaining_human']}")

# 5. Wait for completion (or poll periodically)
while status["status"] == "running":
    time.sleep(1)
    status = check_job_status(job_id)

# 6. Get results
if status["status"] == "completed":
    predictions = status["result"]["predictions"]
    print(f"Predictions: {predictions}")
else:
    print(f"Job failed: {status['errors']}")

# 7. Clean up
delete_job(job_id)
```

## Technical Details

### Async Execution

The `fit_predict_async` method uses:

1. **`asyncio.run_in_executor()`** - Runs synchronous sktime operations in a thread pool
2. **`await asyncio.sleep(0.01)`** - Yields control to the event loop every 10ms
3. **`asyncio.run_coroutine_threadsafe()`** - Schedules coroutines on the main event loop

This prevents blocking the MCP server's event loop.

### Thread Safety

The `JobManager` uses `threading.Lock()` to ensure thread-safe access to job data:

```python
with self.lock:
    self.jobs[job_id] = JobInfo(...)
```

### Progress Updates

Jobs update their status in real-time:

```python
# Step 1: Load data
job_manager.update_job(job_id, 
    completed_steps=0,
    current_step="Loading dataset 'airline'..."
)

# Step 2: Fit model
job_manager.update_job(job_id,
    completed_steps=1,
    current_step="Fitting ARIMA on airline..."
)

# Step 3: Predict
job_manager.update_job(job_id,
    completed_steps=2,
    current_step="Generating predictions..."
)
```

### Memory Management

Jobs are automatically cleaned up after 24 hours (configurable):

```python
# Manual cleanup
cleanup_old_jobs(max_age_hours=12)

# Or delete specific job
delete_job(job_id)
```

## Benefits

✅ **Non-blocking** - Server stays responsive during long operations  
✅ **Progress tracking** - Real-time updates on training progress  
✅ **Time estimation** - Estimated time remaining  
✅ **Error handling** - Graceful failure with error messages  
✅ **Cancellation** - Ability to cancel long-running jobs  
✅ **Memory efficient** - Automatic cleanup of old jobs  

## Comparison: Blocking vs Non-Blocking

### Traditional (Blocking)

```python
# ❌ Server freezes for 30 seconds
result = fit_predict(handle, "large_dataset", horizon=100)
# Server cannot handle any other requests during this time
```

### Async (Non-Blocking)

```python
# ✅ Server returns immediately
job = fit_predict_async(handle, "large_dataset", horizon=100)
# Server continues handling other requests

# Check progress whenever you want
status = check_job_status(job["job_id"])
```

## When to Use Each

### Use `fit_predict` (blocking) when:
- Training on small datasets (< 1 second)
- You need results immediately
- Running in a script/notebook (not via MCP)

### Use `fit_predict_async` (non-blocking) when:
- Training large models
- Working with big datasets
- Server responsiveness is critical
- You want progress updates
- Training might take > 5 seconds

## Future Enhancements

Potential improvements:

- [ ] Support for custom progress callbacks
- [ ] Job persistence across server restarts
- [ ] Job priority queue
- [ ] Parallel job execution limits
- [ ] Streaming progress updates via WebSocket
- [ ] Job result caching

## Troubleshooting

### Job stuck in "running" status

```python
# Cancel and restart
cancel_job(job_id)
new_job = fit_predict_async(handle, dataset, horizon)
```

### Too many old jobs

```python
# Clean up jobs older than 1 hour
cleanup_old_jobs(max_age_hours=1)
```

### Job failed

```python
status = check_job_status(job_id)
if status["status"] == "failed":
    print(f"Errors: {status['errors']}")
```
