"""
Test async data loading functionality.
"""

import asyncio
import sys
import time
import pandas as pd

# Add src to path
sys.path.insert(0, '/home/chandangiri/Downloads/AOOP/Lab Practical/Open Source/sktime-mcp/src')

try:
    from sktime_mcp.data import DataSourceRegistry, PandasAdapter, UrlAdapter
    from sktime_mcp.runtime.executor import get_executor
    from sktime_mcp.runtime.jobs import get_job_manager, JobStatus
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


async def test_base_load_async():
    """Test the default run_in_executor fallback in DataSourceAdapter."""
    print("\n1. Testing base adapter load_async (Pandas fallback)...")
    
    config = {
        "type": "pandas",
        "data": {
            "date": pd.date_range(start='2020-01-01', periods=10, freq='D'),
            "value": list(range(10, 20)),
        },
        "time_column": "date",
        "target_column": "value",
    }
    
    adapter = DataSourceRegistry.create_adapter(config)
    
    # Progress callback tracker
    progress_calls = []
    
    def progress_cb(pct, msg):
        progress_calls.append((pct, msg))
    
    data = await adapter.load_async(progress_callback=progress_cb)
    
    assert len(data) == 10
    assert len(progress_calls) == 2  # Start and End
    assert progress_calls[-1][0] == 100.0
    print("✓ Base load_async fallback works")


async def test_url_adapter_load_async():
    """Test the native aiohttp streaming in UrlAdapter."""
    print("\n2. Testing UrlAdapter native load_async...")
    
    # Simple public CSV file
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
    
    config = {
        "type": "url",
        "url": url,
        "format": "csv",
    }
    
    adapter = DataSourceRegistry.create_adapter(config)
    
    progress_updates = []
    async def progress_cb(pct, msg):
        progress_updates.append((pct, msg))
        if pct > 0 and pct < 100:
            print(f"  {msg}")
    
    start_time = time.time()
    data = await adapter.load_async(progress_callback=progress_cb)
    elapsed = time.time() - start_time
    
    assert len(data) > 0
    assert len(progress_updates) >= 2 # start, download steps, end
    assert progress_updates[-1][0] == 100.0
    print(f"✓ UrlAdapter async streaming works. Loaded {len(data)} rows in {elapsed:.2f}s")


async def test_executor_load_data_source_async():
    """Test the full executor job flow for async data loading."""
    print("\n3. Testing Executor load_data_source_async...")
    
    executor = get_executor()
    job_manager = get_job_manager()
    
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/fmri.csv"
    config = {
        "type": "url",
        "url": url,
        "format": "csv",
    }
    
    # Start async load
    # It returns a result dict just like fit_predict_async, wait for it
    result = await executor.load_data_source_async(config)
    
    assert result["success"] == True
    assert "job_id" in result
    job_id = result["job_id"]
    
    # Check job is complete
    job = job_manager.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.completed_steps == 3
    assert job.progress_percentage == 100.0
    
    print(f"✓ Executor async data source job completed successfully")
    print(f"  Data handle: {result.get('data_handle')}")
    print(f"  Rows: {result.get('metadata', {}).get('rows')}")

    # Check that bad URL fails cleanly
    print("\n4. Testing Executor load_data_source_async failure...")
    
    bad_config = {
        "type": "url",
        "url": "https://this-url-definitely-does-not-exist.com/data.csv",
    }
    
    bad_result = await executor.load_data_source_async(bad_config)
    
    assert bad_result["success"] == False
    assert "job_id" in bad_result
    
    bad_job = job_manager.get_job(bad_result["job_id"])
    assert bad_job is not None
    assert bad_job.status == JobStatus.FAILED
    assert len(bad_job.errors) > 0
    
    print(f"✓ Bad URL gracefully fails the job: {bad_job.errors[0]}")


async def main():
    print("====================================")
    print("Testing Async Data Loading")
    print("====================================")
    
    await test_base_load_async()
    await test_url_adapter_load_async()
    await test_executor_load_data_source_async()
    
    print("====================================")
    print("✅ All tests passed!")
    print("====================================")

if __name__ == "__main__":
    asyncio.run(main())
