"""
Test async data loading for SQL and File adapters.
"""

import asyncio
import sys
import time
import pandas as pd
import tempfile
import os

# Add src to path
sys.path.insert(0, '/home/chandangiri/Downloads/AOOP/Lab Practical/Open Source/sktime-mcp/src')

try:
    from sktime_mcp.data import DataSourceRegistry
    from sktime_mcp.runtime.executor import get_executor
    from sktime_mcp.runtime.jobs import get_job_manager, JobStatus
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


async def test_file_adapter_async():
    print("\n1. Testing FileAdapter async chunking...")
    
    # Create a reasonably large temp CSV
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as f:
        # write header
        f.write("date,value1,value2\n")
        # write 25k lines
        for i in range(25000):
            f.write(f"2023-01-{(i%30)+1},{i},{i*2}\n")
        temp_csv_path = f.name
        
    try:
        config = {
            "type": "file",
            "path": temp_csv_path,
            "format": "csv",
            "time_column": "date",
        }
        
        adapter = DataSourceRegistry.create_adapter(config)
        progress_updates = []
        
        async def progress_cb(pct, msg):
            progress_updates.append((pct, msg))
            
        start_time = time.time()
        data = await adapter.load_async(progress_callback=progress_cb)
        elapsed = time.time() - start_time
        
        assert len(data) == 25000
        assert len(progress_updates) >= 4  # Start, at least one chunk read, concat, and finish callback
        assert progress_updates[-1][0] == 100.0
        print(f"✓ FileAdapter async chunking works. Loaded {len(data)} rows in {elapsed:.2f}s")
        print(f"✓ Generated {len(progress_updates)} progress updates")
    finally:
        os.unlink(temp_csv_path)


async def test_sql_adapter_sqlite_async():
    print("\n2. Testing SQLAdapter native aiosqlite async...")
    import sqlite3
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.db') as f:
        temp_db_path = f.name
        
    try:
        # seed data
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_data (date TEXT, value REAL)")
        
        # Insert 3000 rows
        rows = [(f"2023-01-{(i%30)+1}", i*1.5) for i in range(3000)]
        cursor.executemany("INSERT INTO test_data VALUES (?, ?)", rows)
        conn.commit()
        conn.close()
        
        config = {
            "type": "sql",
            "dialect": "sqlite",
            "database": temp_db_path,
            "table": "test_data",
            "time_column": "date",
        }
        
        adapter = DataSourceRegistry.create_adapter(config)
        progress_updates = []
        
        async def progress_cb(pct, msg):
            progress_updates.append((pct, msg))
            
        start_time = time.time()
        data = await adapter.load_async(progress_callback=progress_cb)
        elapsed = time.time() - start_time
        
        assert len(data) == 3000
        assert len(progress_updates) >= 3 # start, fetching, finish
        assert progress_updates[-1][0] == 100.0
        
        print(f"✓ SQLAdapter async (aiosqlite) works. Loaded {len(data)} rows in {elapsed:.2f}s")
        print(f"✓ Generated {len(progress_updates)} progress updates")
    finally:
        os.unlink(temp_db_path)


async def test_pandas_adapter_async():
    print("\n3. Testing PandasAdapter native async...")
    
    # Create sample dataframe in-memory
    df = pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=10000, freq='h'),
        "value": range(10000)
    })
    
    config = {
        "type": "pandas",
        "data": df,
        "time_column": "date",
    }
    
    adapter = DataSourceRegistry.create_adapter(config)
    progress_updates = []
    
    async def progress_cb(pct, msg):
        progress_updates.append((pct, msg))
        
    start_time = time.time()
    data = await adapter.load_async(progress_callback=progress_cb)
    elapsed = time.time() - start_time
    
    assert len(data) == 10000
    assert len(progress_updates) >= 3  # start, middle, end
    assert progress_updates[-1][0] == 100.0
    print(f"✓ PandasAdapter async works. Processed {len(data)} rows in {elapsed:.2f}s")
    print(f"✓ Generated {len(progress_updates)} progress updates")


async def main():
    print("====================================")
    print("Testing Async Data Loading Phase 2")
    print("====================================")
    
    await test_file_adapter_async()
    await test_sql_adapter_sqlite_async()
    await test_pandas_adapter_async()
    
    print("====================================")
    print("✅ All tests passed!")
    print("====================================")

if __name__ == "__main__":
    asyncio.run(main())
