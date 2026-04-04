import asyncio
import pandas as pd
import pytest
import sqlite3
import os
import tempfile
from unittest.mock import MagicMock, patch
from sktime_mcp.data.adapters.sql_adapter import SQLAdapter
from sktime_mcp.runtime.jobs import get_job_manager, JobStatus

@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database with some data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE sales (date TEXT, value REAL)")
        cursor.execute("INSERT INTO sales VALUES ('2023-01-01', 10.5)")
        cursor.execute("INSERT INTO sales VALUES ('2023-01-02', 20.0)")
        cursor.execute("INSERT INTO sales VALUES ('2023-01-03', 30.0)")
        conn.commit()
        conn.close()
        yield path
    finally:
        os.close(fd)
        if os.path.exists(path):
            os.remove(path)

@pytest.mark.asyncio
async def test_sql_adapter_load_async_sqlite(temp_sqlite_db):
    """Test SQLAdapter.load_async with a real SQLite database using aiosqlite."""
    config = {
        "type": "sql",
        "dialect": "sqlite",
        "database": temp_sqlite_db,
        "query": "SELECT * FROM sales",
        "time_column": "date"
    }
    
    adapter = SQLAdapter(config)
    job_manager = get_job_manager()
    job_id = job_manager.create_job("load_data", "N/A")
    
    df = await adapter.load_async(job_id=job_id)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "value" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    
    job = job_manager.get_job(job_id)
    assert job.status in [JobStatus.RUNNING, JobStatus.PENDING, JobStatus.COMPLETED]
    assert any(step in job.current_step for step in ["Executing", "Processing", "Completed"])

@pytest.mark.asyncio
async def test_sql_adapter_load_async_fallback():
    """Test SQLAdapter.load_async fallback to executor for non-sqlite."""
    config = {
        "type": "sql",
        "dialect": "postgresql",
        "connection_string": "postgresql://user:pass@host/db",
        "query": "SELECT * FROM sales"
    }
    
    adapter = SQLAdapter(config)
    
    with patch.object(SQLAdapter, 'load') as mock_load:
        mock_load.return_value = pd.DataFrame({"val": [1]})
        
        df = await adapter.load_async()
        
        assert mock_load.called
        assert len(df) == 1
