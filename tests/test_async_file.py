import asyncio
import pandas as pd
import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from sktime_mcp.data.adapters.file_adapter import FileAdapter
from sktime_mcp.runtime.jobs import get_job_manager, JobStatus

@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100),
            "value": range(100)
        })
        df.to_csv(path, index=False)
        yield path
    finally:
        os.close(fd)
        if os.path.exists(path):
            os.remove(path)

@pytest.mark.asyncio
async def test_file_adapter_load_async_csv(temp_csv_file):
    """Test FileAdapter.load_async with a CSV file (incremental)."""
    config = {
        "type": "file",
        "path": temp_csv_file,
        "format": "csv",
        "time_column": "date"
    }
    
    adapter = FileAdapter(config)
    job_manager = get_job_manager()
    job_id = job_manager.create_job("load_data", "N/A")
    
    df = await adapter.load_async(job_id=job_id)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert isinstance(df.index, pd.DatetimeIndex)
    
    job = job_manager.get_job(job_id)
    assert "Reading CSV" in job.current_step or "complete" in job.current_step

@pytest.mark.asyncio
async def test_file_adapter_load_async_parquet():
    """Test FileAdapter.load_async with Parquet (executor fallback)."""
    # Just mock the load since we don't want to depend on pyarrow for this test
    config = {
        "type": "file",
        "path": "dummy.parquet",
        "format": "parquet"
    }
    
    adapter = FileAdapter(config)
    
    with patch.object(FileAdapter, 'load') as mock_load:
        mock_load.return_value = pd.DataFrame({"val": [1]})
        
        df = await adapter.load_async()
        
        assert mock_load.called
        assert len(df) == 1
