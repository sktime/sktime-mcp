import asyncio
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch
from sktime_mcp.data.adapters.url_adapter import UrlAdapter
from sktime_mcp.runtime.jobs import get_job_manager, JobStatus
from sktime_mcp.runtime.executor import get_executor

@pytest.mark.asyncio
async def test_url_adapter_load_async_progress():
    """Test that UrlAdapter.load_async updates job progress."""
    job_manager = get_job_manager()
    job_id = job_manager.create_job(
        job_type="load_data",
        estimator_handle="None",
        total_steps=1
    )
    
    config = {
        "type": "url",
        "url": "https://example.com/data.csv",
        "target_column": "target"
    }
    
    adapter = UrlAdapter(config)
    
    # Mock data for the CSV
    csv_content = b"date,target\n2020-01-01,10\n2020-01-02,20\n"
    
    # Mock aiohttp
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"content-length": str(len(csv_content))}
    
    # Simulate chunked iteration
    async def mock_iter_chunked(size):
        yield csv_content[:10]
        yield csv_content[10:]
        
    mock_response.content.iter_chunked = mock_iter_chunked
    
    mock_session = MagicMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.__aenter__.return_value = mock_session
    
    # Mock FileAdapter to avoid actual disk IO for the loaded file
    with patch("aiohttp.ClientSession", return_value=mock_session):
        with patch("sktime_mcp.data.adapters.url_adapter.FileAdapter") as MockFileAdapter:
            mock_file_adapter = MockFileAdapter.return_value
            mock_file_adapter.load_async = AsyncMock(return_value=pd.DataFrame({"target": [10, 20]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"])))
            mock_file_adapter.get_metadata.return_value = {"rows": 2}
            
            df = await adapter.load_async(job_id=job_id)
            
            assert len(df) == 2
            
            # Verify job progress was updated
            job = job_manager.get_job(job_id)
            assert "Downloading:" in job.current_step
            # Since we had 2 chunks, the last status should be 100%
            assert "100.0%" in job.current_step

@pytest.mark.asyncio
async def test_executor_load_data_source_async():
    """Test the Executor integration for async data loading."""
    executor = get_executor()
    config = {
        "type": "url",
        "url": "https://example.com/test.csv"
    }
    
    # Mock the adapter creation and its load_async method
    with patch("sktime_mcp.data.DataSourceRegistry.create_adapter") as MockCreate:
        mock_adapter = MockCreate.return_value
        mock_adapter.load_async = AsyncMock(return_value=pd.DataFrame({"val": [1]}))
        mock_adapter.validate.return_value = (True, {"valid": True})
        mock_adapter.to_sktime_format.return_value = (pd.Series([1], name="val"), None)
        mock_adapter.get_metadata.return_value = {"rows": 1}
        
        result = await executor.load_data_source_async(config)
        
        assert result["success"] is True
        assert "data_handle" in result
        
        # Verify job is completed
        job_id = result.get("job_id")
        if job_id:
            job = executor._job_manager.get_job(job_id)
            assert job.status == JobStatus.COMPLETED
