import asyncio
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sktime_mcp.data.adapters.url_adapter import UrlAdapter
from sktime_mcp.runtime.jobs import get_job_manager, JobStatus

@pytest.mark.asyncio
async def test_url_adapter_load_async():
    """Test UrlAdapter.load_async with a mock response."""
    config = {
        "type": "url",
        "url": "https://example.com/data.csv",
        "format": "csv"
    }
    
    adapter = UrlAdapter(config)
    job_manager = get_job_manager()
    job_id = job_manager.create_job("load_data", "N/A")
    
    # Mock aiohttp
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {'Content-Length': '1024'}
        
        # Mock iter_chunked
        async def mock_iter_chunked(n):
            yield b"date,value\n2023-01-01,10\n2023-01-02,20"
            
        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Mock FileAdapter.load since we don't want to actually parse a file here
        with patch('sktime_mcp.data.adapters.FileAdapter.load') as mock_file_load:
            mock_file_load.return_value = pd.DataFrame({
                "value": [10, 20]
            }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
            
            df = await adapter.load_async(job_id=job_id)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            
            # Check if job was updated
            job = job_manager.get_job(job_id)
            assert "Downloading" in job.current_step or "Parsing" in job.current_step

@pytest.mark.asyncio
async def test_executor_load_data_source_async():
    """Test Executor.load_data_source_async integration."""
    from sktime_mcp.runtime.executor import get_executor
    executor = get_executor()
    
    config = {
        "type": "url",
        "url": "https://example.com/data.csv"
    }
    
    job_manager = get_job_manager()
    job_id = job_manager.create_job("load_data", "N/A")
    
    with patch('sktime_mcp.data.adapters.UrlAdapter.load_async') as mock_load_async:
        mock_load_async.return_value = pd.DataFrame({
            "value": [10, 20]
        }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        
        # Mock adapter creation
        with patch('sktime_mcp.data.DataSourceRegistry.create_adapter') as mock_create_adapter:
            from sktime_mcp.data.adapters.url_adapter import UrlAdapter
            mock_adapter = MagicMock(spec=UrlAdapter)
            mock_adapter.load_async.return_value = mock_load_async.return_value
            mock_adapter.validate.return_value = (True, {})
            mock_adapter.to_sktime_format.return_value = (mock_load_async.return_value["value"], None)
            mock_adapter.get_metadata.return_value = {}
            mock_create_adapter.return_value = mock_adapter
            
            result = await executor.load_data_source_async(config, job_id)
            
            assert result["success"] is True
            assert "data_handle" in result
            
            job = job_manager.get_job(job_id)
            assert job.status == JobStatus.COMPLETED
            assert job.completed_steps == 4
