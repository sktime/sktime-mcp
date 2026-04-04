"""
Data loading tools for sktime MCP.

Provides tools for loading data from various sources.
"""

from typing import Any, Dict
from sktime_mcp.runtime.executor import get_executor


def load_data_source_tool(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load data from any source (pandas, SQL, file, etc.).
    
    Args:
        config: Data source configuration
            {
                "type": "pandas" | "sql" | "file",
                ... (type-specific configuration)
            }
    
    Returns:
        Dictionary with:
        - success: bool
        - data_handle: str (handle ID for the loaded data)
        - metadata: dict (information about the data)
        - validation: dict (validation results)
    
    Examples:
        # Pandas DataFrame
        >>> load_data_source_tool({
        ...     "type": "pandas",
        ...     "data": {"date": [...], "value": [...]},
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
        
        # SQL Database
        >>> load_data_source_tool({
        ...     "type": "sql",
        ...     "connection_string": "postgresql://user:pass@host:5432/db",
        ...     "query": "SELECT date, value FROM sales",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
        
        # CSV File
        >>> load_data_source_tool({
        ...     "type": "file",
        ...     "path": "/path/to/data.csv",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
    """
    executor = get_executor()
    return executor.load_data_source(config)


def list_data_sources_tool() -> Dict[str, Any]:
    """
    List all available data source types.
    
    Returns:
        Dictionary with:
        - success: bool
        - sources: list of available source types
        - descriptions: dict with descriptions for each source type
    """
    from sktime_mcp.data import DataSourceRegistry
    
    sources = DataSourceRegistry.list_adapters()
    
    # Get descriptions for each source
    descriptions = {}
    for source_type in sources:
        info = DataSourceRegistry.get_adapter_info(source_type)
        descriptions[source_type] = {
            "class": info["class"],
            "description": info["docstring"].split("\n")[0] if info["docstring"] else "",
        }
    
    return {
        "success": True,
        "sources": sources,
        "descriptions": descriptions,
    }


def fit_predict_with_data_tool(
    estimator_handle: str,
    data_handle: str,
    horizon: int = 12,
) -> Dict[str, Any]:
    """
    Fit and predict using custom data.
    
    Args:
        estimator_handle: Handle from instantiate_estimator
        data_handle: Handle from load_data_source
        horizon: Forecast horizon (default: 12)
    
    Returns:
        Dictionary with predictions
    
    Example:
        >>> fit_predict_with_data_tool(
        ...     estimator_handle="est_abc123",
        ...     data_handle="data_xyz789",
        ...     horizon=12
        ... )
    """
    executor = get_executor()
    return executor.fit_predict_with_data(
        estimator_handle,
        data_handle,
        horizon,
    )


def release_data_handle_tool(data_handle: str) -> Dict[str, Any]:
    """
    Release a data handle and free memory.
    
    Args:
        data_handle: Data handle to release
    
    Returns:
        Dictionary with success status
    """
    executor = get_executor()
    return executor.release_data_handle(data_handle)
