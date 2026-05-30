# Data Source Support

sktime-mcp now supports loading data from multiple sources beyond the built-in demo datasets!

## Supported Data Sources

1. **SQL Databases** - PostgreSQL, MySQL, SQLite, MSSQL
2. **Files** - CSV, TSV, Excel, Parquet

## Quick Start

### 1. CSV File

```python
result = executor.load_data_source({
    "type": "file",
    "path": "/path/to/data.csv",
    "time_column": "date",
    "target_column": "sales",
    "exog_columns": ["temperature", "promotion"],  # Optional
})
```

### 2. SQL Database

```python
# SQLite
result = executor.load_data_source({
    "type": "sql",
    "connection_string": "sqlite:///path/to/database.db",
    "query": "SELECT date, sales FROM sales_table",
    "time_column": "date",
    "target_column": "sales",
})

# PostgreSQL
result = executor.load_data_source({
    "type": "sql",
    "connection_string": "postgresql://user:pass@host:5432/db",
    "query": "SELECT * FROM sales WHERE date >= '2020-01-01'",
    "time_column": "date",
    "target_column": "sales",
})
```

## MCP Tools

### `load_data_source`

Load data from any supported source.

**Arguments:**
- `config` (dict): Data source configuration

**Returns:**
- `success` (bool): Whether loading succeeded
- `data_handle` (str): Handle to reference the loaded data
- `metadata` (dict): Information about the data (rows, columns, frequency, etc.)
- `validation` (dict): Data quality validation results

**Example:**
```json
{
  "config": {
    "type": "pandas",
    "data": {"date": [...], "value": [...]},
    "time_column": "date",
    "target_column": "value"
  }
}
```

### `fit_predict` (custom data)

After `load_data_source`, call `fit_predict` with `data_handle` set (and omit `dataset`, or pass an empty string). This is the same MCP tool used for demo datasets.

**Arguments:**
- `estimator_handle` (str): Handle from `instantiate_estimator`
- `data_handle` (str): Handle from `load_data_source`
- `horizon` (int): Forecast horizon (default: 12)

**Example:**
```json
{
  "estimator_handle": "est_abc123",
  "data_handle": "data_xyz789",
  "horizon": 7
}
```

### `list_data_sources`

List all available data source types.

**Returns:**
- `sources` (list): Available source types
- `descriptions` (dict): Description for each source

### `list_available_data`

List available data in a single response, including demo datasets and active handles.

**Arguments (optional):**
- `is_demo` (bool):
    - `True` → only demo datasets
    - `False` → only active data handles
    - omitted → both

**Returns:**
- `system_demos` (list): Built-in demo dataset names
- `active_handles` (list): Active data handle information
- `total` (int): Combined count of returned entries

### `release_data_handle`

Release a data handle and free memory.

**Arguments:**
- `data_handle` (str): Handle to release

## Configuration Options

### Pandas Adapter

```python
{
    "type": "pandas",
    "data": df_or_dict,  # DataFrame or dict
    "time_column": "date",  # Optional, will try to detect
    "target_column": "sales",  # Optional, defaults to first column
    "exog_columns": ["temp", "promo"],  # Optional
    "frequency": "D"  # Optional, will try to infer
}
```

### SQL Adapter

```python
{
    "type": "sql",
    
    # Option 1: Connection string
    "connection_string": "postgresql://user:pass@host:5432/db",
    
    # Option 2: Individual components
    "dialect": "postgresql",  # postgresql, mysql, sqlite, mssql
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "pass",
    
    # Query
    "query": "SELECT * FROM sales",  # Direct SQL query
    # OR
    "table": "sales",  # Table name
    "filters": {"region": "North"},  # Optional filters
    
    # Column mapping
    "time_column": "date",
    "target_column": "sales",
    "exog_columns": ["temperature"],
    
    # Optional
    "parse_dates": ["date"],
    "frequency": "D"
}
```

### File Adapter

```python
{
    "type": "file",
    "path": "/path/to/data.csv",
    "format": "csv",  # csv, excel, parquet (auto-detected if not specified)
    
    # Column mapping
    "time_column": "date",
    "target_column": "sales",
    "exog_columns": ["temperature"],
    
    # CSV-specific options
    "csv_options": {
        "sep": ",",
        "header": 0,
        "encoding": "utf-8"
    },
    
    # Excel-specific options
    "excel_options": {
        "sheet_name": 0,
        "header": 0
    },
    
    # Common options
    "parse_dates": True,
    "frequency": "D"
}
```

## Data Validation

All data sources are automatically validated for:

- ✅ DatetimeIndex presence
- ✅ No duplicate time indices
- ✅ Sufficient data points
- ✅ Missing value detection
- ✅ Frequency inference

Validation results are included in the response:

```python
{
    "valid": True,
    "errors": [],  # Critical issues that prevent usage
    "warnings": ["Missing values detected: {'sales': 5.0}"]  # Non-critical issues
}
```

## Installation

### Core (Pandas support included)
```bash
pip install -e .
```

### With SQL support
```bash
pip install -e ".[sql]"
```

### With file format support
```bash
pip install -e ".[files]"
```

### With all optional dependencies
```bash
pip install -e ".[all]"
```

## Examples

See the `examples/` directory for complete working examples:

- `pandas_example.py` - Loading from pandas DataFrames
- `csv_example.py` - Loading from CSV/TSV files
- `sql_example.py` - Loading from SQL databases (SQLite, PostgreSQL, MySQL)

## Architecture

```
Data Source Layer
├── base.py                 # Abstract adapter interface
├── registry.py             # Adapter registry
└── adapters/
    ├── pandas_adapter.py   # In-memory DataFrames
    ├── sql_adapter.py      # SQL databases
    └── file_adapter.py     # CSV, Excel, Parquet
```

Each adapter implements:
- `load()` - Load data from source
- `validate()` - Check data quality
- `to_sktime_format()` - Convert to (y, X) format

## Metadata

Each loaded data source provides rich metadata:

```python
{
    "source": "sql",
    "rows": 100,
    "columns": ["sales", "temperature"],
    "frequency": "D",
    "start_date": "2020-01-01",
    "end_date": "2020-04-09",
    "missing_values": {"sales": 0, "temperature": 2}
}
```

## Error Handling

All operations return structured error information:

```python
{
    "success": False,
    "error": "Could not convert index to datetime",
    "error_type": "ValueError",
    "validation": {
        "valid": False,
        "errors": ["Index must be DatetimeIndex"]
    }
}
```

## 💡 LLM Guidelines & Best Practices

When using these tools to handle user data, follow these best practices to avoid common execution errors:

### 1. Column Identification
*   **Target vs. Features**: Identify which column is the target (`y`) and which are exogenous features (`X`).
*   **Numeric Only**: Ensure the target column contains only numeric data. Forecasting models will fail if string/categorical columns are accidentally included in `y`.
*   **User Confirmation**: If the dataset has multiple columns and the user hasn't specified which one to forecast, **ask for clarification** instead of guessing.

### 2. Strategic Loading
*   **Use `usecols`**: When loading from CSV or Excel, use `csv_options: {"usecols": ["Column1", "Column2"]}` to load only the necessary columns. This prevents string-based columns from being accidentally passed to the model.
*   **Time Column Mapping**: Always map the `time_column` if one exists. If it doesn't, sktime will default to a `RangeIndex`, which is often safer for simple sequences.

### 3. Frequency Hurdles
*   **Month-Start vs Month-End**: If you encounter errors like `<MonthBegin> is not supported as period frequency`, it is likely due to `statsmodels` limitations. In such cases, consider dropping the timestamp index and using a simple integer index (RangeIndex) by not specifying a `time_column`.

### 4. Validation First
*   **Status Check**: Always inspect the `validation` object returned by `load_data_source`. If `valid` is `false`, read the `errors` list to understand why the data might break the model.

## Troubleshooting

### "No module named 'sqlalchemy'"
```bash
pip install "sktime-mcp[sql]"
```

### "No module named 'openpyxl'" (for Excel files)
```bash
pip install "sktime-mcp[files]"
```

### "No module named 'pyarrow'" (for Parquet files)
```bash
pip install "sktime-mcp[files]"
```

### "Could not infer frequency"
Specify frequency explicitly in config:
```python
config = {
    ...
    "frequency": "D"  # Daily, "W" for weekly, "M" for monthly, etc.
}
```

### "Index must be DatetimeIndex"
Ensure your time column is specified correctly:
```python
config = {
    ...
    "time_column": "date",  # Name of your date/time column
    "parse_dates": True
}
```
