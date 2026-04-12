# Lazy Loading & Streaming Data

## Overview

The "Data Wall" is a limitation where sktime-mcp server crashes from out-of-memory errors when loading large datasets (>available RAM). This guide explains how to use **lazy loading and streaming** to handle multi-gigabyte files efficiently.

### What is Lazy Loading?

Instead of loading an entire file into memory:
```
Normal: File (1GB) → Load All → Memory Crisis (FAILS)
```

We load in chunks:
```
Lazy: File (1GB) → Chunk 1 → Process → Chunk 2 → Process (SUCCESS)
```

---

## Quick Start: Load a Large CSV

### 1. Use `load_data_source_streaming` Tool

```json
{
  "tool": "load_data_source_streaming",
  "arguments": {
    "source_type": "streaming",
    "config": {
      "path": "/data/large_sales.csv",
      "format": "csv",
      "chunk_size": 50000,
      "time_column": "date",
      "target_column": "sales"
    }
  }
}
```

### 2. Response

```json
{
  "success": true,
  "handle": "data_abc123",
  "metadata": {
    "format": "csv",
    "file_size_bytes": 5368709120,
    "estimated_total_rows": 1000000,
    "chunk_size": 50000,
    "memory_estimate_mb": 200,
    "columns": ["date", "sales", "cost", "profit"]
  },
  "preview": [
    {"date": "2020-01-01", "sales": 1500.50, ...},
    ...
  ]
}
```

### 3. Use the Data Handle

Pass the `data_handle` to `fit_predict_with_data`:

```json
{
  "tool": "fit_predict_with_data",
  "arguments": {
    "estimator_handle": "est_abc123",
    "data_handle": "data_abc123",
    "horizon": 12
  }
}
```

**Done!** Large file processed without crashing.

---

## Supported Formats

| Format | Supported | Notes |
|--------|-----------|-------|
| **CSV** | Yes | Recommended for large files |
| **Parquet** | Yes | Requires `pyarrow` |
| **Excel** | No | Use `load_data_source` for small files only |
| **SQL** | Yes | Use pagination tools |

---

## Tools Reference

### `load_data_source_streaming`

Load large files with automatic chunking.

**Parameters:**
- `source_type`: `"streaming"` (required)
- `config`: Configuration object
  - `path`: File path (required)
  - `format`: `"csv"` or `"parquet"` (auto-detected if omitted)
  - `chunk_size`: Rows per chunk (default: 10000)
  - `time_column`: Column name for time index (optional)
  - `target_column`: Column name for target variable (optional)
  - `csv_options`: Additional pandas read_csv options (optional)
  ```json
  {
    "csv_options": {
      "sep": ",",
      "encoding": "utf-8",
      "skiprows": 0
    }
  }
  ```

**Returns:**
- `handle`: Data handle for use with other tools
- `metadata`: File statistics and schema
- `preview`: Sample of first rows

**Example: Large CSV**
```json
{
  "source_type": "streaming",
  "config": {
    "path": "/mnt/data/transactions_2020_2024.csv",
    "format": "csv",
    "chunk_size": 100000,
    "time_column": "transaction_date",
    "target_column": "amount",
    "csv_options": {
      "sep": ",",
      "encoding": "utf-8"
    }
  }
}
```

---

### `get_data_source_metadata`

Preview file metadata without loading all data.

Useful for understanding large files before deciding how to process them.

**Parameters:**
- `config`: Same format as `load_data_source_streaming`
- `sample_size`: Rows to sample (default: 1000)

**Returns:**
- `metadata`: File statistics
  - `estimated_total_rows`: Rough estimate based on sample
  - `file_size_bytes`: Total file size
  - `memory_estimate_mb`: Estimated RAM needed  
  - `columns`: Column names and types

**Example:**
```json
{
  "config": {
    "path": "/data/large_file.parquet",
    "format": "parquet"
  },
  "sample_size": 5000
}
```

---

### `load_data_paginated_sql`

Load from SQL databases page-by-page (very large tables).

**Parameters:**
- `connection_config`: SQL connection details
  - `dialect`: `postgresql`, `mysql`, `sqlite`, `mssql`
  - `host`, `port`, `database`, `username`, `password`
  - (OR) `connection_string`: Full connection string
  - `table`: Table name
  - `filters`: Optional WHERE conditions
  
- `page_number`: Page to load (0-indexed, default: 0)
- `page_size`: Rows per page (default: 10000)

**Returns:**
- `page_number`: Current page
- `rows_in_page`: Number of rows in this page
- `data`: Rows as list of dictionaries
- `offset`: Starting row number
- `loader_handle`: Handle for fetching subsequent pages

**Example: Load first page of large sales table**
```json
{
  "connection_config": {
    "dialect": "postgresql",
    "host": "db.example.com",
    "port": 5432,
    "database": "sales_db",
    "username": "analyst",
    "password": "***",
    "table": "transactions",
    "filters": {
      "year": ">=2020"
    }
  },
  "page_number": 0,
  "page_size": 50000
}
```

**Pattern: Load All Pages**
```
Page 0 → Get data
  ↓
Page has 50000 rows? → Load Page 1
  ↓
Page has 50000 rows? → Load Page 2
  ↓
Page is empty? → Done
```

---

### `get_streaming_data_sample`

Get a sample from an already-loaded streaming data handle.

**Parameters:**
- `data_handle`: Handle from `load_data_source_streaming`
- `sample_size`: Number of rows (default: 100)

**Returns:**
- `data`: Sample data as list of dictionaries

---

## Memory Efficiency Comparison

### Scenario: 500 MB CSV File

#### Without Lazy Loading (CRASHES)
```
1. read_csv("large.csv") 
2. (FAILS) MemoryError: Unable to allocate 400 MB
```

#### With Lazy Loading (WORKS)
```
1. load_data_source_streaming(path, chunk_size=50000)
2. Process Chunk 1 (50k rows, ~4MB) → RAM: 4MB used
3. Process Chunk 2 (50k rows, ~4MB) → RAM: 4MB used (freed chunk 1)
4. ... (repeat for all chunks)
5. Total RAM never exceeds 50MB
```

---

## Best Practices

### Do

1. **Preview Before Processing**
   ```json
   // First, understand the file
   {
     "tool": "get_data_source_metadata",
     "arguments": {
       "config": {"path": "/data/large.csv"}
     }
   }
   
   // Then load
   {
     "tool": "load_data_source_streaming",
     "arguments": {...}
   }
   ```

2. **Choose Appropriate Chunk Size**
   - Small chunks (1000 rows): More overhead, but predictable memory
   - Large chunks (100000 rows): Better performance, higher RAM
   - **Recommendation**: Start with chunk_size of 50000 for most use cases
   
3. **Use Parquet for Repeated Analysis**
   - CSV requires parsing each time
   - Parquet is pre-formatted and faster

4. **Verify Column Mapping**
   ```json
   {
     "config": {
       "path": "/data/sales.csv",
       "time_column": "date",      // ← Required for time series
       "target_column": "revenue"  // ← Required for forecasting
     }
   }
   ```

### Don't

1. **Don't Use `load_data_source` for Large Files**
   - Use only for files <500MB
   - Falls back to full in-memory loading
   
   ```json
   AVOID:
   {
     "tool": "load_data_source",
     "arguments": {
       "config": {
         "type": "file",
         "path": "/data/5gb_file.csv"  // Too large!
       }
     }
   }
   
   USE INSTEAD:
   {
     "tool": "load_data_source_streaming",
     "arguments": {...}
   }
   ```

2. **Don't Assume First Column is Time Index**
   - Always explicitly specify `time_column`
   - Series forecasting requires proper time indexing

3. **Don't Load Everything at Once**
   - Use iterative processing via chunked loading
   - Work with what fits in memory

---

## Workflow Examples

### Example 1: Forecast from Large CSV (1GB+)

```
Agent Plan:
1. Get metadata about the file
   → File size: 1.2 GB, ~2M rows
   
2. Load with streaming
   → Handle: data_xyz789
   
3. Create forecaster
   → Handle: est_abc123
   
4. Fit and predict
   → Uses streaming data handle
   → Processes in 50k-row chunks
   
5. Get results
   → Predictions without OOM!
```

**Conversation:**
```
User: "Forecast next 12 months of sales from my large sales.csv (1GB)"

Agent:
1. get_data_source_metadata({path: "/data/sales.csv"})
   → "File has 2M rows, 8 columns"

2. load_data_source_streaming({
     path: "/data/sales.csv",
     chunk_size: 100000,
     time_column: "date",
     target_column: "revenue"
   })
   → Handle: data_92k3j1

3. instantiate_estimator("ExponentialSmoothing", {trend: "add", seasonal: "add"})
   → Handle: est_8h3k2l

4. fit_predict_with_data({
     estimator_handle: "est_8h3k2l",
     data_handle: "data_92k3j1",
     horizon: 12
   })
   -> SUCCESS: Predictions returned
```

---

### Example 2: Process Large SQL Table

```
Agent Plan:
1. Load first page of transactions table
2. Fit model on page 1
3. Load page 2, continue fitting with partial_fit
4. Repeat until all pages processed
5. Generate predictions
```

**Code Pattern:**
```json
// Page 1
{
  "tool": "load_data_paginated_sql",
  "arguments": {
    "connection_config": {...},
    "page_number": 0,
    "page_size": 50000
  }
}

// Page 2
{
  "tool": "load_data_paginated_sql",
  "arguments": {
    "connection_config": {...},
    "page_number": 1,
    "page_size": 50000
  }
}

// ... repeat until empty
```

---

## Troubleshooting

### Issue: "Handle not found"

**Cause**: Data handle expired or invalid
**Solution**: Reload the data
```json
{
  "tool": "load_data_source_streaming",
  "arguments": {...}
}
```

---

### Issue: "Chunk size too large"

**Cause**: chunk_size exceeds available RAM
**Solution**: Reduce chunk_size
```json
{
  "config": {
    "path": "/data/large.csv",
    "chunk_size": 10000  // ← Reduced from 100000
  }
}
```

---

### Issue: "Cannot parse date format"

**Cause**: Time column format not recognized
**Solution**: Omit time_column and use integer index
```json
{
  "config": {
    "path": "/data/sales.csv"
    // Removed "time_column": "date"
    // ↓ Will use RangeIndex(0, N)
  }
}
```

---

## Performance Tips

| Optimization | Impact | Effort |
|--------------|--------|--------|
| **Use Parquet format** | 10x faster | Medium (convert CSV once) |
| **Increase chunk_size** | 2x faster | Low (tune parameter) |
| **Pre-filter in SQL** | 5x faster | High (modify query) |
| **Compress files** | 50% smaller | Medium (storage setup) |

---

## Limits & Limitations

| Aspect | Limit | Notes |
|--------|-------|-------|
| **File Size** | Unlimited | As long as chunks fit in RAM |
| **Chunk Size** | RAM - 1GB | Keep small margin for estimators |
| **Columns** | 1000+ | No hard limit |
| **Formats** | CSV, Parquet, SQL | SQL requires SQLAlchemy |
| **Frequency** | Cannot infer from chunks | Specify manually if needed |

---

## See Also

- [Data Sources](data-sources.md) - Complete data loading guide
- [User Guide](user-guide.md) - All MCP tools and workflows
- [Architecture](architecture.md) - How data flows through the system
