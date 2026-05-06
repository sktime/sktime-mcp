# Data Management

Loading and preparing data is the first step in any forecasting workflow. With `sktime-mcp`, you don't need to write data-loading code; you simply tell your AI assistant where your data is and how it is structured.

## Loading Your Data

You can ask your assistant to load data from several different sources. When you do, be sure to provide the necessary context like file paths or database connection details.

### Local Files
To load a local file, provide the absolute path. Supported formats include CSV, Excel (`.xlsx`), and Parquet.

**What to tell your assistant:**
- The **absolute path** to the file.
- Which column contains the **timestamps** (if any).
- Which column is the **target** you want to forecast.

> *Example: "Load my data from /home/user/sales.csv. Use 'Date' as the time column and 'Total' as the target."*

### SQL Databases
You can also pull data directly from a database.

**What to tell your assistant:**
- The **connection string** (URI).
- The **SQL query** to fetch the data.

> *Example: "Load data from my PostgreSQL database at 'postgresql://user:pass@localhost/db' using 'SELECT * FROM monthly_revenue'."*

### Remote URLs
If your data is hosted online, just provide the URL.

> *Example: "Load the CSV data from https://example.com/timeseries.csv"*

## Working with Handles

When the assistant successfully loads your data, it will mention a `data_handle` (e.g., `data_5f2b3c`). This is an internal identifier that the system uses to track your dataset. You don't need to memorize these; you can simply refer to it as "my loaded data" or "the sales dataset" in your follow-up requests.

## Formatting and Cleaning

If your data isn't perfectly formatted, you can ask the assistant to clean it up for you.

### Automatic Cleanup
You can tell the assistant to:
- **Infer the frequency**: "Fix the frequency of my data."
- **Fill missing values**: "Forward fill any gaps in the time series."
- **Remove duplicates**: "Remove any duplicate timestamps in my dataset."

The assistant will use the `format_time_series` tool behind the scenes to ensure the data meets the requirements for forecasting.

## Managing Resources

The system keeps your loaded data in memory for the duration of your session. To free up resources, you can tell the assistant to "release my data" or "clear the memory" when you are finished with a specific dataset.

## Built-in Demo Datasets

For testing, you can ask to use standard time-series datasets by name (e.g., `airline`, `sunspots`, `lynx`) without needing to provide any files.

> *Example: "What demo datasets are available?" or "Use the airline dataset for this forecast."*
