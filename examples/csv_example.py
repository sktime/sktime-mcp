"""
Example: Loading data from a CSV file.

This example demonstrates how to use the sktime-mcp data loading
functionality with CSV files.
"""

import tempfile
from pathlib import Path

import pandas as pd

from sktime_mcp.runtime.executor import get_executor

# Create a sample CSV file
sample_data = pd.DataFrame(
    {
        "date": pd.date_range(start="2020-01-01", periods=100, freq="D"),
        "sales": [100 + i + (i % 7) * 5 for i in range(100)],
        "temperature": [20 + (i % 10) for i in range(100)],
    }
)

csv_path = Path(tempfile.gettempdir()) / "sample_sales_data.csv"
sample_data.to_csv(csv_path, index=False)
print(f"Created sample CSV file: {csv_path}")
print(f"File size: {csv_path.stat().st_size} bytes")

# Load data from CSV
print("\n" + "=" * 60)
print("Loading data from CSV file")
print("=" * 60)

config = {
    "type": "file",
    "path": str(csv_path),
    "format": "csv",  # Optional, auto-detected from extension
    "time_column": "date",
    "target_column": "sales",
    "exog_columns": ["temperature"],
    "csv_options": {
        "sep": ",",
        "header": 0,
    },
}

executor = get_executor()

# Load data
result = executor.load_data_source(config)
print(f"\nLoad result: {result['success']}")
print(f"Data handle: {result.get('data_handle')}")

if result["success"]:
    metadata = result["metadata"]
    print("\nMetadata:")
    print(f"  Source: {metadata['source']}")
    print(f"  Format: {metadata['format']}")
    print(f"  Rows: {metadata['rows']}")
    print(f"  Columns: {metadata['columns']}")
    print(f"  Frequency: {metadata['frequency']}")
    print(f"  Date range: {metadata['start_date']} to {metadata['end_date']}")

    validation = result["validation"]
    print("\nValidation:")
    print(f"  Valid: {validation['valid']}")
    if validation.get("warnings"):
        print(f"  Warnings: {validation['warnings']}")

    # Instantiate and fit a forecaster
    estimator_result = executor.instantiate("NaiveForecaster", {"strategy": "drift"})

    if estimator_result["success"]:
        predictions = executor.fit_predict(
            estimator_result["handle"],
            "",
            10,
            data_handle=result["data_handle"],
        )

        if predictions["success"]:
            print("\nForecast for next 10 days:")
            for step, value in list(predictions["predictions"].items())[:10]:
                print(f"  Day {step}: {value:.2f}")

        # Clean up
        executor.release_data_handle(result["data_handle"])
        print("\nData handle released")

# Example with TSV file
print("\n" + "=" * 60)
print("Loading data from TSV file")
print("=" * 60)

tsv_path = Path(tempfile.gettempdir()) / "sample_sales_data.tsv"
sample_data.to_csv(tsv_path, sep="\t", index=False)
print(f"Created sample TSV file: {tsv_path}")

config_tsv = {
    "type": "file",
    "path": str(tsv_path),
    "time_column": "date",
    "target_column": "sales",
}

result_tsv = executor.load_data_source(config_tsv)
print(f"\nLoad result: {result_tsv['success']}")
print(f"Metadata: {result_tsv.get('metadata', {}).get('format')}")

if result_tsv["success"]:
    executor.release_data_handle(result_tsv["data_handle"])

print("\nExample completed!")
