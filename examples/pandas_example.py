"""
Example: Loading data from a pandas DataFrame.

This example demonstrates how to use the sktime-mcp data loading
functionality with in-memory pandas DataFrames.
"""

import pandas as pd

from sktime_mcp.data import DataSourceRegistry
from sktime_mcp.runtime.executor import get_executor

# Create sample time series data
dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
values = [100 + i + (i % 7) * 5 for i in range(100)]

# Create DataFrame
df = pd.DataFrame(
    {
        "date": dates,
        "sales": values,
        "temperature": [20 + (i % 10) for i in range(100)],
    }
)

print("Sample DataFrame:")
print(df.head())
print(f"\nShape: {df.shape}")

# Method 1: Using DataSourceRegistry directly
print("\n" + "=" * 60)
print("Method 1: Using DataSourceRegistry")
print("=" * 60)

config = {
    "type": "pandas",
    "data": df,
    "time_column": "date",
    "target_column": "sales",
    "exog_columns": ["temperature"],
}

adapter = DataSourceRegistry.create_adapter(config)
data = adapter.load()
is_valid, validation_report = adapter.validate(data)

print(f"\nData loaded: {len(data)} rows")
print(f"Valid: {is_valid}")
print(f"Validation: {validation_report}")

y, X = adapter.to_sktime_format(data)
print(f"\nTarget (y): {y.shape}")
print(f"Exogenous (X): {X.shape if X is not None else None}")

# Method 2: Using Executor (recommended for MCP tools)
print("\n" + "=" * 60)
print("Method 2: Using Executor")
print("=" * 60)

executor = get_executor()

# Load data
result = executor.load_data_source(config)
print(f"\nLoad result: {result['success']}")
print(f"Data handle: {result.get('data_handle')}")
print(f"Metadata: {result.get('metadata')}")

# Instantiate a forecaster
estimator_result = executor.instantiate("NaiveForecaster", {"strategy": "last"})
print(f"\nEstimator handle: {estimator_result['handle']}")

# Fit and predict
if result["success"] and estimator_result["success"]:
    predictions = executor.fit_predict_with_data(
        estimator_handle=estimator_result["handle"],
        data_handle=result["data_handle"],
        horizon=7,
    )

    print(f"\nPredictions: {predictions['success']}")
    if predictions["success"]:
        print("Forecast for next 7 days:")
        for step, value in list(predictions["predictions"].items())[:7]:
            print(f"  Step {step}: {value:.2f}")

# List all data handles
handles = executor.list_data_handles()
print(f"\nActive data handles: {handles['count']}")

# Clean up
if result["success"]:
    cleanup = executor.release_data_handle(result["data_handle"])
    print(f"Cleanup: {cleanup['message']}")
