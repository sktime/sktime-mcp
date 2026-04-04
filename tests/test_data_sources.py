"""
Simple test to verify the data source implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test imports
print("Testing imports...")
try:
    from sktime_mcp.data import DataSourceRegistry, FileAdapter, PandasAdapter, SQLAdapter

    print("✓ Data module imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    raise Exception(e) from e

# Test registry
print("\nTesting DataSourceRegistry...")
adapters = DataSourceRegistry.list_adapters()
print(f"✓ Available adapters: {adapters}")

for adapter_type in adapters:
    info = DataSourceRegistry.get_adapter_info(adapter_type)
    print(f"  - {adapter_type}: {info['class']}")

# Test pandas adapter with dict data
print("\nTesting PandasAdapter with dict data...")
try:
    import pandas as pd

    config = {
        "type": "pandas",
        "data": {
            "date": pd.date_range(start="2020-01-01", periods=10, freq="D"),
            "value": list(range(10, 20)),
        },
        "time_column": "date",
        "target_column": "value",
    }

    adapter = DataSourceRegistry.create_adapter(config)
    print(f"✓ Adapter created: {type(adapter).__name__}")

    data = adapter.load()
    print(f"✓ Data loaded: {len(data)} rows, {len(data.columns)} columns")

    is_valid, validation = adapter.validate(data)
    print(f"✓ Validation: valid={is_valid}")
    if validation.get("warnings"):
        print(f"  Warnings: {validation['warnings']}")

    y, X = adapter.to_sktime_format(data)
    print(f"✓ Converted to sktime format: y shape={y.shape}, X={X}")

    metadata = adapter.get_metadata()
    print(f"✓ Metadata: {metadata['rows']} rows, frequency={metadata['frequency']}")

except Exception as e:
    print(f"✗ Pandas adapter test failed: {e}")
    import traceback

    traceback.print_exc()

# Test executor integration
print("\nTesting Executor integration...")
try:
    from sktime_mcp.runtime.executor import get_executor

    executor = get_executor()
    print("✓ Executor created")

    # Test load_data_source
    config = {
        "type": "pandas",
        "data": {
            "date": pd.date_range(start="2020-01-01", periods=50, freq="D"),
            "sales": [100 + i for i in range(50)],
        },
        "time_column": "date",
        "target_column": "sales",
    }

    result = executor.load_data_source(config)
    print(f"✓ Data loaded via executor: success={result['success']}")

    if result["success"]:
        print(f"  Data handle: {result['data_handle']}")
        print(f"  Rows: {result['metadata']['rows']}")

        # Test list_data_handles
        handles = executor.list_data_handles()
        print(f"✓ Active data handles: {handles['count']}")

        # Test instantiate and fit_predict_with_data
        est_result = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        if est_result["success"]:
            print(f"✓ Estimator instantiated: {est_result['handle']}")

            pred_result = executor.fit_predict_with_data(
                estimator_handle=est_result["handle"],
                data_handle=result["data_handle"],
                horizon=5,
            )

            if pred_result["success"]:
                print(f"✓ Predictions generated: {len(pred_result['predictions'])} steps")
                print(f"  Sample predictions: {list(pred_result['predictions'].items())[:3]}")
            else:
                print(f"✗ Prediction failed: {pred_result.get('error')}")

        # Test release_data_handle
        cleanup = executor.release_data_handle(result["data_handle"])
        print(f"✓ Data handle released: {cleanup['success']}")

except Exception as e:
    print(f"✗ Executor test failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
