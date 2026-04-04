"""
Example demonstrating the instantiate_pipeline tool.

This shows how to use the new instantiate_pipeline tool to create
complete pipelines rather than individual components.
"""

from sktime_mcp.tools.fit_predict import fit_predict_tool
from sktime_mcp.tools.instantiate import instantiate_pipeline_tool


def example_1_simple_pipeline():
    """Example 1: Create a simple deseasonalization + forecasting pipeline."""
    print("=" * 60)
    print("Example 1: Simple Pipeline (Deseasonalizer → ARIMA)")
    print("=" * 60)

    # Instantiate the pipeline
    result = instantiate_pipeline_tool(components=["ConditionalDeseasonalizer", "ARIMA"])

    print(f"Success: {result['success']}")
    if result["success"]:
        print(f"Handle: {result['handle']}")
        print(f"Pipeline: {result['pipeline']}")
        print(f"Components: {result['components']}")

        # Now we can use this handle for fit_predict
        print("\nFitting and predicting on airline dataset...")
        pred_result = fit_predict_tool(
            estimator_handle=result["handle"], dataset="airline", horizon=12
        )

        if pred_result["success"]:
            print(f"Predictions: {list(pred_result['predictions'].values())[:5]}...")
        else:
            print(f"Error: {pred_result.get('error')}")
    else:
        print(f"Error: {result.get('error')}")

    print()


def example_2_complex_pipeline():
    """Example 2: Create a complex pipeline with deseasonalization, detrending, and forecasting."""
    print("=" * 60)
    print("Example 2: Complex Pipeline (Deseasonalizer → Detrender → ARIMA)")
    print("=" * 60)

    # Instantiate the pipeline
    result = instantiate_pipeline_tool(
        components=["ConditionalDeseasonalizer", "Detrender", "ARIMA"]
    )

    print(f"Success: {result['success']}")
    if result["success"]:
        print(f"Handle: {result['handle']}")
        print(f"Pipeline: {result['pipeline']}")
        print(f"Components: {result['components']}")

        # Now we can use this handle for fit_predict
        print("\nFitting and predicting on airline dataset...")
        pred_result = fit_predict_tool(
            estimator_handle=result["handle"], dataset="airline", horizon=12
        )

        if pred_result["success"]:
            print(f"Predictions: {list(pred_result['predictions'].values())[:5]}...")
        else:
            print(f"Error: {pred_result.get('error')}")
    else:
        print(f"Error: {result.get('error')}")

    print()


def example_3_pipeline_with_params():
    """Example 3: Create a pipeline with custom parameters."""
    print("=" * 60)
    print("Example 3: Pipeline with Custom Parameters")
    print("=" * 60)

    # Instantiate the pipeline with custom parameters
    result = instantiate_pipeline_tool(
        components=["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
        params_list=[
            {},  # Default params for ConditionalDeseasonalizer
            {},  # Default params for Detrender
            {"order": (1, 1, 1), "suppress_warnings": True},  # Custom params for ARIMA
        ],
    )

    print(f"Success: {result['success']}")
    if result["success"]:
        print(f"Handle: {result['handle']}")
        print(f"Pipeline: {result['pipeline']}")
        print(f"Components: {result['components']}")
        print(f"Parameters: {result['params_list']}")

        # Now we can use this handle for fit_predict
        print("\nFitting and predicting on airline dataset...")
        pred_result = fit_predict_tool(
            estimator_handle=result["handle"], dataset="airline", horizon=12
        )

        if pred_result["success"]:
            print(f"Predictions: {list(pred_result['predictions'].values())[:5]}...")
        else:
            print(f"Error: {pred_result.get('error')}")
    else:
        print(f"Error: {result.get('error')}")
        if "validation_errors" in result:
            print(f"Validation errors: {result['validation_errors']}")
        if "traceback" in result:
            print(f"Traceback:\n{result['traceback']}")

    print()


def example_4_invalid_pipeline():
    """Example 4: Try to create an invalid pipeline (should fail validation)."""
    print("=" * 60)
    print("Example 4: Invalid Pipeline (ARIMA → Detrender)")
    print("=" * 60)

    # This should fail because you can't put a transformer after a forecaster
    result = instantiate_pipeline_tool(components=["ARIMA", "Detrender"])

    print(f"Success: {result['success']}")
    if not result["success"]:
        print(f"Error: {result.get('error')}")
        if "validation_errors" in result:
            print(f"Validation errors: {result['validation_errors']}")
        if "suggestions" in result:
            print(f"Suggestions: {result['suggestions']}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INSTANTIATE_PIPELINE TOOL EXAMPLES")
    print("=" * 60 + "\n")

    example_1_simple_pipeline()
    example_2_complex_pipeline()
    example_3_pipeline_with_params()
    example_4_invalid_pipeline()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
