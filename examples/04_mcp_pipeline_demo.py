"""
Demonstration: Using instantiate_pipeline via MCP

This script shows how an LLM can use the instantiate_pipeline tool
to create complete pipelines in a single MCP call.
"""

import json


def simulate_llm_workflow():
    """
    Simulate how an LLM would use the MCP tools to create a pipeline.
    """

    print("=" * 70)
    print("SCENARIO: User asks 'Create a forecaster with deseasonalization")
    print("           and detrending'")
    print("=" * 70)
    print()

    # Step 1: LLM decides what components are needed
    print("Step 1: LLM determines components needed")
    print("  - ConditionalDeseasonalizer (for deseasonalization)")
    print("  - Detrender (for detrending)")
    print("  - ARIMA (for forecasting)")
    print()

    # Step 2: LLM validates the pipeline (optional but recommended)
    print("Step 2: LLM validates the composition (optional)")
    print("  MCP Call: validate_pipeline")
    print("  Input:")
    validate_request = {"components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"]}
    print(f"    {json.dumps(validate_request, indent=4)}")
    print()

    # Simulate the validation
    from sktime_mcp.composition.validator import get_composition_validator

    validator = get_composition_validator()
    validation = validator.validate_pipeline(validate_request["components"])

    print("  Output:")
    print(f"    {json.dumps(validation.to_dict(), indent=4)}")
    print()

    # Step 3: LLM creates the pipeline
    print("Step 3: LLM creates the complete pipeline")
    print("  MCP Call: instantiate_pipeline")
    print("  Input:")
    pipeline_request = {
        "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
        "params_list": [
            {},  # Default params for ConditionalDeseasonalizer
            {},  # Default params for Detrender
            {"suppress_warnings": True},  # Custom params for ARIMA
        ],
    }
    print(f"    {json.dumps(pipeline_request, indent=4)}")
    print()

    # Execute the pipeline creation
    from sktime_mcp.tools.instantiate import instantiate_pipeline_tool

    result = instantiate_pipeline_tool(**pipeline_request)

    print("  Output:")
    print(f"    {json.dumps(result, indent=4)}")
    print()

    if not result["success"]:
        print("  ❌ FAILED to create pipeline")
        return

    # Step 4: LLM uses the pipeline
    print("Step 4: LLM uses the pipeline for forecasting")
    print("  MCP Call: fit_predict")
    print("  Input:")
    fit_request = {"estimator_handle": result["handle"], "dataset": "airline", "horizon": 12}
    print(f"    {json.dumps(fit_request, indent=4)}")
    print()

    # Execute fit_predict
    from sktime_mcp.tools.fit_predict import fit_predict_tool

    pred_result = fit_predict_tool(**fit_request)

    # Show abbreviated output
    if pred_result["success"]:
        predictions = pred_result["predictions"]
        abbreviated_preds = {k: v for i, (k, v) in enumerate(predictions.items()) if i < 3}
        abbreviated_preds["..."] = f"... ({len(predictions) - 3} more predictions)"

        print("  Output:")
        print("    {")
        print('      "success": true,')
        print('      "predictions": {')
        for k, v in abbreviated_preds.items():
            if k == "...":
                print(f"        {v}")
            else:
                print(f'        "{k}": {v},')
        print("      },")
        print(f'      "horizon": {pred_result["horizon"]}')
        print("    }")
    else:
        print("  Output:")
        print(f"    {json.dumps(pred_result, indent=4)}")

    print()
    print("=" * 70)
    print("✅ SUCCESS! LLM created and used a complete pipeline in 2 MCP calls:")
    print("   1. instantiate_pipeline → got handle")
    print("   2. fit_predict → got predictions")
    print("=" * 70)


def show_mcp_json_rpc_format():
    """
    Show the exact JSON-RPC format for MCP protocol.
    """
    print()
    print("=" * 70)
    print("JSON-RPC FORMAT FOR MCP")
    print("=" * 70)
    print()

    print("1. Instantiate Pipeline:")
    print("-" * 70)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "instantiate_pipeline",
            "arguments": {
                "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
                "params_list": [{}, {}, {"suppress_warnings": True}],
            },
        },
    }
    print(json.dumps(request, indent=2))
    print()

    print("2. Fit and Predict:")
    print("-" * 70)
    request2 = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "fit_predict",
            "arguments": {
                "estimator_handle": "est_abc123def456",  # From previous response
                "dataset": "airline",
                "horizon": 12,
            },
        },
    }
    print(json.dumps(request2, indent=2))
    print()


def compare_before_after():
    """
    Show the before/after comparison.
    """
    print()
    print("=" * 70)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 70)
    print()

    print("BEFORE (Without instantiate_pipeline):")
    print("-" * 70)
    print("LLM could:")
    print("  ✅ validate_pipeline(['ConditionalDeseasonalizer', 'Detrender', 'ARIMA'])")
    print("  ✅ instantiate_estimator('ConditionalDeseasonalizer') → handle1")
    print("  ✅ instantiate_estimator('Detrender') → handle2")
    print("  ✅ instantiate_estimator('ARIMA') → handle3")
    print()
    print("LLM could NOT:")
    print("  ❌ Combine handle1, handle2, handle3 into a single pipeline")
    print("  ❌ Use the pipeline for forecasting")
    print()
    print("Result: User got 3 separate components, not a usable forecaster!")
    print()

    print("AFTER (With instantiate_pipeline):")
    print("-" * 70)
    print("LLM can:")
    print("  ✅ instantiate_pipeline(['ConditionalDeseasonalizer', 'Detrender', 'ARIMA'])")
    print("     → Returns single handle for complete pipeline")
    print("  ✅ fit_predict(handle, 'airline', horizon=12)")
    print("     → Returns predictions")
    print()
    print("Result: User gets a complete, working forecaster! ✨")
    print()


if __name__ == "__main__":
    simulate_llm_workflow()
    show_mcp_json_rpc_format()
    compare_before_after()
