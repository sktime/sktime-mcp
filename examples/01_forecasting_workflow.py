#!/usr/bin/env python
"""
Example 1: Complete Forecasting Workflow

This example demonstrates all capabilities of the sktime-mcp for individual models:
1. Discovery - Finding estimators by task and tags
2. Description - Understanding estimator capabilities
3. Instantiation - Creating estimator instances
4. Execution - Fitting and predicting
5. Pipeline Validation - Checking composition validity

Run this example:
    python examples/01_forecasting_workflow.py
"""

import sys

sys.path.insert(0, "src")

from sktime_mcp.composition.validator import get_composition_validator
from sktime_mcp.tools.describe_estimator import describe_estimator_tool
from sktime_mcp.tools.fit_predict import fit_predict_tool, list_datasets_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool, list_handles_tool
from sktime_mcp.tools.list_estimators import get_available_tags, list_estimators_tool


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print("\n🚀 sktime-mcp: Complete Forecasting Workflow Demo")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Discovery - List Available Datasets
    # =========================================================================
    print_section("STEP 1: Discover Available Datasets")

    datasets = list_datasets_tool()
    print(f"Available datasets: {datasets['datasets']}")

    # =========================================================================
    # STEP 2: Discovery - Find Forecasting Estimators
    # =========================================================================
    print_section("STEP 2: Discover Forecasting Estimators")

    # List all forecasters
    result = list_estimators_tool(task="forecasting", limit=10)

    if result["success"]:
        print(f"Found {result['total']} forecasting estimators")
        print("\nFirst 10 forecasters:")
        for est in result["estimators"]:
            print(f"  - {est['name']}")
    else:
        print(f"Error: {result.get('error')}")

    # =========================================================================
    # STEP 3: Discovery with Tag Filtering
    # =========================================================================
    print_section("STEP 3: Find Probabilistic Forecasters")

    # Find forecasters that can produce prediction intervals
    result = list_estimators_tool(
        task="forecasting",
        tags={"capability:pred_int": True},
        limit=10,
    )

    if result["success"]:
        print(f"Found {result['total']} probabilistic forecasters")
        print("\nExamples:")
        for est in result["estimators"][:5]:
            print(f"  - {est['name']}")

    # =========================================================================
    # STEP 4: Describe an Estimator
    # =========================================================================
    print_section("STEP 4: Describe NaiveForecaster")

    desc = describe_estimator_tool("NaiveForecaster")

    if desc["success"]:
        print(f"Name: {desc['name']}")
        print(f"Task: {desc['task']}")
        print(f"Module: {desc['module']}")
        print("\nHyperparameters:")
        for param, info in list(desc["hyperparameters"].items())[:5]:
            default = info.get("default", "N/A")
            print(f"  - {param}: default={default}")
        print("\nKey Tags:")
        for tag, value in list(desc["tags"].items())[:5]:
            print(f"  - {tag}: {value}")

    # =========================================================================
    # STEP 5: Validate a Pipeline
    # =========================================================================
    print_section("STEP 5: Validate Pipeline Compositions")

    validator = get_composition_validator()

    # Valid pipeline: Transformer -> Forecaster
    print("\n✅ Testing: ['Imputer', 'NaiveForecaster']")
    result = validator.validate_pipeline(["Imputer", "NaiveForecaster"])
    print(f"   Valid: {result.valid}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")

    # Invalid pipeline: Forecaster -> Forecaster
    print("\n❌ Testing: ['NaiveForecaster', 'ExponentialSmoothing']")
    result = validator.validate_pipeline(["NaiveForecaster", "ExponentialSmoothing"])
    print(f"   Valid: {result.valid}")
    if result.errors:
        print(f"   Errors: {result.errors}")

    # =========================================================================
    # STEP 6: Instantiate an Estimator
    # =========================================================================
    print_section("STEP 6: Instantiate NaiveForecaster")

    inst_result = instantiate_estimator_tool(
        "NaiveForecaster", params={"strategy": "last", "sp": 12}
    )

    if inst_result["success"]:
        handle = inst_result["handle"]
        print(f"Created estimator with handle: {handle}")
        print(f"Parameters: {inst_result['params']}")
    else:
        print(f"Error: {inst_result.get('error')}")
        return

    # =========================================================================
    # STEP 7: Fit and Predict
    # =========================================================================
    print_section("STEP 7: Fit and Predict on Airline Dataset")

    pred_result = fit_predict_tool(
        estimator_handle=handle,
        dataset="airline",
        horizon=12,
    )

    if pred_result["success"]:
        print(f"Forecast horizon: {pred_result['horizon']} periods")
        print("\nPredictions:")
        predictions = pred_result["predictions"]
        if isinstance(predictions, dict):
            for step, value in list(predictions.items())[:6]:
                print(f"  Step {step}: {value:.2f}")
            if len(predictions) > 6:
                print(f"  ... ({len(predictions) - 6} more steps)")
    else:
        print(f"Error: {pred_result.get('error')}")

    # =========================================================================
    # STEP 8: List Active Handles
    # =========================================================================
    print_section("STEP 8: List Active Handles")

    handles = list_handles_tool()
    print(f"Active handles: {handles['count']}")
    for h in handles["handles"]:
        print(f"  - {h['handle_id']}: {h['estimator_name']} (fitted: {h['fitted']})")

    # =========================================================================
    # STEP 9: Show Available Tags
    # =========================================================================
    print_section("STEP 9: Available Capability Tags")

    tags_result = get_available_tags()
    if tags_result["success"]:
        print(f"Total tags available: {len(tags_result['tags'])}")
        print("\nSample tags:")
        for tag in tags_result["tags"][:15]:
            print(f"  - {tag}")

    print("\n" + "=" * 60)
    print("  ✅ Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
