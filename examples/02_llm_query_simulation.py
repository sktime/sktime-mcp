#!/usr/bin/env python
"""
Example 2: LLM-Style Query Simulation

This example simulates how an LLM would interact with sktime-mcp
to answer user queries about time series forecasting.

It demonstrates:
1. Natural language query → MCP tool calls
2. Reasoning over estimator capabilities
3. Constraint-based model selection
4. Safe execution without code generation

Run this example:
    python examples/02_llm_query_simulation.py
"""

import sys

sys.path.insert(0, "src")

from sktime_mcp.composition.validator import get_composition_validator
from sktime_mcp.tools.describe_estimator import describe_estimator_tool
from sktime_mcp.tools.fit_predict import fit_predict_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool
from sktime_mcp.tools.list_estimators import list_estimators_tool


def print_llm_thought(thought: str):
    """Print LLM reasoning in a distinct style."""
    print(f"\n🤖 LLM Thought: {thought}")


def print_tool_call(name: str, args: dict):
    """Print a tool call."""
    import json

    print(f"\n📞 Tool Call: {name}")
    print(f"   Args: {json.dumps(args, indent=2)}")


def print_result(result: dict):
    """Print tool result."""
    import json

    # Truncate large results
    result_str = json.dumps(result, indent=2, default=str)
    if len(result_str) > 500:
        result_str = result_str[:500] + "\n   ... (truncated)"
    print(f"   Result: {result_str}")


def simulate_query_1():
    """
    Query: "I need to forecast monthly sales. The data has missing values.
    I want prediction intervals. What should I use?"
    """
    print("\n" + "=" * 70)
    print("  QUERY 1: Forecasting with Missing Data and Prediction Intervals")
    print("=" * 70)
    print('\nUser: "I need to forecast monthly sales. The data has missing values.')
    print('       I want prediction intervals. What should I use?"')

    # Step 1: LLM reasons about requirements
    print_llm_thought(
        "User needs forecasting with: 1) missing data handling, 2) prediction intervals"
    )

    # Step 2: Search for capability tags
    print_llm_thought("Let me find estimators with prediction interval capability...")

    print_tool_call(
        "list_estimators",
        {"task": "forecasting", "tags": {"capability:pred_int": True}, "limit": 10},
    )

    result = list_estimators_tool(task="forecasting", tags={"capability:pred_int": True}, limit=10)
    print_result(result)

    # Step 3: LLM picks candidates
    if result["success"] and result["estimators"]:
        candidates = [e["name"] for e in result["estimators"][:3]]
        print_llm_thought(f"Found candidates: {candidates}. Let me check the first one...")

        # Step 4: Describe the candidate
        print_tool_call("describe_estimator", {"estimator": candidates[0]})
        desc = describe_estimator_tool(candidates[0])
        print_result(desc)

        # Step 5: Generate recommendation
        print("\n🤖 LLM Response:")
        print(f"   Based on your requirements, I recommend '{candidates[0]}'.")
        if desc["success"]:
            print("   It supports prediction intervals (capability:pred_int = True)")
            print(f"   Module: {desc['module']}")


def simulate_query_2():
    """
    Query: "Compare ARIMA and Theta for my sunspot data"
    """
    print("\n" + "=" * 70)
    print("  QUERY 2: Compare Two Forecasters")
    print("=" * 70)
    print('\nUser: "Compare NaiveForecaster and ThetaForecaster for sunspot data"')

    # Step 1: LLM plans comparison
    print_llm_thought("I'll describe both estimators and run them on sunspot data")

    # Step 2: Describe first estimator
    print_tool_call("describe_estimator", {"estimator": "NaiveForecaster"})
    desc1 = describe_estimator_tool("NaiveForecaster")
    print_result(
        {"name": desc1.get("name"), "task": desc1.get("task"), "success": desc1["success"]}
    )

    # Step 3: Describe second estimator
    print_tool_call("describe_estimator", {"estimator": "ThetaForecaster"})
    desc2 = describe_estimator_tool("ThetaForecaster")
    print_result(
        {"name": desc2.get("name"), "task": desc2.get("task"), "success": desc2["success"]}
    )

    if desc1["success"] and desc2["success"]:
        # Step 4: Instantiate both
        print_llm_thought("Both are valid forecasters. Let me run them...")

        print_tool_call(
            "instantiate_estimator", {"estimator": "NaiveForecaster", "params": {"sp": 11}}
        )
        inst1 = instantiate_estimator_tool("NaiveForecaster", {"sp": 11})
        h1 = inst1.get("handle") if inst1["success"] else None

        print_tool_call("instantiate_estimator", {"estimator": "ThetaForecaster", "params": {}})
        inst2 = instantiate_estimator_tool("ThetaForecaster", {})
        h2 = inst2.get("handle") if inst2["success"] else None

        # Step 5: Run predictions
        if h1:
            print_tool_call(
                "fit_predict", {"estimator_handle": h1, "dataset": "sunspots", "horizon": 6}
            )
            pred1 = fit_predict_tool(h1, "sunspots", 6)
            print_result({"success": pred1["success"], "horizon": pred1.get("horizon")})

        if h2:
            print_tool_call(
                "fit_predict", {"estimator_handle": h2, "dataset": "sunspots", "horizon": 6}
            )
            pred2 = fit_predict_tool(h2, "sunspots", 6)
            print_result({"success": pred2["success"], "horizon": pred2.get("horizon")})

        # Step 6: Generate comparison
        print("\n🤖 LLM Response:")
        print("   Comparison of NaiveForecaster vs ThetaForecaster on Sunspots:")
        print("   - NaiveForecaster: Simple baseline, uses last season's values")
        print("   - ThetaForecaster: Decomposition-based, better for trended data")
        if h1 and pred1["success"]:
            print("   - NaiveForecaster predictions generated successfully")
        if h2 and pred2["success"]:
            print("   - ThetaForecaster predictions generated successfully")


def simulate_query_3():
    """
    Query: "Can I use ARIMA after LogTransformer?"
    """
    print("\n" + "=" * 70)
    print("  QUERY 3: Validate Pipeline Composition")
    print("=" * 70)
    print('\nUser: "Can I build a pipeline with Imputer -> Detrend -> NaiveForecaster?"')

    # Step 1: LLM uses composition validator
    print_llm_thought("Let me validate this pipeline composition...")

    validator = get_composition_validator()

    pipeline = ["Imputer", "Detrend", "NaiveForecaster"]
    print_tool_call("validate_pipeline", {"components": pipeline})
    result = validator.validate_pipeline(pipeline)
    print_result(result.to_dict())

    # Step 2: Generate response
    print("\n🤖 LLM Response:")
    if result.valid:
        print(f"   ✅ Yes! The pipeline {pipeline} is valid.")
        print("   - Imputer handles missing values (transformer)")
        print("   - Detrend removes trend (transformer)")
        print("   - NaiveForecaster generates forecasts (forecaster)")
    else:
        print("   ❌ This pipeline has issues:")
        for error in result.errors:
            print(f"      - {error}")


def simulate_query_4():
    """
    Query: "Find me estimators related to exponential smoothing"
    """
    print("\n" + "=" * 70)
    print("  QUERY 4: Semantic Search")
    print("=" * 70)
    print('\nUser: "Find me estimators related to exponential smoothing"')

    # Step 1: Search
    print_llm_thought("Searching for 'exponential' in estimator names and docs...")

    print_tool_call("list_estimators", {"query": "exponential", "limit": 5})
    result = list_estimators_tool(query="exponential", limit=5)
    print_result(result)

    # Step 2: Generate response
    print("\n🤖 LLM Response:")
    if result["success"] and result["estimators"]:
        print(f"   Found {result['count']} estimators related to exponential smoothing:")
        for est in result["estimators"]:
            print(f"   - {est['name']} ({est['task']})")
    else:
        print("   No estimators found matching 'exponential'")


def simulate_query_5():
    """
    Query: "What's the difference in capabilities between classification and forecasting?"
    """
    print("\n" + "=" * 70)
    print("  QUERY 5: Task Comparison")
    print("=" * 70)
    print('\nUser: "How many estimators are there for classification vs forecasting?"')

    print_llm_thought("Let me count estimators for each task...")

    # Count forecasters
    print_tool_call("list_estimators", {"task": "forecasting"})
    forecasters = list_estimators_tool(task="forecasting")

    # Count classifiers
    print_tool_call("list_estimators", {"task": "classification"})
    classifiers = list_estimators_tool(task="classification")

    # Count transformers
    print_tool_call("list_estimators", {"task": "transformation"})
    transformers = list_estimators_tool(task="transformation")

    print("\n🤖 LLM Response:")
    print("   sktime estimator counts by task:")
    if forecasters["success"]:
        print(f"   - Forecasting: {forecasters['total']} estimators")
    if classifiers["success"]:
        print(f"   - Classification: {classifiers['total']} estimators")
    if transformers["success"]:
        print(f"   - Transformation: {transformers['total']} estimators")


def main():
    print("\n" + "🤖" * 30)
    print("  sktime-mcp: LLM Query Simulation Demo")
    print("  Demonstrating how LLMs interact with sktime via MCP")
    print("🤖" * 30)

    # Run simulations
    simulate_query_1()  # Find forecasters with constraints
    simulate_query_2()  # Compare two models
    simulate_query_3()  # Validate pipeline
    simulate_query_4()  # Semantic search
    simulate_query_5()  # Task comparison

    print("\n" + "=" * 70)
    print("  ✅ All LLM Query Simulations Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. LLM uses MCP tools instead of generating code directly")
    print("  2. Registry provides ground truth about capabilities")
    print("  3. Tag-based filtering enables constraint satisfaction")
    print("  4. Pipeline validation prevents invalid compositions")
    print("  5. Handle-based execution is safe and reproducible")


if __name__ == "__main__":
    main()
