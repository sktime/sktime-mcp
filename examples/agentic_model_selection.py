#!/usr/bin/env python
"""
Example: Agentic Model Selection Workflow

This example demonstrates how an LLM agent uses sktime-mcp to perform
a complete model selection loop without human intervention.
The agent performs these steps:
1. Instantiates candidate models.
2. Evaluates the candidates on a dataset using cross-validation.
3. Compares their test errors (Mean Absolute Error).
4. Picks the winner and generates Python code for it.

Run this example:
    python examples/agentic_model_selection.py
"""

import sys
sys.path.insert(0, "src")

from sktime_mcp.tools.instantiate import instantiate_estimator_tool
from sktime_mcp.tools.evaluate import evaluate_estimator_tool
from sktime_mcp.tools.codegen import export_code_tool

def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print("\n🤖 sktime-mcp: Agentic Model Selection Workflow Demo")
    print("=" * 60)

    dataset_name = "airline"
    candidate_configs = [
        {"name": "NaiveForecaster", "params": {"strategy": "last"}},
        {"name": "ExponentialSmoothing", "params": {"trend": "add", "seasonal": "mul", "sp": 12}}
    ]

    print(f"Goal: Find the best forecaster for the '{dataset_name}' dataset.")

    # =========================================================================
    # STEP 1: Instantiate Candidates
    # =========================================================================
    print_section("STEP 1: Instantiate Candidates")
    
    handles = {}
    for config in candidate_configs:
        print(f"Instantiating {config['name']}...")
        result = instantiate_estimator_tool(config['name'], params=config['params'])
        if result["success"]:
            handles[config['name']] = result["handle"]
            print(f"  ✅ Created handle: {result['handle']}")
        else:
            print(f"  ❌ Failed: {result.get('error')}")

    # =========================================================================
    # STEP 2: Evaluate Candidates
    # =========================================================================
    print_section("STEP 2: Evaluate Candidates")
    
    evaluation_results = {}
    
    for name, handle in handles.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_estimator_tool(
            estimator_handle=handle,
            dataset=dataset_name,
            cv_folds=3, # 3-fold cross validation
        )
        if result["success"]:
            # Extract Mean Absolute Error (MAE) for comparison
            metrics = result["metrics"]
            mae = metrics.get("MeanAbsoluteError", float('inf'))
            evaluation_results[name] = mae
            print(f"  ✅ Evaluation successful. MAE: {mae:.2f}")
        else:
            print(f"  ❌ Failed: {result.get('error')}")

    if not evaluation_results:
        print("\nNo models were successfully evaluated. Exiting.")
        return

    # =========================================================================
    # STEP 3: Select the Winner
    # =========================================================================
    print_section("STEP 3: Select the Winner")
    
    winner_name = min(evaluation_results, key=evaluation_results.get)
    winner_mae = evaluation_results[winner_name]
    winner_handle = handles[winner_name]
    
    print(f"🏆 Winner: {winner_name}")
    print(f"   Lowest MAE: {winner_mae:.2f}")

    # =========================================================================
    # STEP 4: Export Code for the Winner
    # =========================================================================
    print_section("STEP 4: Export Reproducible Code")
    
    print(f"Exporting Python code for the winning model ({winner_name})...\n")
    
    code_result = export_code_tool(
        handle=winner_handle,
        var_name="best_model",
        include_fit_example=True,
        dataset=dataset_name
    )
    
    if code_result["success"]:
        print(code_result["code"])
    else:
        print(f"  ❌ Failed to export code: {code_result.get('error')}")

    print("\n" + "=" * 60)
    print("  ✅ Agentic Workflow Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
