#!/usr/bin/env python
"""
Example: External Agent Integration Workflow

This script demonstrates how an external LLM agent (or its backend)
should programmatically interact with the `sktime_mcp` server tools
when using the raw module API.

It covers the complete workflow:
1. Loading custom data
2. Reading returned metadata
3. Instantiating a forecaster
4. Evaluating the forecaster on the loaded data

Run this example:
    python examples/external_agent_integration.py
"""

import sys
import json
import tempfile
from pathlib import Path
import pandas as pd

sys.path.insert(0, "src")

from sktime_mcp.tools.data_tools import load_data_source_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool
from sktime_mcp.tools.evaluate import evaluate_estimator_tool

def print_step(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print("\n🤖 External Agent Integration Workflow Demo")
    
    # Create some dummy external data
    db_path = Path(tempfile.gettempdir()) / "agent_dummy_data.csv"
    sample_data = pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=100, freq="D"),
        "sales": [100 + i + (i % 7) * 5 for i in range(100)]
    })
    sample_data.to_csv(db_path, index=False)
    
    # ---------------------------------------------------------
    # STEP 1: Load Custom Data
    # ---------------------------------------------------------
    print_step("STEP 1: Load Custom Data")
    print(f"Loading data from {db_path}...")
    
    data_config = {
        "type": "file",
        "path": str(db_path),
        "time_column": "date",
        "target_column": "sales"
    }
    
    load_res = load_data_source_tool(data_config)
    if not load_res["success"]:
        print("Failed to load data:", load_res)
        return
        
    data_handle = load_res["data_handle"]
    print("✅ Data loaded successfully.")
    print(f"Handle assigned: {data_handle}")
    print("Metadata:", json.dumps(load_res["metadata"], indent=2))
    
    # ---------------------------------------------------------
    # STEP 2: Instantiate Forecaster
    # ---------------------------------------------------------
    print_step("STEP 2: Instantiate Forecaster")
    
    print("Instantiating NaiveForecaster...")
    inst_res = instantiate_estimator_tool("NaiveForecaster", params={"strategy": "last"})
    
    if not inst_res["success"]:
        print("Failed to instantiate:", inst_res)
        return
        
    estimator_handle = inst_res["handle"]
    print("✅ Forecaster instantiated successfully.")
    print(f"Handle assigned: {estimator_handle}")
    
    # ---------------------------------------------------------
    # STEP 3: Evaluate Estimator
    # ---------------------------------------------------------
    print_step("STEP 3: Evaluate Estimator on Custom Data")
    
    print("Running 3-fold cross-validation...")
    eval_res = evaluate_estimator_tool(
        estimator_handle=estimator_handle,
        data_handle=data_handle,
        cv_folds=3
    )
    
    if not eval_res["success"]:
        print("Failed to evaluate:", eval_res)
        return
        
    print("✅ Evaluation completed successfully.")
    print("\nMetrics:")
    for metric, value in eval_res["metrics"].items():
        print(f"  - {metric}: {value:.4f}")
        
    print("\nWorkflow completed successfully!")

if __name__ == "__main__":
    main()
