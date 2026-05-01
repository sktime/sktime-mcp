#!/usr/bin/env python
"""
Example 8: External Agent Integration Workflow

This example shows a minimal pattern for integrating sktime-mcp tool
functions inside a custom agent or orchestration layer.

It demonstrates:
1. Loading user-provided data and reading agent-friendly metadata
2. Fitting and predicting from a custom data handle
3. Calling evaluate_estimator with the currently supported inputs

Run this example:
    python examples/08_external_agent_integration.py
"""

import sys

import pandas as pd

sys.path.insert(0, "src")

from sktime_mcp.tools.data_tools import load_data_source_tool, release_data_handle_tool
from sktime_mcp.tools.evaluate import evaluate_estimator_tool
from sktime_mcp.tools.fit_predict import fit_predict_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool


def print_step(title: str):
    """Print a visible step heading."""
    print(f"\n== {title} ==")


def main():
    """Run a simple external-agent style workflow."""
    print("External Agent Integration Workflow")
    print("Use editable install first: pip install -e .")
    print("Fallback during source-only integration: PYTHONPATH=/path/to/sktime-mcp/src")

    print_step("1. Load custom data and inspect metadata")
    load_result = load_data_source_tool(
        {
            "type": "pandas",
            "data": {
                "date": pd.date_range(start="2021-01-01", periods=24, freq="D"),
                "sales": [100 + i for i in range(24)],
                "promo": [0, 1] * 12,
            },
            "time_column": "date",
            "target_column": "sales",
            "exog_columns": ["promo"],
        }
    )
    assert load_result["success"], load_result
    data_handle = load_result["data_handle"]
    metadata = load_result["metadata"]
    print(f"Loaded handle: {data_handle}")
    print(
        "Metadata summary: "
        f"rows={metadata.get('rows')}, "
        f"frequency={metadata.get('frequency')}, "
        f"columns={metadata.get('columns')}, "
        f"exog_columns={metadata.get('exog_columns', [])}"
    )

    print_step("2. Fit and predict from a custom data handle")
    inst_result = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
    assert inst_result["success"], inst_result
    fit_result = fit_predict_tool(
        inst_result["handle"],
        dataset="",
        horizon=3,
        data_handle=data_handle,
    )
    assert fit_result["success"], fit_result
    print(f"Custom-data forecast horizon: {fit_result['horizon']}")

    print_step("3. Evaluate on a demo dataset using current MCP contract")
    eval_result = evaluate_estimator_tool(inst_result["handle"], dataset="airline", cv_folds=3)
    assert eval_result["success"], eval_result
    first_row = eval_result["results"][0]
    metric_keys = sorted(k for k in first_row if k.startswith("test_"))
    print(f"Evaluation folds requested: {eval_result['cv_folds_requested']}")
    print(f"Returned metric keys: {', '.join(metric_keys[:3])}")

    cleanup = release_data_handle_tool(data_handle)
    assert cleanup["success"], cleanup
    print("\nExternal agent workflow complete!")


if __name__ == "__main__":
    main()
