#!/usr/bin/env python
"""
Example 7: Agentic Model Selection Workflow

This example demonstrates a simple LLM-style selection loop using the current
sktime-mcp tool surface:
1. Discover forecasting estimators
2. Inspect candidate capabilities
3. Instantiate and evaluate candidates
4. Select the best-performing model
5. Generate a forecast
6. Export reproducible Python code

Run this example:
    python examples/07_agentic_model_selection.py
"""

from __future__ import annotations

import statistics
import sys
from typing import Any

sys.path.insert(0, "src")

from sktime_mcp.tools.codegen import export_code_tool
from sktime_mcp.tools.describe_estimator import describe_estimator_tool
from sktime_mcp.tools.evaluate import evaluate_estimator_tool
from sktime_mcp.tools.fit_predict import fit_predict_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool
from sktime_mcp.tools.list_estimators import list_estimators_tool

PREFERRED_CANDIDATES: dict[str, dict[str, Any]] = {
    "NaiveForecaster": {"strategy": "last", "sp": 12},
    "ThetaForecaster": {},
    "ExponentialSmoothing": {"trend": "add", "seasonal": "add", "sp": 12},
}


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def summarize_metric(result: dict[str, Any], metric_key: str) -> float:
    """Return the mean value of a fold metric from evaluate_estimator output."""
    scores = [fold[metric_key] for fold in result["results"] if metric_key in fold]
    return statistics.fmean(scores)


def main() -> None:
    """Run the agentic workflow example."""
    print("\n🤖 sktime-mcp: Agentic Model-Selection Workflow Demo")
    print("=" * 72)
    print("Scenario: an LLM needs to pick a forecasting model for the airline dataset.")

    print_section("STEP 1: Discover forecasting estimators")
    discovery = list_estimators_tool(task="forecasting", limit=200)
    if not discovery["success"]:
        print(f"Discovery failed: {discovery.get('error')}")
        return

    available_names = {est["name"] for est in discovery["estimators"]}
    candidate_names = [name for name in PREFERRED_CANDIDATES if name in available_names]
    print(f"Discovered {discovery['total']} forecasting estimators.")
    print(f"Shortlisted candidates: {candidate_names}")

    print_section("STEP 2: Inspect candidate capabilities")
    for name in candidate_names:
        details = describe_estimator_tool(name)
        if not details["success"]:
            print(f"- {name}: description unavailable")
            continue
        pred_int = details["tags"].get("capability:pred_int")
        print(f"- {name}: module={details['module']}, pred_int={pred_int}")

    print_section("STEP 3: Instantiate and evaluate candidates")
    leaderboard: list[dict[str, Any]] = []

    for name in candidate_names:
        params = PREFERRED_CANDIDATES[name]
        instantiation = instantiate_estimator_tool(name, params)
        if not instantiation["success"]:
            print(f"- {name}: instantiation failed -> {instantiation.get('error')}")
            continue

        handle = instantiation["handle"]
        evaluation = evaluate_estimator_tool(handle, "airline", 3)
        if not evaluation["success"]:
            print(f"- {name}: evaluation failed -> {evaluation.get('error')}")
            continue

        metric_key = "test_MeanAbsolutePercentageError"
        mean_mape = summarize_metric(evaluation, metric_key)
        leaderboard.append(
            {
                "name": name,
                "handle": handle,
                "params": params,
                "mean_mape": mean_mape,
                "cv_folds": evaluation["cv_folds_run"],
            }
        )
        print(f"- {name}: mean MAPE={mean_mape:.4f} across {evaluation['cv_folds_run']} folds")

    if not leaderboard:
        print("No candidate evaluations succeeded.")
        return

    leaderboard.sort(key=lambda item: item["mean_mape"])

    print_section("STEP 4: Rank candidates and select a winner")
    print("Leaderboard (lower mean MAPE is better):")
    for idx, row in enumerate(leaderboard, start=1):
        print(f"{idx}. {row['name']}: {row['mean_mape']:.4f}")

    winner = leaderboard[0]
    print(f"\nSelected winner: {winner['name']}")
    print(f"Chosen params: {winner['params']}")

    print_section("STEP 5: Generate forecast with the winner")
    forecast = fit_predict_tool(
        estimator_handle=winner["handle"],
        dataset="airline",
        horizon=12,
    )
    if not forecast["success"]:
        print(f"Forecast failed: {forecast.get('error')}")
        return

    print(f"Forecast generated for {forecast['horizon']} steps.")
    for step, value in list(forecast["predictions"].items())[:5]:
        print(f"- {step}: {value:.2f}")

    print_section("STEP 6: Export reproducible code")
    exported = export_code_tool(
        winner["handle"],
        var_name="selected_model",
        include_fit_example=True,
        dataset="airline",
    )
    if not exported["success"]:
        print(f"Code export failed: {exported.get('error')}")
        return

    code_preview = "\n".join(exported["code"].splitlines()[:12])
    print("Exported code preview:")
    print(code_preview)

    print("\n" + "=" * 72)
    print("✅ Agentic workflow complete!")
    print("=" * 72)
    print("The example used only the existing MCP tool surface to choose a model,")
    print("evaluate it, forecast, and export reproducible code.")


if __name__ == "__main__":
    main()
