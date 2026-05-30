#!/usr/bin/env python
"""
Demo: reproduce missing forecast horizon validation in fit_predict_tool.

This script shows what happens today when invalid horizon values are passed
through MCP tooling before sktime is called.

Run:
    python examples/07_fh_validation_bug_demo.py
"""

import sys

sys.path.insert(0, "src")

from sktime_mcp.tools.fit_predict import fit_predict_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool


def run_case(handle: str, horizon: int) -> None:
    print("\n" + "-" * 72)
    print(f"Case: fit_predict_tool(..., horizon={horizon})")
    result = fit_predict_tool(
        estimator_handle=handle,
        dataset="airline",
        horizon=horizon,
    )
    print(f"success: {result.get('success')}")

    if result.get("success"):
        print("Unexpected success. This indicates invalid horizon was accepted.")
        print(f"horizon in response: {result.get('horizon')}")
        return

    # This error currently comes from deeper stack behavior rather than
    # early, explicit input validation at tool boundary.
    print("error returned:")
    print(f"  {result.get('error')}")


def main() -> None:
    print("=" * 72)
    print("BUG DEMO: fh/horizon not validated before sktime call")
    print("=" * 72)

    inst = instantiate_estimator_tool(
        "NaiveForecaster",
        params={"strategy": "last", "sp": 12},
    )

    if not inst.get("success"):
        print("Failed to instantiate estimator:")
        print(inst)
        return

    handle = inst["handle"]
    print(f"Estimator handle: {handle}")

    # Invalid horizons that should be rejected with a clear tool-level message.
    run_case(handle, 0)
    run_case(handle, -3)

    print("\n" + "=" * 72)
    print("Expected fix behavior:")
    print("  Invalid fh=... . fh must be a positive integer.")
    print("=" * 72)


if __name__ == "__main__":
    main()
