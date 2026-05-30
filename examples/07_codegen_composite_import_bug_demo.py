"""
Demo for codegen bug: missing step-class imports in composite estimators.

This script creates a ForecastingPipeline-like export payload where step
estimators appear inside params["steps"], then validates that generated code
contains imports for both outer and nested classes.
"""

from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.difference import Differencer

from sktime_mcp.tools.codegen import _generate_single_estimator_code


def run_demo() -> None:
    result = _generate_single_estimator_code(
        "ForecastingPipeline",
        {
            "steps": [
                ("differencer", Differencer()),
                ("forecaster", NaiveForecaster()),
            ]
        },
        var_name="pipeline",
    )

    print("success:", result["success"])
    if not result["success"]:
        print("error:", result.get("error"))
        return

    code = result["code"]
    print("\nGenerated code:\n")
    print(code)

    required_imports = [
        "import ForecastingPipeline",
        "import Differencer",
        "import NaiveForecaster",
    ]
    missing = [imp for imp in required_imports if imp not in code]

    print("\nValidation:")
    if missing:
        print("FAIL - missing imports:", missing)
    else:
        print("PASS - all expected imports are present")


if __name__ == "__main__":
    run_demo()
