import sys

sys.path.insert(0, "src")
from sktime_mcp.tools.evaluate import evaluate_estimator_tool
from sktime_mcp.runtime.executor import get_executor
from sktime.forecasting.naive import NaiveForecaster

executor = get_executor()
handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
result = evaluate_estimator_tool(handle, "airline", cv_folds=2)
print("SUCCESS:", result.get("success"))
if "error" in result:
    print("ERROR:", result["error"])
else:
    print("RESULTS LENGTH:", len(result.get("results", [])))
    print("FIRST ITEM KEYS:", list(result.get("results", [{}])[0].keys()))
