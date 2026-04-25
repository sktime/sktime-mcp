from sktime.datasets import load_airline
from sktime_mcp.tools import get_timeseries_diagnostics

# load example dataset
y = load_airline()

diagnostics = get_timeseries_diagnostics(y)

print("Time Series Diagnostics")
print(diagnostics)