from sktime_mcp.tools import get_timeseries_diagnostics

series = [112,118,132,129,121,135,148,148,136,119,104,118]

print(get_timeseries_diagnostics(series))