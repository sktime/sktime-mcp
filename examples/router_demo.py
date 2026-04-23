from sktime_mcp.coordination.router import MCPToolRouter

router = MCPToolRouter()

series = [112,118,132,129,121,135,148,148,136,119,104,118]

result = router.route(
    "get_timeseries_diagnostics",
    series
)

print("Diagnostics:", result)
print("Stored memory:", router.memory.get("last_diagnostics"))