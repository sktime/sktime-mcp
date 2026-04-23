from sktime_mcp.tools import get_timeseries_diagnostics


class MCPMemory:
    """
    Shared memory for MCP workflows.
    """

    def __init__(self):
        self.memory = {}

    def store(self, key, value):
        self.memory[key] = value

    def get(self, key):
        return self.memory.get(key)

    def clear(self):
        self.memory = {}


class MCPToolRouter:
    """
    Simple router that executes tools and stores results.
    """

    def __init__(self):
        self.memory = MCPMemory()

    def route(self, tool_name, payload):

        if tool_name == "get_timeseries_diagnostics":

            result = get_timeseries_diagnostics(payload)

            self.memory.store("last_diagnostics", result)

            return result

        raise ValueError(f"Unknown tool: {tool_name}")