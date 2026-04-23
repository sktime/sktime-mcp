import time

from sktime_mcp.tools.get_agent_trace import record_trace
from sktime_mcp.tools import get_timeseries_diagnostics


class MCPMemory:
    """
    Shared memory for MCP workflows.
    """

    def __init__(self):
        self.memory = {}

    def store(self, key, value):
        """Store a value in memory."""
        self.memory[key] = value

    def get(self, key):
        """Retrieve a value from memory."""
        return self.memory.get(key)

    def clear(self):
        """Clear memory."""
        self.memory = {}


class MCPToolRouter:
    """
    Simple router that executes MCP tools and stores results.
    Also records execution traces for debugging agent workflows.
    """

    def __init__(self):
        self.memory = MCPMemory()

    def route(self, session_id: str, tool_name: str, payload):
        """
        Execute a tool and record its trace.

        Parameters
        ----------
        session_id : str
            Unique identifier for the agent session.
        tool_name : str
            Name of the tool to execute.
        payload : dict
            Input payload for the tool.

        Returns
        -------
        dict
            Tool execution result.
        """

        start_time = time.time()

        if tool_name == "get_timeseries_diagnostics":

            result = get_timeseries_diagnostics(payload)

            # store result in shared memory
            self.memory.store("last_diagnostics", result)

            # record trace
            record_trace(
                session_id=session_id,
                tool=tool_name,
                input_data=payload,
                output_data=result,
                start_time=start_time,
            )

            return result

        raise ValueError(f"Unknown tool: {tool_name}")