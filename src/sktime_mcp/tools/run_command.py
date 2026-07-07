import subprocess
from typing import Any


def run_command_tool(command: str) -> dict[str, Any]:
    """
    Run an arbitrary CLI/bash command inside the sktime-mcp container.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"

        return {
            "success": result.returncode == 0,
            "output": output.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command execution timed out after 120 seconds."}
    except Exception as e:
        return {"success": False, "error": str(e)}
