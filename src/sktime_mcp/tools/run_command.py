import subprocess
from typing import Any

# Cap on returned output so a command like `seq 1 100000` (688k chars observed)
# can't overflow the client (BUG-23).
_MAX_OUTPUT_CHARS = 20_000


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text, False
    head = _MAX_OUTPUT_CHARS // 2
    tail = _MAX_OUTPUT_CHARS - head
    omitted = len(text) - _MAX_OUTPUT_CHARS
    return (
        f"{text[:head]}\n...[{omitted} characters truncated]...\n{text[-tail:]}",
        True,
    )


def run_command_tool(command: str) -> dict[str, Any]:
    """
    Run an arbitrary CLI/bash command on the host, in the server's working
    directory, as the user that launched the server.

    Note: this executes on the host machine (not a container or sandbox) with
    the server's own permissions. Output is capped; see ``truncated``.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"

        output, truncated = _truncate(output.strip())
        response = {
            "success": result.returncode == 0,
            "output": output,
            "returncode": result.returncode,
            "truncated": truncated,
        }
        # Every other tool reports failures under "error"; do the same on a
        # non-zero exit so callers can branch on it uniformly.
        if result.returncode != 0:
            response["error"] = (
                result.stderr.strip() or f"Command exited with code {result.returncode}"
            )
        return response
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command execution timed out after 120 seconds."}
    except Exception as e:
        return {"success": False, "error": str(e)}
