import os
import signal
import subprocess
from typing import Any

# Cap on returned output so a command like `seq 1 100000` (688k chars observed)
# can't overflow the client (BUG-23).
_MAX_OUTPUT_CHARS = 20_000

# Wall-clock budget for a command. Module-level so tests can shrink it.
_TIMEOUT_SECONDS = 120


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


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the shell and everything it spawned (they share a session, F-50)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - non-POSIX
            proc.kill()
    except ProcessLookupError:
        pass


def _format_output(stdout: str, stderr: str) -> tuple[str, bool]:
    output = stdout or ""
    if stderr:
        output += f"\nErrors:\n{stderr}"
    return _truncate(output.strip())


def run_command_tool(command: str) -> dict[str, Any]:
    """
    Run an arbitrary CLI/bash command on the host, in the server's working
    directory, as the user that launched the server.

    Note: this executes on the host machine (not a container or sandbox) with
    the server's own permissions. Output is capped; see ``truncated``.

    The command gets ``_TIMEOUT_SECONDS`` of wall-clock time. On timeout the
    whole process group is killed (so a backgrounded child such as
    ``sleep 8 &`` cannot keep the stdout pipe open forever) and whatever was
    captured so far is returned with ``timed_out: true``.
    """
    try:
        # start_new_session puts the shell and its children in their own
        # process group so a timeout can kill all of them, not just the shell.
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

    try:
        stdout, stderr = proc.communicate(timeout=_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        # communicate() keeps the partial output it already read; once the
        # group is dead the pipes close and this returns everything captured.
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
            stdout, stderr = proc.communicate()
        output, truncated = _format_output(stdout, stderr)
        return {
            "success": False,
            "error": (
                f"Command timed out after {_TIMEOUT_SECONDS} seconds; the process "
                "group was killed. Partial output is in 'output'."
            ),
            "output": output,
            "returncode": proc.returncode,
            "truncated": truncated,
            "timed_out": True,
        }
    except Exception as e:
        _kill_process_group(proc)
        return {"success": False, "error": str(e)}

    output, truncated = _format_output(stdout, stderr)
    response = {
        "success": proc.returncode == 0,
        "output": output,
        "returncode": proc.returncode,
        "truncated": truncated,
        "timed_out": False,
    }
    # Every other tool reports failures under "error"; do the same on a
    # non-zero exit so callers can branch on it uniformly.
    if proc.returncode != 0:
        response["error"] = stderr.strip() or f"Command exited with code {proc.returncode}"
    return response
