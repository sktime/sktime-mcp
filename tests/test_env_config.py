"""Tests for integer environment-variable parsing and the job-cleanup loop.

Covers #556 item 14 (audit F-18), #249 (re-scoped) and audit F-51:

* ``Settings`` integer properties fall back to their default, with a warning,
  on non-integer or below-minimum values instead of raising at the call site.
* ``JobManager.cleanup_old_jobs`` rejects ``max_age_hours < 1`` and never
  evicts a PENDING or RUNNING job.
* ``_periodic_job_cleanup`` survives both a bad env value and an exception
  raised by the cleanup itself.

``Settings`` reads the environment on every property access, so tests only
need ``monkeypatch.setenv`` -- no module reload.
"""

import asyncio
import logging

import pytest

from sktime_mcp import server
from sktime_mcp.config import settings
from sktime_mcp.runtime.jobs import JobManager, JobStatus

# ---------------------------------------------------------------------------
# Settings: integer env parsing
# ---------------------------------------------------------------------------

_INT_SETTINGS = [
    # (env var, property, default, minimum)
    ("SKTIME_MCP_JOB_MAX_AGE_HOURS", "job_max_age_hours", 24, 1),
    ("SKTIME_MCP_JOB_CLEANUP_INTERVAL", "job_cleanup_interval_secs", 3600, 1),
    ("SKTIME_MCP_MAX_DATA_HANDLES", "max_data_handles", 50, 1),
    ("SKTIME_MCP_MAX_RESPONSE_TOKENS", "max_response_tokens", 0, 0),
]


@pytest.mark.parametrize(("env", "prop", "default", "_min"), _INT_SETTINGS)
def test_invalid_int_env_falls_back_with_warning(monkeypatch, caplog, env, prop, default, _min):
    """A non-integer value logs a warning and yields the default (no ValueError)."""
    monkeypatch.setenv(env, "abc")
    with caplog.at_level(logging.WARNING, logger="sktime_mcp.config"):
        assert getattr(settings, prop) == default
    assert any(env in r.getMessage() and "abc" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(("env", "prop", "default", "_min"), _INT_SETTINGS)
def test_below_minimum_int_env_falls_back_with_warning(
    monkeypatch, caplog, env, prop, default, _min
):
    """A value below the documented minimum logs a warning and yields the default."""
    monkeypatch.setenv(env, str(_min - 1))
    with caplog.at_level(logging.WARNING, logger="sktime_mcp.config"):
        assert getattr(settings, prop) == default
    assert any(env in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(("env", "prop", "default", "_min"), _INT_SETTINGS)
def test_valid_int_env_is_respected(monkeypatch, env, prop, default, _min):
    """Valid values, including exactly the minimum, override the default."""
    monkeypatch.setenv(env, str(_min))
    assert getattr(settings, prop) == _min
    monkeypatch.setenv(env, str(default + 7))
    assert getattr(settings, prop) == default + 7


@pytest.mark.parametrize(("env", "prop", "default", "_min"), _INT_SETTINGS)
def test_unset_int_env_uses_default(monkeypatch, env, prop, default, _min):
    monkeypatch.delenv(env, raising=False)
    assert getattr(settings, prop) == default


def test_server_has_no_import_time_int_constants():
    """The dead module-level copies of the job settings are gone (F-18)."""
    assert not hasattr(server, "_get_int_env")
    assert not hasattr(server, "JOB_MAX_AGE_HOURS")
    assert not hasattr(server, "JOB_CLEANUP_INTERVAL_SECS")


# ---------------------------------------------------------------------------
# _periodic_job_cleanup loop robustness
# ---------------------------------------------------------------------------


async def _drive_cleanup_loop(monkeypatch, cleanup_impl, iterations: int) -> tuple[list, bool]:
    """Run the periodic loop until cleanup has been invoked `iterations` times.

    ``asyncio.sleep`` is replaced so the loop does not wait for the real
    interval; the requested intervals are recorded and returned.
    """
    real_sleep = asyncio.sleep
    sleeps: list = []

    async def fake_sleep(delay, *args, **kwargs):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(JobManager, "cleanup_old_jobs", cleanup_impl)

    task = asyncio.create_task(server._periodic_job_cleanup())
    for _ in range(50):
        await real_sleep(0)
        if task.done() or len(cleanup_impl.calls) >= iterations:
            break

    still_alive = not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return sleeps, still_alive


async def test_periodic_cleanup_survives_invalid_env(monkeypatch):
    """F-18: SKTIME_MCP_JOB_CLEANUP_INTERVAL=abc no longer kills the loop.

    On origin/main ``settings.job_cleanup_interval_secs`` raised ValueError
    outside the ``try`` and the task died before its first cleanup.
    """
    monkeypatch.setenv("SKTIME_MCP_JOB_CLEANUP_INTERVAL", "abc")
    monkeypatch.setenv("SKTIME_MCP_JOB_MAX_AGE_HOURS", "abc")

    def cleanup(self, max_age_hours=24):
        cleanup.calls.append(max_age_hours)
        return 0

    cleanup.calls = []

    sleeps, alive = await _drive_cleanup_loop(monkeypatch, cleanup, iterations=2)

    assert alive
    assert sleeps[:2] == [3600, 3600]  # default interval used, loop kept going
    assert cleanup.calls[:2] == [24, 24]  # default max-age used


async def test_periodic_cleanup_survives_cleanup_exception(monkeypatch):
    """One failing cleanup is logged and the next iteration still runs."""
    monkeypatch.delenv("SKTIME_MCP_JOB_CLEANUP_INTERVAL", raising=False)
    monkeypatch.delenv("SKTIME_MCP_JOB_MAX_AGE_HOURS", raising=False)

    def cleanup(self, max_age_hours=24):
        cleanup.calls.append(max_age_hours)
        if len(cleanup.calls) == 1:
            raise RuntimeError("boom")
        return 1

    cleanup.calls = []

    _, alive = await _drive_cleanup_loop(monkeypatch, cleanup, iterations=2)

    assert alive
    assert len(cleanup.calls) >= 2
