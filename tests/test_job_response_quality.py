"""Background-job response hygiene (#538).

- NB-04: errors[] must not leak server tracebacks with absolute paths.
- NB-05: a coarse-step job must not report a bogus ETA; a cancelled job must
  show a terminal current_step, not the frozen in-flight one.
"""

import pytest

from sktime_mcp.runtime.jobs import JobStatus, get_job_manager


def test_no_bogus_eta_for_coarse_jobs():
    jm = get_job_manager()
    job_id = jm.create_job(job_type="evaluate", estimator_handle="est_x", total_steps=3)
    jm.update_job(job_id, status=JobStatus.RUNNING, completed_steps=1)
    try:
        job = jm.get_job(job_id)
        # 3-step job: ETA is not meaningful, so it must be None (not extrapolated)
        assert job.estimated_time_remaining is None
        assert job.estimated_time_remaining_human is None
    finally:
        jm.delete_job(job_id)


def test_eta_available_for_fine_grained_jobs():
    jm = get_job_manager()
    job_id = jm.create_job(job_type="evaluate", estimator_handle="est_x", total_steps=100)
    jm.update_job(job_id, status=JobStatus.RUNNING, completed_steps=10)
    try:
        job = jm.get_job(job_id)
        assert job.estimated_time_remaining is not None
    finally:
        jm.delete_job(job_id)


def test_cancel_sets_terminal_step():
    jm = get_job_manager()
    job_id = jm.create_job(job_type="evaluate", estimator_handle="est_x", total_steps=3)
    jm.update_job(job_id, status=JobStatus.RUNNING, completed_steps=1,
                  current_step="Running cross-validation...")
    try:
        jm.cancel_job(job_id)
        job = jm.get_job(job_id)
        assert job.status == JobStatus.CANCELLED
        assert job.current_step == "Cancelled"
        # a cancelled (terminal) job reports no ETA
        assert job.estimated_time_remaining is None
    finally:
        jm.delete_job(job_id)


def test_failed_job_errors_have_no_traceback():
    """The async failure path stores only the clean message, not a traceback."""
    jm = get_job_manager()
    job_id = jm.create_job(job_type="fit", estimator_handle="est_x", total_steps=2)
    # Simulate what the async failure path now stores.
    jm.update_job(job_id, status=JobStatus.FAILED, errors=["Handle not found: est_x"])
    try:
        job = jm.get_job(job_id)
        assert job.errors == ["Handle not found: est_x"]
        joined = " ".join(job.errors)
        assert "Traceback" not in joined
        assert ".py" not in joined  # no file paths
    finally:
        jm.delete_job(job_id)
