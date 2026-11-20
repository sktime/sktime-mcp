"""Job status/list responses must cap embedded fold_results.

A completed evaluate job can carry hundreds of per-fold rows; embedding
them all in check_job_status (and once per job in list_jobs) floods the
MCP client. Aggregate metrics/summary stay intact.
"""

import pytest

from sktime_mcp.runtime.jobs import JobStatus, get_job_manager
from sktime_mcp.tools.job_tools import check_job_status_tool, list_jobs_tool


@pytest.fixture
def evaluate_job_with_120_folds():
    job_manager = get_job_manager()
    job_id = job_manager.create_job(
        job_type="evaluate",
        estimator_handle="est_test",
        estimator_name="ThetaForecaster",
        dataset_name="airline",
        total_steps=3,
    )
    result = {
        "success": True,
        "metrics": {"test_MeanAbsolutePercentageError": 0.03},
        "summary": {"test_MeanAbsolutePercentageError": {"mean": 0.03}},
        "fold_results": [
            {"test_MeanAbsolutePercentageError": 0.01 * i, "len_train_window": 24 + i}
            for i in range(120)
        ],
        "cv_folds_run": 120,
    }
    job_manager.update_job(job_id, status=JobStatus.COMPLETED, result=result)
    yield job_id
    job_manager.delete_job(job_id)


def test_check_job_status_caps_fold_results(evaluate_job_with_120_folds):
    res = check_job_status_tool(evaluate_job_with_120_folds)
    assert res["success"]
    assert len(res["result"]["fold_results"]) == 10
    assert res["result"]["fold_results_truncated"]["total"] == 120
    # aggregates untouched
    assert res["result"]["metrics"]["test_MeanAbsolutePercentageError"] == 0.03
    assert res["result"]["cv_folds_run"] == 120


def test_list_jobs_drops_fold_results(evaluate_job_with_120_folds):
    res = list_jobs_tool(status="completed")
    assert res["success"]
    job = next(j for j in res["jobs"] if j["job_id"] == evaluate_job_with_120_folds)
    assert job["result"]["fold_results"] == []
    assert job["result"]["fold_results_truncated"]["total"] == 120
    assert job["result"]["metrics"]["test_MeanAbsolutePercentageError"] == 0.03


def test_stored_job_result_not_mutated(evaluate_job_with_120_folds):
    check_job_status_tool(evaluate_job_with_120_folds)
    stored = get_job_manager().get_job(evaluate_job_with_120_folds)
    assert len(stored.result["fold_results"]) == 120


def test_small_results_pass_through_unchanged():
    job_manager = get_job_manager()
    job_id = job_manager.create_job(
        job_type="predict",
        estimator_handle="est_test",
        estimator_name="NaiveForecaster",
        total_steps=2,
    )
    job_manager.update_job(
        job_id,
        status=JobStatus.COMPLETED,
        result={"success": True, "predictions": {"1961-01": 417.0}},
    )
    try:
        res = check_job_status_tool(job_id)
        assert res["result"] == {"success": True, "predictions": {"1961-01": 417.0}}
        assert "fold_results_truncated" not in res["result"]
    finally:
        job_manager.delete_job(job_id)
