"""
Agentic Iterative Evaluation Tools.
Provides fine-grained tools for LLM agents to iteratively reason about data, test models on holdouts, and commit final winners.
"""

from typing import Any
from sktime_mcp.runtime.executor import get_executor

def describe_data_tool(dataset: str | None = None, data_handle: str | None = None) -> dict[str, Any]:
    """
    Fingerprint a dataset to help the agent reason about its characteristics before choosing an estimator.
    Returns length, frequency, missingness, mean, std, and a trend slope proxy.
    
    Args:
        dataset: Name of a builtin dataset (e.g. 'airline')
        data_handle: ID of a loaded dataset (from load_data_source)
        
    Returns:
        Dictionary with statistical summary of the data.
    """
    executor = get_executor()
    try:
        return executor.describe_data(dataset=dataset, data_handle=data_handle)
    except Exception as e:
        return {"success": False, "error": str(e)}

def score_on_holdout_tool(
    estimator_handle: str,
    dataset: str | None = None,
    data_handle: str | None = None,
    holdout_size: int = 12,
    metric: str = "MeanAbsoluteError"
) -> dict[str, Any]:
    """
    Evaluate a single estimator candidate quickly on a holdout tail.
    Fits the model on the in-sample portion and scores it against the holdout.
    
    Args:
        estimator_handle: The ID of the instantiated estimator
        dataset: Builtin dataset name
        data_handle: Loaded data handle ID
        holdout_size: Number of steps to hold out at the end of the series
        metric: Performance metric to use (default: 'MeanAbsoluteError')
        
    Returns:
        Dictionary with the scalar score.
    """
    executor = get_executor()
    try:
        return executor.score_on_holdout(
            estimator_handle=estimator_handle,
            dataset=dataset,
            data_handle=data_handle,
            holdout_size=holdout_size,
            metric=metric
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

def commit_estimator_tool(
    estimator_handle: str,
    dataset: str | None = None,
    data_handle: str | None = None,
    rationale: str | None = None
) -> dict[str, Any]:
    """
    Commit an estimator as the final winner by refitting it on the entire historical dataset.
    This prepares the handle for future 'predict_only' calls.
    
    Args:
        estimator_handle: The ID of the instantiated estimator
        dataset: Builtin dataset name
        data_handle: Loaded data handle ID
        rationale: Optional text explanation from the agent for why this model was chosen
        
    Returns:
        Dictionary confirming the model was successfully fitted and committed.
    """
    executor = get_executor()
    try:
        return executor.commit_estimator(
            estimator_handle=estimator_handle,
            dataset=dataset,
            data_handle=data_handle,
            rationale=rationale
        )
    except Exception as e:
        return {"success": False, "error": str(e)}
