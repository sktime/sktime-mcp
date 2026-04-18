"""
recommend_estimators tool for sktime MCP.

Provides a lightweight recommendation layer for LLM agents by combining:
- hard constraints (task + required tags)
- soft preferences (preferred tags)
- query keyword inference
"""

from typing import Any, Optional

from sktime_mcp.registry.interface import EstimatorNode, get_registry


def _infer_constraints_from_query(query: str) -> dict[str, Any]:
    """Infer task and tag constraints from free-text query."""
    text = query.lower()

    inferred_task = None
    if any(token in text for token in ["forecast", "forecasting", "horizon"]):
        inferred_task = "forecasting"
    elif any(token in text for token in ["classify", "classification", "classifier"]):
        inferred_task = "classification"
    elif any(token in text for token in ["regression", "regressor", "predict value"]):
        inferred_task = "regression"
    elif any(token in text for token in ["transform", "preprocess", "transformation"]):
        inferred_task = "transformation"

    inferred_required_tags: dict[str, Any] = {}

    if any(token in text for token in ["prediction interval", "interval", "uncertainty"]):
        inferred_required_tags["capability:pred_int"] = True

    if any(token in text for token in ["missing values", "missing data", "na values"]):
        inferred_required_tags["handles-missing-data"] = True

    if "multivariate" in text:
        inferred_required_tags["capability:multivariate"] = True

    return {
        "task": inferred_task,
        "required_tags": inferred_required_tags,
    }


def _tokenize_query(query: str) -> list[str]:
    """Tokenize user query for lightweight lexical scoring."""
    return [token for token in query.lower().split() if len(token) >= 4]


def _score_estimator(
    node: EstimatorNode,
    preferred_tags: dict[str, Any],
    query_tokens: list[str],
) -> tuple[int, dict[str, Any]]:
    """Score estimator with transparent score breakdown."""
    score = 0

    matched_preferences = []
    for tag_name, expected_value in preferred_tags.items():
        if node.tags.get(tag_name) == expected_value:
            score += 3
            matched_preferences.append(tag_name)

    name_matches = []
    for token in query_tokens:
        if token in node.name.lower():
            score += 2
            name_matches.append(token)

    doc_matches = []
    if node.docstring:
        doc_text = node.docstring.lower()
        for token in query_tokens:
            if token in doc_text:
                score += 1
                doc_matches.append(token)

    rationale = {
        "matched_preferred_tags": sorted(set(matched_preferences)),
        "name_keyword_matches": sorted(set(name_matches)),
        "doc_keyword_matches": sorted(set(doc_matches)),
    }
    return score, rationale


def recommend_estimators_tool(
    query: Optional[str] = None,
    task: Optional[str] = None,
    required_tags: Optional[dict[str, Any]] = None,
    preferred_tags: Optional[dict[str, Any]] = None,
    limit: int = 5,
) -> dict[str, Any]:
    """
    Recommend estimators using hard constraints and soft preferences.

    Args:
        query: Optional natural-language requirement text.
        task: Optional hard task filter.
        required_tags: Optional hard capability constraints.
        preferred_tags: Optional soft capability preferences used for ranking.
        limit: Maximum number of recommendations to return.

    Returns:
        Dictionary with ranked recommendations and score explanations.
    """
    if limit <= 0:
        return {
            "success": False,
            "error": "limit must be a positive integer.",
        }

    inferred = {"task": None, "required_tags": {}}
    if query:
        inferred = _infer_constraints_from_query(query)

    effective_task = task or inferred["task"]
    effective_required_tags = dict(inferred["required_tags"])
    if required_tags:
        effective_required_tags.update(required_tags)

    effective_preferred_tags = preferred_tags or {}

    registry = get_registry()

    try:
        candidates = registry.get_all_estimators(
            task=effective_task,
            tags=effective_required_tags if effective_required_tags else None,
        )

        query_tokens = _tokenize_query(query) if query else []

        scored = []
        for node in candidates:
            score, rationale = _score_estimator(node, effective_preferred_tags, query_tokens)
            scored.append(
                {
                    "name": node.name,
                    "task": node.task,
                    "module": node.module,
                    "score": score,
                    "rationale": rationale,
                    "tags": node.tags,
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["name"]))
        recommendations = scored[:limit]

        return {
            "success": True,
            "recommendations": recommendations,
            "count": len(recommendations),
            "total_candidates": len(scored),
            "applied_filters": {
                "task": effective_task,
                "required_tags": effective_required_tags,
                "preferred_tags": effective_preferred_tags,
            },
            "inferred_from_query": inferred,
            "query": query,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
