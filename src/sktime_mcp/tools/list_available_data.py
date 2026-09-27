from typing import Any

from sktime_mcp.runtime.executor import get_executor


def list_available_data_tool(is_demo: bool | None = None) -> dict[str, Any]:
    """
    List all data available for use — system demo datasets and active data handles.

    Aggregates the output of list_datasets and list_data_handles into a single
    unified response, with clear labelling for each category.

    Args:
        is_demo: Optional boolean filter.
            - True  -> return only system demo datasets (no active_handles key)
            - False -> return only active (user-loaded) data handles (no system_demos key)
            - None  -> return both (default)

    Returns:
        Dictionary with:
        - success: bool
        - system_demos: dict of demo dataset task -> list of names (omitted if is_demo=False)
        - active_handles: list of dicts with handle id and metadata (omitted if is_demo=True)
        - total: int — combined count of items returned
    """
    executor = get_executor()

    result: dict[str, Any] = {"success": True}
    total = 0

    if is_demo is None or is_demo is True:
        system_demos_raw = executor.list_datasets()
        # Categorize system demo datasets
        demos_dict = {
            "forecasting": [],
            "classification": [],
            "regression": [],
        }
        classification_names = {"arrow_head", "gunpoint", "basic_motions", "italy_power_demand"}
        regression_names = {"covid_3month", "cardano_sentiment"}

        for name in system_demos_raw:
            if name in classification_names:
                demos_dict["classification"].append(name)
            elif name in regression_names:
                demos_dict["regression"].append(name)
            else:
                demos_dict["forecasting"].append(name)

        result["system_demos"] = demos_dict
        total += len(system_demos_raw)

    if is_demo is None or is_demo is False:
        handles_result = executor.list_data_handles()
        active_handles = handles_result.get("handles", [])
        result["active_handles"] = active_handles
        total += len(active_handles)

    result["total"] = total
    return result
