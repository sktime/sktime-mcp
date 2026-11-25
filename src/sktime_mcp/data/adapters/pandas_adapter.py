"""In-memory adapter for native sktime containers and JSON tables."""

import contextlib
from typing import Any

import pandas as pd

from ..base import DataSourceAdapter
from ..contracts import describe_data, map_index, materialize_inline, scientific_type, validate_data


def _safe_infer_freq(index: pd.Index) -> str | None:
    if len(index) < 3:
        return None
    with contextlib.suppress(ValueError, TypeError):
        return pd.infer_freq(index)
    return None


class PandasAdapter(DataSourceAdapter):
    """Load a native sktime container or JSON table, preserving values and row order.

    Explicit ``mtype``/``scitype`` resolve array and table semantics. Column/index
    mappings are applied only when requested. No automatic cleaning is performed.
    """

    def load(self) -> Any:
        if self.config.get("data") is None:
            raise ValueError("Config must contain 'data' key")
        data = materialize_inline(self.config["data"], self.config)
        data, notes = map_index(data, self.config)
        scitype = scientific_type(data, self.config)
        self._data = data
        self._metadata = {"source": "memory", **describe_data(data, scitype)}
        self._metadata["validation"]["warnings"].extend(notes)
        return data

    def validate(self, data: Any) -> tuple[bool, dict[str, Any]]:
        report = validate_data(data, scientific_type(data, self.config), self.config.get("mtype"))
        return report["valid"], report
