"""
Handle Manager for sktime MCP.

Manages references to instantiated estimator objects.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class HandleInfo:
    """Information about a managed handle."""

    handle_id: str
    estimator_name: str
    instance: Any
    params: dict[str, Any]
    created_at: datetime
    fitted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle_id": self.handle_id,
            "estimator_name": self.estimator_name,
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "fitted": self.fitted,
            "metadata": self.metadata,
        }


class HandleManager:
    """Manager for estimator instance handles."""

    def __init__(self, max_handles: int = 100):
        self._handles: dict[str, HandleInfo] = {}
        self._max_handles = max_handles

    def create_handle(
        self,
        estimator_name: str,
        instance: Any,
        params: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        if len(self._handles) >= self._max_handles:
            self._cleanup_oldest()

        handle_id = f"est_{uuid.uuid4().hex[:12]}"
        handle_info = HandleInfo(
            handle_id=handle_id,
            estimator_name=estimator_name,
            instance=instance,
            params=params or {},
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        self._handles[handle_id] = handle_info
        return handle_id

    def get_instance(self, handle_id: str) -> Any:
        if handle_id not in self._handles:
            raise KeyError(f"Handle not found: {handle_id}")
        return self._handles[handle_id].instance

    def get_info(self, handle_id: str) -> HandleInfo:
        if handle_id not in self._handles:
            raise KeyError(f"Handle not found: {handle_id}")
        return self._handles[handle_id]

    def exists(self, handle_id: str) -> bool:
        return handle_id in self._handles

    def mark_fitted(self, handle_id: str) -> None:
        if handle_id in self._handles:
            self._handles[handle_id].fitted = True

    def is_fitted(self, handle_id: str) -> bool:
        if handle_id not in self._handles:
            return False
        return self._handles[handle_id].fitted

    def release_handle(self, handle_id: str) -> bool:
        if handle_id in self._handles:
            del self._handles[handle_id]
            return True
        return False

    def list_handles(self) -> list[dict[str, Any]]:
        return [info.to_dict() for info in self._handles.values()]

    def clear_all(self) -> int:
        count = len(self._handles)
        self._handles.clear()
        return count

    def _cleanup_oldest(self, count: int = 10) -> None:
        sorted_handles = sorted(
            self._handles.items(),
            key=lambda x: x[1].created_at,
        )
        for handle_id, _ in sorted_handles[:count]:
            del self._handles[handle_id]


_handle_manager_instance: Optional[HandleManager] = None


def get_handle_manager() -> HandleManager:
    global _handle_manager_instance
    if _handle_manager_instance is None:
        _handle_manager_instance = HandleManager()
    return _handle_manager_instance
