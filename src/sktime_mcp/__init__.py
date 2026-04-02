"""
sktime-mcp: MCP (Model Context Protocol) layer for sktime.

A semantic engine that exposes sktime's native registry and semantics to LLMs,
enabling discovery, reasoning, composition, and execution of time series workflows.
"""

__version__ = "0.1.0"
__author__ = "sktime-mcp contributors"

from sktime_mcp.composition.validator import CompositionValidator
from sktime_mcp.registry.interface import (
    EstimatorNode,
    RegistryInterface,
)
from sktime_mcp.registry.tag_resolver import TagResolver
from sktime_mcp.runtime.executor import Executor
from sktime_mcp.runtime.handles import HandleManager

__all__ = [
    "EstimatorNode",
    "RegistryInterface",
    "TagResolver",
    "CompositionValidator",
    "Executor",
    "HandleManager",
    "__version__",
]
