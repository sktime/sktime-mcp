"""
sktime-mcp: MCP (Model Context Protocol) layer for sktime.

A semantic engine that exposes sktime's native registry and semantics to LLMs,
enabling discovery, reasoning, composition, and execution of time series workflows.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("sktime-mcp")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__author__ = "sktime-mcp contributors"

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
    "Executor",
    "HandleManager",
    "__version__",
]
