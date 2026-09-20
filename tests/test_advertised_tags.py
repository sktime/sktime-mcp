"""Tags named in the ``query_registry`` description must be real and applicable.

That description is the only place an LLM learns which tags it may filter by.
A name sktime has since renamed is rejected outright. A name that is valid but
belongs to another scitype is worse: it passes validation and returns an empty
page with ``success: true``, which #316 identified as indistinguishable from
"no estimator matches". Neither failure reaches a test that calls
``query_registry_tool`` with known-good tags, so the description itself is
checked against the live registry.
"""

import asyncio
import re

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.server import list_tools

_TAG_PATTERN = re.compile(r"'([^']+)' \((?:bool|str)\)")


def _advertised_tags():
    """Return the tag names documented on the ``query_registry`` tool."""
    tools = {tool.name: tool for tool in asyncio.run(list_tools())}
    return _TAG_PATTERN.findall(tools["query_registry"].description)


def test_description_advertises_tags():
    """Guard the extraction, so a reworded description cannot silently pass the rest."""
    assert _advertised_tags(), "no tags parsed out of the query_registry description"


def test_advertised_tags_exist_in_the_registry():
    valid = {entry["tag"] for entry in get_registry().get_available_tags()}
    unknown = [name for name in _advertised_tags() if name not in valid]
    assert not unknown, (
        "query_registry advertises tag names that sktime does not define, so the "
        f"tool rejects its own documented filters: {unknown}"
    )


def test_advertised_tags_are_carried_by_forecasters():
    forecasters = get_registry().get_all_estimators(task="forecaster")
    absent = [
        name
        for name in _advertised_tags()
        if not any(estimator.tags.get(name) is not None for estimator in forecasters)
    ]
    assert not absent, (
        "query_registry advertises tags that no forecaster carries, so filtering by "
        f"them returns an empty page an agent cannot tell from 'no match' (#316): {absent}"
    )
