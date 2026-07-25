sktime-mcp
==========

.. image:: https://img.shields.io/pypi/v/sktime-mcp.svg
   :target: https://pypi.org/project/sktime-mcp/
   :alt: PyPI Version

.. image:: https://img.shields.io/github/license/sktime/sktime-mcp.svg
   :target: https://github.com/sktime/sktime-mcp/blob/main/LICENSE
   :alt: License

The Semantic Engine for Time-Series with Large Language Models.

`sktime-mcp` is a Model Context Protocol (MCP) server that exposes the full power of the `sktime` ecosystem to AI assistants. It provides a stateful runtime for discovering, instantiating, and executing time-series workflows — forecasting, classification, regression, transformation, detection, and splitting.

Forecasting is the most polished path today; the other scitypes are reachable through
generic instantiation and ``call_method``. See :doc:`tool-reference` for what is available.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   intro
   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide
   tool-reference

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/architecture
   developer/contributing
   api

.. toctree::
   :maxdepth: 2
   :caption: Project Links
   :hidden:

   GitHub Repository <https://github.com/sktime/sktime-mcp>
   sktime Project <https://www.sktime.net/>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
