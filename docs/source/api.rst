API Reference
=============

This page documents the Python API of the ``sktime-mcp`` package, for
contributors and for anyone embedding the server in another application.

If you are looking for the **MCP tools** exposed to an AI client — their names,
arguments, and return values — see :doc:`tool-reference` instead.

.. automodule:: sktime_mcp
   :no-members:

Server
------

.. automodule:: sktime_mcp.server
   :members:
   :undoc-members:
   :show-inheritance:

Tools
-----

.. automodule:: sktime_mcp.tools
   :members:
   :undoc-members:
   :show-inheritance:

Registry
--------

Each class below is documented at its canonical location. ``sktime_mcp.registry``
re-exports them for convenience.

.. automodule:: sktime_mcp.registry.interface
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sktime_mcp.registry.tag_resolver
   :members:
   :undoc-members:
   :show-inheritance:

Runtime
-------

.. automodule:: sktime_mcp.runtime.executor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sktime_mcp.runtime.handles
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sktime_mcp.runtime.jobs
   :members:
   :undoc-members:
   :show-inheritance:

Data
----

.. automodule:: sktime_mcp.data.contracts
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. automodule:: sktime_mcp.config
   :members:
   :undoc-members:
   :show-inheritance:

Preparation contracts
---------------------

.. automodule:: sktime_mcp.data.preparation
   :members:

.. automodule:: sktime_mcp.tools.prepare_data
   :members:

Python compatibility
--------------------

``load_data_source_tool`` and ``Executor.load_data_source`` remain Python aliases
for loading under the new contract. Legacy Python split/transform helpers delegate
to preparation. Their names are not advertised MCP tools. See :doc:`migration`.
