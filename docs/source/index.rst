sktime through MCP
==================

Discover components, prepare data explicitly, and run sktime workflows from an
MCP client. The server exposes **25 tools** and retains models, datasets and
background jobs behind reusable handles.

.. note::

   These development docs describe the current source checkout. The new
   ``load_data`` / ``prepare_data`` interface is not part of the published 0.1.3
   release. See :doc:`migration` before updating an existing client.

Start with :doc:`quickstart` for a first forecast, :doc:`data-guide` for your own
data, or :doc:`support-matrix` for the actual scope of each tool.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Your task
     - Where to go
   * - Connect an assistant
     - :doc:`installation` and :doc:`quickstart`
   * - Work with data
     - :doc:`data-guide`: load, inspect, prepare, save
   * - Build a workflow
     - :doc:`user-guide`: forecasting, panels and classification
   * - Check a tool contract
     - :doc:`tool-reference` and :doc:`support-matrix`
   * - Change the implementation
     - :doc:`developer/architecture`, :doc:`developer/contributing` and :doc:`api`

.. toctree::
   :maxdepth: 1
   :caption: Start here

   intro
   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 1
   :caption: Use the server

   data-guide
   user-guide
   tool-reference
   support-matrix
   migration

.. toctree::
   :maxdepth: 1
   :caption: Develop

   developer/architecture
   developer/contributing
   api

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
