===================
SLURP Configuration
===================

SLURP can be configured with JSON files that allows you to set various parameters.

The main configuration file is given to the SLURP's API or CLI through the `main_config` argument.

.. tip::

    Some arguments defined by the JSON file can be overwritten by command line arguments or by API's arguments.
    For more information, please refer to `API's documentation <usage_api.html>`_ or refer to the help of your CLI command (`cli_command_name --help`).

Below is an example of a configuration file:

.. include:: main_config_descr.md
   :parser: myst_parser.sphinx_


