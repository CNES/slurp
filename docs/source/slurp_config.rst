===================
SLURP Configuration
===================

SLURP can be configured with YAML files that allows you to set various parameters.

The main configuration file is given to the SLURP's API or CLI through the `main_config` argument
Below is an example of a main configuration file:

.. include:: main_config_descr.md
   :parser: myst_parser.sphinx_

The arguments of `main_config` can be completed (and/or overwritten) by a second YAML file provided to the SLURP's API or CLI in the `user_config` argument.
Below is an example of a user configuration file:

.. include:: user_config_descr.md
   :parser: myst_parser.sphinx_

Finally, some arguments defined by the YAML file can be overwritten by command line arguments or by API's arguments.
For more information, please refer to the `API's documentation` or refer to the help of your CLI command (`cli_command_name --help`).


