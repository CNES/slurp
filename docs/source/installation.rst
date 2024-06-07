.. highlight:: shell

============
Installation
============


Stable release
--------------

If deployed in Pypi, to install slurp, run this command in your terminal:

.. code-block:: console

    $ pip install slurp

This is the preferred method to install slurp, as it will always install the most recent stable release.

Consider using a virtualenv to separate and test the installation.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for slurp can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    # To update with real URL
    $ git clone git://github.com/CNES/slurp

Or download the `tarball`_:

.. code-block:: console

    # To update with real URL
    $ curl -OJL https://github.com/CNES/slurp/tarball/master

Once you have a copy of the source, you can install it in a virtualenv with:

.. code-block:: console

    $ make install
    $ source venv/bin/activate


.. _Github repo: https://github.com/CNES/slurp
.. _tarball: https://github.com/CNES/slurp/tarball/master
