.. highlight:: shell

============
Installation
============


Stable release
--------------

If deployed in Pypi, to install slum, run this command in your terminal:

.. code-block:: console

    $ pip install slum

This is the preferred method to install slum, as it will always install the most recent stable release.

Consider using a virtualenv to separate and test the installation.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for slum can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    # To update with real URL
    $ git clone git://github.com/CNES/slum

Or download the `tarball`_:

.. code-block:: console

    # To update with real URL
    $ curl -OJL https://github.com/CNES/slum/tarball/master

Once you have a copy of the source, you can install it in a virtualenv with:

.. code-block:: console

    $ make install
    $ source venv/bin/activate


.. _Github repo: https://github.com/CNES/slum
.. _tarball: https://github.com/CNES/slum/tarball/master
