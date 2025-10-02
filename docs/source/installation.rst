.. highlight:: shell

============
Installation
============


Stable release
--------------

SLURP is deployed on Pypi. To install slurp, please create a python 3.10 virtual environment :

.. code-block:: console

    $ python3.10 -m venv slurp_env
    $ pip install slurp-masks

This is the preferred method to install slurp, as it will always install the most recent stable release.

.. warning::

    Depending on which computation you want to perform (cf. `Use cases <use_cases.html>`_ page), you may need to install OTB on your system as an additional dependencies.
    Please refer to the OTB installation guide provided `here <https://www.orfeo-toolbox.org/CookBook-develop/Installation.html#create-an-healthy-python-environment-for-otb>`_  for more details.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for SLURP can be downloaded from the `Github repo`_.

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
