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

Install (from the README.md)
-------

You need to clone the repository and pip install SLURP.

.. code-block:: console

   git clone git@gitlab.cnes.fr:pluto/slurp.git

To install SLURP, you need OTB,
`EOScale <https://gitlab.cnes.fr/pluto/eoscale>`__ and some libraries
already installed on VRE OT.

Otherwise, if you are are connected to TREX, or working on your personal
computer (Linux), you may set the environment as mentioned below. ###
Create a virtual env with all libraries (if you don’t use VRE OT) On
TREX, connect to a computing node to create & compile the virtual
environment (needed to compile rasterio at install time)

.. code-block:: console

   sinter -A cnes_level2 -N 1 -n 8 --time=02:00:00 --mem=64G --x11 --pty bash

Load OTB and create a virtual env with some Python libraries. Compile
and install EOScale and then SLURP

.. code-block:: console

   module load otb/9.0.0-python3.8
   # Creates a virtual env base on Python 3.8.13
   python -m venv slurp_env
   . slurp_env/bin/activate
   # upgrade pip and install several libraries
   pip install pip --upgrade
   cd <EOScale source folder>
   pip install .
   cd <SLURP source folder>
   pip install .
   # for validation tests
   pip install pytest

Your environment is ready, you can compute SLURP masks with
slurp_watermask, slurp_urbanmask, etc.


.. _Github repo: https://github.com/CNES/slurp
.. _tarball: https://github.com/CNES/slurp/tarball/master
