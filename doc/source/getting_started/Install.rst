======================
Setup and Installation
======================

Installation
############

``IceAdvect`` is currently only available for download from the `GitHub repository <https://github.com/tsutterley/IceAdvect>`_.

Development Install
###################

To use the development repository, please fork ``IceAdvect`` into your own account and then clone onto your system:

.. code-block:: bash

    git clone https://github.com/tsutterley/IceAdvect.git

``IceAdvect`` can then be installed within the package directory using ``pip``:

.. code-block:: bash

    python3 -m pip install --user .

To include all optional dependencies:

.. code-block:: bash

   python3 -m pip install --user .[all]

The development version of ``IceAdvect`` can also be installed directly from GitHub using ``pip``:

.. code-block:: bash

    python3 -m pip install --user git+https://github.com/tsutterley/IceAdvect.git

Package Management with ``pixi``
################################

Alternatively ``pixi`` can be used to create a `streamlined environment <https://pixi.sh/>`_ after cloning the repository:

.. code-block:: bash

    pixi install

``pixi`` maintains isolated environments for each project, allowing for different versions of
``IceAdvect`` and its dependencies to be used without conflict. The ``pixi.lock`` file within the
repository defines the required packages and versions for the environment.

``pixi`` can also create shells for running programs within the environment:

.. code-block:: bash

    pixi shell

To see the available tasks within the ``IceAdvect`` workspace:

.. code-block:: bash

    pixi task list

.. note::

    ``pixi`` is under active development and may change in future releases
