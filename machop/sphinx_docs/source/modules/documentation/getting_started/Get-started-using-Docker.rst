Getting Started using Docker
=============================

Install Conda (for the first time)
-----------------------------------

If you don't have ``Docker`` installed yet, follow the instructions `here <https://docs.docker.com/engine/install/>`_ to install on your platform.

Install environment using Docker
-----------------------------------

1. Start by cloning mase:

.. code-block:: shell

  git clone git@github.com:DeepWok/mase.git

2. Create your own branch to work on:

.. code-block:: shell

  cd mase
  git checkout -b your_branch_name

Launching Docker
-----------------------------------

<<<<<<< HEAD
.. hint::

  If you're an Imperial College student requiring Vivado/Vitis tools for your MSc/MEng project, replace all following steps with the command ``make shell``. This downloads and launches the container with the Vivado installation path pre-mounted.

3. You can pull the required Docker image directly from Dockerhub as follows.

.. code-block:: shell

  docker pull docker.io/deepwok/mase-docker:latest

4. Now launch the container by running the following command.

.. code-block:: shell

  docker run -it --shm-size 256m \
    # set the workspace to the /workspace directory in the container
    -w /workspace \
    # mount git and ssh directories to use those commands in the container
    -v ~/.gitconfig:/root/.gitconfig \
    -v ~/.ssh:/root/.ssh \
    # mount the head of the repository to the workspace directory
    -v $(pwd):/workspace:z \
    # define the required image
    deepwok/mase-docker:latest \
    /bin/bash

.. hint::

  If you need access to Vivado/Vitis toolflows, you can map your Vivado installation to the container by using the following additional command-line argument: `-v <path/to/vivado>:<container/path>` and adding a line like `source <container/path/settings64.sh>` to the container's `bashrc`.
=======
3. If you're an Imperial College student running MASE on a CAS server (ee-beholder/ee-kraken), you can just run the following commands to use the existing docker container.

.. code-block:: shell

  make shell

If you do not have access to one of our servers and need access to Vivado/Vitis toolflows, you need to rebuild your docker container from scratch by running the following command:

.. code-block:: shell

  make shell vhls=$YOUR_VHLS_PATH vhls_version=$YOUR_VHLS_VERSION local=1

The first argument points to your Xilinx tool directory and you should see the following folder under the path:
 
.. code-block:: shell

  DocNav  Model_Composer  Vitis  Vitis_HLS  Vivado  xic

The second argument indicates your Xilinx tool version, e.g. 2023.1.
The last argument asks to build the docker container locally with the new arguments.
>>>>>>> main
