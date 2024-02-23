GPU-accelerated AFM simulations
===============================

The command-line interface to ppafm can only utilize the CPU.
The simulations can be sped-up by at least a couple of orders of magnitude by utilizing a graphics processing unit (GPU).
In order to make use of the GPU acceleration in ppafm, install it with `pip` using the ``[opencl]`` option, as detailed on the `installation page on the wiki <https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm>`_.

The tutorial here concerns how to run GPU-accelerated AFM simulations programmatically via the Python API, which is most useful when running a large number of simulations in a batch.
If you are looking to study an individual system and experiment with the simulation parameters, the `graphical user interface <https://github.com/Probe-Particle/ppafm/wiki/PPAFM-GUI>`_
may be better suited for that purpose.
This tutorial also assumes that you are already familiar with the basic of the structure of the ppafm simulation. If not, take a look at the wiki pages explaining the `basics of the simulation <https://github.com/Probe-Particle/ppafm/wiki#probe-particle-model>`_ and the `different force field models <https://github.com/Probe-Particle/ppafm/wiki/Forces>`_.

The AFMulator API
-----------------

Lennard-Jones
^^^^^^^^^^^^^

Hartree electrostatics
^^^^^^^^^^^^^^^^^^^^^^

Full-density based model
^^^^^^^^^^^^^^^^^^^^^^^^
