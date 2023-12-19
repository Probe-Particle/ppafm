[![codecov](https://codecov.io/gh/Probe-Particle/ppafm/graph/badge.svg?token=bsFIxZhLJd)](https://codecov.io/gh/Probe-Particle/ppafm)
[![Build Status](https://github.com/Probe-Particle/ppafm/actions/workflows/ci.yml/badge.svg)](https://github.com/Probe-Particle/ppafm/actions)
[![PyPI version](https://badge.fury.io/py/ppafm.svg)](https://badge.fury.io/py/ppafm)
[![Documentation Status](https://readthedocs.org/projects/ppafm/badge/?version=latest)](https://ppafm.readthedocs.io/en/latest/?badge=latest)
[![PyPI-downloads](https://img.shields.io/pypi/dm/ppafm.svg?style=flat)](https://pypistats.org/packages/ppafm)


# Probe Particle Model (PPM)

Simple and efficient **simulation software for high-resolution atomic force microscopy** (**HR-AFM**) and other scanning probe microscopy (SPM) techniques with sub-molecular resolution (STM, IETS, TERS). It simulates deflection of the particle attached to the tip (typically CO molecule, but also e.g. Xe, Cl-, H2O and others).

## Installation

The standard way of installing `ppafm` is:

```bash
$ pip install ppafm
```

This should install the package and all its dependencies.

The most up-to-date installation guide can be found on the [dedicated wiki page](https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm).

## Command line interface (CLI)

Once `ppafm` is installed, a collection of command-line tools will become available to the user.
Their names start with `ppafm-` preffix.
To get more information about a given tool, run it with `-h` option, e.g.:

```bash
ppafm-generate-ljff -h
```

For more information, please consult the [dedicated page](https://github.com/Probe-Particle/ppafm/wiki/Command-line-interface) on command line interface of `ppafm`.

## Graphical User Interface (GUI)
The package comes with a convenient graphical user interface.
Unlike CLI, this interface needs to be explicitly enabled during the installation.
To enable it, check the [dedicated section](https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm#enable-gpugui-support) on the [Install ppafm](https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm#enable-gpugui-support) wiki page.
To know more about the GUI interface, please consult [PPAFM GUI](https://github.com/Probe-Particle/ppafm/wiki/PPAFM-GUI) wiki page.

## Usage examples

We provide a set of examples in the `examples` directory.
To run them, navigate to the directory and run the `run.sh` script.
For example:

```bash
$ cd examples/PTCDA_single
$ ./run.sh
```

You can study the script to see how to run the simulation.
Also, have a look at the `params.ini` file and [the wiki](https://github.com/Probe-Particle/ppafm/wiki/Params) to see how to set up the simulation parameters.

Once the simulation is finished, a number of files and folders will be created.



### Run GPU generator for machine learning

* `examples/CorrectionLoopGraphene` use GPU accelerated PPM to iteratively improve the estimate of molecular geometry by comparing simulated AFM images with reference. This is work-in-progress. Currently, modification of estimate geometry is random (Monte-Carlo), while later we plan to develop a more clever (e.g. Machine-Learned) heuristic for more efficient improvment.
* `examples/Generator` quickly generates a batch of simulated AFM images (resp. 3D data stacks) which can be further used for machine learning. Especially in connection with (https://github.com/SINGROUP/ASD-AFM).

## Flavors of PPM

Since 2014 PPM developed into the toolbox of various methodologies adjusted for a particular use case.

1. **CPU version:** - Original implementation using Python & C/C++. It can simulate a typical AFM experiment (3D stack of AFM images) in ~1 minute. It is the base version for the development of new features and methodology. All available simulation models are implemented in this version, including:
   1. **Point charge electrostatics + Lennard-Jones:** Original fully classical implementation allows the user to set up calculation without any ab-initio input simply by specifying atomic positions, types and charges.
   2. **Hartree-potential electrostatics + Lennard-Jones:** Electrostatics is considerably improved by using Hartree potential from DFT calculation (e.g. LOCPOT from VASP) and using the Quadrupole model for CO-tip. We found this crucial to properly simulate polar molecules (e.g. H2O clusters, carboxylic acids, PTCDA) which exhibit strong electrostatic distortions of AFM images. Thanks to implementation using fast Fourier transform (FFT) this improvement does not increase the computational time (still ~1 minute), as long as the input electrostatic field is accessible.
   3. **Hartree-potential electrostatics + Density overlap:** Further accuracy improvement is achieved when Pauli repulsion between electron shells of atoms is modeled by the overlap between electron density of tip and sample. This repulsive term replaces the repulsive part of Lennard-Jones while the attractive part (C6) remains. This modification considerably improves especially simulation of molecules with electron pairs (-NH-, -OH, =O group), triple bonds and other strongly concentrated electrons. Calculation of the overlap repulsive potential is again accelerated by FFT to achieve minimal computational overhead (2-3 minutes) as long as input densities of tip and sample are available.
2. **GPU version:** - Version specially designed for generation of training data for machine learning. Implementation using `pyOpenCL` can parallelize the evaluation of forcefield and relaxation of probe-particle positions over hundreds or thousands of stream-processors of the graphical accelerator. Further speed-up is achieved by using hardware accelerated trilinear interpolation of 3D textures available in most GPUs. This allows simulating 10-100 AFM experiments per second on consumer-grade desktop GPU.
   * GPU version is designed to work in collaboration with machine-learning software for AFM (https://github.com/SINGROUP/ASD-AFM) and use various generators of molecular geometry.
3. **GUI @ GPU** - The speed of GPU implementation also allows to make interactive GUI where AFM images of molecules can be updated on the fly (<<0.1s) on a common laptop computer while the user is editing molecular geometry or parameters of the tip. This provides an invaluable tool especially to experimentalists trying to identify and interpret the structure and configuration of molecules in experiments on-the-fly while running the experiment.

## Other branches

* **master_backup** - Old `master` branch was recently significantly updated and named `main`. For users who miss the old master branch, we provided a backup copy. However, this version is very old and its use is discouraged. If you miss some functionality or are not satisfied with the behavior of current `main` branch please let us know by creating an *issue*.
* **PhotonMap** - implements the latest developments concerning sub-molecular scanning probe combined with Raman spectroscopy (TERS)y and fluorescent spectroscopy (LSTM).
* **complex_tip** - Modification of probe-particle model with 2 particles allows a better fit to experimental results at the cost of additional fitting parameters.


## For developers

If you would like to contribute to the development of the ppafm code, please read the [Developer's Guide](https://github.com/Probe-Particle/ppafm/wiki/For-Developers) wiki page.

## Further information
- Wiki: https://github.com/Probe-Particle/ProbeParticleModel/wiki
- Python API documentation: https://ppafm.readthedocs.io/en/latest/

## Notable publications using Probe Particle Model

* [Prokop Hapala, Georgy Kichin, Christian Wagner, F. Stefan Tautz, Ruslan Temirov, and Pavel Jelínek, Mechanism of high-resolution STM/AFM imaging with functionalized tips, Phys. Rev. B 90, 085421 – Published 19 August 2014](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.085421)
* [Prokop Hapala, Ruslan Temirov, F. Stefan Tautz, and Pavel Jelínek, Origin of High-Resolution IETS-STM Images of Organic Molecules with Functionalized Tips, Phys. Rev. Lett. 113, 226101 – Published 25 November 2014,](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.226101)

## License
MIT
