[![codecov](https://codecov.io/gh/Probe-Particle/ppafm/graph/badge.svg?token=bsFIxZhLJd)](https://codecov.io/gh/Probe-Particle/ppafm)
[![Build Status](https://github.com/Probe-Particle/ppafm/actions/workflows/ci.yml/badge.svg)](https://github.com/Probe-Particle/ppafm/actions)
[![PyPI version](https://badge.fury.io/py/ppafm.svg)](https://badge.fury.io/py/ppafm)
[![Documentation Status](https://readthedocs.org/projects/ppafm/badge/?version=latest)](https://ppafm.readthedocs.io/en/latest/?badge=latest)
[![PyPI-downloads](https://img.shields.io/pypi/dm/ppafm.svg?style=flat)](https://pypistats.org/packages/ppafm)

# Probe-Particle Model

Simple and efficient simulation software for high-resolution atomic force microscopy (HR-AFM) and other scanning probe microscopy (SPM) techniques with sub-molecular resolution (STM, IETS, TERS).
It simulates the deflection of the probe particle attached to the tip, where the probe particle represents a flexible tip apex (typically CO molecule, but also e.g. Xe, Cl-, H2O and others).
The Python package is named as `ppafm`.

## Documentation

* [Wiki](https://github.com/Probe-Particle/ProbeParticleModel/wiki)
* [readthedocs](https://ppafm.readthedocs.io/en/latest/?badge=latest)
* Auto-generated:
   * [DeepWiki](https://deepwiki.com/Probe-Particle/ppafm)
   * [Context7](https://context7.com/probe-particle/ppafm/llms.txt)

## Installation

The standard way of installing `ppafm` is:

```bash
pip install ppafm
```

This should install the package and all its dependencies including **CPU** version and fully working CLI:

The most up-to-date installation guide can be found on the [dedicated wiki page](https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm).

## Command line interface (CLI)

Once `ppafm` is installed, a collection of command-line tools will become available to the user.
Their names start with the `ppafm-` prefix.
To get more information about a given tool, run it with the `-h` option, e.g.:

```bash
ppafm-generate-ljff -h
```

For more information, please consult the [dedicated page](https://github.com/Probe-Particle/ppafm/wiki/Command-line-interface) on the command line interface of `ppafm`.

## GPU version and Graphical User Interface (GUI)
The package comes with a **GPU** speedup version and convenient graphical user interface.
Unlike CLI, these needs to be explicitly enabled during the installation.
To enable it, check the [dedicated section](https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm#enable-gpugui-support) on the [Install ppafm](https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm#enable-gpugui-support) wiki page.
To know more about the GUI interface, please consult the [ppafm GUI](https://github.com/Probe-Particle/ppafm/wiki/PPAFM-GUI) wiki page.
Additional information about the **GPU** version can be found [bellow](https://github.com/Probe-Particle/ppafm?tab=readme-ov-file#run-gpu-generator-for-machine-learning), at [models](https://github.com/Probe-Particle/ppafm/wiki#ppafm-simulation-models-and-implementations) and [models differences](https://github.com/Probe-Particle/ppafm/wiki/Differences-between-the-CPU-and-GPU-implementations) wiki pages.

## Usage examples

We provide a set of examples in the `examples` directory.
To run them, navigate to the directory and run the `run.sh` script.
For example:

```bash
cd examples/PTCDA_single
./run.sh
```

You can study the script to see how to run the simulation.
Also, have a look at the `params.ini` file and [the wiki](https://github.com/Probe-Particle/ppafm/wiki/Params) to see how to set up the simulation parameters.

Once the simulation is finished, several files and folders will be created.

See also the tutorial for using the Python API for [running GPU-accelerated simulations](https://ppafm.readthedocs.io/en/latest/tutorials/afmulator-tutorial.html).

### Run GPU generator for machine learning

* `examples/CorrectionLoopGraphene` use GPU accelerated ppafm to iteratively improve the estimate of molecular geometry by comparing simulated AFM images with reference.
This is a work in progress.
Currently, modification of estimate geometry is random (Monte-Carlo), while later we plan to develop a more clever (e.g. Machine-Learned) heuristic for more efficient improvement.
* `examples/Generator` quickly generates a batch of simulated AFM images (resp. 3D data stacks) which can be further used for machine learning.
Especially in connection with (https://github.com/SINGROUP/ASD-AFM).

## For contributors
If you miss some functionality or have discovered issues with the latest release - let us know by creating [an issue](https://github.com/Probe-Particle/ppafm/issues/new).
If you would like to contribute to the development of the ppafm code, please read the [Developer's Guide](https://github.com/Probe-Particle/ppafm/wiki/For-Developers) wiki page.
Small improvements in the documentation or minor bug fixes are always welcome.

## Publications describing the Probe-Particle Model

If you have used `ppafm` in your research, please cite the following articles:
* [Niko Oinonen, Aliaksandr V. Yakutovich, Aurelio Gallardo, Martin Ondracek, Prokop Hapala, Ondrej Krejci, Advancing Scanning Probe Microscopy Simulations: A Decade of Development in Probe-Particle Models, Comput. Phys. Commun. 305, 109341 - Available online 10 August 2024](https://doi.org/10.1016/j.cpc.2024.109341)
* [Prokop Hapala, Georgy Kichin, Christian Wagner, F. Stefan Tautz, Ruslan Temirov, and Pavel Jelínek, Mechanism of high-resolution STM/AFM imaging with functionalized tips, Phys. Rev. B 90, 085421 – Published 19 August 2014](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.085421)
* [Prokop Hapala, Ruslan Temirov, F. Stefan Tautz, and Pavel Jelínek, Origin of High-Resolution IETS-STM Images of Organic Molecules with Functionalized Tips, Phys. Rev. Lett. 113, 226101 – Published 25 November 2014](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.226101)

## License
MIT
