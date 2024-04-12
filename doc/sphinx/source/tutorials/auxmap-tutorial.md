# Molecule image descriptors

Machine learning in atomic force microscopy often uses so-called image descriptors as the learning target.
An image descriptor represents the atomic structure or some other property of a molecule as a 2D map that is more easily interpretable than an AFM image.
Examples of such descriptors include the vdW Spheres, Atomic Disks, Height Map, and Electrostatic Map descriptors used by [Alldritt et al.](https://www.science.org/doi/10.1126/sciadv.aay6913), and [Oinonen et al.](https://pubs.acs.org/doi/10.1021/acsnano.1c06840):

[//]: # (TODO: Image here)

The GPU-accelerated Python API in ppafm provides the ability to generate these image descriptors, which in ppafm are called {mod}`.AuxMap`s.
The following shows how to generate the image descriptors in the above figure.
We will use the [1-bromo-3,5-dichlorobenzene molecule](https://github.com/Probe-Particle/ppafm/blob/main/examples/Generator/example_molecules/bcb.xyz) as the example.
