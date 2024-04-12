# Generating machine learning training data

The previous tutorials described the [`AFMulator`](afmulator-tutorial) for producing AFM simulations and the [`AuxMap`](auxmap-tutorial) image descriptors.
The simulations and image descriptors can be used as input/output pairs for training a machine learning model for atomic structure prediction.
To make the generation of large databases such training data easier, the ppafm Python API provides a generator that wraps both the AFM simulation and the image descriptor calculation into a single object that produces batches of training data.

The generator comes in two forms, the {class}`.InverseAFMtrainer` which can be used for simple Lennard-Jones (+ point-charge electrostatics) simulations based on xyz files, and the {class}`.GeneratorAFMtrainer` which works with any of the force-field models, but requires a bit more setup.
The following describes how to use both of them.


## Data generation from xyz files

In the simplest case you may have just a directory full of xyz files that contain molecule geometries and possibly point charges for the atoms.

Note: Height Map

## General data generation
