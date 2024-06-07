# Generating machine learning training data

The previous tutorials described the [`AFMulator`](afmulator-tutorial) for producing AFM simulations and the [`AuxMap`](auxmap-tutorial) image descriptors.
The simulations and image descriptors can be used as input/output pairs for training a machine learning model for atomic structure prediction.
To make the generation of large databases such training data easier, the ppafm Python API provides a generator that wraps both the AFM simulation and the image descriptor calculation into a single object that produces batches of training data.

The generator comes in two forms, the {class}`.InverseAFMtrainer` which can be used for simple Lennard-Jones (+ point-charge electrostatics) simulations based on xyz files, and the {class}`.GeneratorAFMtrainer` which works with any of the force-field models, but requires a bit more setup.
The following describes how to use both of them.


## Data generation from xyz files

In the simplest case you may have just a directory full of xyz files that contain molecule geometries and possibly point charges for the atoms.
In this case we can use the {class}`.InverseAFMtrainer` as follows, assuming that the xyz files are in the folder `example_molecules`:

```python
from pathlib import Path

import numpy as np

from ppafm.common import sphereTangentSpace
from ppafm.ml.AuxMap import AtomicDisks, AtomRfunc, Bonds, HeightMap, vdwSpheres
from ppafm.ml.Generator import InverseAFMtrainer
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env


class ExampleTrainer(InverseAFMtrainer):
    # We override this callback method in order to augment the samples with randomized tip distance and tilt
    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)


if __name__ == "__main__":

    init_env(i_platform=0)

    # Define simulator with parameters
    afmulator = AFMulator(
        pixPerAngstrome=10,
        scan_dim=(128, 128, 19),
        scan_window=((2.0, 2.0, 5.0), (18.0, 18.0, 6.9)),
        df_steps=10,
        npbc=(0, 0, 0),
    )

    # Define descriptors to generate
    auxmap_args = {"scan_window": ((2.0, 2.0), (18.0, 18.0)), "scan_dim": (128, 128)}
    aux_maps = [
        vdwSpheres(**auxmap_args),
        AtomicDisks(**auxmap_args),
        HeightMap(afmulator.scanner),
        Bonds(**auxmap_args),
        AtomRfunc(**auxmap_args),
    ]

    # Paths to molecule xyz files
    xyzs_dir = Path("example_molecules")
    paths = list(xyzs_dir.glob("*.xyz"))

    # Combine everything to one
    trainer = ExampleTrainer(
        afmulator,
        aux_maps,
        paths,
        batch_size=20,
        distAbove=4.5,
        iZPPs=[8],
        QZs=[[0.1, 0, -0.1, 0]],
        Qs=[[-10, 20, -10, 0]],
    )

    # Augment molecule list with rotations and shuffle
    trainer.augment_with_rotations_entropy(sphereTangentSpace(n=100), 30)
    trainer.shuffle_molecules()

    # Iterate over batches
    for ib, (afm, desc, mols) in enumerate(trainer):
        np.savez(f"batch_{ib}.npz", afm=afm, desc=desc, mols=mols)

```

Here we first define the {class}`.AFMulator` and various `AuxMap` instances as before.
Additionally we gather a list of paths to the xyz files in the variable `paths`.
We combine all of the above into the generator class:
```python
trainer = ExampleTrainer(
    afmulator,
    aux_maps,
    paths,
    batch_size=20,
    distAbove=4.5,
    iZPPs=[8], # CO
    QZs=[[0.1, 0, -0.1, 0]],
    Qs=[[-10, 20, -10, 0]],
)
```
There are a couple of things to note here.
First is that the `iZPPs`, `QZs`, and `Qs` which define the properties of the probe particle (PP) are defined here instead of the `AFMulator`.
This is because it is possible to generate the AFM image for multiple different PPs at once.
Notice that the arguments here are lists.
If we wanted to add for example a Xe PP in addition to CO, we could just add the corresponding parameters to the lists:
```python
trainer = ExampleTrainer(
    ...
    iZPPs=[8, 54], # CO, Xe
    QZs=[[0.1, 0, -0.1, 0], [0.3, 0.0, 0.0, 0.0]],
    Qs=[[-10, 20, -10, 0], [0.0, 0.0, 0.0, 0.0]],
    ...
)
```
```{note}
The ordering of the PPs matters for the {class}`.HeightMap` descriptor which uses the generated force field. It will use the force field of the last PP in the list.
```

The other two important arguments are `batch_size`, which sets the number of samples in a batch (last batch might be smaller), and `distAbove` which sets the tip-sample distance.
The tip-sample distance here is measured as the distance between the top of the scan and the top atom in the sample subtracted by the vdW radius of the PP and the top atom of the sample.
This is a heuristic which gives a roughly similar level of contrast in the simulated AFM images regardless of the sample.


Secondly, instead of using the {class}`.InverseAFMtrainer` directly, we subclass it:
```python
class ExampleTrainer(InverseAFMtrainer):
    # We override this callback method in order to augment the samples with randomized tip distance and tilt
    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)
```
This enables us to add random augmentations to the samples as they are generated. This is done by overriding callback methods that are automatically called at various parts of the generation process. The overridable methods are {meth}`.InverseAFMtrainer.on_afm_start`, {meth}`.InverseAFMtrainer.on_sample_start`, and {meth}`.InverseAFMtrainer.on_batch_start`.
In pseudo-code the generation procedure looks like the following:
```
on_batch_start()
for each sample:
    on_sample_start()
    for each PP:
        set_afmulator_PP_params()
        on_afm_start()
        afm = afmulator(...)
    for each descriptor:
        desc = descriptor(...)
```
By default the callbacks don't do anything.
In this case we randomize the tip-sample distance and the tip tilt for each sample by overriding the `on_sample_start` callback.
When writing the overriding methods, it is useful to know that the `AFMulator` instance is available in `self.afmulator`, the list of `AuxMap` is available in `self.aux_maps`, and the currently set tip-sample distance is available in `self.distAboveActive`.

We additionally augment the molecule list before the generation starts by taking random rotations for each molecule:
```python
trainer.augment_with_rotations_entropy(sphereTangentSpace(n=100), 30)
trainer.shuffle_molecules()
```
The {func}`.sphereTangentSpace` function first generates `n = 100` rotations (more-or-less) uniformly over a sphere and then the {meth}`InverseAFMtrainer.augment_with_rotations_entropy` method picks the 30 best ones by a certain measure of "entropy" that prefers directions that have more atoms in a plane parallel to the scan plane.
We also shuffle the list of molecules with `InverseAFMtrainer.shuffle_molecules` so that the samples within a batch are uncorrelated.

The constructed instance of the generator class is an iterator which we can use in a for-loop:
```python
for ib, (afm, desc, mols) in enumerate(trainer):
    np.savez(f"batch_{ib}.npz", afm=afm, desc=desc, mols=mols)
```
The returned arrays contain the simulated AFM images `afm`, the image descriptors `desc`, and the arrays of coordinates and types of the atoms `mols`.

A complete example with the script and the data can be found [in the examples folder](https://github.com/Probe-Particle/ppafm/blob/main/examples/Generator/inverse_trainer_xyz.py).

## General data generation
