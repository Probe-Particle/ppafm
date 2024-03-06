import os

import matplotlib.pyplot as plt
import numpy as np

from ppafm import io
from ppafm.ml.AuxMap import AuxMaps
from ppafm.ml.Generator import GeneratorAFMtrainer
from ppafm.ocl import oclUtils as oclu
from ppafm.ocl.AFMulator import AFMulator


class ExampleTrainer(GeneratorAFMtrainer):
    # Override the on_sample_start method to randomly modify simulation parameters for each sample.
    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)


class SampleGenerator:
    """A simple generator that yields samples from xyz files."""

    def __init__(self, sample_paths):
        self.sample_paths = sample_paths

    def __iter__(self):
        self.sample_count = 0
        return self

    def __next__(self):
        if self.sample_count >= len(self.sample_paths):
            raise StopIteration

        # Fetch a sample from file
        sample_path = self.sample_paths[self.sample_count]
        xyzs, Zs, qs, _ = io.loadXYZ(sample_path)

        # We yield samples as dicts containing the input arguments to AFMulator. Here we only
        # have atomic coordinates, atomic numbers, and charges, but we could also include
        # the Hartree potential or electron density depending on what simulation model
        # we want to use. We could also include rotations of the molecules.
        sample_dict = {"xyzs": xyzs, "Zs": Zs, "qs": qs}

        self.sample_count += 1

        return sample_dict


oclu.init_env(i_platform=0)

save_dir = "./test_images_generator/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create the simulator
afmulator = AFMulator(
    pixPerAngstrome=10,  # Force field grid density
    scan_dim=(201, 201, 19),  # Output scan dimensions (z dimension gets reduced by df_steps - 1)
    scan_window=((0, 0, 5), (20, 20, 6.9)),  # Physical limits of the scan
    tipR0=[0, 0, 4],  # Tip equilibrium position (x, y, R)
    tipStiffness=[0.25, 0.25, 0.0, 20.0],  # Tip spring constants (x, y, z, R)
    npbc=(0, 0, 0),  # Periodic images of atoms in (x, y, z) directions
    df_steps=10,  # Oscillation amplitude in number of scan steps
)

# Create auxmap objects to generate image descriptors of the samples
auxmap_args = {"scan_window": ((0, 0), (20, 20)), "scan_dim": (201, 201)}
spheres = AuxMaps("vdwSpheres", auxmap_args)
disks = AuxMaps("AtomicDisks", auxmap_args)
height_map = AuxMaps("HeightMap", {"scanner": afmulator.scanner})
bonds = AuxMaps("Bonds", auxmap_args)
atomrfunc = AuxMaps("AtomRfunc", auxmap_args)
aux_maps = [spheres, disks, height_map, bonds, atomrfunc]

# Create a sample generator that yields samples by loading them from the disk
molecules = ["out2", "benzeneBrCl2", "out3"]
paths = [f"{mol}/pos.xyz" for mol in molecules]
sample_generator = SampleGenerator(paths)

# Combine everything into a trainer object
trainer = ExampleTrainer(
    afmulator,
    aux_maps,
    sample_generator,
    batch_size=2,  # Number of samples per batch
    distAbove=3.2,  # Tip-sample distance, taking into account the effective size of the tip and sample atoms
    iZPPs=[8, 54],  # Tip atomic number(s)
    rhos=[{"dz2": -0.1}, {"s": 0.3}],  # Tip charge distribution(s)
)

# Get samples from the trainer by iterating over it
counter = 0
for ib, (Xs, Ys, mols, sws) in enumerate(trainer):
    print(f"Batch {ib+1}")

    # Loop over samples in the batch
    for X, Y, mol, sw in zip(Xs, Ys, mols, sws):
        # X: AFM images
        # Y: Image descriptors
        # mol: Sample atom coordinates, atomic numbers, and charges
        # sw: Scan window bounds

        # Plot AFM images
        for i, x in enumerate(X):
            rows, cols = 2, 5
            fig = plt.figure(figsize=(3.2 * cols, 2.5 * rows))
            for k in range(X.shape[-1]):
                fig.add_subplot(rows, cols, k + 1)
                plt.imshow(x[..., k].T, cmap="afmhot", origin="lower")
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{counter}_afm{i}.png"))
            plt.close()

        # Plot images descriptors
        fig, axes = plt.subplots(1, len(aux_maps))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
        fig.set_size_inches(3 * len(aux_maps), 3)
        for y, ax in zip(Y, axes):
            im = ax.imshow(y.T, origin="lower")
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{counter}_auxmaps.png"))
        plt.close()

        # Save molecule into a xyz file
        io.saveXYZ(os.path.join(save_dir, f"{counter}_mol.xyz"), mol[:, :3], mol[:, 4].astype(np.int32), mol[:, 3])

        counter += 1
