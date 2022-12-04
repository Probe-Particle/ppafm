
import sys

sys.path.append('../..')

import os

import matplotlib.pyplot as plt
import numpy as np

from ppafm import basUtils
from ppafm import common as PPU
from ppafm.ml.AuxMap import AuxMaps
from ppafm.ml.Generator import InverseAFMtrainer
from ppafm.ocl import field as FFcl
from ppafm.ocl import oclUtils as oclu
from ppafm.ocl import relax as oclr
from ppafm.ocl.AFMulator import AFMulator


class ExampleTrainer(InverseAFMtrainer):

    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)

env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

afmulator = AFMulator(
    pixPerAngstrome=10,
    scan_dim=(128, 64, 19),
    scan_window=((2.0, 2.0, 5.0), (18.0, 10.0, 7.0)),
    iZPP=8,
    QZs=[ 0.1,  0, -0.1, 0 ],
    Qs=[ -10, 20,  -10, 0 ],
    df_steps=10,
    npbc=(0, 0, 0)
)

auxmap_args = {'scan_window': ((2.0, 2.0), (18.0, 10.0)), 'scan_dim': (128, 64)}
spheres = AuxMaps('vdwSpheres', auxmap_args)
disks = AuxMaps('AtomicDisks', auxmap_args)
height_map = AuxMaps('HeightMap', {'scanner': afmulator.scanner})
bonds = AuxMaps('Bonds', auxmap_args)
atomrfunc = AuxMaps('AtomRfunc', auxmap_args)
aux_maps = [spheres, disks, height_map, bonds, atomrfunc]

molecules = ['out2', 'benzeneBrCl2', 'out3']
paths = [f'{mol}/pos.xyz' for mol in molecules]

trainer = ExampleTrainer(afmulator, aux_maps, paths, batch_size=20, distAbove=4.5)
trainer.bRuntime = True

rotations = PPU.sphereTangentSpace(n=100)
trainer.augment_with_rotations_entropy(rotations, 30)
trainer.shuffle_molecules()

save_dir = './test_images/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

counter = 0
for Xs, Ys, mols in trainer:

    print('Plotting...')

    for j in range(len(Ys[0])):

        for i, X in enumerate(Xs):
            rows, cols = 2, 5
            fig = plt.figure(figsize=(3.2*cols,2.5*rows))
            for k in range(X.shape[-1]):
                fig.add_subplot(rows, cols, k+1)
                plt.imshow(X[j,:,:,k].T, cmap='afmhot', origin="lower")
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{counter}_afm{i}.png'))
            plt.close()

        fig, axes = plt.subplots(1, len(aux_maps))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
        fig.set_size_inches(3*len(aux_maps), 3)
        for i, ax in enumerate(axes):
            im = ax.imshow(Ys[i][j].T, origin='lower')
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{counter}_auxmaps.png'))
        plt.close()

        mol = mols[j]
        basUtils.saveXYZ(os.path.join(save_dir, f'{counter}_mol.xyz'), mol[:, :3], mol[:, 4].astype(np.int32), mol[:, 3])

        counter += 1
