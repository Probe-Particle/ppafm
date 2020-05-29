
import sys
sys.path.append('../..')

from pyProbeParticle import oclUtils     as oclu 
from pyProbeParticle import fieldOCL     as FFcl 
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle import common       as PPU
from pyProbeParticle import basUtils
from pyProbeParticle.AFMulatorOCL_Simple import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2 import InverseAFMtrainer
from pyProbeParticle.AuxMap import AuxMaps

import os
import time
import numpy as np
import matplotlib.pyplot as plt

class ExampleTrainer(InverseAFMtrainer):

    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)
        print(self.distAboveActive, self.afmulator.tipR0)

env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

args = {
    'pixPerAngstrome'   : 8,
    'lvec'              : np.array([
                            [ 0.0,  0.0,  0.0],
                            [30.0,  0.0,  0.0],
                            [ 0.0, 30.0,  0.0],
                            [ 0.0,  0.0, 30.0]
                            ]),
    'scan_dim'          : (128, 128, 20),
    'scan_window'       : ((-8.0,-8.0), ( 8.0, 8.0)),
    'iZPP'              : 8,
    'QZs'               : [ 0.1,  0, -0.1, 0 ],
    'Qs'                : [ -10, 20,  -10, 0 ],
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

afmulator = AFMulator(**args)
afmulator.distAbove = 8.8
afmulator.npbc = (0,0,0)

spheres = AuxMaps('vdwSpheres')
disks = AuxMaps('AtomicDisks')
height_map = AuxMaps('HeightMap', {'scanner': afmulator.scanner})
bonds = AuxMaps('Bonds')
atomrfunc = AuxMaps('AtomRfunc')
aux_maps = [spheres, disks, height_map, bonds, atomrfunc]

molecules = ['out2', 'benzeneBrCl2', 'out3']
paths = [f'{mol}/pos.xyz' for mol in molecules]

trainer = ExampleTrainer(afmulator, aux_maps, paths, batch_size=3, distAbove=5.0)

rotations = PPU.sphereTangentSpace(n=100)
trainer.augment_with_rotations_entropy(rotations, 2)
trainer.shuffle_molecules()

save_dir = './test_images/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

counter = 0
t0 = time.time()
for Xs, Ys, mols in trainer:

    print(f'Batch generation time: {time.time() - t0}')

    for j in range(len(Xs[0])):

        for i, X in enumerate(Xs):
            rows, cols = 2, 5
            fig = plt.figure(figsize=(3.2*cols,2.5*rows))
            for k in range(X.shape[-1]):
                fig.add_subplot(rows, cols, k+1)
                plt.imshow(X[j,:,:,k], cmap='afmhot', origin="lower")
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{counter}_afm{i}.png'))
            plt.close()

        fig, axes = plt.subplots(1, len(aux_maps))
        fig.set_size_inches(2.5*len(aux_maps), 3)
        for i, ax in enumerate(axes):
            ax.imshow(Ys[i][j], origin='lower')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{counter}_auxmaps.png'))
        plt.close()
        
        mol = mols[j]
        basUtils.saveXyz(os.path.join(save_dir, f'{counter}_mol.xyz'), mol[:,4].astype(np.int32), mol[:,:4])

        counter += 1
        t0 = time.time()
