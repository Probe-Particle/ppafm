#!/usr/bin/env python3

import os
import sys
from subprocess import run
import matplotlib.pyplot as plt

sys.path.append('../..')
from ppafm.ocl.AFMulator import AFMulator, hartreeFromFile

if not os.path.exists('LOCPOT.xsf'):
    run(['wget', '--no-check-certificate', 'https://www.dropbox.com/s/18eg89l89npll8x/LOCPOT.xsf.zip'])
    run(['unzip', 'LOCPOT.xsf.zip'])

afmulator = AFMulator(
    scan_dim=(200, 200, 60),
    scan_window=((0.0, 0.0, 16.0), (19.875, 19.875, 22.0)),
    iZPP=8,
    df_steps=10, # Amplitude = 10 * (22.0Å - 16.0Å) / 60 = 1.0Å
    tipStiffness=(0.37, 0.37, 0.0, 20.0),
    rho={'dz2': -0.05},
    tipR0=[0.0, 0.0, 4.0],
    npbc=(1, 1, 0)
)

print('Loading potential')
pot, xyzs, Zs = hartreeFromFile('./LOCPOT.xsf')

print('Running simulation')
X = afmulator(xyzs, Zs, pot)

print('Plotting images')
outdir = './PTCDA_CO_dz2-0.05_K0.37_A1.0'
if not os.path.exists(outdir):
    os.makedirs(outdir)

for i in range(X.shape[-1]):
    plt.imshow(X[:, :, i].T, origin='lower', cmap='gray')
    plt.savefig(os.path.join(outdir, f'{i}.png'))
    plt.close()
