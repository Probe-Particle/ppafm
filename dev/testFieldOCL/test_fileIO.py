#!/usr/bin/python

import sys

import numpy as np

sys.path.append("/home/prokop/git/ProbeParticleModel_OCL")

from ppafm.io import loadXSF, saveXSF

xs    = np.linspace(0.1,1.0,5)
ys    = np.linspace(0.1,1.0,5)
zs    = np.linspace(0.1,1.0,5)
Xs,Ys,Zs = np.meshgrid(xs,ys,zs)

F = Xs*Ys*Zs

lvec_OUT = (
    [0.0,0.0,0.0],
    [10.0,0.0,0.0],
    [0.0,10.0,0.0],
    [0.0,0.0,6.0]
)

saveXSF( 'mini.xsf',  F, lvec_OUT )

E,lvec, nDim, head = loadXSF( 'mini.xsf' )
