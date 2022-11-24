#!/usr/bin/python 

'''
Version of conv_rho.py ( i.e. convolution of electron densities density ) which is optimized to minimize size of datafiles and therefore also run-time of hard-disk I/O operations. 
The goal is achieve efficient calculation of AFM images with density-overlap for thousands of molecules (e.g. fr Machine-Learning application).
'''


import os
import sys
import __main__ as main
import numpy as np
import matplotlib.pyplot as plt
#import GridUtils as GU
#sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")

import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT
from optparse import OptionParser

parser = OptionParser()
parser.add_option( "-s", "--sample", action="store", type="string", default="CHGCAR.xsf", help="sample 3D data-file (.xsf)")
parser.add_option( "-t", "--tip",    action="store", type="string", default="./tip/CHGCAR.xsf", help="tip 3D data-file (.xsf)")
parser.add_option( "-o", "--output", action="store", type="string", default="pauli", help="output 3D data-file (.xsf)")
(options, args) = parser.parse_args()

rho1, lvec1, nDim1, head1 = GU.loadXSF( options.sample )
rho2, lvec2, nDim2, head2 = GU.loadXSF( options.tip    )

Fx,Fy,Fz,E = fFFT.potential2forces_mem( rho1, lvec1, nDim1, rho=rho2, doForce=True, doPot=True, deleteV=True )

PQ = -1.0

namestr = options.output
print("save to ", namestr)

zmin = 3.0
zmax = 7.0
dz  = lvec1[3,2]/rho1.shape[0]
izmin=int(zmin/dz)
izmax=int(zmax/dz)+1

lvec_ = lvec1.copy()
lvec_[0,2] = izmin*dz
lvec_[3,2] = (izmax - izmin)*dz

print("izmin,izmax ", izmin,izmax)

Ecut = E[izmin:izmax,:,:]

print(Ecut.shape)

# Density Overlap Model
GU.saveXSF( "E"+namestr+"_mini.xsf", Ecut*(PQ*-1.0), lvec_, head=head1 )

#np.fft.fft(  )

np.save( "E"+namestr+"_mini.npy", Ecut*(PQ*-1.0) )
np.savez_compressed( "E"+namestr+"_mini.npz", Ecut*(PQ*-1.0) )

#GU.saveXSF( "FF"+namestr+"_x.xsf", Fx*PQ,       lvec1, head=head1 )
#GU.saveXSF( "FF"+namestr+"_y.xsf", Fy*PQ,       lvec1, head=head1 )
#GU.saveXSF( "FF"+namestr+"_z.xsf", Fz*PQ,       lvec1, head=head1 )
#Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)






