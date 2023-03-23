#!/usr/bin/python

#import matplotlib
#matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

LWD = '/home/prokop/git/ProbeParticleModel/code'

print(" # ========== make & load  ProbeParticle C++ library ")


# sys.path.append( LWD )
print(" sys.path =  ", sys.path)
sys.path = [ LWD ]
print(" sys.path = ", sys.path)

import elements
import ProbeParticle as PP

from ppafm import io

print(" ============= RUN  ")

print(" >> WARNING!!! OVEWRITING SETTINGS by params.ini  ")

#PP.loadParams( 'params_carbox.ini' )

print(" load Electrostatic Force-field ")
FFel_x,lvec,nDim,head=io.loadXSF('FFel_x.xsf')
PP.params['gridA'] = lvec[ 1,:  ].copy()
PP.params['gridB'] = lvec[ 2,:  ].copy()
PP.params['gridC'] = lvec[ 3,:  ].copy()
PP.params['gridN'] = nDim.copy()

print(" compute Lennard-Jones Force-filed ")
atoms     = io.loadAtoms('geom.bas')
if os.path.isfile( 'atomtypes.ini' ):
	print(">> LOADING LOCAL atomtypes.ini")
	FFparams=PPU.loadSpecies( 'atomtypes.ini' )
else:
	FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
iZs,Rs,Qs = parseAtoms( atoms, autogeom = False, PBC = True, FFparams=FFparams )
FFLJ      = PP.computeLJ( iZs, Rs, FFLJ=None, cell=None, autogeom = False, PBC =
                         True, FFparams=FFparams)

print("impose 4fold symmetry on FFLJ ")
FFLJ4 = np.zeros(np.shape( FFLJ ))
FFLJ4[:,:,:,0] = 0.25*( FFLJ[:,:,:,0] - FFLJ[:,:,::-1,0] + FFLJ[:,::-1,:,0] - FFLJ[:,::-1,::-1,0] )
FFLJ4[:,:,:,1] = 0.25*( FFLJ[:,:,:,1] + FFLJ[:,:,::-1,1] - FFLJ[:,::-1,:,1] - FFLJ[:,::-1,::-1,1] )
FFLJ4[:,:,:,2] = 0.25*( FFLJ[:,:,:,2] + FFLJ[:,:,::-1,2] + FFLJ[:,::-1,:,2] + FFLJ[:,::-1,::-1,2] )

print("save FFLJ to .xsf ")
io.saveVecFieldXsf( 'FFLJ', FFLJ4, lvec, head )


print(" ***** ALL DONE ***** ")


#plt.show()
