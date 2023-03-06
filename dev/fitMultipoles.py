#!/usr/bin/python

# TODO === remains not converted

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

#import elements

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
import ppafm as PPU

#import ppafm.ProbeParticle as PP
import ppafm.cpp_utils as cpp_utils
import ppafm.Multipoles as MP
from ppafm.io import getFromHead_PRIMCOORD, loadXSF, saveXSF

# ---- Load potential

V, lvec, nDim, head = loadXSF( 'LOCPOT.xsf' )

cell = np.array( [ lvec[1], lvec[2], lvec[3] ]);
print(V.flags)
#MP.setGrid( V, cell );
MP.setGrid_shape  ( V.shape, cell )
MP.setGrid_Pointer( V )

# ---- prepare atoms

atom_types,atom_pos = getFromHead_PRIMCOORD( head )   # load atoms from header of xsf file

# set sample region around atom atom_Rmin, atom_Rmax
if os.path.isfile( 'atomtypes.ini' ):
    print(">> LOADING LOCAL atomtypes.ini")
    FFparams=PPU.loadSpecies( 'atomtypes.ini' )
else:
    FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )

#print FFparams
atom_Rmin, atom_Rmax = MP.make_Ratoms( atom_types, np.array([ p[0] for p in FFparams]),   fmax = 3.0   )

# mask atoms which should not to be included into the expansion
natoms          = len( atom_types )
atom_mask       = np.array( [ True ] * natoms );
#atom_mask[ 2: ] = False     #    first two atoms

# ---- do the fitting

sampled_val, sampled_pos = MP.sampleGridArroundAtoms( atom_pos, atom_Rmin, atom_Rmax, atom_mask, pbc=False, show_where=True )
#saveXSF( 'LOCPOT_sample.xsf', V, lvec, head );

centers = atom_pos[atom_mask].copy(); print("centers", centers)
types   = np.array([3]*len(centers)).astype(np.uint32); print("types", types)

B,BB = MP.buildLinearSystemMultipole(  sampled_pos, sampled_val, centers, types ); print("B=",B); print("BB=",BB);

coefs = np.linalg.solve( BB, B ); print("coefs:", coefs)
#coefs = MP.regularizedLinearFit( B, BB, nMaxSteps=100 ); print "coefs:", coefs

#exit()
#raw_input("Press Enter to continue...")


MP.evalMultipoleComb( coefs, V )
saveXSF( 'LOCPOT_eval.xsf', V, lvec, head )

# ============== output results

#for i in range( len( coefs ) ):
#	print basis_assignment[i], coefs[i]
