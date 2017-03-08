#!/usr/bin/python

# TODO === remains not converted

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
#import basUtils
#import elements

from   pyProbeParticle            import basUtils
import pyProbeParticle.GridUtils     as GU
import pyProbeParticle.Multipoles    as MP
#import pyProbeParticle.ProbeParticle as PP
import pyProbeParticle.cpp_utils     as cpp_utils


# ============== Setup

V, lvec, nDim, head = GU.loadXSF( 'LOCPOT.xsf' )

cell = np.array( [ lvec[1], lvec[2], lvec[3] ]); 

#MP.setGrid( V, cell );
MP.setGrid_shape  ( V.shape, cell )
MP.setGrid_Pointer( V )

# ============== prepare atoms

atom_types,atom_pos = GU.getFromHead_PRIMCOORD( head )   # load atoms from header of xsf file

# set sample region around atom atom_Rmin, atom_Rmax
if os.path.isfile( 'atomtypes.ini' ):
    print ">> LOADING LOCAL atomtypes.ini"  
    FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
else:
    FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
atom_Rmin, atom_Rmax = MP.make_Ratoms( atom_types, FFparams[:,0] ) 

# mask atoms which should not to be included into the expansion 
natoms          = len( atom_types )
atom_mask       = np.array( [ True ] * natoms ); 
atom_mask[ 2: ] = False     #    first two atoms

# set basiset for each atom 
atom_basis = MP.make_bas_list( [ len( atom_pos ) ],  basis=[ ['s','px','py','pz'] ] )

#print "atom_pos:   ", atom_pos
#print "atom_Rmin:  ", atom_Rmin
#print "atom_Rmax:  ", atom_Rmax
#print "atom_mask:  ", atom_mask
#print "atom_basis: ", atom_basis

# ============== do the fitting

coefs, basis_assignment   =   MP.fitMultipolesPotential( atom_pos, atom_basis, atom_Rmin, atom_Rmax, atom_mask=atom_mask, show_where=True );

# ============== output results

for i in range( len( coefs ) ):
	print basis_assignment[i], coefs[i]

#print "saving LOCPOT_debug.xsf "
#GU.saveXSF( 'LOCPOT_debug.xsf', V, lvec, head );


