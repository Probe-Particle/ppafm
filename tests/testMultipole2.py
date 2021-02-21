#!/usr/bin/python

# TODO === remains not converted

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
import pyProbeParticle.basUtils
import pyProbeParticle.elements
import pyProbeParticle.GridUtils     as GU
import pyProbeParticle.Multipoles    as MP
import pyProbeParticle.common        as PP

# ============== recompile & load C++ libraries

def makeclean( ):
	LIB_PATH = os.path.dirname( os.path.realpath(__file__) )
	print(" ProbeParticle Library DIR = ", LIB_PATH)
	CWD=os.getcwd()
	os.chdir(LIB_PATH)
	os.system("make clean")
	os.chdir(CWD)

makeclean( )

# ============== Setup

WORK_DIR =  '/home/prokop/Desktop/Probe_Particle_Simulations/Multipoles/COCu4/'
# NOTE: Data for COCu4 tip example are on tarkil  /auto/praha1/prokop/STHM/vasp/COCu4

# ============== load reference grid

V, lvec, nDim, head = GU.loadXSF( WORK_DIR + 'LOCPOT.xsf' )

cell = np.array( [ lvec[1], lvec[2], lvec[3] ]); 

MP.setGrid( V, cell );

# ============== prepare atoms

atom_types,atom_pos = GU.getFromHead_PRIMCOORD( head )   # load atoms from header of xsf file

# set sample region around atom atom_Rmin, atom_Rmax
spacies              = PP.loadSpecies( './defaults/atomtypes.ini' )
R_type               = spacies[:,0]
atom_Rmin, atom_Rmax = MP.make_Ratoms( atom_types, R_type ) 

# mask atoms which should not to be included into the expansion 
natoms          = len( atom_types )
atom_mask       = np.array( [ True ] * natoms ); 
atom_mask[ 2: ] = False

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
	print(basis_assignment[i], coefs[i])

print("saving LOCPOT_debug.xsf ")
GU.saveXSF( WORK_DIR + 'LOCPOT_debug.xsf', V, lvec, head );


