#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import basUtils
import elements

def makeclean( ):
	LIB_PATH = os.path.dirname( os.path.realpath(__file__) )
	print " ProbeParticle Library DIR = ", LIB_PATH
	CWD=os.getcwd()
	os.chdir(LIB_PATH)
	os.system("make clean")
	os.chdir(CWD)

makeclean( )

import GridUtils     as GU
import Multipoles    as MP
import ProbeParticle as PP

# ============== LOAD REFFERENCE GRID

V, lvec, nDim, head = GU.loadXSF('/home/prokop/Desktop/Probe_Particle_Simulations/Multipoles/COCu4/LOCPOT.xsf')

cell = np.array( [ lvec[1], lvec[2], lvec[3] ]); 

MP.setGrid( V, cell );

# ============== LOAD ATOMS

atom_types,atom_pos = GU.getFromHead_PRIMCOORD( head )

spacies = PP.loadSpecies( './defaults/atomtypes.ini' )
R_type = spacies[:,0]
atom_Rmin, atom_Rmax = MP.make_Ratoms( atom_types, R_type ) 

natoms = len( atom_types )
atom_mask = np.array( [ True ] * natoms ); 

print "atom_pos:  " , atom_pos
print "atom_Rmin: ", atom_Rmin
print "atom_Rmax: ", atom_Rmax
print "atom_mask: ", atom_mask


sampled_val, sampled_pos = MP.sampleGridArroundAtoms( atom_pos, atom_Rmin, atom_Rmax, atom_mask )




'''

atom_bas = MP.make_bas_list( [ len( atom_pos ) ] )
print "bas_list:", atom_bas


X = sampled_pos[:,0]
Y = sampled_pos[:,1] 
Z = sampled_pos[:,2] 
basis_set,basis_assignment = MP.make_matrix( atom_pos, atom_bas, X, Y, Z, radial_func = None, beta=1.0 )

print "basis_assignment: ", basis_assignment

# M     = np.dot( basis_set, np.transpose(basis_set) )
# coefs = np.linalg.solve( M , sampled_val )
# print "basis_set: ", np.shape( basis_set ), "sampled_val: ", np.shape( sampled_val )

fit_result =  np.linalg.lstsq( np.transpose( basis_set ), sampled_val ) 
coefs = fit_result[0]

for i in range( len( coefs ) ):
	print basis_assignment[i], coefs[i]
'''

