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

def makeclean( ):
	LIB_PATH = os.path.dirname( os.path.realpath(__file__) )
	print(" ProbeParticle Library DIR = ", LIB_PATH)
	CWD=os.getcwd()
	os.chdir(LIB_PATH)
	os.system("make clean")
	os.chdir(CWD)

makeclean( )

atom_pos = np.array( [ 
[ 0.0, 0.0, 0.0 ],
[ 1.0, 0.0, 0.0 ],
[ 2.0, 0.0, 0.0 ],
[ 3.0, 0.0, 0.0 ],
] );

'''
atom_bas = [ 
[ 's', 'px', 'py', 'pz' ],
[ 's'                   ],
[ 's'                   ],
[ 's', 'dz2'            ],
]
'''

atom_bas = MP.make_bas_list( [ len( atom_pos ) ] )
print("bas_list:", atom_bas)

atom_Rmin = np.array( [ 1.0,   1.0,  1.0,   1.0 ] );
atom_Rmax = np.array( [ 2.0,   2.0,  2.0,   2.0 ] ); 
atom_mask = np.array( [ True, True, True, False ] ); 

V    = np.zeros( (50,50,50) )
cell = np.array([
[ 10.0,  0.0,  0.0 ],
[  0.0, 10.0,  0.0 ],
[  0.0,  0.0, 10.0 ]
]); 

MP.setGrid( V, cell );

sampled_val, sampled_pos = MP.sampleGridArroundAtoms( atom_pos, atom_Rmin, atom_Rmax, atom_mask )









'''
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

