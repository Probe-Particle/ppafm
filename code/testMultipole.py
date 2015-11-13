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

import GridUtils  as GU
import Multipoles as MP

atom_pos = np.array( [ 
[ 0.0, 0.0, 0.0 ],
[ 1.0, 0.0, 0.0 ],
[ 2.0, 0.0, 0.0 ],
[ 3.0, 0.0, 0.0 ],
] );

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

MP.sampleGridArroundAtoms( atom_pos, atom_Rmin, atom_Rmax, atom_mask );
