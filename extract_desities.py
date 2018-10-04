#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys

import numpy as np
import matplotlib.pyplot as plt   
import pyProbeParticle               as PPU
import pyProbeParticle.GridUtils     as GU
import pyProbeParticle.basUtils      as BU
from   optparse import OptionParser

parser = OptionParser()
parser.add_option( "-i",   action="store", type="string", help="input file",                   default='CHGCAR.xsf'        )
(options, args) = parser.parse_args()

fname, fext = os.path.splitext( options.i ); fext = fext[1:]

F,lvec,nDim=GU.load_scal_field(fname,data_format=fext)

if( fext == 'cube' ):
	F /= GU.Hartree2eV

zs = np.linspace( 0, lvec[3,2], nDim[0] )
print lvec

atoms,nDim,lvec     = BU.loadGeometry( options.i, params=PPU.params )

print lvec
print atoms
print zs

GU.lib.setGridN   ( np.array( nDim[::-1], dtype=np.int32 )   )
GU.lib.setGridCell( np.array( lvec[1:],   dtype=np.float64 ) )

dlines = [ zs, ]

'''
ipivot = 0
print "============="
print atoms[0][ipivot]
x = atoms[1][ipivot]
y = atoms[2][ipivot]
p1   = ( x,y, zs[ 0] )
p2   = ( x,y, zs[-1] )
print p1, p2
vals = GU.interpolateLine( F, p1, p2, sz=nDim[2], cartesian=True )
print vals
dlines.append(vals)
#vals = F[ :, ix, iy ]
'''


for i in range( len(atoms[0]) ):
	x = atoms[1][i]
	y = atoms[2][i]
	p1   = ( x,y, zs[0]  )
	p2   = ( x,y, zs[-1] )
	vals = GU.interpolateLine( F, p1, p2, sz=nDim[2], cartesian=True )
	dlines.append(vals)



np.savetxt( fname+"_zlines.dat", np.transpose( np.array(dlines) ) )
