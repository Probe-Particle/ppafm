#!/usr/bin/python

#import matplotlib
#matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

LWD = '/home/prokop/git/ProbeParticleModel/code' 

print " # ========== make & load  ProbeParticle C++ library " 


'''
def makeclean( ):
	import os
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".so") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".o") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".pyc") ]

CWD = os.getcwd()
os.chdir(LWD);       print " >> WORKDIR: ", os.getcwd()
makeclean( )
sys.path.insert(0, "./")
import basUtils
import elements
import GridUtils as GU
import ProbeParticle as PP
os.chdir(CWD);  print " >> WORKDIR: ", os.getcwd()
'''

# sys.path.append( LWD )
print " sys.path =  ", sys.path
sys.path = [ LWD ]
print " sys.path = ", sys.path

import basUtils
import elements
import GridUtils as GU
import ProbeParticle as PP

print " ============= RUN  "

print " >> WARNING!!! OVEWRITING SETTINGS by params.ini  "

#PP.loadParams( 'params_carbox.ini' )

print " load Electrostatic Force-field "
FFel_x,lvec,nDim,head=GU.loadXSF('FFel_x.xsf')
PP.params['gridA'] = lvec[ 1,:  ].copy()
PP.params['gridB'] = lvec[ 2,:  ].copy()
PP.params['gridC'] = lvec[ 3,:  ].copy()
PP.params['gridN'] = nDim.copy()

print " compute Lennard-Jones Force-filed "
atoms     = basUtils.loadAtoms('geom.bas', elements.ELEMENT_DICT )
iZs,Rs,Qs = parseAtoms( atoms, autogeom = False, PBC = True )
FFLJ      = PP.computeLJ( iZs, Rs, FFLJ=None, cell=None, autogeom = False, PBC = True )

print "impose 4fold symmetry on FFLJ "
FFLJ4 = np.zeros(np.shape( FFLJ ))
FFLJ4[:,:,:,0] = 0.25*( FFLJ[:,:,:,0] - FFLJ[:,:,::-1,0] + FFLJ[:,::-1,:,0] - FFLJ[:,::-1,::-1,0] )
FFLJ4[:,:,:,1] = 0.25*( FFLJ[:,:,:,1] + FFLJ[:,:,::-1,1] - FFLJ[:,::-1,:,1] - FFLJ[:,::-1,::-1,1] )
FFLJ4[:,:,:,2] = 0.25*( FFLJ[:,:,:,2] + FFLJ[:,:,::-1,2] + FFLJ[:,::-1,:,2] + FFLJ[:,::-1,::-1,2] )

print "save FFLJ to .xsf "
GU.saveVecFieldXsf( 'FFLJ', FFLJ4, lvec, head )


print " ***** ALL DONE ***** "


#plt.show()




