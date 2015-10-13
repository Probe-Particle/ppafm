#!/usr/bin/python

#import matplotlib
#matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import os
import numpy as np
import matplotlib.pyplot as plt
import elements
#import XSFutils
import basUtils


from memory_profiler import profile


print " # ========== make & load C++ library " 

LWD = '/home/prokop/git/ProbeParticleModel/code' 

def makeclean( ):
	import os
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".so") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".o") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".pyc") ]

CWD = os.getcwd()
os.chdir(LWD);       print " >> WORKDIR: ", os.getcwd()
makeclean( )
sys.path.insert(0, ".")
import GridUtils     as GU
import ProbeParticle as PP
os.chdir(CWD);  print " >> WORKDIR: ", os.getcwd()


print " # ========== SETUP " 

print " >> WARNING!!! OVEWRITING SETTINGS by params.ini  "
PP.loadParams( 'params_carbox.ini' )


PP.setTip()

atoms    = basUtils.loadAtoms('carboxylics.bas', elements.ELEMENT_DICT )

xTips, yTips, zTips, lvecScan = PP.prepareGrids( )

print " # ========== Force-Field preparation " 

PP.prepareForceFields(  )

print " # ========== main simulation run " 

#Ks = [ 0.125, 0.25, 0.5, 1.0 ]
#Qs = [ -0.4, -0.3, -0.2, -0.1, 0.0, +0.1, +0.2, +0.3, +0.4 ]
#Amps = [ 2.0 ]

Ks   = [ 0.25, 0.5, 1.0 ]
Qs   = [ -0.2, 0.0, +0.2 ]
Amps = [ 2.0 ]

for iq,Q in enumerate( Qs ):
	FF = FFLJ + FFel * Q
	PP.setFF_Pointer( FF )
	for ik,K in enumerate( Ks ):
		dirname = "Q%1.2fK%1.2f" %(Q,K)
		os.makedirs( dirname )
		PP.setTip( kSpring = np.array((K,K,0.0))/-PP.eVA_Nm )
		fzs = relaxedScan3D( xTips, yTips, zTips )
		PP.saveXSF( dirname+'/OutFz.xsf', headScan, lvecScan, fzs )
		for iA,Amp in enumerate( Amps ):
			AmpStr = "/Amp%2.2f" %Amp
			print "Amp= ",AmpStr
			os.makedirs( dirname+AmpStr )
			dfs = PP.Fz2df( fzs, dz = dz, k0 = PP.params['kCantilever'], f0=PP.params['f0Cantilever'], n=Amp/dz )
			plotImages( dirname+AmpStr+"/df", dfs, slices = range( 0, len(dfs) ) )

print " ***** ALL DONE ***** "

#plt.show()




