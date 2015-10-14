#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

print " # ========== make & load  ProbeParticle C++ library " 

LWD = '/home/prokop/git/ProbeParticleModel/code' 
sys.path = [ LWD ]

import basUtils
import elements
import GridUtils as GU
import ProbeParticle as PP

print " ============= RUN  "

print " >> WARNING!!! OVEWRITING SETTINGS by params.ini  "

PP.loadParams( 'params.ini' )

print " load Electrostatic Force-field "
FFel, lvec, nDim, head = loadVecFieldXsf( "FFel" )
print " load Lenard-Jones Force-field "
FFLJ, lvec, nDim, head = loadVecFieldXsf( "FFLJ" )
PP.params['gridA'] = lvec[ 1,:  ].copy()
PP.params['gridB'] = lvec[ 2,:  ].copy()
PP.params['gridC'] = lvec[ 3,:  ].copy()
PP.params['gridN'] = nDim.copy()

xTips,yTips,zTips,lvecScan = prepareGrids( )

#Ks   = [ 0.25, 0.5, 1.0 ]
#Qs   = [ -0.2, 0.0, +0.2 ]
#Amps = [ 2.0 ]

Ks   = [  0.5 ]
Qs   = [ -0.1 ]
Amps = [  1.0 ]

def main():
	for iq,Q in enumerate( Qs ):
		FF = FFLJ + FFel * Q
		PP.setFF_Pointer( FF )
		for ik,K in enumerate( Ks ):
			dirname = "Q%1.2fK%1.2f" %(Q,K)
			os.makedirs( dirname )
			PP.setTip( kSpring = np.array((K,K,0.0))/-PP.eVA_Nm )
			fzs = PP.relaxedScan3D( xTips, yTips, zTips )
			PP.saveXSF( dirname+'/OutFz.xsf', headScan, lvecScan, fzs )
			for iA,Amp in enumerate( Amps ):
				AmpStr = "/Amp%2.2f" %Amp
				print "Amp= ",AmpStr
				os.makedirs( dirname+AmpStr )
				dfs = PP.Fz2df( fzs, dz = dz, k0 = PP.params['kCantilever'], f0=PP.params['f0Cantilever'], n=Amp/dz )
				PP.plotImages( dirname+AmpStr+"/df", dfs, slices = range( 0, len(dfs) ) )


print " ***** ALL DONE ***** "

#plt.show()


