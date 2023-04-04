#!/usr/bin/python

import os

import matplotlib.pyplot as plt
import numpy as np

print(" # ========== make & load  ProbeParticle C++ library ")

import ppafm as PP
from ppafm import PPPlot, elements, io

print(" ============= RUN  ")

print(" >> WARNING!!! OVEWRITING SETTINGS by params.ini  ")

PP.loadParams( 'params.ini' )

PPPlot.params = PP.params

print(" load Electrostatic Force-field ")
FFel, lvec, nDim, head = io.loadVecFieldXsf( "FFel" )
print(" load Lenard-Jones Force-field ")
FFLJ, lvec, nDim, head = io.loadVecFieldXsf( "FFLJ" )
PP.lvec2params( lvec )
PP.setFF( FFel )

xTips,yTips,zTips,lvecScan = PP.prepareScanGrids( )

#Ks   = [ 0.25, 0.5, 1.0 ]
#Qs   = [ -0.2, 0.0, +0.2 ]
#Amps = [ 2.0 ]

Ks   = [  0.5 ]
Qs   = [  0.0 ]
Amps = [  1.0 ]

for iq,Q in enumerate( Qs ):
	FF = FFLJ + FFel * Q
	PP.setFF_Pointer( FF )
	for ik,K in enumerate( Ks ):
		dirname = "Q%1.2fK%1.2f" %(Q,K)
		os.makedirs( dirname )
		PP.setTip( kSpring = np.array((K,K,0.0))/-PP.eVA_Nm )
		fzs = PP.relaxedScan3D( xTips, yTips, zTips )
		io.saveXSF( dirname+'/OutFz.xsf', fzs, lvecScan, io.XSF_HEAD_DEFAULT )
		for iA,Amp in enumerate( Amps ):
			AmpStr = "/Amp%2.2f" %Amp
			print("Amp= ",AmpStr)
			os.makedirs( dirname+AmpStr )
			dz  = PP.params['scanStep'][2]
			dfs = PP.Fz2df( fzs, dz = dz, k0 = PP.params['kCantilever'], f0=PP.params['f0Cantilever'], n=Amp/dz )
			extent=( xTips[0], xTips[-1], yTips[0], yTips[-1] )
			PPPlot.plotImages( dirname+AmpStr+"/df", dfs, slices = list(range( 0, len(dfs))), extent=extent )

print(" ***** ALL DONE ***** ")

#plt.show()
