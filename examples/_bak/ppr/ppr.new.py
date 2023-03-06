#!/usr/bin/python

import os
from optparse import OptionParser

import elements
import GridUtils as GU
import matplotlib.pyplot as plt
import numpy as np
import PPPlot
import ProbeParticle as PP

parser = OptionParser()
parser.add_option( "-k",       action="store", type="float", help="k parameter", default=0.5)
parser.add_option( "--krange", action="store", type="float", help="k parameter range", nargs=3)
parser.add_option( "-q",       action="store", type="float", help="charge", default=0.0)
parser.add_option( "--qrange", action="store", type="float", help="charge range", nargs=3)
parser.add_option( "-a",       action="store", type="float", help="amplitude", default=0.0)
parser.add_option( "--arange", action="store", type="float", help="amplitude range", nargs=3)
(options, args) = parser.parse_args()


Ks   = [  0.3, 0.5, 0.7 ]
Qs   = [ -0.2, -0.1, 0.0, 0.1, 0.2]
Amps = [  1.0 ]

print(" ============= RUN  ")

print(" >> WARNING!!! OVEWRITING SETTINGS by params.ini  ")
PP.loadParams( 'params.ini' )

PPPlot.params = PP.params

print(" load Electrostatic Force-field ")
FFel, lvec, nDim, head = GU.loadVecFieldXsf( "FFel" )
print(" load Lenard-Jones Force-field ")
FFLJ, lvec, nDim, head = GU.loadVecFieldXsf( "FFLJ" )
PP.lvec2params( lvec )
PP.setFF( FFel )

xTips,yTips,zTips,lvecScan = PP.prepareScanGrids( )

#Ks   = [ 0.25, 0.5, 1.0 ]
#Qs   = [ -0.2, 0.0, +0.2 ]
#Amps = [ 2.0 ]



for iq,Q in enumerate( Qs ):
	FF = FFLJ + FFel * Q
	PP.setFF_Pointer( FF )
	for ik,K in enumerate( Ks ):
		dirname = "Q%1.2fK%1.2f" %(Q,K)
		os.makedirs( dirname )
		PP.setTip( kSpring = np.array((K,K,0.0))/-PP.eVA_Nm )
		#GU.saveVecFieldXsf( 'FFtot', FF, lvec, head )
		fzs = PP.relaxedScan3D( xTips, yTips, zTips )
		GU.saveXSF( dirname+'/OutFz.xsf', fzs, lvecScan, GU.XSF_HEAD_DEFAULT )
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
