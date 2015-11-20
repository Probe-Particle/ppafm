#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import basUtils
import elements
import GridUtils as GU
import ProbeParticle as PP
import PPPlot

# =============== arguments definition

from optparse import OptionParser
parser = OptionParser()
parser.add_option( "-k",       action="store", type="float", help="tip stiffenss [N/m]" )
parser.add_option( "--krange", action="store", type="float", help="tip stiffenss range (min,max,n) [N/m]", nargs=3)
parser.add_option( "-q",       action="store", type="float", help="tip charge [e]" )
parser.add_option( "--qrange", action="store", type="float", help="tip charge range (min,max,n) [e]", nargs=3)
parser.add_option( "-a",       action="store", type="float", help="oscilation amplitude [A]" )
parser.add_option( "--arange", action="store", type="float", help="oscilation amplitude range (min,max,n) [A]", nargs=3)

parser.add_option( "--img",    action="store_true", default=False, help="save images for dfz " )
parser.add_option( "--df" ,    action="store_true", default=False, help="save frequency shift as df.xsf " )
parser.add_option( "--dxy",    action="store_true", default=False, help="save lateral deflection as dy.xsf dx.xsf" )

(options, args) = parser.parse_args()
opt_dict = vars(options)

# =============== Setup

print " >> OVEWRITING SETTINGS by params.ini  "
PP.loadParams( 'params.ini' )

print " >> OVEWRITING SETTINGS by command line arguments  "
print opt_dict
# Ks
if opt_dict['krange'] is not None:
	Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], opt_dict['krange'][2] )
elif opt_dict['krange'] is not None:
	Ks = [ opt_dict['k'] ]
else:
	Ks = [ PP.params['stiffness'][0] ]
# Qs
if opt_dict['qrange'] is not None:
	Qs = np.linspace( opt_dict['qrange'][0], opt_dict['qrange'][1], opt_dict['qrange'][2] )
elif opt_dict['q'] is not None:
	Qs = [ opt_dict['q'] ]
else:
	Qs = [ PP.params['charge'] ]
# Amps
if opt_dict['arange'] is not None:
	Amps = np.linspace( opt_dict['arange'][0], opt_dict['arange'][1], opt_dict['arange'][2] )
elif opt_dict['a'] is not None:
	Amps = [ opt_dict['a'] ]
else:
	Amps = [ PP.params['Amplitude'] ]

print "Ks   =", Ks 
print "Qs   =", Qs 
print "Amps =", Amps 

#Ks   = [  0.3, 0.5, 0.7 ]
#Qs   = [ -0.2, -0.1, 0.0, 0.1, 0.2]
#Amps = [  1.0 ]
#Ks   = [ 0.5 ]
#Qs   = [ 0.0 ]
#Amps = [ 1.0 ]

#sys.exit("  STOPPED ")

print " ============= RUN  "

PPPlot.params = PP.params

print " load Electrostatic Force-field "
FFel, lvec, nDim, head = GU.loadVecFieldXsf( "FFel" )
print " load Lenard-Jones Force-field "
FFLJ, lvec, nDim, head = GU.loadVecFieldXsf( "FFLJ" )
PP.lvec2params( lvec )
PP.setFF( FFel )

xTips,yTips,zTips,lvecScan = PP.prepareScanGrids( )

for iq,Q in enumerate( Qs ):
	FF = FFLJ + FFel * Q
	PP.setFF_Pointer( FF )
	for ik,K in enumerate( Ks ):
		dirname = "Q%1.2fK%1.2f" %(Q,K)
		if not os.path.exists( dirname ):
			os.makedirs( dirname )
		PP.setTip( kSpring = np.array((K,K,0.0))/-PP.eVA_Nm )
		#GU.saveVecFieldXsf( 'FFtot', FF, lvec, head )
		fzs,dxs,dys = PP.relaxedScan3D( xTips, yTips, zTips )
		GU.saveXSF( dirname+'/OutFz.xsf', fzs, lvecScan, GU.XSF_HEAD_DEFAULT )
		if opt_dict['dxy']:
			GU.saveXSF( dirname+'/dX.xsf', dxs, lvecScan, GU.XSF_HEAD_DEFAULT )
			GU.saveXSF( dirname+'/dY.xsf', dys, lvecScan, GU.XSF_HEAD_DEFAULT )
		if opt_dict['df'] or opt_dict['img']:
			for iA,Amp in enumerate( Amps ):
				AmpStr = "/Amp%2.2f" %Amp
				print "Amp= ",AmpStr
				dirNameAmp = dirname+AmpStr
				if not os.path.exists( dirNameAmp ):
					os.makedirs( dirNameAmp )
				dz  = PP.params['scanStep'][2]
				dfs = PP.Fz2df( fzs, dz = dz, k0 = PP.params['kCantilever'], f0=PP.params['f0Cantilever'], n=Amp/dz )
				if opt_dict['df']:
					GU.saveXSF( dirNameAmp+'/df.xsf', dfs, lvecScan, GU.XSF_HEAD_DEFAULT )
				if opt_dict['img']:
					extent=( xTips[0], xTips[-1], yTips[0], yTips[-1] )
					PPPlot.plotImages( dirNameAmp+"/df", dfs, slices = range( 0, len(dfs) ), extent=extent )

print " ***** ALL DONE ***** "

#plt.show()



