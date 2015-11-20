#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import basUtils
import elements
import GridUtils as GU
import ProbeParticle      as PP;    PPU = PP.PPU;

#import PPPlot 		# we do not want to make it dempendent on matplotlib

print "Amplitude ", PPU.params['Amplitude']

# =============== arguments definition

from optparse import OptionParser
parser = OptionParser()
parser.add_option( "-k",       action="store", type="float", help="tip stiffenss [N/m]" )
parser.add_option( "--krange", action="store", type="float", help="tip stiffenss range (min,max,n) [N/m]", nargs=3)
parser.add_option( "-q",       action="store", type="float", help="tip charge [e]" )
parser.add_option( "--qrange", action="store", type="float", help="tip charge range (min,max,n) [e]", nargs=3)
parser.add_option( "-a",       action="store", type="float", help="oscilation amplitude [A]" )
parser.add_option( "--arange", action="store", type="float", help="oscilation amplitude range (min,max,n) [A]", nargs=3)

#parser.add_option( "--img",    action="store_true", default=False, help="save images for dfz " )
parser.add_option( "--df" ,    action="store_true", default=False, help="save frequency shift as df.xsf " )
parser.add_option( "--pos",    action="store_true", default=False, help="save probe particle positions" )

(options, args) = parser.parse_args()
opt_dict = vars(options)

# =============== Setup

# dgdfgdfg

print " >> OVEWRITING SETTINGS by params.ini  "
PPU.loadParams( 'params.ini' )

print " >> OVEWRITING SETTINGS by command line arguments  "
print opt_dict
# Ks
if opt_dict['krange'] is not None:
	Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], opt_dict['krange'][2] )
elif opt_dict['krange'] is not None:
	Ks = [ opt_dict['k'] ]
else:
	Ks = [ PPU.params['stiffness'][0] ]
# Qs
if opt_dict['qrange'] is not None:
	Qs = np.linspace( opt_dict['qrange'][0], opt_dict['qrange'][1], opt_dict['qrange'][2] )
elif opt_dict['q'] is not None:
	Qs = [ opt_dict['q'] ]
else:
	Qs = [ PPU.params['charge'] ]
# Amps
if opt_dict['arange'] is not None:
	Amps = np.linspace( opt_dict['arange'][0], opt_dict['arange'][1], opt_dict['arange'][2] )
elif opt_dict['a'] is not None:
	Amps = [ opt_dict['a'] ]
else:
	Amps = [ PPU.params['Amplitude'] ]

print "Ks   =", Ks 
print "Qs   =", Qs 
print "Amps =", Amps 

print " ============= RUN  "

#PPPlot.params = PPU.params 			# now we dont use PPPlot here

print " load Electrostatic Force-field "
FFel, lvec, nDim, head = GU.loadVecFieldXsf( "FFel" )
print " load Lenard-Jones Force-field "
FFLJ, lvec, nDim, head = GU.loadVecFieldXsf( "FFLJ" )
PPU.lvec2params( lvec )
PP.setFF( FFel )

xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )

for iq,Q in enumerate( Qs ):
	FF = FFLJ + FFel * Q
	PP.setFF_Pointer( FF )
	for ik,K in enumerate( Ks ):
		dirname = "Q%1.2fK%1.2f" %(Q,K)
		print " relaxed_scan for ", dirname
		if not os.path.exists( dirname ):
			os.makedirs( dirname )
		PP.setTip( kSpring = np.array((K,K,0.0))/-PPU.eVA_Nm )
		fzs,PPpos = PP.relaxedScan3D( xTips, yTips, zTips )
		GU.saveXSF( dirname+'/OutFz.xsf', fzs, lvecScan, GU.XSF_HEAD_DEFAULT )
		if opt_dict['pos']:
			GU.saveVecFieldXsf( dirname+'/PPpos', PPpos, lvec, GU.XSF_HEAD_DEFAULT )
		# the rest is done in plot_results.py
		'''
		if opt_dict['df'] or opt_dict['img']:
			for iA,Amp in enumerate( Amps ):
				AmpStr = "/Amp%2.2f" %Amp
				print "Amp= ",AmpStr
				dirNameAmp = dirname+AmpStr
				if not os.path.exists( dirNameAmp ):
					os.makedirs( dirNameAmp )
				dz  = PPU.params['scanStep'][2]
				dfs = PPU.Fz2df( fzs, dz = dz, k0 = PPU.params['kCantilever'], f0=PPU.params['f0Cantilever'], n=Amp/dz )
				if opt_dict['']:
					GU.saveXSF( dirNameAmp+'/df.xsf', dfs, lvecScan, GU.XSF_HEAD_DEFAULT )
				if opt_dict['img']:
					extent=( xTips[0], xTips[-1], yTips[0], yTips[-1] )
					PPPlot.plotImages( dirNameAmp+"/df", dfs, slices = range( 0, len(dfs) ), extent=extent )
		'''


print " ***** ALL DONE ***** "

#plt.show()




