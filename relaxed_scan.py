#!/usr/bin/python -u

import os
import numpy as np
#import matplotlib.pyplot as plt
import sys

'''
import basUtils
import elements
import GridUtils as GU
import ProbeParticle      as PP;    PPU = PP.PPU;
'''

import pyProbeParticle                as PPU     
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.core           as PPC
import pyProbeParticle.HighLevel      as PPH

#import PPPlot 		# we do not want to make it dempendent on matplotlib

print "Amplitude ", PPU.params['Amplitude']

# =============== arguments definition

from optparse import OptionParser
parser = OptionParser()
parser.add_option( "-k",       action="store", type="float", help="tip stiffenss [N/m]" )
parser.add_option( "--krange", action="store", type="float", help="tip stiffenss range (min,max,n) [N/m]", nargs=3)
parser.add_option( "-q",       action="store", type="float", help="tip charge [e]" )
parser.add_option( "--qrange", action="store", type="float", help="tip charge range (min,max,n) [e]", nargs=3)
#parser.add_option( "-a",       action="store", type="float", help="oscilation amplitude [A]" )
#parser.add_option( "--arange", action="store", type="float", help="oscilation amplitude range (min,max,n) [A]", nargs=3)
#parser.add_option( "--img",    action="store_true", default=False, help="save images for dfz " )
#parser.add_option( "--df" ,    action="store_true", default=False, help="save frequency shift as df.xsf " )
parser.add_option( "-b", "--boltzmann" ,action="store_true", default=False, help="calculate forces with boltzmann particle" )

parser.add_option( "--pos",       action="store_true", default=False, help="save probe particle positions" )
parser.add_option( "--disp",      action="store_true", default=False, help="save probe particle displacements")
parser.add_option( "--tipspline", action="store", type="string", help="file where spline is stored", default=None )

(options, args) = parser.parse_args()
opt_dict = vars(options)

# =============== Setup

# dgdfgdfg

PPU.loadParams( 'params.ini' )

print opt_dict
# Ks
if opt_dict['krange'] is not None:
	Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], opt_dict['krange'][2] )
elif opt_dict['k'] is not None:
	Ks = [ opt_dict['k'] ]
else:
	Ks = [ PPU.params['stiffness'][0] ]
# Qs

charged_system=False
if opt_dict['qrange'] is not None:
	Qs = np.linspace( opt_dict['qrange'][0], opt_dict['qrange'][1], opt_dict['qrange'][2] )
elif opt_dict['q'] is not None:
	Qs = [ opt_dict['q'] ]
else:
	Qs = [ PPU.params['charge'] ]

for iq,Q in enumerate(Qs):
	if ( abs(Q) > 1e-7):
		charged_system=True

if options.tipspline is not None :
	try:
		S = np.genfromtxt(options.tipspline )
		print " loading tip spline from "+options.tipspline
		xs   = S[:,0].copy();  print "xs: ",   xs
		ydys = S[:,1:].copy(); print "ydys: ", ydys
		PPC.setTipSpline( xs, ydys )
		#Ks   = [0.0]
	except:
		print "cannot load tip spline from "+options.tipspline
		sys.exit()
	
# Amps
#if opt_dict['arange'] is not None:
#	Amps = np.linspace( opt_dict['arange'][0], opt_dict['arange'][1], opt_dict['arange'][2] )
#elif opt_dict['a'] is not None:
#	Amps = [ opt_dict['a'] ]
#else:
#	Amps = [ PPU.params['Amplitude'] ]

print "Ks   =", Ks 
print "Qs   =", Qs 
#print "Amps =", Amps 

print " ============= RUN  "

#PPPlot.params = PPU.params 			# now we dont use PPPlot here
if ( charged_system == True):
        print " load Electrostatic Force-field "
        FFel, lvec, nDim, head = GU.loadVecFieldXsf( "FFel" )

if options.boltzmann :
        print " load Boltzmann Force-field "
        FFboltz, lvec, nDim, head = GU.loadVecFieldXsf( "FFboltz" )


print " load Lenard-Jones Force-field "
FFLJ, lvec, nDim, head = GU.loadVecFieldXsf( "FFLJ" )
PPU.lvec2params( lvec )
PPC.setFF( FFLJ )

xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )

for iq,Q in enumerate( Qs ):
	if ( charged_system == True):
		FF = FFLJ + FFel * Q
	else:
		FF = FFLJ
	if options.boltzmann :
		FF += FFboltz
	PPC.setFF_Fpointer( FF )
	for ik,K in enumerate( Ks ):
		dirname = "Q%1.2fK%1.2f" %(Q,K)
		print " relaxed_scan for ", dirname
		if not os.path.exists( dirname ):
			os.makedirs( dirname )
		PPC.setTip( kSpring = np.array((K,K,0.0))/-PPU.eVA_Nm )
		fzs,PPpos = PPH.relaxedScan3D( xTips, yTips, zTips )
		GU.saveXSF( dirname+'/OutFz.xsf', fzs, lvecScan, GU.XSF_HEAD_DEFAULT )
		#print "SHAPE", PPpos.shape, xTips.shape, yTips.shape, zTips.shape
		if opt_dict['disp']:
			PPdisp=PPpos.copy()
			nx=PPdisp.shape[2]
			ny=PPdisp.shape[1]
			nz=PPdisp.shape[0]
			test=np.meshgrid(xTips,yTips,zTips)
			#print "TEST SHAPE", np.array(test).shape
			#print nx,ny,nz
			i=0
			while i<nx:
				j=0
				while j<ny:
				    k=0
				    while k<nz:
				        PPdisp[k][j][i]-=np.array([xTips[i],xTips[j],zTips[k]])+ np.array([PPU.params['r0Probe'][0],PPU.params['r0Probe'][1],-PPU.params['r0Probe'][2]])
				        k+=1
				    j+=1
				i+=1
			GU.saveVecFieldXsf( dirname+'/PPdisp', PPdisp, lvec, head )
		if opt_dict['pos']:
			GU.saveVecFieldXsf( dirname+'/PPpos', PPpos, lvec, head )
		# the rest is done in plot_results.py; For df, go to plot_results.py
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
				if opt_dict['df']:
					GU.saveXSF( dirNameAmp+'/df.xsf', dfs, lvecScan, head )
				
				if opt_dict['img']:
					extent=( xTips[0], xTips[-1], yTips[0], yTips[-1] )
					PPPlot.plotImages( dirNameAmp+"/df", dfs, slices = range( 0, len(dfs) ), extent=extent )
				
		'''

print " ***** ALL DONE ***** "

#plt.show()
