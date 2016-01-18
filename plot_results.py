#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

'''
import basUtils
import elements
import GridUtils as GU
import ProbeParticleUtils as PPU
import PPPlot
'''

import pyProbeParticle                as PPU     
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.PPPlot         as PPPlot
from   pyProbeParticle            import basUtils
from   pyProbeParticle            import elements 
#import pyProbeParticle.core           as PPC
#import pyProbeParticle.HighLevel      as PPH

# =============== arguments definition

from optparse import OptionParser
parser = OptionParser()
parser.add_option( "-k",       action="store", type="float", help="tip stiffenss [N/m]" )
parser.add_option( "--krange", action="store", type="float", help="tip stiffenss range (min,max,n) [N/m]", nargs=3)
parser.add_option( "-q",       action="store", type="float", help="tip charge [e]" )
parser.add_option( "--qrange", action="store", type="float", help="tip charge range (min,max,n) [e]", nargs=3)
parser.add_option( "-a",       action="store", type="float", help="oscilation amplitude [A]" )
parser.add_option( "--arange", action="store", type="float", help="oscilation amplitude range (min,max,n) [A]", nargs=3)

parser.add_option( "--df",       action="store_true", default=False,  help="plot images for dfz " )
parser.add_option( "--save_df" , action="store_true", default=False, help="save frequency shift as df.xsf " )
parser.add_option( "--pos",      action="store_true", default=False, help="save probe particle positions" )
parser.add_option( "--atoms",    action="store_true", default=False, help="plot atoms to images" )
parser.add_option( "--bonds",    action="store_true", default=False, help="plot bonds to images" )
parser.add_option( "--cbar",     action="store_true", default=False, help="plot bonds to images" )

(options, args) = parser.parse_args()
opt_dict = vars(options)
print "opt_dict: "
print opt_dict

# =============== Setup

# dgdfgdfg

print " >> OVEWRITING SETTINGS by params.ini  "
PPU.loadParams( 'params.ini' )

#PPPlot.params = PPU.params

print " >> OVEWRITING SETTINGS by command line arguments  "
# Ks
if opt_dict['krange'] is not None:
	Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], opt_dict['krange'][2] )
elif opt_dict['k'] is not None:
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

#sys.exit("  STOPPED ")

print " ============= RUN  "

dz  = PPU.params['scanStep'][2]
xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )
extent = ( xTips[0], xTips[-1], yTips[0], yTips[-1] )

atoms_str=""
atoms = None
bonds = None
if opt_dict['atoms'] or opt_dict['bonds']:
	atoms_str="_atoms"
	atoms = basUtils.loadAtoms( 'input_plot.xyz' )
	print "atoms ", atoms
	atom_colors = basUtils.getAtomColors( atoms )
	atoms[ 4 ] = atom_colors
	#print "atom_colors: ", atom_colors
if opt_dict['bonds']:
	bonds = basUtils.findBonds( atoms, 1.0 )
	#print "bonds ", bonds
atomSize = 0.15

cbar_str =""
if opt_dict['cbar']:
	cbar_str="_cbar"

for iq,Q in enumerate( Qs ):
	for ik,K in enumerate( Ks ):
		dirname = "Q%1.2fK%1.2f" %(Q,K)
		if opt_dict['pos']:
			try:
				PPpos, lvec, nDim, head = GU.loadVecFieldXsf( dirname+'/PPpos' )
				print " plotting PPpos : "
				PPPlot.plotDistortions( dirname+"/xy"+atoms_str+cbar_str, PPpos[:,:,:,0], PPpos[:,:,:,1], slices = range( 0, len(PPpos) ), BG=PPpos[:,:,:,2], extent=extent, atoms=atoms, bonds=bonds, atomSize=atomSize, markersize=2.0, cbar=opt_dict['cbar'] )
				del PPpos
			except:
				print "error: ", sys.exc_info()
				print "cannot load : " + ( dirname+'/PPpos_?.xsf' ) 
		if ( ( opt_dict['df'] or opt_dict['save_df'] ) ):
			try :
				fzs, lvec, nDim, head = GU.loadXSF( dirname+'/OutFz.xsf' )
				for iA,Amp in enumerate( Amps ):
					AmpStr = "/Amp%2.2f" %Amp
					print "Amp= ",AmpStr
					dirNameAmp = dirname+AmpStr
					if not os.path.exists( dirNameAmp ):
						os.makedirs( dirNameAmp )
					dfs = PPU.Fz2df( fzs, dz = dz, k0 = PPU.params['kCantilever'], f0=PPU.params['f0Cantilever'], n=Amp/dz )
					if opt_dict['save_df']:
						GU.saveXSF( dirNameAmp+'/df.xsf', PPpos, lvec, GU.XSF_HEAD_DEFAULT )
					if opt_dict['df']:
						print " plotting df : "
						PPPlot.plotImages( dirNameAmp+"/df"+atoms_str+cbar_str, dfs, slices = range( 0, len(dfs) ), extent=extent, atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar'] )
					del dfs
				del fzs
			except:
				print "error: ", sys.exc_info()
				print "cannot load : ", dirname+'/OutFz.xsf'
		
print " ***** ALL DONE ***** "

#plt.show()



