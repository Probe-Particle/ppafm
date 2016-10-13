#!/usr/bin/python
import sys
import numpy as np
import os
import __main__ as main


import pyProbeParticle                as PPU     
from   pyProbeParticle            import basUtils
from   pyProbeParticle            import elements   
import pyProbeParticle.GridUtils      as GU
#import pyProbeParticle.core          as PPC
import pyProbeParticle.HighLevel      as PPH
import pyProbeParticle.fieldFFT       as fFFT

HELP_MSG="""Use this program in the following way:
%s -i <filename> 

Supported file fromats are:
   * xyz 
""" %os.path.basename(main.__file__)


from optparse import OptionParser

parser = OptionParser()
parser.add_option( "--noProbab", action="store_false",  help="probability False", default=True )
parser.add_option( "--noForces", action="store_false",  help="Forces False"     , default=True )
parser.add_option( "--current" , action="store_true" ,  help="current True"     , default=False)
(options, args) = parser.parse_args()
opt_dict = vars(options)
    
print options

kBoltz = 8.617332478e-5   # [ eV / K ]

# ============= functions

def getXYZ( nDim, cell ):
	dcell = np.array( [ cell[0]/nDim[0], cell[1]/nDim[1], cell[2]/nDim[2] ]  )
	print " dcell ", dcell 
	ABC = np.mgrid[0:nDim[0],0:nDim[1],0:nDim[2]].astype(float)
	#print "ABC[0]", ABC[0]
	#print "ABC[1]", ABC[1]
	#print "ABC[2]", ABC[2]
	#X = ABC[2]*mat[0, 0] + ABC[1]*mat[0, 1] + ABC[0]*mat[0, 2]
	#Y = ABC[2]*mat[1, 0] + ABC[1]*mat[1, 1] + ABC[0]*mat[1, 2]
	#Z = ABC[2]*mat[2, 0] + ABC[1]*mat[2, 1] + ABC[0]*mat[2, 2]
	X = ABC[2]*dcell[0, 0] + ABC[1]*dcell[1, 0] + ABC[0]*dcell[2, 0]
	Y = ABC[2]*dcell[0, 1] + ABC[1]*dcell[1, 1] + ABC[0]*dcell[2, 1]
	Z = ABC[2]*dcell[0, 2] + ABC[1]*dcell[1, 2] + ABC[0]*dcell[2, 2]
	#print "X : \n", X
	#print "Y : \n", Y
	#print "Z : \n", Z
	return X,Y,Z

def getProbeDensity( pos,  X, Y, Z, sigma ):
	r2      = (X-pos[0])**2 + (Y-pos[1])**2 + (Z-pos[2])**2
	radial  = np.exp( -r2/( sigma**2 ) )
	return radial
	#return (X-pos[0])**2

def getProbeTunelling( pos,  X, Y, Z, beta=1.0 ):
	r2      = (X-pos[0])**2 + (Y-pos[1])**2 + (Z-pos[2])**2
	radial  = np.exp( -beta * np.sqrt(r2) )
	return radial

def limitE( E, E_cutoff ):
	Emin    = E.min()
	E      -= Emin
	mask    = ( E > E_cutoff )
	E[mask] = E_cutoff

def W_cut(W,nz=100,side='up',sm=10):
	'''
	W_cut(W,nz=100,side='up',sm=10):
	W - incomming potential
	nz - z segment, where is the turning point of the fermi function
	'up' cuts up than nz
	'down' cuts down than nz
	sm - smearing = width of the fermi function in number of z segments
	'''
	ndim=W.shape
	print ndim
	if (side=='up'):
		for iz in range(ndim[0]):
		    #print iz, 1/(np.exp((-iz+nz)*1.0/sm) + 1)
		    W[iz,:,:] *= 1/(np.exp((-iz+nz)*1.0/sm) + 1)
	if (side=='down'):
		for iz in range(ndim[0]):
		    #print iz, 1/(np.exp((iz-nz)*1.0/sm) + 1)
		    W[iz,:,:] *= 1/(np.exp((iz-nz)*1.0/sm) + 1)
	return W;

# ============== setup 

T = 10.0 # [K]

beta = 1/(kBoltz*T)   # [eV]
print "T= ", T, " [K] => beta ", beta/1000.0, "[meV] " 

#E_cutoff = 32.0 * beta

E_cutoff = 18.0 * beta

wGauss =  2.0
Egauss = -0.01

pos = ( 13.014300307738408, 7.513809785987396, 6.5 )  # postition of atom to tunneling
#pos = ( 0.0, 0.0,0.0 )

# =============== main

if options.noProbab :
	print " ==== calculating probabilties ===="
	V_tip,   lvec, nDim, head = GU.loadXSF('tip/VLJ.xsf')
	#cell   = np.array( [ lvec[1],lvec[2],lvec[3] ] ); print "nDim ", nDim, "\ncell ", cell
	#X,Y,Z  = getXYZ( nDim, cell )
	#E_tip += Egauss * getProbeDensity( pos, X, Y, Z, wGauss )
	#E_tip = E_tip*0 + Egauss * getProbeDensity( pos, X, Y, Z, wGauss )
	limitE( V_tip,  E_cutoff ) 
	W_tmp  = np.exp( -beta * V_tip  )
	W_tip = W_cut(W_tmp,nz=95,side='down',sm=5)
	del V_tip, W_tmp;
	GU.saveXSF ( 'W_tip.xsf',  W_tip,    lvec)#, echo=True )


	# --- sample
	V_surf,  lvec, nDim, head = GU.loadXSF('sample/VLJ.xsf')
	limitE( V_surf, E_cutoff ) 
	W_tmp = np.exp( -beta * V_surf )
	W_surf=W_cut(W_tmp,nz=50,side='up',sm=1)
	del V_surf; 
	GU.saveXSF ( 'W_surf.xsf', W_surf,   lvec)#, echo=True )

#=================== Force

if options.noForces :
	print " ==== calculating Forces ====" 
	if (options.noProbab==False) :
		print " ==== loading probabilties ====" 
		# --- tip
		W_tmp,   lvec, nDim, head = GU.loadXSF('W_tip.xsf')
		W_tip = np.roll(np.roll(np.roll(W_tmp.copy(),nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2)

		#GU.saveXSF        ( 'W_tmp.xsf', W_tip, lvec, echo=True )
		del W_tmp;
	
		# --- sample
		W_surf,  lvec, nDim, head = GU.loadXSF('W_surf.xsf')

	# Fz:
	Fz_tmp, lvec, nDim, head = GU.loadXSF('tip/FFLJ_z.xsf')
	Fz_tip = np.roll(np.roll(np.roll(Fz_tmp.copy(),nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2)
	del Fz_tmp;

	F1=fFFT.Average_tip( Fz_tip, W_surf, W_tip )
	GU.saveXSF        ( 'FFboltz_z.xsf', F1, lvec)#, echo=True )
	del F1; del Fz_tip;

	# Fx:
	Fx_tmp, lvec, nDim, head = GU.loadXSF('tip/FFLJ_x.xsf')
	Fx_tip = np.roll(np.roll(np.roll(Fx_tmp.copy(),nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2)
	del Fx_tmp;

	F1=fFFT.Average_tip( Fx_tip, W_surf, W_tip )
	GU.saveXSF        ( 'FFboltz_x.xsf', F1, lvec)#, echo=True )
	del F1; del Fx_tip

	# Fy:
	Fy_tmp, lvec, nDim, head = GU.loadXSF('tip/FFLJ_y.xsf')
	Fy_tip = np.roll(np.roll(np.roll(Fy_tmp.copy(),nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2)
	del Fy_tmp;
	F1=fFFT.Average_tip( Fy_tip, W_surf, W_tip )
	GU.saveXSF        ( 'FFboltz_y.xsf', F1, lvec)#, echo=True )
	del F1; del Fy_tip;

	print "x,y & z forces for the Boltzmann distribution of moving particle stored"

'''
# surface just for debugging
#Fz_surf, lvec, nDim, head = GU.loadXSF('sample/FFLJ_z.xsf')
#F2=Average_surf( Fz_surf, W_surf, W_tip )
#GU.saveXSF        ( 'Fz_surf.xsf', F2, lvec, echo=True )
'''
#=================== Current


if options.current :
	if ((options.noProbab==False)or(options.noForces==False)) :
		print " ==== loading probabilties ====" 
		# --- tip
		W_tmp,   lvec, nDim, head = GU.loadXSF('W_tip.xsf')
		W_tip = np.roll(np.roll(np.roll(W_tmp.copy(),nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2)

		#GU.saveXSF        ( 'W_tmp.xsf', W_tip, lvec, echo=True )
		del W_tmp;
	
		# --- sample
		W_surf,  lvec, nDim, head = GU.loadXSF('W_surf.xsf')

	print " ==== calculating current ====" 
	cell   = np.array( [ lvec[1],lvec[2],lvec[3] ] ); print "nDim ", nDim, "\ncell ", cell
	X,Y,Z  = getXYZ( nDim, cell )
	I_tip = getProbeTunelling( pos,  X, Y, Z, beta=2.0 )  #beta decay in in eV/Angstom
	I=fFFT.Average_tip( I_tip, W_surf, W_tip )
	del I_tip;
	GU.saveXSF        ( 'I_conv.xsf', I, lvec)#, echo=True )



