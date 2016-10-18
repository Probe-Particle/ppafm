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
	'''
	getXYZ( nDim, cell ):
	X,Y,Z - output: three dimensional arrays with x, y, z coordinates as value
	'''
	dcell = np.array( [ cell[0]/nDim[2], cell[1]/nDim[1], cell[2]/nDim[0] ]  )
	print " dcell ", dcell 
	CBA = np.mgrid[0:nDim[0],0:nDim[1],0:nDim[2]].astype(float) # grid going: CBA[z,x,y]
	X = CBA[2]*dcell[0, 0] + CBA[1]*dcell[1, 0] + CBA[0]*dcell[2, 0]
	Y = CBA[2]*dcell[0, 1] + CBA[1]*dcell[1, 1] + CBA[0]*dcell[2, 1]
	Z = CBA[2]*dcell[0, 2] + CBA[1]*dcell[1, 2] + CBA[0]*dcell[2, 2]
	return X,Y,Z

def getProbeDensity( pos,  X, Y, Z, sigma ):
	'''
	getProbeDensity( pos,  X, Y, Z, sigma ):
	pos - position of the tip
	X,Y,Z - input: three dimensional arrays with x, y, z coordinates as value
	sigma - FWHM of the Gaussian function
	'''
	r2      = (X-pos[0])**2 + (Y-pos[1])**2 + (Z-pos[2])**2
	radial  = np.exp( -r2/( sigma**2 ) )
	return radial
	#return (X-pos[0])**2

def getProbeTunelling( pos,  X, Y, Z, beta=1.0 ):
	'''
	getProbeTunelling( pos,  X, Y, Z, sigma ):
	pos - position of the tip
	X,Y,Z - input: three dimensional arrays with x, y, z coordinates as value
	beta - decay in eV/Angstrom
	'''
	r2      = (X-pos[0])**2 + (Y-pos[1])**2 + (Z-pos[2])**2
	radial  = np.exp( -beta * np.sqrt(r2) )
	return radial

def limitE( E, E_cutoff ):
	'''
	limitE( E, E_cutoff ):
	exclude too high or infinite energies
	'''
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


# =============== main

if options.noProbab :
	print " ==== calculating probabilties ===="
	V_tip,   lvec, nDim, head = GU.loadXSF('tip/VLJ.xsf')
	#cell   = np.array( [ lvec[1],lvec[2],lvec[3] ] ); print "nDim ", nDim, "\ncell ", cell
	#X,Y,Z  = getXYZ( nDim, cell )
	#V_tip = V_tip*0 + Egauss * getProbeDensity( (cell[0,0]/2.+cell[1,0]/2.,cell[1,1]/2,cell[2,2]/2.-3.8), X, Y, Z, wGauss ) # works for tip (the last flexible tip apex atom) in the middle of the cell
	limitE( V_tip,  E_cutoff ) 
	W_tip  = np.exp( -beta * V_tip  )
	#W_tip = W_cut(W_tip,nz=95,side='down',sm=5)
	del V_tip;
	GU.saveXSF ( 'W_tip.xsf',  W_tip,    lvec)#, echo=True )


	# --- sample
	V_surf,  lvec, nDim, head = GU.loadXSF('sample/VLJ.xsf')
	limitE( V_surf, E_cutoff ) 
	W_surf = np.exp( -beta * V_surf )
	#W_surf=W_cut(W_surf,nz=50,side='up',sm=1)
	del V_surf; 
	GU.saveXSF ( 'W_surf.xsf', W_surf,   lvec)#, echo=True )

#=================== Force

if options.noForces :
	print " ==== calculating Forces ====" 
	if (options.noProbab==False) :
		print " ==== loading probabilties ====" 
		# --- tip
		W_tip,   lvec, nDim, head = GU.loadXSF('W_tip.xsf')
		# --- sample
		W_surf,  lvec, nDim, head = GU.loadXSF('W_surf.xsf')

	W_tip = np.roll(np.roll(np.roll(W_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell

	# Fz:
	Fz_tmp, lvec, nDim, head = GU.loadXSF('tip/FFLJ_z.xsf')
	Fz_tip = np.roll(np.roll(np.roll(Fz_tmp.copy(),nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
	del Fz_tmp;

	F1=fFFT.Average_tip( Fz_tip , W_surf, W_tip  )
	GU.saveXSF        ( 'FFboltz_z.xsf', F1, lvec)#, echo=True )
	del F1; del Fz_tip;

	# Fx:
	Fx_tmp, lvec, nDim, head = GU.loadXSF('tip/FFLJ_x.xsf')
	Fx_tip = np.roll(np.roll(np.roll(Fx_tmp.copy(),nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
	del Fx_tmp;

	F1=fFFT.Average_tip( Fx_tip , W_surf, W_tip  )
	GU.saveXSF        ( 'FFboltz_x.xsf', F1, lvec)#, echo=True )
	del F1; del Fx_tip

	# Fy:
	Fy_tmp, lvec, nDim, head = GU.loadXSF('tip/FFLJ_y.xsf')
	Fy_tip = np.roll(np.roll(np.roll(Fy_tmp.copy(),nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
	del Fy_tmp;
	F1=fFFT.Average_tip( Fy_tip , W_surf, W_tip  )
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
	if ((options.noProbab==False)and(options.noForces==False)) :
		print " ==== loading probabilties ====" 
		# --- tip
		W_tip,   lvec, nDim, head = GU.loadXSF('W_tip.xsf')
		W_tip = np.roll(np.roll(np.roll(W_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
		# --- sample
		W_surf,  lvec, nDim, head = GU.loadXSF('W_surf.xsf')

	if ((options.noProbab)and(options.noForces==False)) :
		W_tip = np.roll(np.roll(np.roll(W_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell

	print " ==== calculating current ====" 
	cell   = np.array( [ lvec[1],lvec[2],lvec[3] ] ); print "nDim ", nDim, "\ncell ", cell
	X,Y,Z  = getXYZ( nDim, cell )
	T_tip = getProbeTunelling( (cell[0,0]/2.+cell[1,0]/2.,cell[1,1]/2,cell[2,2]/2.) ,  X, Y, Z, beta=1.14557 )  #beta decay in eV/Angstom for WF = 5.0 eV;  works for tip (the last flexible tip apex atom) in the middle of the cell
	T_tip = np.roll(np.roll(np.roll(T_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2)
	T=fFFT.Average_tip( (-1)*T_tip, W_surf, W_tip )                                             # T stands for hoppings
	del T_tip;
	GU.saveXSF        ( 'I_boltzmann.xsf', T**2 , lvec)#, echo=True ) # I ~ T**2 

print " ***** ALL DONE ***** "
