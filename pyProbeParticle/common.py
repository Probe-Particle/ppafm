#!/usr/bin/python

import numpy as np
import os

# ====================== constants

eVA_Nm               =  16.0217657
CoulombConst         = -14.3996448915;

# default parameters of simulation
params={
'PBC': False,
'nPBC' :       np.array( [      1,        1,        1 ] ),
'gridN':       np.array( [ 150,     150,   50   ] ).astype(np.int),
'gridA':       np.array( [ 12.798,  -7.3889,  0.00000 ] ),
'gridB':       np.array( [ 12.798,   7.3889,  0.00000 ] ),
'gridC':       np.array( [      0,        0,      5.0 ] ),
'moleculeShift':  np.array( [  0.0,      0.0,    0.0 ] ),
'probeType':   8,
'charge':      0.00,
'useLJ':True,
'r0Probe'  :  np.array( [ 0.00, 0.00, 4.00] ),
'stiffness':  np.array( [ 0.5,  0.5, 20.00] ),

'scanStep': np.array( [ 0.10, 0.10, 0.05 ] ),
'scanMin': np.array( [   0.0,     0.0,    5.0 ] ),
'scanMax': np.array( [  20.0,    20.0,    8.0 ] ),
'kCantilever'  :  1800.0, 
'f0Cantilever' :  30300.0,
'Amplitude'    :  1.0,
'plotSliceFrom':  16,
'plotSliceTo'  :  22,
'plotSliceBy'  :  1,
'imageInterpolation': 'bicubic',
'colorscale'   : 'gray',

}

# ==============================
# ============================== Pure python functions
# ==============================

def Fz2df( F, dz=0.1, k0 = 1800.0, f0=30300.0, n=4, units=16.0217656 ):
	'''
	conversion of vertical force Fz to frequency shift 
	according to:
	Giessibl, F. J. A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy Appl. Phys. Lett. 78, 123 (2001)
	oscialltion amplitude of cantilever is A = n * dz
	'''
	x  = np.linspace(-1,1,n+1)
	y  = np.sqrt(1-x*x)
	dy =  ( y[1:] - y[:-1] )/(dz*n)
	fpi    = (n-2)**2; prefactor = ( 1 + fpi*(2/np.pi) ) / (fpi+1) # correction for small n
	dFconv = prefactor * np.apply_along_axis( lambda m: np.convolve(m, dy, mode='valid'), axis=0, arr=F )
	return dFconv*units*f0/k0

# ==============================
# ==============================  server interface file I/O
# ==============================

# overide default parameters by parameters read from a file 
def loadParams( fname ):
        print " >> OVERWRITING SETTINGS by "+fname
	fin = open(fname,'r')
	FFparams = []
	for line in fin:
		words=line.split()
		if len(words)>=2:
			key = words[0]
			if key in params:
				val = params[key]
				print key,' is class ', val.__class__
				if   isinstance( val, bool ):
					word=words[1].strip()
					if (word[0]=="T") or (word[0]=="t"):
						params[key] = True
					else:
						params[key] = False
					print key, params[key], ">>",word,"<<"
				elif isinstance( val, float ):
					params[key] = float( words[1] )
					print key, params[key], words[1]
				elif   isinstance( val, int ):
					params[key] = int( words[1] )
					print key, params[key], words[1]
				elif isinstance( val, str ):
					params[key] = words[1]
					print key, params[key], words[1]
				elif isinstance(val, np.ndarray ):
					if val.dtype == np.float:
						params[key] = np.array([ float(words[1]), float(words[2]), float(words[3]) ])
						print key, params[key], words[1], words[2], words[3]
					elif val.dtype == np.int:
						print key
						params[key] = np.array([ int(words[1]), int(words[2]), int(words[3]) ])
						print key, params[key], words[1], words[2], words[3]
	fin.close()


# load atoms species parameters form a file ( currently used to load Lenard-Jones parameters )
def loadSpecies( fname ):
	fin = open(fname,'r')
	FFparams = []
	for line in fin:
		words=line.split()
		if len(words)>=2:
			FFparams.append( ( float(words[0]), float(words[1]) ) )
	fin.close()
	return np.array( FFparams )


def autoGeom( Rs, shiftXY=False, fitCell=False, border=3.0 ):
	'''
	set Force-Filed and Scanning supercell to fit optimally given geometry
	then shifts the geometry in the center of the supercell
	'''
	zmax=max(Rs[2]); 	Rs[2] -= zmax
	print " autoGeom substracted zmax = ",zmax
	xmin=min(Rs[0]); xmax=max(Rs[0])
	ymin=min(Rs[1]); ymax=max(Rs[1])
	if fitCell:
		params[ 'gridA' ][0] = (xmax-xmin) + 2*border
		params[ 'gridA' ][1] = 0
		params[ 'gridB' ][0] = 0
		params[ 'gridB' ][1] = (ymax-ymin) + 2*border
		params[ 'scanMin' ][0] = 0
		params[ 'scanMin' ][1] = 0
		params[ 'scanMax' ][0] = params[ 'gridA' ][0]
		params[ 'scanMax' ][1] = params[ 'gridB' ][1]
		print " autoGeom changed cell to = ", params[ 'scanMax' ]
	if shiftXY:
		dx = -0.5*(xmin+xmax) + 0.5*( params[ 'gridA' ][0] + params[ 'gridB' ][0] ); Rs[0] += dx
		dy = -0.5*(ymin+ymax) + 0.5*( params[ 'gridA' ][1] + params[ 'gridB' ][1] ); Rs[1] += dy;
		print " autoGeom moved geometry by ",dx,dy

def PBCAtoms( Zs, Rs, Qs, avec, bvec, na=None, nb=None ):
	'''
	multiply atoms of sample along supercell vectors
	the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
	'''
	Zs_ = []
	Rs_ = []
	Qs_ = []
	if na is None:
		na=params['nPBC'][0]
	if nb is None:
		nb=params['nPBC'][1]
	for i in range(-na,na+1):
		for j in range(-nb,nb+1):
			for iatom in range(len(Zs)):
				x = Rs[iatom][0] + i*avec[0] + j*bvec[0]
				y = Rs[iatom][1] + i*avec[1] + j*bvec[1]
				#if (x>xmin) and (x<xmax) and (y>ymin) and (y<ymax):
				Zs_.append( Zs[iatom]          )
				Rs_.append( (x,y,Rs[iatom][2]) )
				Qs_.append( Qs[iatom]          )
	return np.array(Zs_).copy(), np.array(Rs_).copy(), np.array(Qs_).copy()	

def get_C612( i, j, FFparams ):
	'''
	compute Lenard-Jones coefitioens C6 and C12 pair of atoms i,j
	'''
	#print i, j, FFparams[i], FFparams[j]
	Rij = FFparams[i][0] + FFparams[j][0]
	Eij = np.sqrt( FFparams[i][1] * FFparams[j][1] )
	return 2*Eij*(Rij**6), Eij*(Rij**12)

def getAtomsLJ(  iZprobe, iZs,  FFparams ):
	'''
	compute Lenard-Jones coefitioens C6 and C12 for interaction between atoms in list "iZs" and probe-particle "iZprobe"
	'''
	n   = len(iZs)
	C6  = np.zeros(n)
	C12 = np.zeros(n)
	for i in range(n):
		C6[i],C12[i] = get_C612( iZprobe-1, iZs[i]-1, FFparams )
	return C6,C12

# ============= Hi-Level Macros

def prepareScanGrids( ):
	zTips  = np.arange( params['scanMin'][2], params['scanMax'][2]+0.00001, params['scanStep'][2] )[::-1];
	xTips  = np.arange( params['scanMin'][0], params['scanMax'][0]+0.00001, 0.1 )
	yTips  = np.arange( params['scanMin'][1], params['scanMax'][1]+0.00001, 0.1 )
	extent=( xTips[0], xTips[-1], yTips[0], yTips[-1] )
	lvecScan =np.array([
	params['scanMin'],
	[        params['scanMax'][0],0.0,0.0],
	[0.0,    params['scanMax'][1],0.0    ],
	[0.0,0.0,params['scanMax'][2]        ]
	]).copy() 
	return xTips,yTips,zTips,lvecScan

def lvec2params( lvec ):
	params['gridA'] = lvec[ 1,: ].copy()
	params['gridB'] = lvec[ 2,: ].copy()
	params['gridC'] = lvec[ 3,: ].copy()

def params2lvec( ):
	lvec = np.array([
	[ 0.0, 0.0, 0.0 ],
	params['gridA'],
	params['gridB'],
	params['gridC'],
	]).copy
	return lvec




