#!/usr/bin/python

import numpy as np
from   ctypes import c_int, c_double
import ctypes
import os

# ====================== constants

eVA_Nm =  16.0217657

# ==============================
# ============================== Pure python functions
# ==============================

def multArray( F, nx=2,ny=2 ):
	'''
	multiply data array "F" along second two axis (:, :*nx, :*ny ) 
	it is usefull to visualization of images computed in periodic supercell ( PBC )
	'''
	nF = np.shape(F)
	print "nF: ",nF
	F_ = np.zeros( (nF[0],nF[1]*ny,nF[2]*nx) )
	for iy in range(ny):
		for ix in range(nx):
			F_[:, iy*nF[1]:(iy+1)*nF[1], ix*nF[2]:(ix+1)*nF[2]  ] = F
	return F_

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
		params[ 'scanMax' ][1] = params[ 'gridA' ][0]
		params[ 'scanMin' ][0] = 0
		params[ 'scanMax' ][1] = params[ 'gridB' ][1]
		print " autoGeom changed cell to = ", params[ 'scanMax' ]
	if shiftXY:
		dx = -0.5*(xmin+xmax) + 0.5*( params[ 'gridA' ][0] + params[ 'gridB' ][0] ); Rs[0] += dx
		dy = -0.5*(ymin+ymax) + 0.5*( params[ 'gridA' ][1] + params[ 'gridB' ][1] ); Rs[1] += dy;
		print " autoGeom moved geometry by ",dx,dy

def PBCAtoms( Zs, Rs, Qs, avec, bvec ):
	'''
	multiply atoms of sample along supercell vectors
	the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
	'''
	Zs_ = []
	Rs_ = []
	Qs_ = []
	for i in [-1,0,1]:
		for j in [-1,0,1]:
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

# default parameters of simulation
params={
'PBC': False,
'gridN':       np.array( [ 150,     150,   50   ] ).astype(np.int),
'gridA':       np.array( [ 12.798,  -7.3889,  0.00000 ] ),
'gridB':       np.array( [ 12.798,   7.3889,  0.00000 ] ),
'gridC':       np.array( [      0,        0,      5.0 ] ),
'moleculeShift':  np.array( [  0.0,      0.0,    -2.0 ] ),
'probeType':   8,
'charge':      0.00,
'r0Probe'  :  np.array( [ 0.00, 0.00, 4.00] ),
'stiffness':  np.array( [ 0.5,  0.5, 20.00] ),

'scanStep': np.array( [ 0.10, 0.10, 0.10] ),
'scanMin': np.array( [   0.0,     0.0,    5.0 ] ),
'scanMax': np.array( [  20.0,    20.0,    8.0 ] ),
'kCantilever'  :  1800.0, 
'f0Cantilever' :  30300.0,
'Amplitude'    :  1.0,
'plotSliceFrom':  16,
'plotSliceTo'  :  22,
'plotSliceBy'  :  1,
'imageInterpolation': 'nearest',
'colorscale'   : 'gray',

}

# overide default parameters by parameters read from a file 
def loadParams( fname ):
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

# ==============================
# ============================== interface to C++ core 
# ==============================

name='ProbeParticle'
ext='_lib.so'

# recompilation of C++ dynamic librady ProbeParticle_lib.so from ProbeParticle.cpp
def recompile( 
		LFLAGS="",
		#FFLAGS="-Og -g -Wall"
		#FFLAGS="-std=c99 -O3 -ffast-math -ftree-vectorize"
		FFLAGS="-std=c++11 -O3 -ffast-math -ftree-vectorize"
	):
	import os
	print " ===== COMPILATION OF : "+name+".cpp"+"   "+name+ext
	print  os.getcwd()
	os.system("g++ "+FFLAGS+" -c -fPIC "+name+".cpp -o "+name+".o"+LFLAGS)
	os.system("g++ "+FFLAGS+" -shared -Wl,-soname,"+name+ext+" -o "+name+ext+" "+name+".o"+LFLAGS)


# if binary of ProbeParticle_lib.so is deleted => recompile it
if not os.path.exists("./"+name+ext ):
	recompile()

lib    = ctypes.CDLL("./"+name+ext )    # load dynamic librady object using ctypes 

# define used numpy array types for interfacing with C++
array1i = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')
array4d = np.ctypeslib.ndpointer(dtype=np.double, ndim=4, flags='CONTIGUOUS')

# ========
# ======== Python warper function for C++ functions
# ========

# void setFF( int * n, double * grid, double * step,  )
lib.setFF.argtypes = [array1i,array4d,array2d]
lib.setFF.restype  = None
def setFF( grid, mat ):
	n_    = np.shape(grid)
	n     = np.array( (n_[2],n_[1],n_[0]) ).astype(np.int32)
	lib.setFF( n, grid, mat )

# void setFF( int * n, double * grid, double * step,  )
lib.setFF_Pointer.argtypes = [array4d]
lib.setFF_Pointer.restype  = None
def setFF_Pointer( grid ):
	lib.setFF_Pointer( grid )

#void setRelax( int maxIters, double convF2, double dt, double damping )
lib.setRelax.argtypes = [ c_int, c_double, c_double, c_double ]
lib.setRelax.restype  = None
def setRelax( maxIters  = 1000, convF2 = 1.0e-4, dt = 0.1, damping = 0.1 ):
	lib.setRelax( maxIters, convF*convF, dt, damping )

#void setFIRE( double finc, double fdec, double falpha )
lib.setFIRE.argtypes = [ c_double, c_double, c_double ]
lib.setFIRE.restype  = None
def setFIRE( finc = 1.1, fdec = 0.5, falpha  = 0.99 ):
	lib.setFIRE( finc, fdec, falpha )


#void setTip( double lRad, double kRad, double * rPP0, double * kSpring )
lib.setTip.argtypes = [ c_double, c_double, array1d, array1d ]
lib.setTip.restype  = None
def setTip( lRadial=None, kRadial=None, rPP0=None, kSpring=None	):
	if lRadial==None:
		lRadial=params['r0Probe'][2]
	if kRadial==None:
		kRadial=params['stiffness'][2]/-eVA_Nm
	if rPP0==None:
		rPP0=np.array((params['r0Probe'][0],params['r0Probe'][1],0.0))
	if kSpring==None: 
		kSpring=np.array((params['stiffness'][0],params['stiffness'][1],0.0))/-eVA_Nm 
	print " IN setTip !!!!!!!!!!!!!! "
	print " lRadial ", lRadial
	print " kRadial ", kRadial
	print " rPP0 ", rPP0
	print " kSpring ", kSpring
	lib.setTip( lRadial, kRadial, rPP0, kSpring )


# void getClassicalFF       (    int natom,   double * Rs_, double * C6, double * C12 )
lib.getLenardJonesFF.argtypes  = [ c_int,       array2d,      array1d,     array1d     ]
lib.getLenardJonesFF.restype   = None
def getLenardJonesFF( Rs, C6, C12 ):
	natom = len(Rs) 
	lib.getLenardJonesFF( natom, Rs, C6, C12 )

# void getCoulombFF       (    int natom,   double * Rs_, double * C6, double * C12 )
lib.getCoulombFF.argtypes  = [ c_int,       array2d,      array1d   ]
lib.getCoulombFF.restype   = None
def getCoulombFF( Rs, kQQs ):
	natom = len(Rs) 
	lib.getCoulombFF( natom, Rs, kQQs )

# int relaxTipStroke ( int probeStart, int nstep, double * rTips_, double * rs_, double * fs_ )
lib.relaxTipStroke.argtypes  = [ c_int, c_int, c_int,  array2d, array2d, array2d ]
lib.relaxTipStroke.restype   = c_int
def relaxTipStroke( rTips, rs, fs, probeStart=1, relaxAlg=1 ):
	n = len(rTips) 
	return lib.relaxTipStroke( probeStart, relaxAlg, n, rTips, rs, fs )



