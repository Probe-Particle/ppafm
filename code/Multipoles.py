#!/usr/bin/python

import numpy as np
from   ctypes import c_int, c_double, c_char_p, c_bool
import ctypes
import os
import GridUtils as GU
import libFFTfin

# ====================== constants


# ==============================
# ============================== Pure python functions
# ==============================

LIB_PATH = os.path.dirname( os.path.realpath(__file__) )
print " ProbeParticle Library DIR = ", LIB_PATH

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

def getSphericalHarmonic( X, Y, Z, kind='1' ):
	# TODO: renormalization should be probaby here
	if    kind=='s':
		return 1.0
	# p-functions
	elif  kind=='px':
		return Z
	elif  kind=='py':
		return Y
	elif  kind=='pz':
		return Z
	# d-functions
	if    kind=='dz2' :
		return X**2 + Y**2 - 2*Z**2
	else:
		return 0.0

def getProbeDensity(sampleSize, X, Y, Z, sigma, dd, multipole_dict=None ):
	'returns probe particle potential'
	mat = getNormalizedBasisMatrix(sampleSize).getT()
	rx = X*mat[0, 0] + Y*mat[0, 1] + Z*mat[0, 2]
	ry = X*mat[1, 0] + Y*mat[1, 1] + Z*mat[1, 2]
	rz = X*mat[2, 0] + Y*mat[2, 1] + Z*mat[2, 2]
	rquad  = rx**2 + ry**2 + rz**2
	radial       = np.exp( -(rquad)/(1*sigma**2) )
	radial_renom = np.sum(radial)*np.abs(np.linalg.det(mat))*dd[0]*dd[1]*dd[2]  # TODO analytical renormalization may save some time ?
	radial      /= radial_renom
	if multipole_dict is not None:	# multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
		rho = np.zeros( shape(radial) )
		for kind, coef in multipole_dict.iteritems():
			rho += radial * coef * getSphericalHarmonic( X, Y, Z, kind=kind )    # TODO renormalization should be probaby inside getSphericalHarmonic if possible ?
	else:
		rho = radial
	return rho

# make_matrix 
#	compute values of analytic basis functions set at given points in space and store in matrix "basis_set"  
#	basis functions are sperical harmonics around atoms positioned in atom_pos[i] and 
#	for each of them there is various number of multipole expansion atom_bas which is e.g. atom_bas[i] = [ 's', 'px', 'py', 'pz', 'dz2' ]  
#	you can set specific radial function radial_func( R, beta ) as optional paramenter, othervise exp( -beta*R ) is used
def make_matrix( atom_pos, atom_bas, X, Y, Z, radial_func = None, beta=1.0 ):
	natom = len(atom_pos)
	basis_set = [ ]
	for apos in enumerate( natom ):
		dX = X - apos[0]
		dY = Y - apos[1]
		dZ = Z - apos[2]
		r  = sqrt( dX**2 + dY**2 + dZ**2 )
		if radial_func is None:
			radial = radial_func( R, beta )
		else:
			radial = exp( -beta*R )	
		for kind in atom_bas:
			basis_func = radial * getSphericalHarmonic( X, Y, Z, kind=kind )
			basis_set.append( basis_func )
	return np.array( basis_set )


# ==============================
# ============================== interface to C++ core 
# ==============================

name='Multipoles'
ext='_lib.so'

def makeclean( ):
	CWD=os.getcwd()
	os.chdir(LIB_PATH)
	os.system("make clean")
	os.chdir(CWD)

# recompilation of C++ dynamic librady ProbeParticle_lib.so from ProbeParticle.cpp
def recompile():
	CWD=os.getcwd()
	os.chdir(LIB_PATH)
	os.system("make MP")
	os.chdir(CWD)

# if binary of ProbeParticle_lib.so is deleted => recompile it

#makeclean()

if not os.path.exists(LIB_PATH+"/"+name+ext):
	recompile()

lib    = ctypes.CDLL(LIB_PATH+"/"+name+ext )    # load dynamic librady object using ctypes 

# define used numpy array types for interfacing with C++

array1b = np.ctypeslib.ndpointer(dtype=np.bool  , ndim=1, flags='CONTIGUOUS')
array1i = np.ctypeslib.ndpointer(dtype=np.int32 , ndim=1, flags='CONTIGUOUS')
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

# ========
# ======== Python warper function for C++ functions
# ========

# void setFF( int * n, double * grid, double * step,  )
lib.setGrid.argtypes = [array1i,array3d,array2d]
lib.setGrid.restype  = None
def setGrid( grid, cell ):
	n_    = np.shape(grid)
	n     = np.array( (n_[2],n_[1],n_[0]) ).astype(np.int32)
	lib.setGrid( n, grid, cell )

# void setGrid_Pointer( int * n, double * grid, double * step,  )
lib.setGrid_Pointer.argtypes = [array3d]
lib.setGrid_Pointer.restype  = None
def setGrid_Pointer( grid ):
	lib.setGrid_Pointer( grid )

# int sampleGridArroundAtoms( 
# 	int natoms, double * atom_pos, double * atom_Rmin, double * atom_Rmax, bool * atom_mask, 
# 	double * sampled_val, double * sampled_pos, bool canStore )
lib.sampleGridArroundAtoms.argtypes  = [ c_int, array2d, array1d, array1d, array1b, array1d, array2d, c_bool ]
lib.sampleGridArroundAtoms.restype   = c_int
def sampleGridArroundAtoms( atom_pos, atom_Rmin, atom_Rmax, atom_mask ):
	natom = len( atom_pos ) 
	#points_found = lib.sampleGridArroundAtoms( natom, atom_pos, atom_Rmin, atom_Rmax, atom_mask, None, None,              False )
	sampled_val  = np.zeros(  1    )
	sampled_pos  = np.zeros( (1,3) )
	points_found = lib.sampleGridArroundAtoms( natom, atom_pos, atom_Rmin, atom_Rmax, atom_mask, sampled_val, sampled_pos, False )
	print " found ",points_found," points "  
	sampled_val  = np.zeros(  points_found    )
	sampled_pos  = np.zeros( (points_found,3) )
	points_found = lib.sampleGridArroundAtoms( natom, atom_pos, atom_Rmin, atom_Rmax, atom_mask, sampled_val, sampled_pos, True )
	return sampled_val, sampled_pos






















