#!/usr/bin/python

import numpy as np
from   ctypes import c_int, c_double, c_char_p, c_bool
import ctypes
import os

# ====================== constants

# ==============================
# ============================== Pure python functions
# ==============================

LIB_PATH = os.path.dirname( os.path.realpath(__file__) )
print " ProbeParticle Library DIR = ", LIB_PATH

def getSphericalHarmonic( X, Y, Z, R, kind='1' ):
	# TODO: renormalization should be probaby here
	# TODO: radial dependence of multipole is probably wrong
	if    kind=='s':
		return 1.0/R
	# p-functions
	elif  kind=='px':
		return Z/(R**2)
	elif  kind=='py':
		return Y/(R**2)
	elif  kind=='pz':
		return Z/(R**2)
	# d-functions
	if    kind=='dz2' :
		return ( X**2 + Y**2 - 2*Z**2 )/( R**4 )
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
	radial_renom = np.sum(radial)*np.abs(np.linalg.det(mat))*dd[0]*dd[1]*dd[2]   # TODO analytical renormalization may save some time ?
	radial      /= radial_renom
	if multipole_dict is not None:	   # multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
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
def sample_basis( atom_pos, atom_bas, atom_mask, X, Y, Z, radial_func = None, beta=1.0 ):
	basis_set = [ ]
	basis_assignment = [ ]
	for iatom, apos in enumerate( atom_pos ):
		if atom_mask[ iatom ]:
			dX = X - apos[0]
			dY = Y - apos[1]
			dZ = Z - apos[2]
			R  = np.sqrt( dX**2 + dY**2 + dZ**2 )
			radial = 1
			if radial_func is not None:
				radial = radial_func( R, beta )	# TODO: problem is that radial function is different for  monopole, dipole, quadrupole ... 
			for kind in atom_bas[iatom]:
				basis_func = radial * getSphericalHarmonic( X, Y, Z, R, kind=kind )
				basis_set.append( basis_func )
				basis_assignment.append( ( iatom, kind ) )
	return np.array( basis_set ), basis_assignment

# make_bas_list
# create list of basises for each atom 
# e.g. make_bas_list( ns=[3,5], bas=[ ['s','px','py','pz'], ['s', 'dz2'] ] ) will create first 3 atoms with ['s','px','py','pz'] basiset,  than 5 atoms with ['s', 'dz2'] basiset
def make_bas_list( ns, basis=[['s']] ):
	bas_list = []
	for i,n in enumerate(ns):
		for j in range(n):
			bas_list.append( basis[i] )
	return bas_list

'''
def make_Ratoms( atom_types, type_R,  fmin = 0.9 , fmax = 1.3 ):
	natoms =  len( atom_types )
	R_min = np.zeros( natoms )
	R_max = np.zeros( natoms )
	for i,typ in enumerate(atom_types):
		R_min[i] = type_R[ typ ] * fmin
		R_max[i] = type_R[ typ ] * fmax
	return R_min,R_max
'''

# make_Ratoms
def make_Ratoms( atom_types, type_R,  fmin = 0.9 , fmax = 1.3 ):
	atom_R = type_R[atom_types]
	return atom_R*fmin,atom_R*fmax

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
# 	double * sampled_val, Vec3d * sampled_pos_, bool canStore, bool pbc, bool show_where )
lib.sampleGridArroundAtoms.argtypes  = [ c_int, array2d, array1d, array1d, array1b, array1d, array2d, c_bool, c_bool, c_bool ]
lib.sampleGridArroundAtoms.restype   = c_int
def sampleGridArroundAtoms( atom_pos, atom_Rmin, atom_Rmax, atom_mask, pbc = False, show_where=False ):
	natom = len( atom_pos ) 
	#points_found = lib.sampleGridArroundAtoms( natom, atom_pos, atom_Rmin, atom_Rmax, atom_mask, None, None,              False )
	sampled_val  = np.zeros(  1    )
	sampled_pos  = np.zeros( (1,3) )
	points_found = lib.sampleGridArroundAtoms( natom, atom_pos, atom_Rmin, atom_Rmax, atom_mask, sampled_val, sampled_pos, False, pbc, False  )
	print " found ",points_found," points "  
	sampled_val  = np.zeros(  points_found    )
	sampled_pos  = np.zeros( (points_found,3) )
	points_found = lib.sampleGridArroundAtoms( natom, atom_pos, atom_Rmin, atom_Rmax, atom_mask, sampled_val, sampled_pos, True, pbc, show_where )
	return sampled_val, sampled_pos


# ============= Hi-Level Macros


# fitMultipoles
#	multipole expansion of given potential ( sampled on 3D grid ) in distance "atom_Rmin".."atom_Rmax" around atoms placed in "atom_pos" 
#	expand to multipoles defined by "atom_basis" which is list of lists e.g. "atom_basis" =  [ ['s'], ['s','dz2'] ] for 1st atom with just s-basis and 2nd atom with basis s,dz2 
#	"atom_mask" is list of booleans, "True" if atoms is inclueded into the axapnsion and "False" if is exclueded
#   periodic boundary conditions with "pbc" ( True/False ) 
#   "show_where" will set values in the sampled grid at sampled position so it can be saved and visuzalized ( useful for debugging )
def fitMultipolesPotential( atom_pos, atom_basis, atom_Rmin, atom_Rmax, atom_mask=None, pbc=False, show_where = False ):
	if atom_mask is None:
		atom_mask = np.array( [ True ] * natoms )
	sampled_val, sampled_pos = sampleGridArroundAtoms( atom_pos, atom_Rmin, atom_Rmax, atom_mask, pbc=pbc, show_where=show_where )
	#print "bas_list:", atom_bas
	X = sampled_pos[:,0]
	Y = sampled_pos[:,1] 
	Z = sampled_pos[:,2] 
	basis_set, basis_assignment = sample_basis( atom_pos, atom_basis, atom_mask, X, Y, Z, radial_func = None, beta=1.0 )
	#print "basis_assignment: ", basis_assignment
	fit_result = np.linalg.lstsq( np.transpose( basis_set ), sampled_val ) 
	coefs      = fit_result[0]
	return coefs, basis_assignment














