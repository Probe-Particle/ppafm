#!/usr/bin/python

import ctypes
import os
from ctypes import c_bool, c_double, c_int

import numpy as np

from . import cpp_utils

'''

Approximates potential using small number of basis-functions of multipole kind

derivation:

Err = Integral_r{ ( V(r) - Sum_i{ a_i phi_i(r) } )**2 }

dErr_{a_j}      =  Integral_r{   -2*( V(r) - Sum_i{ a_i phi_i(r) } )   phi_j(r)        }
dErr_{a_j}/(-2) =  Integral_r{ V(r)*phi_j(r) }  - Integral_r{ phi_j(r) Sum_i{ a_i phi_i(r) } }

dErr_{a_j}/(-2) = <V|phi_j> - Sum_i{<phi_i|phi_j(r)>}

b_j     = <V|phi_j>
A_{i,j} = <phi_i|phi_j(r)>

A a = b

with regularization Err = <V-a.phi|V-a.phi> - a.a
dErr_{a_j}/(-2) = <V|phi_j> - Sum_i{<phi_i|phi_j(r)>} - 2*aj

'''


# ====================== constants

# ==============================
# ============================== Pure python functions
# ==============================

LIB_PATH = os.path.dirname( os.path.realpath(__file__) )
if(verbose>0): print(" ProbeParticle Library DIR = ", LIB_PATH)

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
                basis_func = radial * SH.getSphericalHarmonic( X, Y, Z, R, kind=kind )
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

# make_Ratoms
def make_Ratoms( atom_types, type_R,  fmin = 0.9 , fmax = 2.0 ):
    atom_R = type_R[atom_types]
    return atom_R*fmin,atom_R*fmax


def BB2symMat( nbas, BB ):
    np.zeros(nbas,nbas)
    return

# ==============================
# ============================== interface to C++ core
# ==============================

cpp_name='Multipoles'
cpp_utils.make("MP")
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )    # load dynamic librady object using ctypes

# define used numpy array types for interfacing with C++
array1b  = np.ctypeslib.ndpointer(dtype=bool     , ndim=1, flags='CONTIGUOUS')
array1i  = np.ctypeslib.ndpointer(dtype=np.int32 , ndim=1, flags='CONTIGUOUS')
array1ui = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')
array1d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')


# ========
# ======== Python warper function for C++ functions
# ========


#void setGridN( int * n ){
lib.setGridN.argtypes = [array1i]
lib.setGridN.restype  = None

#void setGridCell( double * cell ){
lib.setGridCell.argtypes = [array2d]
lib.setGridCell.restype  = None

def setGrid_shape( n_, cell ):
    n     = np.array( (n_[2],n_[1],n_[0]) ).astype(np.int32)
    lib.setGridN    ( n    )
    lib.setGridCell ( cell )

# void setGrid_Pointer( int * n, double * grid, double * step,  )
lib.setGrid_Pointer.argtypes = [array3d]
lib.setGrid_Pointer.restype  = None
def setGrid_Pointer( grid ):
    lib.setGrid_Pointer( grid )

# int setCenters( int nCenters_, double * centers_, uint32_t * types_ )
lib.setCenters.argtypes = [ c_int, array2d, array1ui ]
lib.setCenters.restype  = c_int
def setCenters( centers, types ):
    return lib.setCenters( len(centers), centers, types  )

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
    if(verbose>0): print(" found ",points_found," points ")
    sampled_val  = np.zeros(  points_found    )
    sampled_pos  = np.zeros( (points_found,3) )
    points_found = lib.sampleGridArroundAtoms( natom, atom_pos, atom_Rmin, atom_Rmax, atom_mask, sampled_val, sampled_pos, True, pbc, show_where )
    return sampled_val, sampled_pos

# int buildLinearSystemMultipole(
#    int npos,     double * poss_, double * V,
#    int ncenter,  double * centers_, uint32_t * type,
#    double * bas, double * BB
lib.buildLinearSystemMultipole.argtypes  = [ c_int, array2d, array1d, array1d,  c_int, array1d, array2d ]
lib.buildLinearSystemMultipole.restype   = c_double
def buildLinearSystemMultipole( poss, vals, centers, types, Ws=None ):
    nbas = setCenters(centers, types)
    B=np.zeros(nbas)
    BB=np.zeros((nbas,nbas))
    if Ws is None:
        Ws = np.ones(vals.shape)
    Wsum = lib.buildLinearSystemMultipole(len(poss),poss,vals,Ws, nbas, B, BB )
    if(verbose>0): print("Wsum = ", Wsum)
    return B/Wsum,BB/Wsum


# int regularizedLinearFit(
#    int nbas,  double * coefs,  double * kReg, double * B, double * BB,
#    double dt, double damp, double convF, int nMaxSteps
lib.regularizedLinearFit.argtypes = [c_int,array1d,array1d,array1d,array2d,   c_double, c_double, c_double, c_int ]
lib.regularizedLinearFit.restype  = c_int
def regularizedLinearFit(B, BB, coefs=None, kReg=1.0,  dt=0.1, damp=0.1, convF=1e-6, nMaxSteps=1000):
    n = len(B)
    if not isinstance(kReg,np.ndarray):
        try:
            kReg = np.ones(n) * kReg
            if(verbose>0): print("kReg",kReg)
        except:
            try:
                kReg = np.array(kReg)
                if(verbose>0): print("kReg",kReg)
            except:
                if(verbose>0): print(" kReg must be either array,list,scalar")
                return
    if coefs is None:
        coefs = np.zeros(n)
    if(verbose>0): print("kReg",kReg)
    lib.regularizedLinearFit( n, coefs, kReg, B, BB,    dt, damp, convF, nMaxSteps )
    return coefs

#int evalMultipoleComb( double * coefs_, double * gridV_ )
lib.evalMultipoleComb.argtypes = [array1d,array3d]
lib.evalMultipoleComb.restype  = None
def evalMultipoleComb( coefs, V ):
    lib.evalMultipoleComb( coefs, V )

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
    X = sampled_pos[:,0]
    Y = sampled_pos[:,1]
    Z = sampled_pos[:,2]
    basis_set, basis_assignment = sample_basis( atom_pos, atom_basis, atom_mask, X, Y, Z, radial_func = None, beta=1.0 )
    fit_result = np.linalg.lstsq( np.transpose( basis_set ), sampled_val )
    coefs      = fit_result[0]
    return coefs, basis_assignment
