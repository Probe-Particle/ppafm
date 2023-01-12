#!/usr/bin/python

import ctypes
from ctypes import c_bool, c_double, c_int

import numpy as np

from . import cpp_utils

# ==============================
# ============================== interface to C++ core
# ==============================

cpp_name='atomfit'
cpp_utils.make( "atomfit"  )
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )    # load dynamic librady object using ctypes

# define used numpy array types for interfacing with C++
array1i = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')
array4d = np.ctypeslib.ndpointer(dtype=np.double, ndim=4, flags='CONTIGUOUS')

# ========
# ======== Python warper function for C++ functions
# ========

#void setGrid(double* npix_, double* dpix_, double* gridFF_ ){
lib.setGridFF.argtypes = [array1i,array1d,array3d]
lib.setGridFF.restype  = None
def setGridFF(gridFF,dpix):
    dpix = np.array(dpix        , dtype=np.float64)
    npix = np.array(gridFF.shape, dtype=np.int32)
    lib.setGridFF(npix,dpix,gridFF)

#void setAtoms(int natom_, double* pos_, double* vel_, double* force_ ){
lib.setAtoms.argtypes = [c_int, array2d,array2d,array2d]
lib.setAtoms.restype  = None
def setAtoms(pos,vel=None,force=None):
    n=len(pos)
    if vel is None:
        vel = np.zeros((n,2))
    if force is None:
        force = np.zeros((n,2))
    lib.setAtoms(n, pos, vel, force)
    return vel,force

# void setupParams(double eps_, double rmin_ ){
lib.setParams.argtypes  = [ c_double, c_double ]
lib.setParams.restype   = None
def setParams( eps, rmin ):
    lib.setParams( eps, rmin );

# bool relaxAtoms( int n, double dt, double damp, double F2conv ){
lib.relaxAtoms.argtypes  = [ c_int, c_double, c_double, c_double ]
lib.relaxAtoms.restype   = c_bool
def relaxAtoms( n, dt, damp, F2conv ):
    return lib.relaxAtoms( n, dt, damp, F2conv );

# int relaxParticlesUnique( int np, Vec2d* poss, int nstep, double dt, double damp, double F2conv ){
lib.relaxParticlesUnique.argtypes  = [ c_int, array2d,  c_int,    c_double, c_double, c_double ]
lib.relaxParticlesUnique.restype   = c_int
def relaxParticlesUnique( poss, nstep, dt, damp, F2conv ):
    return lib.relaxParticlesUnique( len(poss), poss, nstep, dt, damp, F2conv )

# int relaxParticlesUnique( int np, Vec2d* poss, int nstep, double dt, double damp, double F2conv ){
lib.relaxParticlesRepel.argtypes  = [ c_int, array2d,  c_int,    c_double, c_double, c_double ]
lib.relaxParticlesRepel.restype   = c_int
def relaxParticlesRepel( poss, nstep, dt, damp, F2conv ):
    return lib.relaxParticlesRepel( len(poss), poss, nstep, dt, damp, F2conv )
