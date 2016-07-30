#!/usr/bin/python

import numpy as np
from   ctypes import c_int, c_double, c_char_p
import ctypes
import os
import common as PPU
import cpp_utils

# ==============================
# ============================== interface to C++ core 
# ==============================

cpp_name='ProbeParticle'
#cpp_utils.compile_lib( cpp_name  )
cpp_utils.make( "PP"  )
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

# void setFF( int * n, double * grid, double * step,  )
lib.setFF.argtypes = [array1i,array4d,array2d]
lib.setFF.restype  = None
def setFF( grid, cell = None ):
	n_    = np.shape(grid)
	n     = np.array( (n_[2],n_[1],n_[0]) ).astype(np.int32)
	if cell is None:
		cell = np.array([
		PPU.params['gridA'],
		PPU.params['gridB'],
		PPU.params['gridC'],
		]).copy() 
	lib.setFF( n, grid, cell )

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
	if lRadial is None:
		lRadial=PPU.params['r0Probe'][2]
	if kRadial is  None:
		kRadial=PPU.params['stiffness'][2]/-PPU.eVA_Nm
	if rPP0 is  None:
		rPP0=np.array((PPU.params['r0Probe'][0],PPU.params['r0Probe'][1],0.0))
	if kSpring is  None: 
		kSpring=np.array((PPU.params['stiffness'][0],PPU.params['stiffness'][1],0.0))/-PPU.eVA_Nm 
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

# void subsample_uniform_spline( double x0, double dx, int n, double * ydys, int m, double * xs_, double * ys_ )
lib.subsample_uniform_spline.argtypes  = [ c_double, c_double, c_int, array2d, c_int, array1d, array1d ]
lib.subsample_uniform_spline.restype   = None
def subsample_uniform_spline( x0, dx, ydys, xs_, ys_=None ):
	n = len(ydys)
	m = len(xs_)
	if ys_ is None :
		ys_ = np.zeros(m)
	lib.subsample_uniform_spline( x0, dx, n, ydys, m, xs_, ys_ );
	return ys_;

# void subsample_nonuniform_spline( int n, double * xs, double * ydys, int m, double * xs_, double * ys_ )
lib.subsample_nonuniform_spline.argtypes  = [ c_int, array1d, array2d, c_int, array1d, array1d ]
lib.subsample_nonuniform_spline.restype   = None
def subsample_nonuniform_spline( xs, ydys, xs_, ys_=None ):
	n = len(xs )
	m = len(xs_)
	if ys_ is None :
		ys_ = np.zeros(m)
	lib.subsample_nonuniform_spline( n, xs, ydys, m, xs_, ys_ );
	return ys_;
