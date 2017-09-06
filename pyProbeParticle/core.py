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


lib.setFFC_shape.argtypes = [array1i,array2d]
lib.setFFC_shape.restype  = None
def setFFC_shape( n_, cell ):
	n     = np.array( (n_[2],n_[1],n_[0]) ).astype(np.int32)
	lib.setFFC_shape( n, cell )
lib.setFFO_shape.argtypes = [array1i,array2d]
lib.setFFO_shape.restype  = None
def setFFO_shape( n_, cell ):
	n     = np.array( (n_[2],n_[1],n_[0]) ).astype(np.int32)
	lib.setFFO_shape( n, cell )

# void setFFC_pointer( double * gridF, double * gridE  )
lib.setFFC_Fpointer.argtypes = [array4d]
lib.setFFC_Fpointer.restype  = None
def setFFC_Fpointer( gridF ):
	lib.setFFC_Fpointer( gridF )
# void setFFO_pointer( double * gridF, double * gridE  )
lib.setFFO_Fpointer.argtypes = [array4d]
lib.setFFO_Fpointer.restype  = None
def setFFO_Fpointer( gridF ):
	lib.setFFO_Fpointer( gridF )

# void setFFC_pointer( double * gridF, double * gridE  )
lib.setFFC_Epointer.argtypes = [array3d]
lib.setFFC_Epointer.restype  = None
def setFFC_Epointer( gridE ):
	lib.setFFC_Epointer( gridE )

# void setFFO_pointer( double * gridF, double * gridE  )
lib.setFFO_Epointer.argtypes = [array3d]
lib.setFFO_Epointer.restype  = None
def setFFO_Epointer( gridE ):
	lib.setFFO_Epointer( gridE )



def setFFC( gridF=None, cell=None, gridE=None ):
	n_ = None
	if gridF is not None:
		setFFC_Fpointer( gridF )
		n_    = np.shape(gridF)
	if gridE is not None:
		setFFC_Epointer( gridE )
		n_    = np.shape(gridF)
	if cell is None:
		cell = np.array([
		PPU.params['gridA'],
		PPU.params['gridB'],
		PPU.params['gridC'],
		]).copy() 	
	if n_ is not None:
		setFFC_shape( n_, cell )
	else:
		"Warrning : setFFC shape not set !!! "

def setFFO( gridF=None, cell=None, gridE=None ):
	n_ = None
	if gridF is not None:
		setFFO_Fpointer( gridF )
		n_    = np.shape(gridF)
	if gridE is not None:
		setFFO_Epointer( gridE )
		n_    = np.shape(gridF)
	if cell is None:
		cell = np.array([
		PPU.params['gridA'],
		PPU.params['gridB'],
		PPU.params['gridC'],
		]).copy() 	
	if n_ is not None:
		setFFO_shape( n_, cell )
	else:
		"Warrning : setFFO shape not set !!! "

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
#void setTip( double TClRad, double COlRad, double TCkRad, double COkRad, double * rC0, double *rO0, double * CkSpring, double * OkSpring ){  
lib.setTip.argtypes = [ c_double, c_double, c_double, c_double, array1d, array1d, array1d, array1d  ]
lib.setTip.restype  = None
def setTip(ClRadial=None, OlRadial=None, CkRadial=None, OkRadial=None,
           rC0=None, rO0=None, CkSpring=None, OkSpring=None):
	if ClRadial is None:
		ClRadial=PPU.params['rC0'][2]
	if OlRadial is None:
		OlRadial=PPU.params['rO0'][2]
	if CkRadial is  None:
		CkRadial=PPU.params['Ckrad']/-PPU.eVA_Nm
	if OkRadial is  None:
		OkRadial=PPU.params['Okrad']/-PPU.eVA_Nm
	if rC0 is  None:
		rC0=np.array((PPU.params['rC0'][0],PPU.params['rC0'][1],0.0))
	if rO0 is  None:
		rO0=np.array((PPU.params['rO0'][0],PPU.params['rO0'][1],0.0))
	if CkSpring is  None: 
		CkSpring=np.array(PPU.params['Cklat'],PPU.params['Cklat'][1],0.0) 
	if OkSpring is  None: 
		OkSpring=np.array(PPU.params['Oklat'],PPU.params['Oklat'][1],0.0)
	print " IN setTip !!!!!!!!!!!!!! "
	print " ClRadial ", ClRadial
	print " OlRadial ", OlRadial
	print " CkRadial ", CkRadial
	print " ORadial ", OkRadial
	print " rC0 ", rC0
	print " rO0 ", rO0
	print " CkSpring ", CkSpring
	print " OkSpring ", OkSpring
	lib.setTip(ClRadial,OlRadial, CkRadial, OkRadial, rC0,rO0,
                   CkSpring, OkSpring )

#void setTipSpline( int n, double * xs, double * ydys ){  
lib.setTipSpline.argtypes = [ c_int, array1d, array2d ]
lib.setTipSpline.restype  = None
def setTipSpline( xs, ydys	):
	n = len(xs)
	lib.setTipSpline( n, xs, ydys )

# void getClassicalFF       (    int natom,   double * Rs_, double * C6, double * C12 )
lib.getCLenardJonesFF.argtypes  = [ c_int,       array2d,      array1d,     array1d     ]
lib.getCLenardJonesFF.restype   = None
def getCLenardJonesFF( Rs, C6, C12 ):
	natom = len(Rs) 
	lib.getCLenardJonesFF( natom, Rs, C6, C12 )
lib.getOLenardJonesFF.argtypes  = [ c_int,       array2d,      array1d,     array1d     ]
lib.getOLenardJonesFF.restype   = None
def getOLenardJonesFF( Rs, C6, C12 ):
	natom = len(Rs) 
	lib.getOLenardJonesFF( natom, Rs, C6, C12 )

# void getCCoulombFF       (    int natom,   double * Rs_, double * C6, double * C12 )
lib.getCCoulombFF.argtypes  = [ c_int,       array2d,      array1d   ]
lib.getCCoulombFF.restype   = None
def getCCoulombFF( Rs, kQQs ):
	natom = len(Rs) 
	lib.getCCoulombFF( natom, Rs, kQQs )
# void getOCoulombFF       (    int natom,   double * Rs_, double * C6, double * C12 )
lib.getOCoulombFF.argtypes  = [ c_int,       array2d,      array1d   ]
lib.getOCoulombFF.restype   = None
def getOCoulombFF( Rs, kQQs ):
	natom = len(Rs) 
	lib.getOCoulombFF( natom, Rs, kQQs )

#int relaxTipStroke ( int probeStart, int relaxAlg, int nstep, double * rTips_, double * rC_,  double * rO_, double * fC_ , double *fO_){
lib.relaxTipStroke.argtypes  = [ c_int, c_int, c_int,  array2d, array2d,
        array2d, array2d, array2d ]
lib.relaxTipStroke.restype   = c_int
def relaxTipStroke( rTips, rCs, rOs, fCs, fOs, probeStart=1, relaxAlg=1 ):
	n = len(rTips) 
	return lib.relaxTipStroke( probeStart, relaxAlg, n, rTips, rCs, rOs, fCs,fOs )

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
#
# void test_force( int type, int n, double * r0_, double * dr_, double * R_, double * fs_ ){
#lib.test_force.argtypes  = [ c_int, c_int, array1d, array1d, array1d, array2d ]
#lib.test_force.restype   = None
#def test_force( typ, r0, dr, R, fs ):
#	n = len( fs )
#	lib.test_force( typ, n, r0, dr, R, fs );
#	return fs;


