#from unittest.mock import NonCallableMagicMock
import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os
from . import cpp_utils

cpp_name='ChargeRings'
cpp_utils.make('chrings')
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

array1ui = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')
array1i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array2i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=2, flags='CONTIGUOUS')
array1d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

c_double_p = ctypes.POINTER(c_double)
c_float_p  = ctypes.POINTER(c_float)
c_int_p    = ctypes.POINTER(c_int)
c_bool_p   = ctypes.POINTER(c_bool)

def _np_as(arr,atype):
    if arr is None:
        return None
    else: 
        return arr.ctypes.data_as(atype)


# ========= C functions interface



# double solveSiteOccupancies( int npos, double* ptips_, double* Qtips,  int nsite, double* spos, const double* rot, const double* MultiPoles, const double* Esite, double* Qout, double E_mu, double cCouling, int niter, double tol, double dt, int* nitrs ){
lib.solveSiteOccupancies.argtypes = [ c_int, array2d, array1d, c_int, array2d, c_double_p, c_double_p, array1d, array2d, c_double, c_double, c_int, c_double, c_double, array1i ]
lib.solveSiteOccupancies.restype  = c_double
def solveSiteOccupancies( ptips, Qtips, spos, Esite, Qout=None, rot=None, MultiPoles=None, E_mu=0.0, cCouling=1.0, niter=100, tol=1e-6, dt=0.1, niters=None ):
    npos = len(ptips)
    nsite= len(spos)
    ptips = np.array(ptips)
    Qtips  = np.array(Qtips)
    spos   = np.array(spos)
    Esite  = np.array(Esite)
    if(Qout is None): Qout = np.zeros( (npos,nsite) )
    if(niters is None): niters = np.zeros(npos, dtype=np.int32)
    lib.solveSiteOccupancies( npos, ptips, Qtips, nsite, spos, _np_as(rot,c_double_p), _np_as(MultiPoles,c_double_p), Esite, Qout, E_mu, cCouling, niter, tol, dt, niters )
    return Qout, niters

# void setVerbosity(int verbosity_){
lib.setVerbosity.argtypes = [ c_int ]
lib.setVerbosity.restype  = None
def setVerbosity( verbosity ):
    lib.setVerbosity( verbosity )
    
# ========= Python functions