import ctypes
from ctypes import c_double, c_int

import numpy as np

from . import cpp_utils

cpp_name='PolyCycles'
cpp_utils.make("PolyCycles")
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes

array1ui = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')
array1i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array2i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=2, flags='CONTIGUOUS')
array1d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

# ========= C functions

'''
#double* getVpos  (){
lib.getVpos.argtypes = []
lib.getVpos.restype  = ctypes.POINTER(c_double)
def getVpos(nv):
    return np.ctypeslib.as_array( lib.getVpos(), shape=(nv,2) )

#double* getCpos  (){
lib.getCpos.argtypes = []
lib.getCpos.restype  = ctypes.POINTER(c_double)
def getCpos(nc):
    return np.ctypeslib.as_array( lib.getCpos(), shape=(nc,2) )
'''

#double* getPos  (){
lib.getPos.argtypes = []
lib.getPos.restype  = ctypes.POINTER(c_double)
def getPos(nc,nv):
    arr = np.ctypeslib.as_array( lib.getPos(), shape=(nc+nv,2) )
    return arr[:nc],arr[nc:]

# double setupOpt( double dt, double damp, double f_limit, double v_limit ){
lib.setupOpt.argtypes = [c_double,c_double,c_double,c_double]
lib.setupOpt.restype  = None
def setupOpt(dt=1.0,damping=0.05,f_limit=10.0,v_limit=10.0):
    lib.setupOpt(dt,damping,f_limit,v_limit );

#void setup( int ncycles, int* nvs ){
lib.setup.argtypes = [c_int,array1i]
lib.setup.restype  = c_int
def setup(nvs):
    return lib.setup(len(nvs),nvs)

#void init( double* angles ){
lib.init.argtypes = [array1d]
lib.init.restype  = None
def init(angles=None):
    lib.init(angles)

#double relaxNsteps( int kind, int nsteps, double F2conf, double dt, double damp ){
lib.relaxNsteps.argtypes = [c_int, c_int,c_double]
lib.relaxNsteps.restype  = c_double
def relaxNsteps(kind=1, nsteps=10, F2conf=-1.0):
    return lib.relaxNsteps(kind, nsteps, F2conf )
