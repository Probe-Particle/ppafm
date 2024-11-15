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

# void setVerbosity(int verbosity_){
lib.setVerbosity.argtypes = [ c_int ]
lib.setVerbosity.restype  = None
def setVerbosity( verbosity ):
    lib.setVerbosity( verbosity )

# # double solveSiteOccupancies( int npos, double* ptips_, double* Qtips,  int nsite, double* spos, const double* rot, const double* MultiPoles, const double* Esite, double* Qout, double E_mu, double cCoupling, int niter, double tol, double dt, int* nitrs ){
# lib.solveSiteOccupancies_old.argtypes = [ c_int, array2d, array1d, c_int, array2d, c_double_p, c_double_p, array1d, array2d, c_double, c_double, c_int, c_double, c_double, array1i ]
# lib.solveSiteOccupancies_old.restype  = c_double
# def solveSiteOccupancies_old( ptips, Qtips, spos, Esite, Qout=None, rot=None, MultiPoles=None, E_fermi=0.0, cCoupling=1.0, niter=100, tol=1e-6, dt=0.1, niters=None ):
#     npos   = len(ptips)
#     nsite  = len(spos)
#     ptips  = np.array(ptips)
#     Qtips  = np.array(Qtips)
#     spos   = np.array(spos)
#     Esite  = np.array(Esite)
#     if(Qout is None): Qout = np.zeros( (npos,nsite) )
#     if(niters is None): niters = np.zeros(npos, dtype=np.int32)
#     lib.solveSiteOccupancies_old( npos, ptips, Qtips, nsite, spos, _np_as(rot,c_double_p), _np_as(MultiPoles,c_double_p), Esite, Qout, E_fermi, cCoupling, niter, tol, dt, niters )
#     return Qout, niters

# # STM map calculation
# lib.STM_map.argtypes = [c_int, array2d, array1d, c_int, array2d, array2d, c_double_p, c_double_p, array1d, c_double, c_double, c_double, c_double]
# lib.STM_map.restype = None
# def getSTM_map(ptips, Qs, spos, Esite, rot=None, MultiPoles=None, Q_tip=1.0, E_Fermi=0.0, cCoupling=1.0, beta=1.0):
#     npos = len(ptips)
#     nsite = len(spos)
#     ptips = np.array(ptips)
#     spos = np.array(spos)
#     Esite = np.array(Esite)
#     Qs = np.array(Qs)
#     I_stm = np.zeros(npos)
#     lib.STM_map(npos, ptips, I_stm, nsite, spos, Qs, _np_as(rot,c_double_p), _np_as(MultiPoles,c_double_p), Esite, Q_tip, E_Fermi, cCoupling, beta)
#     return I_stm

# Define the ctypes interface for the new function
lib.solveHamiltonians.argtypes = [c_int, array2d, array1d, array2d, array3d, c_double_p, c_double_p ]
lib.solveHamiltonians.restype = None
def solveHamiltonians(ptips, Qtips, evals=None, evecs=None, Gs=None, Hs=None, bH=False, bG=False, bVec=True ):
    npos = len(ptips)
    ptips = np.array(ptips)
    Qtips = np.array(Qtips)
    if ( evals  is None )          : evals = np.zeros((npos, 3   ))
    if ( evecs  is None ) and bVec : evecs = np.zeros((npos, 3, 3))
    if ( Hs     is None ) and bH   : Hs    = np.zeros((npos, 3, 3))
    if ( Gs     is None ) and bG   : Gs    = np.zeros((npos, 3, 3))
    lib.solveHamiltonians( npos, ptips, Qtips, evals, evecs, _np_as(Hs,c_double_p), _np_as(Gs,c_double_p) )
    return evals, evecs, Hs, Gs

# void solveSiteOccupancies_old( int npos, double* ptips_, double* Qtips, int nsite, double* spos, const double* rot, const double* MultiPoles, const double* Esite, double* Qout, double E_Fermi, double cCoupling, double temperature=100.0 ){
lib.solveSiteOccupancies_old.argtypes = [ c_int, array2d, array1d, c_int, array2d, c_double_p, c_double_p, array1d, array2d, c_double, c_double, c_double ]
lib.solveSiteOccupancies_old.restype  = c_double
def solveSiteOccupancies_old( ptips, Qtips, spos, Esite, Qout=None, rot=None, MultiPoles=None, E_fermi=0.0, cCoupling=1.0, temperature=1.0 ):
    npos   = len(ptips)
    nsite  = len(spos)
    ptips  = np.array(ptips)
    Qtips  = np.array(Qtips)
    spos   = np.array(spos)
    Esite  = np.array(Esite)
    if(Qout is None): Qout = np.zeros( (npos,nsite) )
    lib.solveSiteOccupancies_old( npos, ptips, Qtips, nsite, spos, _np_as(rot,c_double_p), _np_as(MultiPoles,c_double_p), Esite, Qout, E_fermi, cCoupling, temperature )
    return Qout

# Initialize ring parameters
lib.initRingParams.argtypes = [c_int, array2d, c_double_p, c_double_p, array1d, c_double, c_double, c_double]
lib.initRingParams.restype = None
def initRingParams(spos, Esite, rot=None, MultiPoles=None, E_Fermi=0.0, cCouling=1.0, temperature=100.0 ):
    global nsite, spos_, Esite_, rot_, MultiPoles_
    nsite = len(spos)
    #spos_  = np.array(spos)
    #Esite_ = np.array(Esite)
    # spos_       = spos.copy()
    # Esite_      = Esite.copy()
    # rot_        = rot.copy()
    # MultiPoles_ = MultiPoles.copy()
    spos_   = np.array(spos)
    Esite_  = np.array(Esite)
    rot_    = np.array(rot)
    MultiPoles_ = np.array(MultiPoles)
    lib.initRingParams(nsite, spos_, _np_as(rot_,c_double_p), _np_as(MultiPoles_,c_double_p), Esite_, E_Fermi, cCouling, temperature )

# Solve site occupancies
lib.solveSiteOccupancies.argtypes = [c_int, array2d, array1d, array2d]
lib.solveSiteOccupancies.restype = None
def solveSiteOccupancies(ptips, Qtips, Qout=None):
    npos  = len(ptips)
    #nsite = len(spos)  # Using global spos from initRingParams
    if Qout is None: Qout = np.zeros((npos, nsite))
    lib.solveSiteOccupancies(npos, ptips, Qtips, Qout)
    return Qout

# Calculate STM map
lib.STM_map.argtypes = [c_int, array2d, array1d, array2d, array1d, c_double, c_bool ]
lib.STM_map.restype = None
def getSTM_map(ptips, Qtips, Qsites, Iout=None, decay=1.0, bOccupied=False ):
    npos = len(ptips)
    if Iout is None:  Iout = np.zeros(npos)
    lib.STM_map(npos, ptips, Qtips, Qsites, Iout, decay, bOccupied )
    return Iout
    
# ========= Python functions

def makePosXY(n=100, L=10.0, z0=5.0):
    x = np.linspace(-L,L,n)
    y = np.linspace(-L,L,n)
    Xs,Ys = np.meshgrid(x,y)
    ps = np.zeros((n*n,3))
    ps[:,0] = Xs.flatten()
    ps[:,1] = Ys.flatten()
    ps[:,2] = z0
    return ps
    
def makeRotMats(phi, nsite=3 ):
    rot = np.zeros((nsite,3,3))
    ca = np.cos(phi)
    sa = np.sin(phi)
    rot[:,0,0] = ca
    rot[:,1,1] = ca
    rot[:,0,1] = -sa
    rot[:,1,0] = sa
    rot[:,2,2] = 1.0
    return rot

def getLine(spos, comb1, comb2, n=10):
    " Get line between linear combination of the charge sites"
    ps = np.zeros((n, spos.shape[1]))
    ts = np.linspace(0.0, 1.0, n, endpoint=False)
    ms = 1 - ts
    for i in range(spos.shape[0]):
        ps += spos[i] * (comb1[i] * ms[:, None] + comb2[i] * ts[:, None])
    return ps

def makePosQscan( ps, qs ):
    npoint  = ps.shape[0]
    ncharge = qs.shape[0]
    Ps = np.expand_dims(ps, axis=0)                 
    Ps = np.broadcast_to(Ps, (ncharge, npoint, 3))  
    Qs = np.expand_dims(qs, axis=1)  
    Qs = np.broadcast_to(Qs, (ncharge, npoint))  
    print("npoint ", Ps.shape, " ncharge ", ncharge )
    print("Ps.shape ", Ps.shape, " Qs.shape", Qs.shape)
    return  Ps.copy(), Qs.copy()