import numpy as np
from ctypes import c_int, c_double, c_bool, c_void_p
import ctypes
import os
from . import cpp_utils

cpp_name = 'LandauerQD_new'
cpp_utils.make('landauer_new')
lib = ctypes.CDLL(cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext)

c_double_p = ctypes.POINTER(c_double)
c_int_p    = ctypes.POINTER(c_int)

array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array1c = np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='CONTIGUOUS')
array2c = np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='CONTIGUOUS')

def _np_as(arr,atype):
    if arr is None:
        return None
    else: 
        return arr.ctypes.data_as(atype)

# Global variable to store number of quantum dots
_n_qds = None

# Initialize system
lib.initLandauerQDs.argtypes = [c_int, array2d, array1d, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int]
lib.initLandauerQDs.restype = None
def init(QDpos, Esite, K=0.01, decay=1.0, tS=0.01, E_sub=0.0, E_tip=0.0, tA=0.1, 
         eta=0.0, Gamma_tip=1.0, Gamma_sub=1.0, debug=0, verbosity=0):
    """Initialize the LandauerQDs system.
    
    Args:
        QDpos: array[n_qds, 2] - Positions of quantum dots (x,y coordinates)
        Esite: array[n_qds] - On-site energies
        K: float - Coulomb interaction between QDs
        decay: float - Decay constant for tip-QD coupling
        tS: float - Coupling strength to substrate
        E_sub: float - Substrate energy level
        E_tip: float - Tip energy level
        tA: float - Tip coupling strength prefactor
        eta: float - Infinitesimal broadening parameter
        Gamma_tip: float - Broadening of tip state
        Gamma_sub: float - Broadening of substrate state
        debug: int - Debug level
        verbosity: int - Verbosity level
    """
    global _n_qds
    _n_qds = len(QDpos)
    QDpos = np.ascontiguousarray(QDpos, dtype=np.float64)
    Esite = np.ascontiguousarray(Esite, dtype=np.float64)
    lib.initLandauerQDs(_n_qds, QDpos, Esite, K, decay, tS, E_sub, E_tip, tA, eta, Gamma_tip, Gamma_sub, debug, verbosity)

# Clean up
lib.deleteLandauerQDs.argtypes = []
lib.deleteLandauerQDs.restype = None
def cleanup():
    """Clean up the LandauerQDs system."""
    lib.deleteLandauerQDs()

# double calculate_transmission(double E, double* tip_pos, double Q_tip, double* Hqd ) {
lib.calculate_transmission.argtypes = [c_double, array1d, c_double_p ]
lib.calculate_transmission.restype = c_double
def calculate_transmission(energy, tip_pos, Hqd=None):
    if (Hqd is not None) and (Hqd.dtype != np.complex128) : Hqd = Hqd.astype(np.complex128) 
    return lib.calculate_transmission(energy, tip_pos, _np_as(Hqd,c_double_p))

#void calculate_transmissions( int nE, double* energies, int npos, double* ptips_, double* Qtips,  double* Hqds_, double* transmissions) {
lib.calculate_transmissions.argtypes = [ c_int, array1d, c_int, array2d, array1d, c_double_p, array2d]
lib.calculate_transmissions.restype = None
def calculate_transmissions( energies, ptips, Qtips, Hqds=None):
    transmissions = np.zeros((npos, nE), dtype=np.float64)
    if (Hqds is not None) and (Hqds.dtype != np.complex128) : Hqds = Hqds.astype(np.complex128) 
    lib.calculate_transmissions(  nE, energies, npos, ptips, Qtips, _np_as(Hqds,c_double_p), transmissions)
    return transmissions