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

# void setSiteConfBasis(int nconf, double* siteConfs_){
lib.setSiteConfBasis.argtypes = [ c_int, array2d ]
lib.setSiteConfBasis.restype  = None
def setSiteConfBasis(siteConfs):
    global nconfs
    nconfs = len(siteConfs)
    siteConfs = np.array(siteConfs)
    if siteConfs.shape[1] != nsite: raise ValueError(f"siteConfs must have shape [nConf, {nsite}]")
    lib.setSiteConfBasis( nconfs, siteConfs )
    return nconfs

# void initRingParams(int nsite, double* spos, double* rots, double* MultiPoles, double* Esites, double E_Fermi, double cCouling, double onSiteCoulomb, double temperature ) {
lib.initRingParams.argtypes = [c_int, array2d, c_double_p, c_double_p, array1d, c_double, c_double, c_double, c_double]
lib.initRingParams.restype = None
def initRingParams(spos, Esite, rot=None, MultiPoles=None, E_Fermi=0.0, cCouling=1.0, onSiteCoulomb=3.0, temperature=100.0 ):
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
    lib.initRingParams(nsite, spos_, _np_as(rot_,c_double_p), _np_as(MultiPoles_,c_double_p), Esite_, E_Fermi, cCouling, onSiteCoulomb, temperature )

# void solveSiteOccupancies(int npos, double* ptips_, double* Qtips, double* Qout, double* Econf, double* dEdQ, int solver_type) {
lib.solveSiteOccupancies.argtypes = [c_int, array2d, array1d, array2d, c_double_p, c_double_p, c_int]
lib.solveSiteOccupancies.restype = None
def solveSiteOccupancies(ptips, Qtips, Qout=None, Econf=None, Esite=None, bEsite=False, bEconf=False, solver_type=0):
    npos = len(ptips)
    ptips = np.array(ptips)
    Qtips = np.array(Qtips)
    if solver_type==0:
        nconfs_ = 1<<(nsite*2)  # 2^(2*nsite) configurations
    else:
        nconfs_ = nconfs
    if Qout is None: 
        Qout = np.zeros((npos, nsite))
    if bEconf:
        if Econf is None: 
            Econf = np.zeros((npos, nconfs_))
        else:
            if Econf.shape != (npos, nconfs_):  raise ValueError(f"Econf array must have shape ({npos}, {nconfs_})")
    if bEsite:
        if Esite is None:
            Esite = np.zeros((npos, nsite))
        else:
            if Esite.shape != (npos, nsite): raise ValueError(f"Esite array must have shape ({npos}, {nsite})")
    lib.solveSiteOccupancies(npos, ptips, Qtips, Qout, _np_as(Econf, c_double_p), _np_as(Esite, c_double_p), solver_type)
    return Qout, Esite, Econf

# void solveSiteOccupancies_old( int npos, double* ptips_, double* Qtips, int nsite, double* spos, const double* rot, const double* MultiPoles, const double* Esite, double* Qout, double E_Fermi, double cCoupling, double temperature ){
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

# void STM_map(int npos, double* ptips_, double* Qtips, double* Qsites, double* Iout, double decay, bool bOccupied ){
lib.STM_map.argtypes = [c_int, array2d, array1d, array2d, array1d, c_double, c_bool ]
lib.STM_map.restype = None
def getSTM_map(ptips, Qtips, Qsites, Iout=None, decay=1.0, bOccupied=False ):
    npos = len(ptips)
    if Iout is None:  Iout = np.zeros(npos)
    lib.STM_map(npos, ptips, Qtips, Qsites, Iout, decay, bOccupied )
    return Iout

# void solveHamiltonians(int npos, double* ptips_, double* Qtips, double* Qsites, double* evals, double* evecs, double* Hs, double* Gs ) {
lib.solveHamiltonians.argtypes = [c_int, array2d, array1d, array2d, array2d, array3d, c_double_p, c_double_p ]
lib.solveHamiltonians.restype = None
def solveHamiltonians(ptips, Qtips, Qsites=None, evals=None, evecs=None, Gs=None, Hs=None, bH=False, bG=False, bVec=True ):
    npos = len(ptips)
    ptips = np.array(ptips)
    Qtips = np.array(Qtips)
    if ( evals  is None )          : evals = np.zeros((npos, 3   ))
    if ( evecs  is None ) and bVec : evecs = np.zeros((npos, 3, 3))
    if ( Hs     is None ) and bH   : Hs    = np.zeros((npos, 3, 3))
    if ( Gs     is None ) and bG   : Gs    = np.zeros((npos, 3, 3))
    lib.solveHamiltonians( npos, ptips, Qtips, Qsites, evals, evecs, _np_as(Hs,c_double_p), _np_as(Gs,c_double_p) )
    return evals, evecs, Hs, Gs

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
    
def makeRotMats(phis, nsite=3 ):
    rot = np.zeros((nsite,3,3))
    ca = np.cos(phis)
    sa = np.sin(phis)
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


def confsFromStrings( strs ):
    '''
    gets list of sting like "010", "001" etc and creates list of numpy arrays like [ [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0] ]
    '''
    nconfs = len(strs)
    confs = [ [ float(ch) for ch in s ] for s in strs ]
    confs = np.array(confs) 
    if (confs.shape[1] != nsite) or (confs.shape[0] != nconfs): raise ValueError(f"confs must have shape ({nconfs}, {nsite})\n", confs.shape, strs)
    return confs

def colorsFromStrings(strs, hi="ff", lo="00"):
    '''
    Gets list of strings like "101", "001" etc. and creates list of color strings like "#FF00FF", "#0000FF"
    '''
    def to_hex(s):
        return ''.join([ hi if c == '1' else lo for c in s])
    colors = ['#' + to_hex(s) for s in strs]
    if any(len(s) != 3 for s in strs):  raise ValueError("Each input string must have exactly 3 characters")
    return colors

def fermi(E, E_fermi, T):
    """Calculate Fermi-Dirac distribution with numerical safeguards."""
    kB = 8.617333262145e-5  # eV/K
    # Add numerical safeguards for large exponentials
    x = (E - E_fermi) / (kB * T)
    # Use log-sum-exp trick for numerical stability
    x_safe = np.minimum(x, 700)  # Prevent overflow in exp
    return 1.0 / (1.0 + np.exp(x_safe))

def calculate_site_current(ps, site_pos, site_E, E_fermi_tip, E_fermi_sub, decay=0.7, T=300, rho_tip=1.0, rho_sub=1.0, M=None ):
    """Calculate tunneling current for a single site using Fermi Golden Rule.
    
    Mathematical Description:
    ------------------------
    For a single site, the current is given by:
    I_site ∝ |M|^2 ρ_tip ρ_sub [f_tip(E_site) - f_sub(E_site)]
    
    where:
    - Matrix element: M = M_0 exp(-κr)
    - κ: decay constant
    - r: tip-site distance
    - E_site: site energy level
    - f_tip, f_sub: Fermi-Dirac distributions
    
    Args:
        ps (ndarray): Tip positions (n_points, 3)
        site_pos (ndarray): Position of the site (3,)
        site_E (ndarray): Site energy at each tip position (n_points,)
        E_fermi_tip (float): Fermi energy of the tip
        E_fermi_sub (float): Fermi energy of the substrate
        decay (float): Decay constant for tunneling matrix element
        T (float): Temperature in Kelvin
        rho_tip (float): Density of states of the tip (assumed constant)
        rho_sub (float): Density of states of the substrate (assumed constant)
    Returns:
        ndarray: Tunneling current through this site for each tip position (n_points,)
    """

    # Calculate matrix elements for all positions
    if M is None:
        # Calculate distances between all tip positions and the site
        dr = ps - site_pos
        distances = np.sqrt(np.sum(dr*dr, axis=1))  # Shape: (n_points,)
        M  = np.exp(-decay * distances)  # Shape: (n_points,)
    
    # Calculate Fermi functions
    f_tip = fermi(site_E, E_fermi_tip, T)  # Shape: (n_points,)
    f_sub = fermi(site_E, E_fermi_sub, T)  # Shape: (n_points,)
    
    # Calculate current:     I ∝ |M|^2 * ρ_tip * ρ_sub * (f_tip - f_sub)
    current = M*M * rho_tip * rho_sub * (f_tip - f_sub)
    
    return current

def calculate_tunneling_current(ps, Esite, E_fermi_tip, E_fermi_sub, decay=0.7, T=300, rho_tip=1.0, rho_sub=1.0):
    """Calculate total tunneling current using Fermi Golden Rule by summing over all sites.
    
    Mathematical Description:
    ------------------------
    The total current is the sum over all sites:
    I_total = Σ_sites I_site
    
    where I_site is calculated by calculate_site_current()
    
    Args:
        ps (ndarray): Tip positions (n_points, 3)
        Esite (ndarray): Site energies (n_points, n_sites)
        E_fermi_tip (float): Fermi energy of the tip
        E_fermi_sub (float): Fermi energy of the substrate
        decay (float): Decay constant for tunneling matrix element
        T (float): Temperature in Kelvin
        rho_tip (float): Density of states of the tip (assumed constant)
        rho_sub (float): Density of states of the substrate (assumed constant)
    Returns:
        ndarray: Total tunneling current for each tip position
    """
    # Get positions of sites from global parameters
    spos = np.array(spos_)  # Shape: (n_sites, 3)
    n_sites = len(spos)
    
    # Initialize total current
    total_current = np.zeros(len(ps))
    
    # Sum contributions from each site
    for i in range(n_sites):
        site_current = calculate_site_current( ps, spos[i], Esite[:,i], E_fermi_tip, E_fermi_sub, decay, T, rho_tip, rho_sub )
        total_current += site_current
    
    return total_current