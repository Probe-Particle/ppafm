#!/usr/bin/env python3

import numpy as np
import os
import ctypes
from ctypes import c_void_p, c_int, c_double, c_bool
from cpp_utils import compile_lib, work_dir, _np_as, c_double_p, c_int_p

verbosity = 0 

# Compile and load the C++ library at module level
def compile_and_load(name='pauli_lib', bASAN=False):
    """Compile and load the C++ library"""
    cpp_dir = os.path.join(work_dir(__file__), '../cpp')
    so_name = name + '.so'
    # remove existing library if it exists
    try:
        os.remove(os.path.join(cpp_dir, so_name))
    except:
        pass
    compile_lib(name, path=cpp_dir, clean=True, bASAN=bASAN, bDEBUG=True)
    
    # When using ASan, ensure it's loaded first
    if bASAN:
        # Get ASan library path
        asan_lib = os.popen('gcc -print-file-name=libasan.so').read().strip()
        # Load ASan first, then our library
        ctypes.CDLL(asan_lib, mode=ctypes.RTLD_GLOBAL)
        
    return ctypes.CDLL(os.path.join(cpp_dir, so_name), mode=ctypes.RTLD_GLOBAL)

# Load the library
lib = compile_and_load()

# New optimized workflow functions
lib.create_solver.argtypes = [c_int, c_int, c_int]
lib.create_solver.restype = c_void_p

# Step 2: Set lead parameters
lib.set_lead.argtypes = [c_void_p, c_int, c_double, c_double]
lib.set_lead.restype = None

# Step 3: Set tunneling amplitudes
lib.set_tunneling.argtypes = [c_void_p, c_double_p]
lib.set_tunneling.restype = None

# Step 4: Set Hsingle (single-particle Hamiltonian)
lib.set_hsingle.argtypes = [c_void_p, c_double_p]
lib.set_hsingle.restype = None

# Step 5: Generate Pauli factors
lib.generate_pauli_factors.argtypes = [c_void_p, c_double, c_int_p]
lib.generate_pauli_factors.restype = None

# Step 6: Generate kernel matrix
lib.generate_kernel.argtypes = [c_void_p]
lib.generate_kernel.restype = None

lib.solve_pauli.argtypes = [c_void_p]
lib.solve_pauli.restype = None

# double solve_hsingle( void* solver_ptr, double* hsingle, double W, int ilead, int* state_order ){
lib.solve_hsingle.argtypes = [c_void_p, c_double_p, c_double, c_int, c_int_p]
lib.solve_hsingle.restype = c_double

# double scan_current(void* solver_ptr, int npoints, double* hsingles, double* Ws, double* VGates, int* state_order, double* out_current) {
#double scan_current(void* solver_ptr, int npoints, double* hsingles, double* Ws, double* VGates, double* TLeads, int* state_order, double* out_current) {
lib.scan_current.argtypes = [c_void_p, c_int, c_double_p, c_double_p, c_double_p, c_double_p, c_int_p, c_double_p, c_bool]
lib.scan_current.restype = c_double


lib.get_kernel.argtypes = [c_void_p, c_double_p]
lib.get_kernel.restype = None

lib.get_probabilities.argtypes = [c_void_p, c_double_p]
lib.get_probabilities.restype = None

lib.get_energies.argtypes = [c_void_p, c_double_p]
lib.get_energies.restype = None

lib.calculate_current.argtypes = [c_void_p, c_int]
lib.calculate_current.restype = c_double

lib.delete_pauli_solver.argtypes = [c_void_p]
lib.delete_pauli_solver.restype = None

lib.get_coupling.argtypes = [c_void_p, c_double_p]
lib.get_coupling.restype = None

lib.get_pauli_factors.argtypes = [c_void_p, c_double_p]
lib.get_pauli_factors.restype = None

# computeCombinedEnergies( int nTip, double* pTips,  double* pSite, double E0, double VBias, double Rtip, double zV0, int order, double* cs, double* Eout ) {
lib.computeCombinedEnergies.argtypes = [c_int, c_double_p, c_double_p, c_double, c_double, c_double, c_double, c_int, c_double_p, c_double_p]
lib.computeCombinedEnergies.restype = None
def computeCombinedEnergies( pTips, VBias, cs, pSite=[0.0,0.0,0.0], E0=0.0, Rtip=1.0, zV0=-2.0, order=1, Eout=None, bMakeArrays=True ):
    nTip = len(pTips)
    if bMakeArrays:
        pSite = np.array(pSite, dtype=np.float64)
        pTips = np.array(pTips, dtype=np.float64)
        cs    = np.array(cs,    dtype=np.float64)
    if Eout is None:
        Eout = np.zeros(nTip, dtype=np.float64)
    lib.computeCombinedEnergies(nTip, _np_as(pTips, c_double_p), _np_as(pSite, c_double_p), E0, VBias, Rtip, zV0, order, _np_as(cs, c_double_p), _np_as(Eout, c_double_p))
    return Eout



class PauliSolver:
    """Python wrapper for C++ PauliSolver class"""
    
    def __init__(self, nSingle=None, nleads=None, verbosity=0):
        self.verbosity = verbosity
        self.solver    = None
        if nSingle is not None and nleads is not None:
            self.create_solver(nSingle, nleads)
    
    # Methods for optimized workflow
    def create_solver(self, nSingle, nleads):
        self.solver = lib.create_solver(nSingle, nleads, self.verbosity)

    def set_lead(self, leadIndex, mu, temp):
        lib.set_lead(self.solver, leadIndex, mu, temp)
    
    def set_tunneling(self, tunneling_amplitudes):
        tunneling_amplitudes = np.ascontiguousarray(tunneling_amplitudes, dtype=np.float64)
        lib.set_tunneling(self.solver, _np_as(tunneling_amplitudes, c_double_p))
    
    def set_hsingle(self, hsingle):
        hsingle = np.ascontiguousarray(hsingle, dtype=np.float64)
        lib.set_hsingle(self.solver, _np_as(hsingle, c_double_p))
    
    def generate_pauli_factors(self, W=0.0, state_order=None):
        lib.generate_pauli_factors(self.solver, W, _np_as(state_order, c_int_p))
    
    def generate_kernel(self):
        lib.generate_kernel(self.solver)
    
    def solve(self):
        lib.solve_pauli(self.solver)

    def solve_hsingle(self, hsingle, W, ilead, state_order):
        return lib.solve_hsingle(self.solver, _np_as(hsingle, c_double_p), W, ilead, _np_as(state_order, c_int_p))
    
    def scan_current(self, hsingles=None, Ws=None, VGates=None, hsingle=None, W=0.0, VGate=0.0, TLeads=None, state_order=None, out_current=None, bOmp=False):
        if hsingle is not None:
            npoints = len(hsingle)
        elif Ws is not None:
            npoints = len(Ws)
        elif VGates is not None:
            npoints = len(VGates)
        if hsingles is None:
            hsingles_ = np.zeros( (npoints,) + hsingles.shape )
            hsingles_[:,:,:] = hsingles[None,:,:]
            hsingles = hsingles_
        if Ws is None:
            Ws = np.ones(npoints, dtype=np.float64)*W
        if VGates is None:
            VGates = np.zeros(npoints, dtype=np.float64)
            VGates[:,:] = VGate[None,:]
        #if TLeads is None:
        #    TLeads = np.zeros((nleads, nSingle), dtype=np.float64)
        if state_order is not None:
            state_order = np.ascontiguousarray(state_order, dtype=np.int32)
        if out_current is None:
            out_current = np.zeros(npoints, dtype=np.float64)
        lib.scan_current(self.solver, npoints, _np_as(hsingles, c_double_p), _np_as(Ws, c_double_p), _np_as(VGates, c_double_p), _np_as(TLeads, c_double_p), _np_as(state_order, c_int_p), _np_as(out_current, c_double_p), bOmp)
        return out_current
    
    def get_energies(self, nstates):
        energies = np.zeros(nstates)
        lib.get_energies(self.solver, _np_as(energies, c_double_p))
        return energies

    def get_kernel(self, nstates):
        kernel = np.zeros((nstates, nstates))
        lib.get_kernel(self.solver, _np_as(kernel, c_double_p))
        return kernel
    
    def get_probabilities(self, nstates):
        probabilities = np.zeros(nstates)
        lib.get_probabilities(self.solver, _np_as(probabilities, c_double_p))
        return probabilities
    
    def calculate_current(self, lead):
        return lib.calculate_current(self.solver, lead)
    
    def get_coupling(self, NLeads, NStates):
        coupling = np.zeros((NLeads, NStates, NStates))
        lib.get_coupling(self.solver, _np_as(coupling, c_double_p))
        return coupling
    
    def get_pauli_factors(self, NLeads, NStates):
        pauli_factors = np.zeros((NLeads, NStates*2, 2))
        lib.get_pauli_factors(self.solver, _np_as(pauli_factors, c_double_p))
        return pauli_factors
    
    def cleanup(self):
        """Clean up C++ solver instance"""
        if self.solver is not None:
            lib.delete_pauli_solver(self.solver)
            self.solver = None

# ========== python

# def prepare_leads_cpp(
#     muS    = 0.0,   # substrate chemical potential
#     muT    = 0.0,   # tip chemical potential
#     Temp   = 0.224, # temperature in meV
#     VTs = [ 0.1, 0.1, 0.1 ]
# ):
#     """Prepare static inputs that don't change with eps"""
#     lead_mu    = np.array([muS, muT ])
#     lead_temp  = np.array([Temp, Temp])
#     lead_gamma = np.array([GammaS, GammaT])
#     TLeads = np.array([
#         [VS, VS, VS],
#         VTs
#     ])    
#     return TLeads, lead_mu, lead_temp, lead_gamma

def prepare_hsingle_cpp(eps1, eps2, eps3, t=0.0 ):
    """Prepare dynamic inputs that change with eps"""
    # Single particle Hamiltonian
    Hsingle = np.array([
        [eps1, t, 0],
        [t, eps2, t],
        [0, t, eps3]
    ])
    return Hsingle

def count_electrons(state):
    """Count number of electrons in a state"""
    return bin(state).count('1')


def run_cpp_scan(params, Es, Ts, scaleE=1.0 ):
    """Run C++ Pauli simulation for current calculation"""
    NSingle = int(params['NSingle'])
    NLeads = 2
    
    # Get parameters
    W     = params['W']*scaleE
    VBias = params['VBias']*scaleE
    Temp  = params['Temp']
    VS    = np.sqrt(params['GammaS']/np.pi)
    VT    = np.sqrt(params['GammaT']/np.pi)
    
    # Initialize solver
    pauli = PauliSolver(NSingle, NLeads, verbosity=verbosity)
    
    # Set up leads
    pauli.set_lead(0, 0.0,  Temp)  # Substrate lead (mu=0)
    pauli.set_lead(1, VBias, Temp)  # Tip lead (mu=VBias)
    
    npoints = len(Es)

    TLeads = np.zeros((npoints, NLeads, NSingle), dtype=np.float64)
    hsingles = np.zeros((npoints, 3, 3))
    for i in range(NSingle):
        hsingles[:,i,i] = Es[:,i]*scaleE
        TLeads  [:,0,i] = VS
        TLeads  [:,1,i] = VT*Ts[:,i]
    
    # Set up other parameters
    Ws = np.full(npoints, W)
    VGates = np.zeros((npoints, NLeads))
    state_order = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)
        
    currents = pauli.scan_current( hsingles=hsingles, Ws=Ws, VGates=VGates, TLeads=TLeads, state_order=state_order )
    return currents

def run_cpp_scan_2D(params, Es, Ts, Vbiases, Vbias0=1.0, scaleE=1.0, bE1d=True, nsize=None, bOmp=False ):
    """Run 2D C++ Pauli simulation with variable bias voltages
    
    Args:
        params: Simulation parameters dictionary
        Es: Array of energies for each point (shape [npoints, NSingle])
        Ts: Array of tunneling amplitudes (shape [npoints, NSingle])
        Vbiases: Array of bias voltages to scan (shape [nbias])
        scaleE: Energy scaling factor
        
    Returns:
        Array of currents for each (point, bias) combination (shape [npoints, nbias])
    """
    NSingle = int(params['NSingle'])
    NLeads = 2
    
    # Get parameters
    W     = params['W']*scaleE
    Temp  = params['Temp']
    VS    = np.sqrt(params['GammaS']/np.pi)
    VT    = np.sqrt(params['GammaT']/np.pi)
    
    if nsize is None:
        npoints = len(Es)
        nbias   = len(Vbiases)
    else:
        nbias, npoints = nsize

    
    # Initialize solver
    pauli = PauliSolver(NSingle, NLeads, verbosity=verbosity)
    
    pauli.set_lead(0, 0.0, Temp)  # Substrate lead (mu=0)
    pauli.set_lead(1, 0.0, Temp)  # Tip lead (mu=VBias)

    # Prepare arrays with shape (nbias, npoints, ...)
    hsingles = np.zeros((nbias, npoints, 3, 3))
    TLeads   = np.zeros((nbias, npoints, NLeads, NSingle), dtype=np.float64)
    VGates   = np.zeros((nbias, npoints, NLeads))
    
    # Set up leads using numpy broadcasting
    #VGates[..., 0] = 0.0  # Lead 1 always at 0
    VGates[..., 1] = Vbiases[:, None] * scaleE  # Lead 2 at VBias
    
    TLeads  [..., 0, :] = VS
    if bE1d:
        # Create energy array with shape (nbias, npoints, NSingle)
        EVs = (Es[None, :, :] * Vbiases[:, None, None] ) * (scaleE/Vbias0)
        TLeads  [..., 1, :] = VT * Ts[None, :, :]
    else:
        EVs = Es
        TLeads  [..., 1, :] = Ts * VT
    
    # Set energies and tunneling amplitudes
    for i in range(NSingle):
        hsingles[..., i, i] = EVs[..., i]
        #TLeads[..., 0, i] = VS
        #TLeads[..., 1, i] = TVs
    
    # Reshape for scan_current (flatten first two dimensions)
    hsingles = hsingles.reshape(-1, 3, 3)
    VGates = VGates.reshape(-1, NLeads)
    TLeads = TLeads.reshape(-1, NLeads, NSingle)
    Ws = np.full(npoints*nbias, W)
    
    state_order = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)

    print("min,max hsingles: ", hsingles.min(), hsingles.max())
    print("min,max Ws:       ", Ws.min(), Ws.max())
    print("min,max VGates:   ", VGates.min(), VGates.max())
    print("min,max TLeads:   ", TLeads.min(), TLeads.max())
    
    # Run scan and reshape results to [npoints, nbias]
    currents = pauli.scan_current(hsingles=hsingles, Ws=Ws, VGates=VGates, TLeads=TLeads, state_order=state_order, bOmp=bOmp)
    return currents.reshape(nbias,npoints)
