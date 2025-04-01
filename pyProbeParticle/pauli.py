#!/usr/bin/env python3

import numpy as np
import os
import ctypes
from ctypes import c_void_p, c_int, c_double
from cpp_utils import compile_lib, work_dir, _np_as, c_double_p, c_int_p

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

# Set up C++ function signatures at module level
# Original solver creation (for backward compatibility)
# lib.create_pauli_solver.argtypes = [c_int, c_int, c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_int]
# lib.create_pauli_solver.restype = c_void_p

# lib.create_pauli_solver_new.argtypes = [
#     c_int, c_int, c_int,
#     c_double_p, c_double, c_double_p,
#     c_double_p, c_double_p, c_double_p, c_int_p, c_int
# ]
# lib.create_pauli_solver_new.restype = c_void_p

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
lib.scan_current.argtypes = [c_void_p, c_int, c_double_p, c_double_p, c_double_p, c_int_p, c_double_p]
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

    # def create_pauli_solver_new(self, nstates, nleads, Hsingle, W, TLeads, lead_mu, lead_temp, lead_gamma, state_order):
    #     # Ensure arrays are C-contiguous and in the correct format
    #     lead_mu     = np.ascontiguousarray(lead_mu,    dtype=np.float64)
    #     lead_temp   = np.ascontiguousarray(lead_temp,  dtype=np.float64)
    #     lead_gamma  = np.ascontiguousarray(lead_gamma, dtype=np.float64)
    #     nSingle = len(Hsingle)
    #     self.solver = lib.create_pauli_solver_new(
    #         nSingle, nstates, nleads,
    #         _np_as(Hsingle, c_double_p), W, _np_as(TLeads, c_double_p),
    #         _np_as(lead_mu, c_double_p), _np_as(lead_temp, c_double_p), _np_as(lead_gamma, c_double_p),
    #         _np_as(state_order, c_int_p),
    #         self.verbosity
    #     )

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
    
    def scan_current(self, hsingles=None, Ws=None, VGates=None, hsingle=None, W=0.0, VGate=0.0,  state_order=None, out_current=None):
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
        if state_order is not None:
            state_order = np.ascontiguousarray(state_order, dtype=np.int32)
        if out_current is None:
            out_current = np.zeros(npoints, dtype=np.float64)
        lib.scan_current(self.solver, npoints, _np_as(hsingles, c_double_p), _np_as(Ws, c_double_p), _np_as(VGates, c_double_p), _np_as(state_order, c_int_p), _np_as(out_current, c_double_p))
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
