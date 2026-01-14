import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import os

# --- Helper to pack float4 arrays ---
def pack_float4(xyz_arr, w_arr=None, w_default=0.0):
    """
    Packs (N,3) or (N,4) arrays into an (N,4) contiguous float32 array
    suitable for OpenCL float4 consumption.
    """
    n = len(xyz_arr)
    packed = np.zeros((n, 4), dtype=np.float32)
    
    # Copy XYZ
    if xyz_arr.shape[1] >= 3:
        packed[:, 0:3] = xyz_arr[:, 0:3]
    
    # Fill W
    if xyz_arr.shape[1] == 4:
        packed[:, 3] = xyz_arr[:, 3]
    elif w_arr is not None:
        packed[:, 3] = w_arr
    else:
        packed[:, 3] = w_default
        
    return np.ascontiguousarray(packed)

class PauliSolverCL:
    def __init__(self, nSingle=4, nLeads=2, verbosity=0, ctx=None):
        self.nSingle = nSingle
        self.nStates = 2**nSingle
        self.nLeads = nLeads
        self.verbosity = verbosity
        
        # OpenCL Setup
        if ctx is None:
            # Select platform/device automatically or ask user
            try:
                self.ctx = cl.create_some_context(interactive=False)
            except:
                print("Warning: Automatic context creation failed. Interactive mode:")
                self.ctx = cl.create_some_context(interactive=True)
        else:
            self.ctx = ctx
            
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # Compile Kernels
        path = os.path.dirname(os.path.abspath(__file__))
        cl_file = os.path.join(path, "../cl/PME.cl")
        cl_file = os.path.normpath(cl_file)
        if not os.path.exists(cl_file):
            raise FileNotFoundError(f"Kernel file not found: {cl_file}")
            
        with open(cl_file, "r") as f:
            src = f.read()
            
        try:
            self.prg = cl.Program(self.ctx, src).build()
        except cl.RuntimeError as e:
            print("Build failed:")
            print(e)
            raise

        self._init_lookups()

    def _init_lookups(self):
        # 1. State ordering
        # The current C++/Python reference for nsite=4 uses identity ordering
        # (see pauli.make_state_order()). To enable 1:1 comparisons, keep identity here.
        self.state_order_host = np.arange(self.nStates, dtype=np.int32)
        self.state_order_dev = cl_array.to_device(self.queue, self.state_order_host)
        
        # 2. Defaults
        self.W = 0.0
        # [mu0, T0, mu1, T1]
        self.lead_params = np.array([0.0, 0.01, 0.0, 0.01], dtype=np.float32)

    def set_lead(self, lead_idx, mu, temp):
        if lead_idx < 2:
            self.lead_params[lead_idx*2] = mu
            self.lead_params[lead_idx*2+1] = temp

    def scan_current_tip(self, pTips, Vtips, pSites, params, order, cs, 
                         state_order=None, rots=None, bOmp=False, 
                         bMakeArrays=True, Ts=None, return_probs=False, 
                         return_state_energies=False, externTs=False, return_curmat=False):
        """
        Main simulation function.
        
        pTips: (N, 3) array of tip positions.
        Vtips: (N,) array of voltages.
        pSites: (4, 3) or (4, 4) array. If (4,3), W component taken from params[3].
        params: [Rtip, zV0, zVd, Esite, beta, Gamma, W, bMirror, bRamp]
        """
        
        # ----------------------------------------------------------------
        # 1. Data Marshalling (Host -> Device)
        # ----------------------------------------------------------------
        n_pixels = len(pTips)
        n_sites = self.nSingle

        # Pack Tips to float4 (x, y, z, 0.0)
        p_tips_packed = pack_float4(pTips)
        p_tips_cl = cl_array.to_device(self.queue, p_tips_packed)
        
        # Pack Sites to float4 (x, y, z, E0)
        # Note: params[3] is Esite default if not in pSites
        pSites = np.array(pSites)
        e0_default = params[3] if len(params) > 3 else 0.0
        p_sites_packed = pack_float4(pSites, w_default=e0_default)
        p_sites_cl = cl_array.to_device(self.queue, p_sites_packed)
        
        v_tips_cl = cl_array.to_device(self.queue, np.array(Vtips, dtype=np.float32))
        cs_cl = cl_array.to_device(self.queue, np.array(cs, dtype=np.float32))
        params_cl = cl_array.to_device(self.queue, np.array(params, dtype=np.float32))
        
        # Prepare intermediate buffers
        # H_shifts and T_factors are computed by kernel 1, used by kernel 2
        h_shifts_cl = cl_array.zeros(self.queue, (n_pixels, n_sites), dtype=np.float32)
        t_factors_cl = cl_array.zeros(self.queue, (n_pixels, n_sites), dtype=np.float32)

        # ----------------------------------------------------------------
        # 2. Kernel 1: Tip Interaction
        # ----------------------------------------------------------------
        global_size_1 = (n_pixels,)
        local_size_1 = None 
        
        evt1 = self.prg.compute_tip_interaction(
            self.queue, global_size_1, local_size_1,
            np.int32(n_pixels), np.int32(n_sites),
            p_tips_cl.data, p_sites_cl.data, v_tips_cl.data, cs_cl.data,
            params_cl.data, np.int32(order),
            h_shifts_cl.data, t_factors_cl.data
        )

        # ----------------------------------------------------------------
        # 3. Kernel 2: Pauli Master Equation
        # ----------------------------------------------------------------
        
        # Prepare Physics Constants
        # params structure: [Rtip, zV0, zVd, Esite, beta, Gamma, W, ...]
        gamma_val = params[5]
        w_val = params[6]
        
        # Gamma in C++ code usually means Gamma/PI in the rate eq context?
        # The kernel uses standard rate = 2*PI * |T|^2.
        # If C++ VS = Gamma/PI, then C++ Rate = VS * 2*PI = 2*Gamma. 
        # Let's adhere to the standard: 
        # C++ input "Gamma" usually implies the broadening Gamma = 2*pi*|V|^2*rho.
        # To match C++ exactly, we pass Gamma/PI as the base factor if C++ does so.
        # Based on C++ `evalSitesTipsTunneling` using `Amp * exp` and solver using `VS=Gamma/PI`,
        # We should pass `Gamma/PI` to the kernel so kernel doing `2*PI*...` results in `2*Gamma`.
        # Wait, exact match check: 
        # C++: TLeads[...] = Gamma/PI * exp(...). Coupling = T^2. 
        # Rate = Coupling * 2PI = (Gamma/PI)^2 * exp^2 * 2PI = 2/PI * Gamma^2 * exp^2. 
        # This seems odd physically (Gamma squared?).
        # 
        # Let's assume the standard: Rate ~ Gamma. 
        # If user passes Gamma, we pass Gamma/(2*PI) to kernel as 'Gamma0'? 
        # No, let's look at C++ `solve_pme`:
        # `pauli_factors[0] = coupling_val * fermi * 2 * PI;`
        # `coupling_val = tij * tji`.
        # `tij` comes from `TLeads`. 
        # `TLeads` initialized as `Gamma/PI` (VS) or `(Gamma/PI)*exp` (VT).
        # So Rate ~ (Gamma/PI)^2 * 2PI. 
        # This is the "C++ Convention" we must keep.
        
        # Match C++ scan_current_tip_ convention:
        #   VS = Gamma/pi; VT = Gamma/pi; coupling ~ (VS)^2, rates use *2*pi
        # Here kernel uses:
        #   rate0 = Gamma0 * (...) * 2*pi
        #   rate1 = (Gamma1 * T^2) * (...) * 2*pi
        # Therefore pass Gamma0=Gamma1=(Gamma/pi)^2.
        gamma_input = (gamma_val / np.pi) ** 2
        
        lead_params_cl = cl_array.to_device(self.queue, self.lead_params)
        H_single_cl = cl_array.zeros(self.queue, (n_sites, n_sites), dtype=np.float32)
        # If Wij is mostly constant W, we handle it via W_scalar in kernel. 
        # If complex Wij is needed, we would upload it here.
        Wij_dummy_cl = cl_array.zeros(self.queue, (n_sites, n_sites), dtype=np.float32)
        
        out_current_cl = cl_array.zeros(self.queue, n_pixels, dtype=np.float32)

        out_probs_cl = None
        out_stateEs_cl = None
        out_K_cl = None
        out_curmat_cl = None
        if return_probs:
            out_probs_cl = cl_array.zeros(self.queue, n_pixels * self.nStates, dtype=np.float32)
        if return_state_energies:
            out_stateEs_cl = cl_array.zeros(self.queue, n_pixels * self.nStates, dtype=np.float32)
        if (return_probs or return_state_energies) and (n_pixels == 1):
            out_K_cl = cl_array.zeros(self.queue, n_pixels * self.nStates * self.nStates, dtype=np.float32)
        if return_curmat and (n_pixels == 1):
            out_curmat_cl = cl_array.zeros(self.queue, n_pixels * self.nStates * self.nStates, dtype=np.float32)

        # Workgroup setup: 1 WG per pixel, 16 threads per WG
        global_size_2 = (n_pixels * 16,)
        local_size_2 = (16,)
        
        evt2 = self.prg.solve_pme(
            self.queue, global_size_2, local_size_2,
            np.int32(n_pixels), np.int32(n_sites), np.int32(self.nStates),
            h_shifts_cl.data, t_factors_cl.data,
            v_tips_cl.data,
            lead_params_cl.data, H_single_cl.data,
            np.float32(w_val),
            np.float32(gamma_input), np.float32(gamma_input),
            self.state_order_dev.data,
            out_current_cl.data,
            out_curmat_cl.data if out_curmat_cl is not None else None,
            out_K_cl.data if out_K_cl is not None else None,
            out_probs_cl.data if out_probs_cl is not None else None,
            out_stateEs_cl.data if out_stateEs_cl is not None else None
        )

        # ----------------------------------------------------------------
        # 4. Fetch Results
        # ----------------------------------------------------------------
        # Use .get() to bring back to host (numpy float32) -> convert to float64 if needed
        currents = out_current_cl.get().astype(np.float64)
        
        Es = None
        Ts = None
        Probs = None
        StateEs = None
        K = None
        CurMat = None
        
        if return_state_energies:
            Es = h_shifts_cl.get().astype(np.float64)
            
        if True: # Always return Ts as per request or debug
            Ts = t_factors_cl.get().astype(np.float64)

        if out_probs_cl is not None:
            Probs = out_probs_cl.get().astype(np.float64).reshape(n_pixels, self.nStates)
        if out_stateEs_cl is not None:
            StateEs = out_stateEs_cl.get().astype(np.float64).reshape(n_pixels, self.nStates)
        if out_K_cl is not None:
            K = out_K_cl.get().astype(np.float64).reshape(n_pixels, self.nStates, self.nStates)
        if out_curmat_cl is not None:
            CurMat = out_curmat_cl.get().astype(np.float64).reshape(n_pixels, self.nStates, self.nStates)

        return currents, Es, Ts, Probs, StateEs, K, CurMat

    def cleanup(self):
        # PyOpenCL handles cleanup via GC, but good to be explicit if needed
        pass