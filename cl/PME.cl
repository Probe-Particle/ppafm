// pauli_kernels.cl
// Single precision (float) implementation for speed.

#define N_STATES 16
#define N_SITES 4
#define PI 3.14159265359f

// ======================================================================
// Helper Functions
// ======================================================================

inline float fermi(float E, float mu, float T) {
    return 1.0f / (1.0f + exp((E - mu) / T));
}

inline float eval_multipole(float3 d, int order, __global const float* cs) {
    float r2 = dot(d, d);
    float ir2 = 1.0f / r2; // 1/r^2
    float E = cs[0]; // Monopole

    if (order > 0) { // Dipole
        E += ir2 * (cs[1]*d.x + cs[2]*d.y + cs[3]*d.z);
    }
    if (order > 1) { // Quadrupole
        float ir4 = ir2 * ir2;
        E += ir4 * ( (cs[4]*d.x + cs[9]*d.y)*d.x +
                     (cs[5]*d.y + cs[7]*d.z)*d.y +
                     (cs[6]*d.z + cs[8]*d.x)*d.z );
    }
    return sqrt(ir2) * E; // 1/r * E
}

// ======================================================================
// Kernel 1: Tip Field Calculation (Standalone)
// ======================================================================
// Computes H_single diagonal shifts and Tunneling factors.
// Optimized with float4 vectors.

__kernel void compute_tip_interaction(
    int n_pixels,
    int n_sites,
    __global const float4* restrict p_tips,      // [n_pixels] (x,y,z, ignored)
    __global const float4* restrict p_sites,     // [n_sites] (x,y,z, E0)
    __global const float*  restrict rots,        // [n_sites * 9] row-major 3x3 per site (identity if unused)
    __global const float*  restrict v_tips,      // [n_pixels]
    __global const float*  restrict multipole_cs,// [10]
    
    // Params: [Rtip, zV0, zVd, Esite_ref, beta, Gamma, W, bMirror, bRamp]
    __global const float* restrict params,
    int order,
    
    // Outputs
    __global float* restrict out_H_shifts,       // [n_pixels * n_sites]
    __global float* restrict out_T_factors       // [n_pixels * n_sites]
) {
    int gid = get_global_id(0);
    if (gid >= n_pixels) return;

    float4 tip_data = p_tips[gid];
    float3 tip_pos = tip_data.xyz;
    float v_bias = v_tips[gid];

    // Local registers for params to avoid global reads
    float Rtip    = params[0];
    float zV0     = params[1];
    float zVd     = params[2];
    float beta    = params[4];
    float bMirror = params[7]; // >0.5 check
    float bRamp   = params[8]; // >0.5 check

    float zV1 = tip_pos.z + zVd; 

    // Loop over sites (usually 4, keep unrolled or simple loop)
    for (int i = 0; i < n_sites; i++) {
        float4 site_data = p_sites[i];
        float3 site_pos = site_data.xyz;
        float E_base = site_data.w;

        // rotation matrix for this site (row-major)
        const float* R = rots + i * 9;

        // --- Electrostatics ---
        float3 d = tip_pos - site_pos;
        float3 tip_mirror = (float3)(tip_pos.x, tip_pos.y, 2.0f*zV0 - tip_pos.z);
        float3 d_mir = tip_mirror - site_pos;

        // rotate into site frame to match C++ evalMultipoleMirror(rot)
        float3 d_rot;
        d_rot.x = R[0]*d.x + R[1]*d.y + R[2]*d.z;
        d_rot.y = R[3]*d.x + R[4]*d.y + R[5]*d.z;
        d_rot.z = R[6]*d.x + R[7]*d.y + R[8]*d.z;

        float3 d_mir_rot;
        d_mir_rot.x = R[0]*d_mir.x + R[1]*d_mir.y + R[2]*d_mir.z;
        d_mir_rot.y = R[3]*d_mir.x + R[4]*d_mir.y + R[5]*d_mir.z;
        d_mir_rot.z = R[6]*d_mir.x + R[7]*d_mir.y + R[8]*d_mir.z;

        float E_val = eval_multipole(d_rot, order, multipole_cs);
        
        if (bMirror > 0.5f) {
            E_val -= eval_multipole(d_mir_rot, order, multipole_cs);
        }
        
        E_val *= (v_bias * Rtip);

        if (bRamp > 0.5f) {
            float ramp = (site_pos.z - zV0) / (zV1 - zV0);
            ramp = clamp(ramp, 0.0f, 1.0f);
            // Monopole term * V_local. multipole_cs[0] is Q (usually 1.0)
            E_val += multipole_cs[0] * v_bias * ramp; 
        }

        // --- Tunneling --- (use unrotated distance, matching C++ evalMultipoleMirror path)
        float r = length(d);
        float t_fac = native_exp(-beta * r);

        out_H_shifts[gid * n_sites + i] = E_base + E_val;
        out_T_factors[gid * n_sites + i] = t_fac;
    }
}

// ======================================================================
// Kernel 2: Pauli Master Equation Solver
// ======================================================================
// 1 Workgroup per Pixel. 
// Uses Shared Memory (LDS) for the 16x16 Matrix.
// Implements Parallel Gauss-Jordan Elimination.

__kernel void solve_pme(
    int n_pixels,
    int n_sites,    // 4
    int n_states,   // 16
    
    // Inputs
    __global const float* restrict H_shifts,    // [n_pixels * n_sites]
    __global const float* restrict T_factors,   // [n_pixels * n_sites]
    __global const float* restrict v_tips,      // [n_pixels] (used as mu1 per pixel)
    
    // Global Config
    __global const float* restrict lead_params, // [mu0, T0, mu1, T1]
    __global const float* restrict H_single_base,// [n_sites * n_sites]
    __global const float* restrict Wij,          // [n_sites * n_sites] or NULL
    float W_scalar,
    float Gamma0,   // Substrate
    float Gamma1,   // Tip
    
    // Lookup
    __global const int* restrict state_order,   // ThreadID -> Bitmask
    
    // Output
    __global float* restrict out_current,
    __global float* restrict out_curmat,        // [n_pixels * n_states * n_states] or NULL; per-transition current contrib (b->c)
    __global float* restrict out_K,             // [n_pixels * n_states * n_states] or NULL
    __global float* restrict out_probs,         // [n_pixels * n_states] or NULL
    __global float* restrict out_stateEs        // [n_pixels * n_states] or NULL
) {
    // Identify Pixel and State
    int pix_id = get_group_id(0);
    int tid    = get_local_id(0); // 0..15

    if (pix_id >= n_pixels) return;

    // --- Local Memory Allocation ---
    // Matrix size 16x16 floats = 1KB. Very small.
    // Use padding if necessary to avoid bank conflicts (16+1), 
    // but for simple broadcast reads on modern GPU, 16 is often fine.
    __local float Mat[N_STATES][N_STATES]; 
    __local float RHS[N_STATES];
    __local float Energies[N_STATES];
    __local float Probs[N_STATES];

    // Initialize Local Memory
    #pragma unroll
    for (int j = 0; j < N_STATES; j++) {
        Mat[tid][j] = 0.0f;
    }
    // Set constraint: Sum(P) = 1. We replace Row 0 later.
    RHS[tid] = (tid == 0) ? 1.0f : 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    // ------------------------------------------------------------------
    // Phase 1: Compute Many-Body Energies
    // ------------------------------------------------------------------
    int mask = state_order[tid];
    float my_energy = 0.0f;
    int nocc = 0;

    // Single Particle (Diagonal + Shift)
    for (int i = 0; i < n_sites; i++) {
        if ((mask >> i) & 1) {
            float h_val = H_single_base[i * n_sites + i] + H_shifts[pix_id * n_sites + i];
            my_energy += h_val;
            nocc++;
        }
    }
    // Coulomb (W/Wij)
    // Match C++ calculate_state_energy(): add (Wij ? Wij[i,j] : W_scalar) for each occupied pair (i<j).
    if (Wij) {
        for (int i = 0; i < n_sites; i++) {
            if ((mask >> i) & 1) {
                for (int j = i + 1; j < n_sites; j++) {
                    if ((mask >> j) & 1) {
                        my_energy += Wij[i * n_sites + j];
                    }
                }
            }
        }
    }
    Energies[tid] = my_energy;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // ------------------------------------------------------------------
    // Phase 2: Build Rate Matrix
    // ------------------------------------------------------------------
    // Thread 'tid' corresponds to State 'b'.
    // We compute transitions entering 'b' from 'c' (Mat[b][c])
    // and transitions leaving 'b' to 'c' (subtracted from Mat[b][b]).

    int b = tid;
    int mask_b = mask;
    float diag_loss = 0.0f;

    // Pre-load parameters into registers
    float mu0 = lead_params[0]; float T0 = lead_params[1];
    float mu1 = v_tips[pix_id]; float T1 = lead_params[3];

    // Iterate over all possible neighbor states 'c'
    #pragma unroll 4
    for (int c = 0; c < N_STATES; c++) {
        if (b == c) continue;

        int mask_c = state_order[c];
        int diff = mask_b ^ mask_c;

        // Check for single electron hop (popcount == 1)
        if (popcount(diff) == 1) {
            // Identify site index
            int site_idx = ctz(diff); // count trailing zeros
            
            bool bit_b = ((mask_b >> site_idx) & 1) != 0;
            bool bit_c = ((mask_c >> site_idx) & 1) != 0;
            bool added = (bit_c && !bit_b); // b->c adds electron to dot

            // Couplings (C++ convention: rate ~ coupling_val * f * 2*pi; coupling_val ~ (Gamma/pi)^2 * T^2)
            float t_val = T_factors[pix_id * n_sites + site_idx];
            float coup0 = Gamma0;
            float coup1 = Gamma1 * t_val * t_val;

            if (added) {
                // b(lower) -> c(higher): energy_diff = E[c]-E[b]
                float E_diff = Energies[c] - Energies[b];
                float f0 = fermi(E_diff, mu0, T0);
                float f1 = fermi(E_diff, mu1, T1);
                float enter = (coup0 * f0 + coup1 * f1) * 2.0f * PI;          // b->c (electron enters dot)  => outflow from b
                float leave = (coup0 * (1.0f - f0) + coup1 * (1.0f - f1)) * 2.0f * PI; // c->b (electron leaves dot) => inflow to b
                diag_loss -= enter;
                Mat[b][c] += leave;
            } else {
                // c(lower) -> b(higher): energy_diff = E[b]-E[c]
                float E_diff = Energies[b] - Energies[c];
                float f0 = fermi(E_diff, mu0, T0);
                float f1 = fermi(E_diff, mu1, T1);
                float enter = (coup0 * f0 + coup1 * f1) * 2.0f * PI;          // c->b (electron enters dot)  => inflow to b
                float leave = (coup0 * (1.0f - f0) + coup1 * (1.0f - f1)) * 2.0f * PI; // b->c (electron leaves dot) => outflow from b
                diag_loss -= leave;
                Mat[b][c] += enter;
            }
        }
    }
    Mat[b][b] += diag_loss;

    barrier(CLK_LOCAL_MEM_FENCE);

    // ------------------------------------------------------------------
    // Phase 3: Normalization (Replace Row 0)
    // ------------------------------------------------------------------
    if (tid == 0) {
        #pragma unroll
        for (int k = 0; k < N_STATES; k++) {
            Mat[0][k] = 1.0f;
        }
        RHS[0] = 1.0f;
    } else {
        RHS[tid] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Optional debug export: kernel matrix (after normalization row, before solve)
    if (out_K) {
        #pragma unroll
        for (int j = 0; j < N_STATES; j++) {
            out_K[pix_id * (N_STATES * N_STATES) + tid * N_STATES + j] = Mat[tid][j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ------------------------------------------------------------------
    // Phase 4: Parallel Gauss-Jordan Elimination
    // ------------------------------------------------------------------
    // We transform Mat into Identity. 
    // Loop over pivot column k.
    
    for (int k = 0; k < N_STATES; k++) {
        
        // --- A. Pivot Selection (Partial Pivoting) ---
        // Crucial for numerical stability even in float.
        // Thread 0 finds the best pivot row in column k.
        if (tid == 0) {
            int pivot_row = k;
            float max_val = fabs(Mat[k][k]);
            
            for (int i = k + 1; i < N_STATES; i++) {
                float val = fabs(Mat[i][k]);
                if (val > max_val) {
                    max_val = val;
                    pivot_row = i;
                }
            }
            
            // Swap rows if necessary
            if (pivot_row != k) {
                // Swap RHS
                float tmp_rhs = RHS[k];
                RHS[k] = RHS[pivot_row];
                RHS[pivot_row] = tmp_rhs;
                
                // Swap Matrix Row
                for (int j = k; j < N_STATES; j++) {
                    float tmp = Mat[k][j];
                    Mat[k][j] = Mat[pivot_row][j];
                    Mat[pivot_row][j] = tmp;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- B. Normalize Pivot Row ---
        // Thread k computes inverse pivot
        float pivot_val = Mat[k][k];
        float inv_pivot = 1.0f / pivot_val;

        // Optimization: We could have all threads help normalize the row,
        // but since N=16, Thread k doing it alone is fine and simpler to sync.
        if (tid == k) {
            #pragma unroll
            for (int j = k; j < N_STATES; j++) {
                Mat[k][j] *= inv_pivot;
            }
            RHS[k] *= inv_pivot;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- C. Elimination ---
        // All threads i != k eliminate their element in column k.
        // Row operation: R_i = R_i - Mat[i][k] * R_k
        
        if (tid != k) {
            float factor = Mat[tid][k];
            // Only need to update columns j >= k. 
            // Actually only j > k because Mat[tid][k] becomes 0, 
            // but we don't strictly need to write the 0 if we don't read it again.
            // For correctness of Identity matrix:
            Mat[tid][k] = 0.0f; 
            
            #pragma unroll
            for (int j = k + 1; j < N_STATES; j++) {
                Mat[tid][j] -= factor * Mat[k][j];
            }
            RHS[tid] -= factor * RHS[k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Solution is now in RHS (since Mat is Identity)
    Probs[tid] = RHS[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Optional debug export: probabilities and many-body energies
    if (out_probs) {
        out_probs[pix_id * N_STATES + tid] = Probs[tid];
    }
    if (out_stateEs) {
        out_stateEs[pix_id * N_STATES + tid] = Energies[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ------------------------------------------------------------------
    // Phase 5: Calculate Current
    // ------------------------------------------------------------------
    // Thread 0 sums contributions.
    // I = Sum ( Rate_in - Rate_out ) for Tip Lead
    
    if (tid == 0) {
        float current = 0.0f;

        // Optional export: current contribution matrix in the same spirit as C++ generate_current()
        // Store contrib only for transitions b(lower charge) -> c(higher charge) (added electron).
        if (out_curmat) {
            for (int ij = 0; ij < N_STATES * N_STATES; ij++) {
                out_curmat[pix_id * (N_STATES * N_STATES) + ij] = 0.0f;
            }
        }
        
        // Iterate transitions to accumulate current
        // Optimization: reuse precomputed factors if memory allowed, 
        // but recalculating here saves register pressure.
        
        // C++ convention (PauliSolver::generate_current):
        // For each transition between charge sectors (b lower -> c higher):
        //   I_tip += +P_b * rate_enter  - P_c * rate_leave
        // where rate_enter uses fermi(Ec-Eb), rate_leave uses (1-fermi(Ec-Eb)).
        for (int b = 0; b < N_STATES; b++) {
            int mask_b = state_order[b];
            float P_b = Probs[b];
            for (int c = 0; c < N_STATES; c++) {
                if (b == c) continue;
                int mask_c = state_order[c];
                int diff = mask_b ^ mask_c;
                if (popcount(diff) != 1) continue;
                int site_idx = ctz(diff);
                bool bit_c = ((mask_c >> site_idx) & 1) != 0;
                bool bit_b = ((mask_b >> site_idx) & 1) != 0;
                bool added = (bit_c && !bit_b);
                if (!added) continue; // only count b(lower)->c(higher) once

                float P_c = Probs[c];
                float E_diff = Energies[c] - Energies[b];

                float t_val = T_factors[pix_id * n_sites + site_idx];
                float coupling1 = Gamma1 * t_val * t_val;
                float f1 = fermi(E_diff, mu1, T1);

                float rate_enter = coupling1 * f1 * 2.0f * PI;
                float rate_leave = coupling1 * (1.0f - f1) * 2.0f * PI;

                float contrib = P_b * rate_enter - P_c * rate_leave;
                current += contrib;
                if (out_curmat) {
                    out_curmat[pix_id * (N_STATES * N_STATES) + b * N_STATES + c] = contrib;
                }
            }
        }
        out_current[pix_id] = current;
    }
}