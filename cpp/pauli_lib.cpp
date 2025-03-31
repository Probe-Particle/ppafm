static int _verbosity = 0;

#include "pauli.hpp"
#include <cstdio>
#include <cstring> // for memcpy
#include "print_utils.hpp"

extern "C" {

// Create a PauliSolver instance with basic initialization but without setting parameters
// This follows step 1 in the optimization scheme
void* create_solver(int nSingle, int nleads, int verbosity = 0) {
    setvbuf(stdout, NULL, _IONBF, 0);  // Disable buffering for stdout
    _verbosity = verbosity;
    int nstates = 1 << nSingle;
    PauliSolver* solver = new PauliSolver(nSingle, nstates, nleads, verbosity);
    return solver;
}

// // Set lead parameters for the solver (step 2 in optimization scheme)
// //void set_leads(void* solver_ptr, double* lead_mu, double* lead_temp, double* lead_gamma) {
// void set_leads(void* solver_ptr, double* lead_mu, double* lead_temp) {
//     PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
//     if (solver) {
//         int nleads = solver->nleads;
//         // Set lead parameters one by one for each lead
//         for (int i = 0; i < nleads; i++) {
//             //solver->setLeadParams(i, lead_mu[i], lead_temp[i], lead_gamma[i]);
//             solver->setLeadParams(i, lead_mu[i], lead_temp[i], lead_gamma[i]);
//         }
//     }
// }

void set_lead(void* solver_ptr, int leadIndex, double mu, double temp) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (solver) { solver->setLeadParams(leadIndex, mu, temp); }
}

// Set tunneling amplitudes (step 3 in optimization scheme)
void set_tunneling(void* solver_ptr, double* tunneling_amplitudes) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (solver) {
        solver->setTLeads(tunneling_amplitudes);
        solver->calculate_tunneling_amplitudes(tunneling_amplitudes);
    }
}

// Set Hsingle (single-particle Hamiltonian) (step 4 in optimization scheme)
void set_hsingle(void* solver_ptr, double* hsingle) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (solver) {
        solver->setHsingle(hsingle);
        
        // Initialize state ordering after setting Hsingle
        solver->init_states_by_charge();
    }
}

// Generate Pauli factors (step 5 in optimization scheme)
void generate_pauli_factors(void* solver_ptr, double W, int* state_order ) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (solver) {
        solver->W = W;
        if(state_order) { solver->setStateOrder(state_order); }
        solver->generate_fct();
    }
}

// Generate kernel matrix (step 6 in optimization scheme)
void generate_kernel(void* solver_ptr) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (solver) {
        solver->generate_kern();
    }
}

// Solve the master equation (step 7 in optimization scheme)
void solve_pauli(void* solver_ptr) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (solver) {
        solver->solve();
    }
}

double solve_hsingle( void* solver_ptr, double* hsingle, double W, int ilead, int* state_order ){
    //printf("solve_hsingle() ilead: %d W: %g\n", ilead, W);
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (solver) {
        solver->W = W;
        solver->setHsingle(hsingle);
        if(state_order) { 
            solver->init_states_by_charge();
            solver->setStateOrder(state_order); 
        }
        solver->generate_fct();
        solver->generate_kern();
        solver->solve();
        if(ilead >= 0) {
            return solver->generate_current(ilead);
        }
    }
    return 0.0;
}

double scan_current(void* solver_ptr, int npoints, double* hsingles, double* Ws, double* VGates, int* state_order, double* out_current) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (!solver) { return 0.0; }
    int n2 = solver->nSingle*solver->nSingle; 
    int nleads = solver->nleads;
    double base_lead_mu[nleads]; 
    for(int l = 0; l < nleads; l++) { base_lead_mu[l] = solver->leads[l].mu; }
    if(state_order) { 
        solver->init_states_by_charge();
        solver->setStateOrder(state_order); 
    }
    for(int i = 0; i < npoints; i++) {
        double  W       = Ws[i];
        double* VGate   = VGates  + (i*solver->nleads);
        for (int l=0; l<nleads; ++l) { solver->leads[l].mu = base_lead_mu[l] + VGate[l]; }
        double* hsingle = hsingles + i*n2;
        double current  = solve_hsingle(solver_ptr, hsingle, W, 0, state_order);
        out_current[i]  = current;
    }
    return 0.0;
}




// Calculate current through a lead (step 8 in optimization scheme)
double calculate_current(void* solver_ptr, int lead_idx) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    return solver->generate_current(lead_idx);
}

// Get the kernel matrix
void get_kernel(void* solver_ptr, double* out_kernel) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    int n = solver->nstates;
    std::memcpy(out_kernel, solver->kernel, n * n * sizeof(double));
}

// Get the probabilities
void get_probabilities(void* solver_ptr, double* out_probs) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    int n = solver->nstates;
    std::memcpy(out_probs, solver->probabilities, n * sizeof(double));
}

// Get the energies
void get_energies(void* solver_ptr, double* out_energies) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    int n = solver->nstates;
    std::memcpy(out_energies, solver->energies, n * sizeof(double));
}

// Get the coupling matrix
void get_coupling(void* solver_ptr, double* out_coupling) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    int n = solver->nleads * solver->nstates * solver->nstates;
    std::memcpy(out_coupling, solver->coupling, n * sizeof(double));
}

// Get the Pauli factors
void get_pauli_factors(void* solver_ptr, double* out_pauli_factors) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    int n = solver->nleads * solver->ndm1 * 2;
    std::memcpy(out_pauli_factors, solver->pauli_factors, n * sizeof(double));
}


/*

// For backward compatibility - will be refactored in the future
void* create_pauli_solver(int nstates, int nleads, 
                         double* energies, double* tunneling_amplitudes,
                         double* lead_mu, double* lead_temp, double* lead_gamma,
                         int verbosity = 0) {
    setvbuf(stdout, NULL, _IONBF, 0);  // Disable buffering for stdout
    _verbosity = verbosity;
    
    // For now, assume nSingle = 3 (can be modified later if needed)
    int nSingle = 3;
    PauliSolver* solver = new PauliSolver(nSingle, nstates, nleads, verbosity);
    
    // Copy energies directly
    for (int i = 0; i < nstates; i++) {
        solver->energies[i] = energies[i];
    }
    
    // Set lead parameters one by one for each lead
    for (int i = 0; i < nleads; i++) {
        solver->setLeadParams(i, lead_mu[i], lead_temp[i], lead_gamma[i]);
    }
    
    // Set up coupling matrix
    solver->setTLeads(tunneling_amplitudes);
    
    return solver;
}

// For backward compatibility - will be refactored in the future
void* create_pauli_solver_new(int nSingle, int nstates, int nleads, 
                             double* Hsingle, double W, double* TLeads, 
                             double* lead_mu, double* lead_temp, double* lead_gamma, 
                             int* state_order, int verbosity = 0) {
    setvbuf(stdout, NULL, _IONBF, 0);  // Disable buffering for stdout
    _verbosity = verbosity;
    
    PauliSolver* solver = new PauliSolver(nSingle, nstates, nleads, verbosity);
    
    // Set up lead parameters one by one for each lead
    for (int i = 0; i < nleads; i++) {
        solver->setLeadParams(i, lead_mu[i], lead_temp[i], lead_gamma[i]);
    }
    
    // Set up Hsingle and W
    solver->setHsingle(Hsingle);
    solver->W = W;
    
    // Set up tunneling amplitudes
    solver->setTLeads(TLeads);
    
    // Copy state order if provided
    if (state_order) {
        for (int i = 0; i < nstates; i++) {
            solver->state_order[i] = state_order[i];
        }
    }
    
    // Initialize energies based on Hsingle and W - using the new method name
    solver->updateStateEnergies();
    
    // Generate coupling terms - now requires a parameter
    for (int b = 0; b < nleads; b++) {
        solver->generate_coupling_terms(b);
    }
    
    return solver;
}
*/

// Cleanup
void delete_pauli_solver(void* solver_ptr) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    delete solver;
}

}
