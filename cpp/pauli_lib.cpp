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

double scan_current(void* solver_ptr, int npoints, double* hsingles, double* Ws, double* VGates, double* TLeads, int* state_order, double* out_current) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (!solver) { return 0.0; }
    int nSingle = solver->nSingle;
    int n2 = nSingle*nSingle; 
    int nleads = solver->nleads;
    double base_lead_mu[nleads];  for(int l = 0; l < nleads; l++) { base_lead_mu[l] = solver->leads[l].mu; }
    if(state_order) { 
        solver->init_states_by_charge();
        solver->setStateOrder(state_order); 
    }

    //solver->print_lead_params();
    //solver->print_state_energies();
    //solver->print_tunneling_amplitudes();

    for(int i = 0; i < npoints; i++) {

        if ( solver->verbosity > 0 ){
            printf("\n#####################\n");
            printf("### C++ pauli_lib.cpp : scan_current()  # i: %i \n", i );
            printf("#####################\n\n");
        }
        
        double  W       = Ws[i];
        double* VGate   = VGates  + (i*nleads);
        double* hsingle = hsingles + i*n2;
        //printf( "### scan_current() #i %i eps %16.8f %16.8f %16.8f \n", i, hsingle[0], hsingle[3+1], hsingle[6+2]  );
        //printf("VGate: "); print_vector(VGate, nleads);

        // When I uncoment this it start to be unstable - maybe VGate is not properly initialized ? ( Check it outside python)
        for (int l=0; l<nleads; ++l) { solver->leads[l].mu = base_lead_mu[l] + VGate[l]; }

        double* TLeads_i = TLeads + i*nleads*nSingle;
        solver->setTLeads(TLeads_i);
        solver->calculate_tunneling_amplitudes(TLeads_i);


        //solver->print_lead_params();
        //double current  = solve_hsingle(solver_ptr, hsingle, W, 0, state_order);
        double current  = solve_hsingle(solver_ptr, hsingle, W, 1, 0);
        out_current[i]  = current;
        fflush(stdout);
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

// Cleanup
void delete_pauli_solver(void* solver_ptr) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    delete solver;
}

}
