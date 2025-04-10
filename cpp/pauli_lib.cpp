static int _verbosity = 0;

#include "pauli.hpp"
#include <cstdio>
#include <cstring> // for memcpy
#include "print_utils.hpp"
#include <omp.h>
#include <thread>

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

double solve_hsingle( void* solver_ptr, const double* hsingle, double W, int ilead, const int* state_order ){
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


// Worker function for a single thread
void solve_batch(
    PauliSolver* solver_template, // Template to copy from
    int thread_id,
    int start_index,
    int end_index, // Exclusive: process indices [start_index, end_index)
    int nleads,
    int nSingle,
    const double* hsingles,
    const double* Ws,
    const double* VGates,
    const double* TLeads,
    const double* base_lead_mu, // Pass the base mu values
    double* out_current)
{
    // 1. Create a thread-local copy of the solver
    PauliSolver solver_local(solver_template); // Use copy constructor
    int n2 = nSingle * nSingle;

    // printf("Thread %d processing indices [%d, %d)\n", thread_id, start_index, end_index); // Debug

    for (int i = start_index; i < end_index; ++i) {
        // Get pointers for the i-th data point
        double  W       = Ws[i];
        const double* VGate   = VGates + (i * nleads);
        const double* hsingle = hsingles + i * n2;
        const double* TLeads_i = TLeads + i * nleads * nSingle;

        // Update local solver state for this point
        for (int l = 0; l < nleads; ++l) {
            solver_local.leads[l].mu = base_lead_mu[l] + VGate[l];
        }
        solver_local.setTLeads(TLeads_i); // Assuming this doesn't modify internal state needed across iterations
        solver_local.calculate_tunneling_amplitudes(TLeads_i); // Assuming this is okay

        // Solve for the current point
        double current = solve_hsingle(&solver_local, hsingle, W, 1, 0); // Pass address of local solver
        out_current[i] = current;
    }
    // No need to delete solver_local if it's a stack object (created without new)
    // If PauliSolver MUST be heap allocated, you'd new/delete it here.
    // But using the copy constructor like above is usually better if possible.
}


double scan_current_manual_threads( PauliSolver* solver, int npoints, double* hsingles, double* Ws, double* VGates, double* TLeads, int* state_order, double* out_current) {

    int nSingle = solver->nSingle;
    int nleads = solver->nleads;
    std::vector<double> base_lead_mu(nleads); // Use std::vector
    for(int l = 0; l < nleads; l++) { base_lead_mu[l] = solver->leads[l].mu; }

    if(state_order) {
        // This setup should ideally happen *before* creating copies if it affects
        // the base state copied by the constructor.
        solver->init_states_by_charge();
        solver->setStateOrder(state_order);
    }

    // --- Manual Threading Setup ---
    // Use std::thread::hardware_concurrency() or a fixed number
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback if detection fails
    num_threads = std::min((unsigned int)npoints, num_threads); // Don't use more threads than points
    num_threads = 16; // Or force like OMP_NUM_THREADS

    //std::cout << "scan_current_manual_threads() nThreads: " << num_threads << " npoints: " << npoints << std::endl;
    printf("scan_current_manual_threads() nThreads: %d npoints: %d \n", num_threads, npoints);
    

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int batch_size = std::ceil(static_cast<double>(npoints) / num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_index = t * batch_size;
        int end_index = std::min(start_index + batch_size, npoints); // Prevent going past the end

        if (start_index >= npoints) break; // Don't launch threads with no work

        threads.emplace_back( // Launch thread
            solve_batch,
            solver,          // Pass the original solver as a template to copy from
            t,               // Thread ID (for debugging)
            start_index,
            end_index,
            nleads,
            nSingle,
            hsingles,
            Ws,
            VGates,
            TLeads,
            base_lead_mu.data(), // Pass pointer to base mu data
            out_current
        );
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    // --- End Manual Threading ---

    return 0.0; // Or return an appropriate value
}

double scan_current_omp( PauliSolver* solver, int npoints, double* hsingles, double* Ws, double* VGates, double* TLeads, int* state_order, double* out_current) {
    
    int nSingle = solver->nSingle;
    int n2 = nSingle*nSingle; 
    int nleads = solver->nleads;
    double base_lead_mu[nleads];  
    for(int l = 0; l < nleads; l++) { base_lead_mu[l] = solver->leads[l].mu; }
    
    if(state_order) { 
        solver->init_states_by_charge();
        solver->setStateOrder(state_order); 
    }

    // Pre-allocate solvers for all threads
    int nThreads = omp_get_max_threads();
    printf("scan_current_omp()  nThreads: %d npoints: %d \n", nThreads, npoints);
    std::vector<PauliSolver*> solvers(nThreads);
    for(int i=0; i<nThreads; i++) {
        solvers[i] = new PauliSolver(solver);
    }

    #pragma omp parallel
    {

        int nThreads_actual = omp_get_num_threads(); // Get actual number of threads used
        int chunk_size = (npoints + nThreads_actual - 1) / nThreads_actual; // Calculate large chunk size
        #pragma omp for schedule(static, chunk_size) // <<-- ADD THIS SCHEDULE
        for(int i = 0; i < npoints; i++) {

            double  W       = Ws[i];
            double* VGate   = VGates  + (i*nleads);
            double* hsingle = hsingles + i*n2;

            int tid = omp_get_thread_num();
            PauliSolver* solver_local = solvers[tid];

            for (int l=0; l<nleads; ++l) { solver_local->leads[l].mu = base_lead_mu[l] + VGate[l]; }

            double* TLeads_i = TLeads + i*nleads*nSingle;
            solver_local->setTLeads(TLeads_i);
            solver_local->calculate_tunneling_amplitudes(TLeads_i);

            double current  = solve_hsingle(solver_local, hsingle, W, 1, 0);
            out_current[i]  = current;
        }
    }
    
    // Clean up solvers
    for(int i=0; i<nThreads; i++) {
        delete solvers[i];
    }
    
    return 0.0;
}

double scan_current_omp_stackalloc( PauliSolver* solver, int npoints, double* hsingles, double* Ws, double* VGates, double* TLeads, int* state_order, double* out_current) {

    int nSingle = solver->nSingle;
    int n2 = nSingle*nSingle;
    int nleads = solver->nleads;
    // Using std::vector might be slightly safer if nleads could change, but C-array is fine too.
    std::vector<double> base_lead_mu(nleads);
    for(int l = 0; l < nleads; l++) { base_lead_mu[l] = solver->leads[l].mu; }

    if(state_order) {
        // Ensure original solver is set up correctly before copying
        solver->init_states_by_charge();
        solver->setStateOrder(state_order);
    }

    // Get max threads for info, but don't pre-allocate vector
    int nThreads_info = omp_get_max_threads();
    printf("scan_current_omp_stackalloc() nThreads: %d npoints: %d \n", nThreads_info, npoints);

    // No vector of PauliSolver pointers needed

    #pragma omp parallel // The original 'solver' is implicitly shared (read-only access for copying is okay)
    {
        // Each thread will execute the loop iterations assigned by OpenMP

        // Let OpenMP handle the scheduling (default static is often fine,
        // or try schedule(dynamic) if solve_hsingle varies significantly)
        #pragma omp for // schedule(static) is default usually
        for(int i = 0; i < npoints; i++) {

            // --- Create solver copy ON STACK inside the loop ---
            // The original 'solver' pointer is shared, dereference it for the copy constr.
            PauliSolver solver_local(*solver);

            // Get pointers for the i-th data point (input arrays are shared)
            // Use const if the data isn't modified
            double  W            = Ws[i];
            const double* VGate  = VGates  + (i * nleads);
            const double* hsingle = hsingles + i * n2; // Mark as const if possible
            const double* TLeads_i = TLeads + i * nleads * nSingle;

            // Update thread-local solver state for this point
            // base_lead_mu.data() gives pointer to vector's buffer (shared read is fine)
            for (int l=0; l<nleads; ++l) {
                 solver_local.leads[l].mu = base_lead_mu[l] + VGate[l];
            }

            // Assuming these modify the state of solver_local only
            solver_local.setTLeads(TLeads_i);
            solver_local.calculate_tunneling_amplitudes(TLeads_i);

            // Solve for the current point using the stack-local solver
            // Apply the const_cast fix if needed, otherwise pass hsingle directly if solve_hsingle accepts const double*
            double current  = solve_hsingle(&solver_local, const_cast<double*>(hsingle), W, 1, 0);

            // Write result to the shared output array (indexed write is okay)
            out_current[i]  = current;

            // solver_local is automatically destroyed here at the end of iteration scope
        } // End of omp for loop
    } // End of omp parallel region

    // No cleanup loop needed

    return 0.0;
}

double scan_current(void* solver_ptr, int npoints, double* hsingles, double* Ws, double* VGates, double* TLeads, int* state_order, double* out_current, bool bOmp) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (!solver) { return 0.0; }

    if (bOmp) { 
        //return scan_current_omp(solver, npoints, hsingles, Ws, VGates, TLeads, state_order, out_current); 
        //return scan_current_omp_stackalloc(solver, npoints, hsingles, Ws, VGates, TLeads, state_order, out_current);
        return scan_current_manual_threads(solver, npoints, hsingles, Ws, VGates, TLeads, state_order, out_current);
    }

    printf("scan_current() npoints: %d bOmp: %d\n", npoints, bOmp);

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
