static int _verbosity = 0;

#include "pauli.hpp"
#include <cstdio>
#include <cstring> // for memcpy
#include <omp.h>
#include <thread>

#include "print_utils.hpp"
#include "TipField.h"

double Tmin_cut = 0.0;
double EW_cut   = 2.0;

extern "C" {

void setLinSolver(void* solver_ptr, int iLinsolveMode, int nMaxLinsolveInter, double LinsolveTolerance) {
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    printf("setLinSolver() iLinsolveMode: %d nMaxLinsolveInter: %d LinsolveTolerance: %g\n", iLinsolveMode, nMaxLinsolveInter, LinsolveTolerance);
    if (solver) { solver->setLinSolver(iLinsolveMode, nMaxLinsolveInter, LinsolveTolerance); }
}

// C wrapper: include zV1 and build Vec2d
void evalSitesTipsMultipoleMirror( int nTip, double* pTips, double* VBias,  int nSites, double* pSite, double* rotSite, double E0, double Rtip, double zV0, double zVd, int order, const double* cs, double* outEs, bool bMirror, bool bRamp ) {
    Vec2d zV{zV0,zVd};
    evalSitesTipsMultipoleMirror( nTip, (Vec3d*)pTips, VBias, nSites, (Vec3d*)pSite, (Mat3d*)rotSite, E0, Rtip, zV, order, cs, outEs, bMirror, bRamp );
}

void evalSitesTipsTunneling( int nTips, const double* pTips, int nSites, const double* pSites, double beta, double Amp, double* outTs ){
    evalSitesTipsTunneling( nTips, (Vec3d*) pTips, nSites, (Vec3d*) pSites, beta, Amp, outTs ); 
}  

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

void set_valid_point_cuts( double Tmin, double EW ){
    Tmin_cut = Tmin;
    EW_cut   = EW;
}

bool is_valid_point( int nSingle, const double* hsingle, const double* TLeads, double W ){
    double Emax=-1e+300;
    double Tmax=-1e+300;
    for(int j=0; j<nSingle; j++) { 
        double Ei = hsingle [j*nSingle + j]; Emax = (Ei>Emax) ? Ei : Emax; 
        double Ti = TLeads[  nSingle + j]; Tmax = (Ti>Tmax) ? Ti : Tmax; 
    }
    return ( ( Emax+(W*EW_cut)<0.0 ) || ( Tmax<Tmin_cut ) );
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
        double  W              = Ws[i];
        const double* VGate    = VGates + (i * nleads);
        const double* hsingle  = hsingles + i * n2;
        const double* TLeads_i = TLeads + i * nleads * nSingle;

        if( is_valid_point( nSingle, hsingle, TLeads_i, W ) ) { out_current[i] = 0.0; continue; }

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

/**
 * @brief Scan current for array of tip positions with integrated site energy and tunneling calculations
 * @param solver_ptr Pauli solver instance
 * @param npoints Number of tip positions
 * @param pTips Array of tip positions (npoints x 3)
 * @param rots Array of rotation matrices for sites (nSites x 3x3)
 * @param nSites Number of sites
 * @param pSites Array of site positions (nSites x 3)
 * @param params Array of parameters:
 *   [0]=VBias, [1]=Rtip, [2]=zV0, [3]=Esite, [4]=beta, [5]=Gamma, [6]=W
 * @param order Multipole order (0=monopole, 1=dipole, 2=quadrupole)
 * @param cs Multipole coefficients array (10 elements)
 * @param state_order State ordering array (optional)
 * @param out_current Output array for currents (npoints)
 * @param bOmp Whether to use OpenMP
 * @return 0 on success
 */

double scan_current_tip_( PauliSolver* solver, int npoints, Vec3d* pTips, double* Vtips, int nSites, Vec3d* pSites, Mat3d* rots, double* params, int order, double* cs,  int* state_order, double* out_current, double* Es, double* Ts, double* Probs, bool externTs ){
    //PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    //if (!solver) return 0.0;
    //printf("scan_current_tip() npoints: %d bOmp: %d state_order: %p \n", npoints, bOmp, state_order );
    //printf("scan_current_tip() nTip: %d nSites: %d E0: %6.3e Rtip: %6.3e zV0: %6.3e order: %d cs: %6.3e %6.3e %6.3e %6.3e %6.3e \n", nTip, nSites, E0, Rtip, zV0, order, cs[0], cs[1], cs[2], cs[3], cs[4] );
    //printf("scan_current_tip() npoints: %d nSites: %d order: %d cs:[ %6.3e, %6.3e, %6.3e, %6.3e ]\n", npoints, nSites, order, cs[0], cs[1], cs[2], cs[3] );
    //if(state_order) { 
    //    int nstates = solver->nstates;
    //    printf("scan_current_tip() state_order: %i %i %i %i %i %i %i %i  \n", state_order[0], state_order[1], state_order[2], state_order[3], state_order[4], state_order[5], state_order[6], state_order[7] );
    //}
    // Extract parameters
    double Rtip  = params[0];
    Vec2d zV{params[1],params[2]};
    double E0    = params[3];
    double beta  = params[4];
    double Gamma = params[5];
    double W     = params[6];
    bool bMirror = params[7] > 0;
    bool bRamp   = params[8] > 0;
    if(solver->verbosity>0) { printf("scan_current_tip() Rtip: %6.3e zV(%6.3e,%6.3e) E0: %6.3e beta: %6.3e Gamma: %6.3e W: %6.3e bMirror: %d bRamp: %d \n", Rtip, zV.x, zV.y, E0, beta, Gamma, W, bMirror, bRamp ); }
    //printf("scan_current_tip() Rtip: nTip: %d nSites: %d E0: %6.3e Rtip: %6.3e VBias[0,-1](%6.3e,%6.3e) pTip.z[0,-1](%6.3e,%6.3e) zV0: %6.3e zV1: %6.3e order: %d cs:[ %6.3e, %6.3e, %6.3e, %6.3e ]\n", npoints, nSites, E0, Rtip, Vtips[0], Vtips[npoints-1], pTips[0].z, pTips[npoints-1].z, zV0, zV1, order, cs[0], cs[1], cs[2], cs[3] );
    // Initialize local solver
    //PauliSolver solver_local(*solver);
    int nleads = 2;
    double base_lead_mu[nleads];
    for (int l = 0; l < nleads; ++l) { base_lead_mu[l] = solver->leads[l].mu; }
    double TLeads[nleads*nSites];
    double hsingle[nSites*nSites];

    double VS = Gamma/M_PI;
    double VT = Gamma/M_PI;

    solver->leads[0].mu = 0.0;
    for (int j = 0; j < nSites;        j++) { TLeads [j] = VS;  } // Lead tunneling rates
    for (int j = 0; j < nSites*nSites; j++) { hsingle[j] = 0.0; } // Initialize hsingle to zero

    solver->W = W;
    solver->setHsingle(hsingle);
    if(state_order) { 
        solver->init_states_by_charge();
        solver->setStateOrder(state_order); 
    }

    // Calculate site energies and tunneling rates for all points
    for (int i = 0; i < npoints; i++) {
        Vec3d tipPos = pTips[i];
        double VBias = Vtips[i];
        solver->leads[1].mu = VBias;
        for (int j = 0; j < nSites; j++) {
            Mat3d* rot = ( rots ) ? ( rots + j ) : nullptr;
            double Ei = evalMultipoleMirror( tipPos, pSites[j], VBias, Rtip, zV, order, cs, E0, rot, bMirror, bRamp );

            hsingle[j*nSites + j] = Ei;
            if( Es ) { Es[i*nSites + j] = Ei; }

            double T=0;
            if( externTs ) { 
                T = Ts[i*nSites + j];
                //if(j!=0) { T = 0.0; }
            }else{
                Vec3d d          = tipPos - pSites[j];
                T         = exp(-beta * d.norm());
                if( Ts ) { Ts[i*nSites + j] = T; }
            }
            TLeads [  nSites + j] = VT*T;
        }
        if( is_valid_point( nSites, hsingle, TLeads, W ) ) { out_current[i] = 0.0; continue; }
        solver->setTLeads(TLeads); // Assuming this doesn't modify internal state needed across iterations
        solver->calculate_tunneling_amplitudes(TLeads); // Assuming this is okay
        double current = solve_hsingle(solver, hsingle, W, 1, 0); // Pass address of local solver
        out_current[i] = current;
        if(Probs){
            int nstates = solver->nstates;
            std::memcpy(Probs + i*nstates, solver->get_probabilities(), nstates * sizeof(double));
        }
    }
        
    return 0.0;
}

double scan_current_tip_threaded( PauliSolver* solver, int npoints, Vec3d* pTips, double* Vtips, int nSites, Vec3d* pSites, Mat3d* rots, double* params, int order, double* cs,  int* state_order, double* out_current, double* Es, double* Ts, double* Probs, bool externTs ){
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback if detection fails
    num_threads = std::min((unsigned int)npoints, num_threads);
    // num_threads = 2; // For debugging
    
    // Important: Initialize the master solver state BEFORE creating thread copies
    // int nSingle = solver->nSingle;
    // int nleads = solver->nleads;
    // std::vector<double> base_lead_mu(nleads); // Use std::vector
    // for(int l = 0; l < nleads; l++) { base_lead_mu[l] = solver->leads[l].mu; }
    if(state_order) {
        // This setup should ideally happen *before* creating copies if it affects
        // the base state copied by the constructor.
        solver->init_states_by_charge();
        solver->setStateOrder(state_order);
    }

    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int batch_size = std::ceil(static_cast<double>(npoints) / num_threads);
    printf("scan_current_tip_threaded() nThreads: %i batch_size %i npoints: %i \n", num_threads, batch_size, npoints);
    
    // Launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        int i0 = t * batch_size;
        int i1 = std::min(i0 + batch_size, npoints);
        if (i0 >= npoints) break;
        threads.emplace_back([=]() { // Capture t by value
            PauliSolver solver_local(*solver); // Create LOCAL copy inside thread
            PauliSolver* solver_local_ptr = &solver_local;
            //PauliSolver* solver_local_ptr = new PauliSolver(*solver);
            scan_current_tip_( solver_local_ptr, i1-i0, pTips+i0, Vtips+i0, nSites, pSites, rots, params, order, cs, state_order, out_current+i0, Es?Es+i0*nSites : nullptr, Ts?Ts+i0*nSites : nullptr, Probs?Probs+i0*solver->nstates : nullptr, externTs );
            //delete solver_local;  // this cause  double free or corruption 
        });
    }

    // Wait for all threads
    for (auto& th : threads) {  if (th.joinable()) { th.join(); } }
    return 0.0;
}


double scan_current_tip_threaded_2( PauliSolver* solver, int npoints, Vec3d* pTips, double* Vtips, int nSites, Vec3d* pSites, Mat3d* rots, double* params, int order, double* cs,  int* state_order, double* out_current, double* Es, double* Ts, double* Probs, bool externTs ){

    // Extract parameters
    double Rtip  = params[0];
    Vec2d zV{params[1],params[2]};
    double E0    = params[3];
    double beta  = params[4];
    double Gamma = params[5];
    double W     = params[6];

    int nleads = 2;
    std::vector<double> base_lead_mu(nleads);
    for (int l = 0; l < nleads; ++l) { base_lead_mu[l] = solver->leads[l].mu; }
    
    // Precompute all site energies and tunneling rates
    double* hsingles = new double[npoints * nSites * nSites];
    double* Ws       = new double[npoints];
    double* VGates   = new double[npoints * nleads];
    double* TLeads   = new double[npoints * nleads * nSites];

    for (int i = 0; i < npoints; i++) {
        Ws[i] = W;
        VGates[i*nleads + 0] = 0.0;
        VGates[i*nleads + 1] = Vtips[i];
        
        Vec3d tipPos = pTips[i];
        for (int j = 0; j < nSites; j++) {
            Mat3d* rot = ( rots ) ? ( rots + j ) : nullptr;
            double Ei = evalMultipoleMirror( tipPos, pSites[j], Vtips[i], Rtip, zV, order, cs, E0, rot );
            hsingles[i*nSites*nSites + j*nSites + j] = Ei;
            if( Es ) { Es[i*nSites + j] = Ei; }

            double T=0;
            if( externTs ) { 
                T = Ts[i*nSites + j];
            }else{
                Vec3d d = tipPos - pSites[j];
                T = exp(-beta * d.norm());
                if( Ts ) { Ts[i*nSites + j] = T; }
            }
            TLeads[i*nleads*nSites + 0*nSites + j] = Gamma/M_PI; // VS
            TLeads[i*nleads*nSites + 1*nSites + j] = (Gamma/M_PI)*T; // VT
        }
    }

    // Setup threading
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    num_threads = std::min((unsigned int)npoints, num_threads);
    
    printf("scan_current_tip_threaded() nThreads: %d npoints: %d \n", num_threads, npoints);
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int batch_size = std::ceil(static_cast<double>(npoints) / num_threads);

    // Setup solver state before threading
    if(state_order) { 
        solver->init_states_by_charge();
        solver->setStateOrder(state_order); 
    }

    // Launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_index = t * batch_size;
        int end_index = std::min(start_index + batch_size, npoints);
        if (start_index >= npoints) break;

        threads.emplace_back(
            solve_batch,
            solver,
            t,
            start_index,
            end_index,
            nleads,
            nSites,
            hsingles,
            Ws,
            VGates,
            TLeads,
            base_lead_mu.data(),
            out_current
        );
    }

    // Wait for all threads
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    // Clean up
    delete[] hsingles;
    delete[] Ws;
    delete[] VGates;
    delete[] TLeads;
    
    return 0.0;
}

double scan_current_tip( void* solver_ptr, int npoints, double* pTips_, double* Vtips, int nSites, double* pSites_, double* rots_, double* params, int order, double* cs,  int* state_order, double* out_current, bool bOmp, double* Es, double* Ts, double* Probs, bool externTs ){
    PauliSolver* solver = static_cast<PauliSolver*>(solver_ptr);
    if (!solver) return 0.0;
    if( bOmp ){
        return scan_current_tip_threaded_2( solver, npoints, (Vec3d*)pTips_, Vtips, nSites, (Vec3d*)pSites_, (Mat3d*)rots_, params, order, cs, state_order, out_current, Es, Ts, Probs, externTs );
    } else {
        return scan_current_tip_( solver, npoints, (Vec3d*)pTips_, Vtips, nSites, (Vec3d*)pSites_, (Mat3d*)rots_, params, order, cs, state_order, out_current, Es, Ts, Probs, externTs );
    }
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
