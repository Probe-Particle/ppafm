#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <numeric>  // Required for std::accumulate
#include <cmath>
#include "gauss_solver.hpp"
//#include "iterative_solver.hpp"
#include "print_utils.hpp"

// Constants should be defined in meV units
const double PI = 3.14159265358979323846;
const double HBAR = 0.6582119;  // Reduced Planck constant in meV*ps
const double KB = 0.08617333;   // Boltzmann constant in meV/K

template<typename T> void swap( T& a, T& b ) {
    T tmp = a;
    a = b;
    b = tmp;
};

void swap_matrix_rows(double* mat, int nrows, int ncols, int row1, int row2) {
    for(int j = 0; j < ncols; j++) {
        swap(mat[row1 * ncols + j], mat[row2 * ncols + j]);
    }
}

void swap_matrix_cols(double* mat, int nrows, int ncols, int col1, int col2) {
    for(int i = 0; i < nrows; i++) {
        swap(mat[i * ncols + col1], mat[i * ncols + col2]);
    }
}

inline static int site_to_state(int site) { return 1 << site;}

inline static bool site_in_state(int site, int state) {  return (state >> site) & 1;}


/*
=== Function calculate_state_energy ====
should reproduce construct_Ea_manybody from  QmeQ construct_Ea_manybody() in /qmeq/qdot.py
def construct_Ea_manybody(valslst, si):
    Ea = np.zeros(si.nmany, dtype=float)
    if si.indexing == 'sz':
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, si.nsingle):
                # Iterate over many-body states for given charge and sz
                szind = sz_to_ind(sz, charge, si.nsingle)
                for ind in range(len(si.szlst[charge][szind])):
                    # The mapping of many-body states is according to szlst
                    Ea[si.szlst[charge][szind][ind]] = valslst[charge][szind][ind]
    elif si.indexing == 'ssq':
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, si.nsingle):
                szind = sz_to_ind(sz, charge, si.nsingle)
                # Iterate over total spin ssq
                for ssq in ssqrange(charge, sz, si.nsingle):
                    ssqind = ssq_to_ind(ssq, sz)
                    # Iterate over many-body states for given charge, sz, and ssq
                    for ind in range(len(si.ssqlst[charge][szind][ssqind])):
                        # The mapping of many-body states is according to ssqlst
                        Ea[si.ssqlst[charge][szind][ssqind][ind]] = valslst[charge][szind][ssqind][ind]
    else:
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over many-body states for given charge
            for ind in range(len(si.chargelst[charge])):
                # The mapping of many-body states is according to chargelst
                Ea[si.chargelst[charge][ind]] = valslst[charge][ind]
    return Ea
*/
double calculate_state_energy(int state, int nSingle, const double* Hsingle, double W) {
    //printf("calculate_state_energy() state: %i nSingle: %i Hsingle: %p W: %f \n", state, nSingle, Hsingle, W );
    double energy = 0.0;
    // Single-particle energies
    for(int i = 0; i < nSingle; i++) {
        int imask = site_to_state(i);
        if(state & imask) {
            // Access diagonal elements correctly
            energy += Hsingle[i * nSingle + i];
        }
    }
    // Hopping terms (t-values)
    for(int i = 0; i < nSingle; i++) {
        int imask = site_to_state(i);
        for(int j = i+1; j < nSingle; j++) {
            int jmask = site_to_state(j);
            // Check if both states are occupied for hopping
            if((state & imask) && (state & jmask)) {
                // Add hopping term (off-diagonal elements)
                energy += Hsingle[i * nSingle + j] + Hsingle[j * nSingle + i];
            }
        }
    }
    // Coulomb interaction - add W for each pair of occupied sites
    for(int i = 0; i < nSingle; i++) {
        int imask = site_to_state(i);
        if(state & imask) {
            for(int j = i+1; j < nSingle; j++) {
                int jmask = site_to_state(j);
                if(state & jmask) {
                    energy += W;
                }
            }
        }
    }
    return energy;
}


// Lead parameters
struct LeadParams {
    double mu;    // Chemical potential
    double temp;  // Temperature
    //double gamma; // Coupling strength
};


template<typename T> bool _reallocate(T*& ptr, int size) {
    bool b = (ptr != nullptr);
    if(b){ delete[] ptr; }
    ptr = new T[size];
    return b;
}


// Calculate number of occupied sites (charge) for a state
inline static int state_to_charge(int state, int nSingle) {
    int charge = 0;
    for(int i = 0; i < nSingle; i++) {
        if(state & (1 << i)) charge++;
    }
    return charge;
}

// Compare states by charge first, then numerically
// struct StateComparator {
//     int nSingle;
//     StateComparator(int nSingle) : nSingle(nSingle) {}
//     bool operator()(int a, int b) {
//         int chargeA = state_to_charge(a, nSingle);
//         int chargeB = state_to_charge(b, nSingle);
//         if(chargeA != chargeB) return chargeA < chargeB;
//         return a < b;
//     }
// };

// ==================================================
//                PauliSolver
// ==================================================

class PauliSolver {
public:
    // Direct integration of SolverParams data
    int nSingle = 0;              // Number of single-particle states
    int nstates = 0;              // Number of states
    int nleads = 0;               // Number of leads
    double* energies = nullptr;   // State energies [nstates]
    LeadParams* leads = nullptr;  // Lead parameters [nleads]
    double* coupling = nullptr;   // Coupling matrix elements [nleads * nstates * nstates]
    int* state_order     = nullptr; // State order [nstates]
    int* state_order_inv = nullptr; // State order [nstates]
    //int* state_order2    = nullptr; // State order [nstates]
    
    // Input parameters that can be modified in a loop
    double* Hsingle = nullptr;    // Single-particle Hamiltonian [nSingle * nSingle]
    double W        = 0.0;               // Coulomb interaction strength
    double* TLeads  = nullptr;     // Lead tunneling amplitudes [nleads * nSingle]
    
    // Working data
    double* kernel = nullptr;                  // Kernel matrix [nstates * nstates]
    double* rhs = nullptr;                     // Right-hand side vector [nstates]
    double* probabilities = nullptr;           // State probabilities [nstates]
    double* pauli_factors = nullptr;   // [nleads][ndm1][2]
    int n_pauli_factors = 0;           // Number of Pauli factors in compact array
    int ndm1 = 0;                              // Number of valid transitions (states differing by 1 charge)
    int verbosity = 0;                         // Verbosity level for debugging
    std::vector<std::vector<int>> states_by_charge;  // States organized by charge number, like Python's statesdm
    // std::vector<int> state_order;              // Maps original index -> ordered index
    // std::vector<int> state_order_inv;          // Maps ordered index -> original index
    std::vector<int> state_order2;             // Maps original index -> ordered index

    // from indexing.py of QmeQ
    std::vector<int> lenlst;      // Number of states in each charge sector
    std::vector<int> dictdm;      // State enumeration within charge sectors
    std::vector<int> shiftlst0;   // Cumulative offsets
    std::vector<int> shiftlst1;   // Cumulative offsets
    std::vector<int> mapdm0;      // State pair mapping
    
    // Flags to track what needs to be recalculated
    bool energies_updated = false;
    bool coupling_updated = false;
    bool kernel_updated = false;

// Simple constructor that only allocates arrays but doesn't initialize values
    // PauliSolver(int nSingle_, int nstates_, int nleads_, int verb = 0) :  nSingle(nSingle_), nstates(nstates_), nleads(nleads_), verbosity(verb) {
        
    //     // Allocate memory for internal arrays
    //     energies        = new double[nstates];
    //     leads           = new LeadParams[nleads];
    //     coupling        = new double[nleads * nstates * nstates];
    //     state_order     = new int[nstates];
    //     state_order_inv = new int[nstates];
    //     //state_order2    = new int[nstates];
    //     kernel          = new double[nstates * nstates];
    //     rhs             = new double[nstates];
    //     probabilities   = new double[nstates];
        
    //     // Allocate input parameter arrays
    //     Hsingle         = new double[nSingle * nSingle];
    //     TLeads          = new double[nleads * nSingle];
        
    //     // Initialize all arrays to zero
    //     std::memset(energies, 0, nstates * sizeof(double));
    //     std::memset(coupling, 0, nleads * nstates * nstates * sizeof(double));
    //     std::memset(kernel,   0, nstates * nstates * sizeof(double));
    //     std::memset(rhs,      0, nstates * sizeof(double));
    //     std::memset(probabilities, 0, nstates * sizeof(double));
    //     std::memset(Hsingle,       0, nSingle * nSingle * sizeof(double));
    //     std::memset(TLeads,        0, nleads * nSingle * sizeof(double));
    //     std::memset(state_order,   0, nstates * sizeof(int));
        
    //     // Don't initialize leads array here - will be set by setLeadParams
        
    //     if (verbosity > 0) { printf("PauliSolver constructed with: nSingle=%d, nstates=%d, nleads=%d\n",  nSingle, nstates, nleads); }
    // }
    

    void alloc( int nSingle_, int nstates_, int nleads_, bool bMemSet=true ) {
        if (verbosity > 0) {  printf("PauliSolver constructing: nSingle=%d, nstates=%d, nleads=%d, verbosity=%d\n",   nSingle, nstates, nleads, verbosity);}
        leads           = new LeadParams[nleads];    for(int i = 0; i < nleads; ++i) { leads[i].mu = 0.0; leads[i].temp = 0.0; }
        state_order     = new int[nstates];          for (int i = 0; i < nstates; ++i) {  state_order[i] = i;}
        state_order_inv = new int[nstates];          for (int i = 0; i < nstates; ++i) { state_order_inv[i] = i; }
        // Or alternatively, just zero it out:
        // std::memset(state_order_inv, 0, nstates * sizeof(int));
        energies = new double[nstates];                         if(bMemSet) std::memset(energies, 0, nstates * sizeof(double));
        coupling      = new double[nleads * nstates * nstates]; if(bMemSet) std::memset(coupling, 0, nleads * nstates * nstates * sizeof(double));
        Hsingle       = new double[nSingle * nSingle];          if(bMemSet) std::memset(Hsingle,       0, nSingle * nSingle * sizeof(double));
        TLeads        = new double[nleads  * nSingle];          if(bMemSet) std::memset(TLeads,        0, nleads  * nSingle * sizeof(double));
        kernel        = new double[nstates * nstates];          if(bMemSet) std::memset(kernel,        0, nstates * nstates * sizeof(double));
        rhs           = new double[nstates          ];          if(bMemSet) std::memset(rhs,           0, nstates * sizeof(double));
        probabilities = new double[nstates          ];          if(bMemSet) std::memset(probabilities, 0, nstates * sizeof(double));
    }   

    // Complete and Corrected Constructor
    PauliSolver(int nSingle_, int nstates_, int nleads_, int verb = 0) :
        nSingle(nSingle_),
        nstates(nstates_),
        nleads(nleads_),
        W(0.0),                // Initialize simple types directly
        pauli_factors(nullptr),
        n_pauli_factors(0),
        ndm1(0),
        verbosity(verb),
        energies_updated(false), // Explicitly initialize flags
        coupling_updated(false),
        kernel_updated(false)
        // std::vectors are default-constructed (empty), which is fine
    {
        alloc(nSingle_, nstates_, nleads_);
        // pauli_factors is initialized to nullptr above, allocated later in generate_fct

        if (verbosity > 0) {  printf("PauliSolver construction finished: Memory allocated and initialized.\n");}
    }

    void deep_clone( const PauliSolver* source ){
        // Copy basic parameters
        W = source->W;
        verbosity = source->verbosity;
        
        // Copy arrays
        std::memcpy(energies,        source->energies,        nstates * sizeof(double) );
        std::memcpy(coupling,        source->coupling,        nleads  * nstates * nstates * sizeof(double) );
        std::memcpy(Hsingle,         source->Hsingle,         nSingle * nSingle * sizeof(double) );
        std::memcpy(TLeads,          source->TLeads,          nleads  * nSingle * sizeof(double) );
        std::memcpy(kernel,          source->kernel,          nstates * nstates * sizeof(double) );
        std::memcpy(rhs,             source->rhs,             nstates * sizeof(double) );
        std::memcpy(probabilities,   source->probabilities,   nstates * sizeof(double) );
        std::memcpy(state_order,     source->state_order,     nstates * sizeof(int) );
        std::memcpy(state_order_inv, source->state_order_inv, nstates * sizeof(int) );
        
        // Copy lead parameters
        for(int i = 0; i < nleads; ++i) {  leads[i] = source->leads[i]; }
        
        // Copy pauli factors
        if(source->pauli_factors) {
            n_pauli_factors = source->n_pauli_factors;
            if(pauli_factors) delete[] pauli_factors;
            pauli_factors = new double[n_pauli_factors];
            std::memcpy(pauli_factors, source->pauli_factors, n_pauli_factors * sizeof(double));
        }
        
        // Copy vectors
        //dm1       = source->dm1;
        ndm1      = source->ndm1;
        lenlst    = source->lenlst;
        dictdm    = source->dictdm;
        shiftlst0 = source->shiftlst0;
        shiftlst1 = source->shiftlst1;
        mapdm0    = source->mapdm0;
        //mapdm1    = source->mapdm1;
        states_by_charge = source->states_by_charge;
        
        // Copy flags
        energies_updated = source->energies_updated;
        coupling_updated = source->coupling_updated;
        kernel_updated = source->kernel_updated;
    }

    PauliSolver( const PauliSolver* source ){
        nSingle = source->nSingle;
        nstates = source->nstates;
        nleads  = source->nleads;
        W       = source->W;
        alloc(nSingle, nstates, nleads, false);
        deep_clone( source );
    }

    // Destructor to free allocated memory
    ~PauliSolver() {
        delete[] kernel;
        delete[] rhs;
        delete[] probabilities;
        delete[] pauli_factors;
        delete[] energies;
        delete[] leads;
        delete[] coupling;
        delete[] state_order;
        delete[] Hsingle;
        delete[] TLeads;
    }

    // Setter methods for modifying parameters in a loop
    
    // Set the single-particle Hamiltonian matrix
    void setHsingle(const double* newHsingle) {
        if (Hsingle && newHsingle) {
            std::memcpy(Hsingle, newHsingle, nSingle * nSingle * sizeof(double));
            energies_updated = false;  // Energy values need to be recalculated
            kernel_updated = false;    // Kernel matrix needs to be recalculated
            if (verbosity > 1) {printf("PauliSolver::setHsingle() - Updated Hsingle matrix\n");}
        }
    }
    
    // Set the Coulomb interaction strength
    void setW(double newW) {
        if (W != newW) {
            W = newW;
            energies_updated = false;  // Energy values need to be recalculated
            kernel_updated = false;    // Kernel matrix needs to be recalculated
            if (verbosity > 1) {printf("PauliSolver::setW() - Updated W to %f\n", W);}
        }
    }
    
    // Set the lead tunneling amplitudes
    void setTLeads(const double* newTLeads) {
        if (TLeads && newTLeads) {
            //printf( "PauliSolver::setTLeads() [nleads=%i, nSingle=%i] \n", nleads, nSingle );
            std::memcpy(TLeads, newTLeads, nleads * nSingle * sizeof(double));
            coupling_updated = false;  // Coupling matrix needs to be recalculated
            kernel_updated   = false;  // Kernel matrix needs to be recalculated
            if (verbosity > 1) {printf("PauliSolver::setTLeads() - Updated TLeads array\n");}
        }
    }
    
    // Set state ordering
    void setStateOrder(const int* newStateOrder) {
        if (state_order && newStateOrder) {
            std::memcpy(state_order, newStateOrder, nstates * sizeof(int));
            energies_updated = false;  // Energy values need to be recalculated using new state order
            coupling_updated = false;  // Coupling matrix needs to be recalculated
            kernel_updated   = false;    // Kernel matrix needs to be recalculated
            if (verbosity > 1) {printf("PauliSolver::setStateOrder() - Updated state ordering\n");}
        }
    }
    
    // Set lead parameters (mu, temp, gamma)
    //void setLeadParams(int leadIndex, double mu, double temp, double gamma) {
    void setLeadParams(int leadIndex, double mu, double temp ) {
        if (leadIndex >= 0 && leadIndex < nleads && leads) {
            leads[leadIndex].mu = mu;
            leads[leadIndex].temp = temp;
            //leads[leadIndex].gamma = gamma;
            kernel_updated = false;  // Kernel matrix needs to be recalculated
            if (verbosity > 1) {
                //printf("PauliSolver::setLeadParams() - Updated lead %d: mu=%f, temp=%f, gamma=%f\n", leadIndex, mu, temp, gamma);
                printf("PauliSolver::setLeadParams() - Updated lead %d: mu=%f temp=%f\n", leadIndex, mu, temp);
            }
        }
    }
    
    // Count number of electrons in a state
    int count_electrons(int state) {
        return __builtin_popcount(state);
    }

    void print_states_by_charge(){
        printf("PauliSolver::print_states_by_charge(): " );
        printf("[");
        for(const auto& states : states_by_charge) {
            printf("[");
            for(size_t i = 0; i < states.size(); i++) {
                printf("%d", states[i]);
                if(i < states.size() - 1) printf(", ");
            }
            printf("]");
        }
        printf("]\n");
    }


    /// Calculate state energies
    void calculate_state_energies() {
        // NOTE: is somewhat equivalent to construct_Ea_manybody(), diagonalise() and set_Ea() in /qmeq/qdot.py 
        if(verbosity > 1) {
            printf("PauliSolver::calculate_state_energies() W=%f\n", W);
            printf("PauliSolver::calculate_state_energies() Hsingle:\n");
            print_matrix(Hsingle, nSingle, nSingle, " %16.8g");
        }
        for(int i = 0; i < nstates; i++) {
            int state_idx = state_order[i];
            energies[i] = calculate_state_energy(state_idx, nSingle, Hsingle, W);
            //printf("calculate_state_energies() i: %i state %i energy=%g \n", i, state_idx, energies[i] );
        }
    }

/*
=== Function eval_lead_coupling ====
should reproduce construct_Tba from QmeQ /home/prokophapala/git/qmeq/qmeq/leadstun.py
def construct_Tba(leads, tleads, Tba_=None):
    si, mtype = leads.si, leads.mtype
    if Tba_ is None:
        Tba = np.zeros((si.nleads, si.nmany, si.nmany), dtype=mtype)
    else:
        Tba = Tba_
    # Iterate over many-body states
    for j1 in range(si.nmany):
        state = si.get_state(j1)
        # Iterate over single particle states
        for j0 in tleads:
            (j3, j2), tamp = j0, tleads[j0]
            # Calculate fermion sign for added/removed electron in a given state
            fsign = np.power(-1, sum(state[0:j2]))
            if state[j2] == 0:
                statep = list(state)
                statep[j2] = 1
                ind = si.get_ind(statep)
                if ind is None:
                    continue
                Tba[j3, ind, j1] += fsign*tamp
            else:
                statep = list(state)
                statep[j2] = 0
                ind = si.get_ind(statep)
                if ind is None:
                    continue
                Tba[j3, ind, j1] += fsign*np.conj(tamp)
    return Tba
*/

    void eval_lead_coupling(int lead, const double* TLead) {
        //if(_verbosity > 3) printf("SolverParams::eval_lead_coupling() Lead %i \n", lead);
        double* coupling_ = coupling + lead * nstates * nstates;
        for(int j1 = 0; j1 < nstates; j1++) {
            int state = j1;
            for(int j2 = 0; j2 < nSingle; j2++) {
                // Reverse bit mapping to match QmeQ convention
                int site = nSingle - 1 - j2;
                
                // Calculate fermionic sign based on QmeQ convention
                // In QmeQ: fsign = (-1)^sum(state[0:j2])
                // We need to count occupied states in positions 0 to j2-1 (in QmeQ indexing)
                // This corresponds to positions (nSingle-1) down to (nSingle-j2) in C++ indexing
                int fsign = 1;
                for(int k = nSingle-1; k > nSingle-1-j2; k--) {
                    if((state >> k) & 1) fsign *= -1;
                }

                double tamp = TLead[j2]; // Keep j2 for TLead access (not reversed)
                double dTba = fsign * tamp;
                
                // Check if site is occupied using reversed bit position
                if(!((state >> site) & 1)) {  // Add electron
                    int ind = state | (1 << site);
                    coupling_[ind * nstates + j1] += dTba;
                    //if(_verbosity > 3) {  printf("add_e lead %i states %3i -> %3i  |  site %3i ind %3i dTba %g tamp %g fsign %i\n",  lead, j1, j2, site, ind, dTba, tamp, fsign ); }
                } else {  // Remove electron
                    int ind = state & ~(1 << site);
                    coupling_[ind * nstates + j1] += dTba;
                    //if(_verbosity > 3) {  printf("sub_e lead %i states %3i -> %3i  |  site %3i ind %3i dTba %g tamp %g fsign %i\n",  lead, j1, j2, site, ind, dTba, tamp, fsign ); }
                }
            }
        }
    }

    // Update tunneling amplitudes after TLeads has changed
    // Seems redundant - perahps we should call eval_lead_coupling() for each lead
    void updateTunnelingAmplitudes() {
        if (!coupling_updated && TLeads && coupling) {
            if (verbosity > 1) {
                printf("PauliSolver::updateTunnelingAmplitudes() - Recalculating tunneling amplitudes\n");
            }
            
            // Zero the coupling matrix
            std::memset(coupling, 0, nleads * nstates * nstates * sizeof(double));

            for (int lead = 0; lead < nleads; lead++) {
                eval_lead_coupling(lead, TLeads + lead * nSingle);
            }
                        
            coupling_updated = true;
        }
    }

    /// Calculate tunneling amplitudes between states
    void calculate_tunneling_amplitudes(const double* TLeads) {
        memset(coupling, 0, nleads * nstates * nstates * sizeof(double));
        // Process all leads to match QmeQ behavior
        for(int lead = 0; lead < nleads; lead++) {
            eval_lead_coupling(lead, TLeads + lead * nSingle);
        }
        if(_verbosity > 1) print_tunneling_amplitudes();
        //exit(0);
        //exit(0);
    }

        // Update functions for recalculating internal data after parameter changes
    
    // Update state energies after Hsingle or W has changed
    void updateStateEnergies() {
        if (!energies_updated && Hsingle && energies && state_order) {
            if (verbosity > 1) {
                printf("PauliSolver::updateStateEnergies() - Recalculating state energies\n");
            }
            
            // Calculate energies for all states
            for (int i = 0; i < nstates; i++) {
                int state_idx = state_order[i];
                energies[i] = calculate_state_energy(state_idx, nSingle, Hsingle, W);
                if (verbosity > 2) {
                    printf("  State %d (raw state %d): Energy = %g\n", i, state_idx, energies[i]);
                }
            }
            
            energies_updated = true;
        }
    }
    
    int count_valid_transitions() {
        if(verbosity > 3) printf("PauliSolver::count_valid_transitions() states_by_charge.size() %li \n", states_by_charge.size()   );
        int ndm1 = 0;
        for(int charge = 0; charge < states_by_charge.size()-1; charge++) {
            int nch0 = states_by_charge[charge  ].size();
            int nch1 = states_by_charge[charge+1].size();
            if(verbosity > 3) printf("PauliSolver::count_valid_transitions() charge %d nch0 %d nch1 %d nch1*nch0 %d ndm1 %d \n", charge, nch0, nch1, nch1*nch0, ndm1 );
            ndm1 += nch1 * nch0;
        }
        return ndm1;
    }

    // Get the site that changed in a transition between two states
    // Returns -1 if more than one site changed or if no site changed
    int get_changed_site(int state1, int state2) {
        int diff = state1 ^ state2;
        if (__builtin_popcount(diff) != 1) {
            return -1;  // More than one site changed or no site changed
        }
        // Find the position of the 1 bit in diff
        return __builtin_ctz(diff);  // Count trailing zeros
    }

    // Calculate Fermi function for given energy difference and lead parameters
    double fermi_func(double energy_diff, double mu, double temp) {
        return 1.0/(1.0 + exp((energy_diff - mu)/temp));
    }

    //inline int get_ind_dm0_0( int i, int iq ){ return dictdm[i] + shiftlst0[iq]; }
    inline int get_ind_dm0(int b, int bp, int charge) {
        // Mirror Python's get_ind_dm0 from indexing.py
        int ib  = dictdm[b];
        int ibp = dictdm[bp];
        int result = ibp + ib * lenlst[charge] + shiftlst0[charge];
        //if(verbosity > 3) {printf("get_ind_dm0(b=%d, bp=%d, charge=%d) = %d (ib=%d, ibp=%d)\n",   b, bp, charge, result, ib, ibp);}
        return result;
    }

    int get_ind_dm1(int c, int b, int bcharge) {
        // Replicate Python logic for mapping transitions to indices
        if (verbosity > 3) {printf("get_ind_dm1(c=%d, b=%d, bcharge=%d)\n",   c, b, bcharge );}
        int ic = dictdm[c];
        int ib = dictdm[b];
        int index = ic * lenlst[bcharge] + ib + shiftlst1[bcharge];
        if (verbosity > 3) {printf("get_ind_dm1(c=%d, b=%d, bcharge=%d) = %d (ic=%d, ib=%d)\n",   c, b, bcharge, index, ic, ib);}
        return index;
    }

    void init_map_dm0() {
        if(verbosity > 3) printf("PauliSolver::init_map_dm0()\n");
        mapdm0.resize(shiftlst0.back(), -1);
        int counter = 0;
        int nq = states_by_charge.size();

        // Diagonal elements
        for(int iq = 0; iq < nq; iq++) {
            for(int b : states_by_charge[iq]) {
                int bbp = get_ind_dm0(b, b, iq);
                if(verbosity > 3) printf("PauliSolver::set_mapdm() diag b,iq,bbp,counter %d %d %d %d \n", b, iq, bbp, counter);
                mapdm0[bbp] = counter++;
            }
        }
        //npauli = counter;

        // Off-diagonal elements using combinations
        for(int iq = 0; iq < nq; iq++) {
            for(int i = 0; i < states_by_charge[iq].size(); i++) {
                for(int j = i + 1; j < states_by_charge[iq].size(); j++) {
                    int b = states_by_charge[iq][i];
                    int bp = states_by_charge[iq][j];
                    
                    int bpb = get_ind_dm0(bp, b, iq);
                    if(verbosity > 3) printf("PauliSolver::set_mapdm() offdiag b,iq,bbp,counter %d %d %d %d \n", bp, iq, bpb, counter);
                    mapdm0[bpb] = counter;
                    
                    int bbp = get_ind_dm0(b, bp, iq);
                    if(verbosity > 3) printf("PauliSolver::set_mapdm() offdiag b,iq,bbp,counter %d %d %d %d \n", b, iq, bbp, counter);
                    mapdm0[bbp] = counter++;
                }
            }
        }
        //ndm0 = counter;
        //ndm0r = npauli + 2*(ndm0 - npauli);
    }

    // Initialize these in init_states_by_charge()
    void init_indexing_maps() {
        if(verbosity > 3) printf("PauliSolver::init_indexing_maps()\n");
        const int n = nstates;
        
        // Clear previous data to avoid duplication
        lenlst.clear();
        dictdm.clear();
        shiftlst0.clear();
        shiftlst1.clear();
        mapdm0.clear();
        
        // Resize arrays properly
        lenlst.resize(states_by_charge.size());
        dictdm.resize(n, 0);  // Initialize with zeros
        shiftlst0.resize(states_by_charge.size() + 1, 0);
        shiftlst1.resize(states_by_charge.size()    , 0);

        int nq = states_by_charge.size();
        if(verbosity > 3) printf("PauliSolver::init_indexing_maps() states_by_charge size: %d\n", nq);
        
        // Fill the mapping arrays following QmeQ's logic
        for(int iq = 0; iq < nq; iq++) {
            lenlst[iq] = states_by_charge[iq].size();
            int counter = 0;
            for(int state : states_by_charge[iq]) {
                dictdm[state] = counter++;
            }
        }

        for(int iq = 0; iq < nq; iq++) { 
            shiftlst0[iq+1] = shiftlst0[iq] + lenlst[iq] * lenlst[iq]; 
        }
        
        for(int iq = 0; iq < nq-1; iq++) { 
            shiftlst1[iq+1] = shiftlst1[iq] + lenlst[iq] * lenlst[iq+1]; 
        }

        init_map_dm0();

        if(verbosity > 3) {
            printf("PauliSolver::init_indexing_maps() len_list    : "); print_vector(lenlst);    
            printf("PauliSolver::init_indexing_maps() dict_dm     : "); print_vector(dictdm);   
            printf("PauliSolver::init_indexing_maps() shift_list0 : "); print_vector(shiftlst0); 
            printf("PauliSolver::init_indexing_maps() shift_list1 : "); print_vector(shiftlst1); 
            printf("PauliSolver::init_indexing_maps() map_dm0     : "); print_vector(mapdm0);    
        }
    }

    void init_state_ordering() {
        if(verbosity > 3) printf("PauliSolver::init_state_ordering() NSingle %d NState %d NCharge %ld \n", nSingle, nstates, states_by_charge.size() );
        
        // Clear previous mappings to prevent duplication
        //state_order.clear();
        //state_order_inv.clear();
        
        // Verify that we're working with the expected number of states (2^nSingle)
        const int expected_nstates = 1 << nSingle; // 2^nSingle
        if (nstates != expected_nstates) {
            if(verbosity > 0) printf("Warning: nstates (%d) doesn't match expected 2^nSingle (%d)\n", nstates, expected_nstates);
        }
        
        // Count total states from charge sectors (should match nstates)
        const int n = states_by_charge.empty() ? 0 : 
            std::accumulate(states_by_charge.begin(), states_by_charge.end(), 0, 
                           [](int sum, const std::vector<int>& vec) { return sum + static_cast<int>(vec.size()); });
        
        if (n != nstates && verbosity > 0) {
            printf("Warning: Total states in charge sectors (%d) doesn't match nstates (%d)\n", n, nstates);
        }
        
        // Initialize state order mappings
        //state_order.resize(nstates, 0);
        //state_order_inv.resize(nstates, 0);
        
        int idx = 0;
        for(int charge = 0; charge < states_by_charge.size(); charge++) {
            for(int state : states_by_charge[charge]) {
                if (state >= 0 && state < nstates) {
                    state_order[state] = idx;
                    idx++;
                } else if(verbosity > 0) {
                    printf("Warning: Invalid state index %d in charge sector %d\n", state, charge);
                }
            }
        }
        
        // Generate inverse mapping
        for(int i = 0; i < nstates; i++) {
            int ordered_idx = state_order[i];
            if (ordered_idx >= 0 && ordered_idx < nstates) {
                state_order_inv[ordered_idx] = i;
            }
        }
        
        if(verbosity > 3) {
            printf("PauliSolver::init_state_ordering() original -> ordered\n");
            for(int i = 0; i < nstates; i++) {
                printf("%d -> %d\n", i, state_order[i]);
            }
        }
    }

    void init_states_by_charge() {
        if(verbosity > 3) printf("PauliSolver::init_states_by_charge()\n");
        const int n = nstates;
        int max_charge = 0;
        
        // Clear previous data first to prevent duplication
        states_by_charge.clear();
        
        // First find maximum charge
        for(int i = 0; i < n; i++) {
            max_charge = std::max(max_charge, count_electrons(i));
        }
        states_by_charge.resize(max_charge + 1);
        
        // Fill states in order
        for(int i = 0; i < n; i++) {
            int charge = count_electrons(i);
            states_by_charge[charge].push_back(i);
        }
        
        // Sort states within each charge sector
        for(auto& states : states_by_charge) {
            std::sort(states.begin(), states.end());
        }

        init_state_ordering();
        init_indexing_maps();

        if(verbosity > 3) {
            printf("PauliSolver::init_states_by_charge() states_by_charge: \n");
            print_vector_of_vectors(states_by_charge);
        }
    }

    /*
    // Python QmeQ version of generate_fct from qmeq/approach/base/pauli.py 
    // self.paulifct = np.zeros((self.si.nleads, self.si.ndm1, 2), dtype=float)
    //    -  self.si.nleads: Number of leads connected to the quantum dot system.
    //    -  self.si.ndm1: Number of valid transitions between states differing by one electron (charge difference of 1).
    //    -  2: Represents the forward and backward transition rates for each lead and transition.
    def generate_fct(self):            
        E, Tba, si = self.qd.Ea, self.leads.Tba, self.si
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm
        itype = self.funcp.itype
        paulifct = self.paulifct
        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)
                Ecb = E[c]-E[b]
                for l in range(nleads):
                    xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                    rez = func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                    paulifct[l, cb, 0] = xcb*rez[0]  # Forward
                    paulifct[l, cb, 1] = xcb*rez[1]  # Backward
    */

    void init_pauli_factors() {
        ndm1 = count_valid_transitions(); 
        if(verbosity > 3) printf("PauliSolver::generate_fct() ndm1 = %d\n", ndm1);
        // Calculate size of compact pauli factors array
        n_pauli_factors = nleads * ndm1 * 2;
        if(verbosity > 3) printf("PauliSolver::generate_fct() n_pauli_factors = %d\n", n_pauli_factors);
        // Free previous pauli_factors if it exists
        delete[] pauli_factors;
        // Allocate compact array with zero initialization
        pauli_factors = new double[n_pauli_factors]();
    }

    void generate_fct() {
        if(verbosity > 3) printf("PauliSolver::generate_fct() - starting\n");
        const int n = nstates;
        
        calculate_state_energies();
        if(verbosity > 3){ 
            printf("PauliSolver::generate_fct() - calculated state energies\n");
            print_state_energies();
        }

        // Initialize states_by_charge ONLY if it's actually empty (prevent redundant initialization)
        static bool initialized = false;
        if(states_by_charge.empty() && !initialized) {
            if(verbosity > 0) printf("Info: First-time initialization of states_by_charge\n");
            init_states_by_charge();
            initialized = true;
        } else if(verbosity > 3) {
            printf("PauliSolver::generate_fct() - using existing states_by_charge\n");
        }
        
        // Ensure coupling matrix is initialized (but only if needed)
        if(!coupling || !coupling_updated) {
            if(verbosity > 0) printf("Info: Initializing coupling matrix\n");
            if(TLeads) {
                calculate_tunneling_amplitudes(TLeads);
                coupling_updated = true;
            } else {
                printf("Error: TLeads not set, cannot generate coupling matrix\n");
                return;
            }
        }
        //leads[0].temp=1.0;
        //leads[1].temp=1.0;
        //printf("PauliSolver::generate_fct() load[0](mu=%f,T=%f) load[1](mu=%f, T=%f) \n", leads[0].mu, leads[0].temp, leads[1].mu, leads[1].temp );
        
        // Count valid transitions
        if( !pauli_factors ){ init_pauli_factors(); }
        
        int n2 = n * n;
        
        // Calculate transitions between adjacent charge states
        for(int charge = 0; charge < states_by_charge.size()-1; charge++) {
            int next_charge = charge + 1;
            
            //if(next_charge >= states_by_charge.size()) { if(verbosity > 0) printf("Warning: next_charge %d out of bounds %zu\n", next_charge, states_by_charge.size()); continue;}
            
            // Process all transitions between these charge sectors
            for(int c : states_by_charge[next_charge]) {
                //if(c < 0 || c >= n) {  if(verbosity > 0) printf("Warning: State index c = %d out of bounds [0, %d]\n", c, n-1); continue;}
                
                for(int b : states_by_charge[charge]) {
                    //if(b < 0 || b >= n) {  if(verbosity > 0) printf("Warning: State index b = %d out of bounds [0, %d]\n", b, n-1); continue;}
                    
                    // Get compact index for this transition
                    int cb = get_ind_dm1(c, b, charge);

                    //if(cb < 0 || cb >= ndm1) { if(verbosity > 0) printf("Warning: Compact index cb = %d out of bounds [0, %d]\n", cb, ndm1-1);continue;}
                    
                    // Calculate energy difference for this transition (same as in Python)
                    double energy_diff = energies[c] - energies[b];
                    if(verbosity > 2) printf("PauliSolver::generate_fct() state energies: E[%d]=%f, E[%d]=%f, diff=%f\n", c, energies[c], b, energies[b], energy_diff);
                    
                    // Process all leads
                    for(int l = 0; l < nleads; l++) {
                        // Calculate array indices safely
                        int idx_ij = l * n2 + b*n + c;
                        int idx_ji = l * n2 + c*n + b;
                        
                        //if(idx_ij < 0 || idx_ij >= nleads * n2 || idx_ji < 0 || idx_ji >= nleads * n2) {if(verbosity > 0) printf("Warning: Coupling index out of bounds: l=%d, b=%d, c=%d\n", l, b, c); continue;}
                        
                        // Get coupling factors
                        double tij = coupling[idx_ij];
                        double tji = coupling[idx_ji];
                        double coupling_val = tij * tji;
                        
                        // Apply Fermi statistics
                        const LeadParams& lead = leads[l];
                        double fermi = fermi_func(energy_diff, lead.mu, lead.temp);
                        
                        // Calculate compact storage index
                        int idx = l * ndm1 * 2 + cb * 2;
                        
                        //if(idx < 0 || idx+1 >= n_pauli_factors) { if(verbosity > 0) printf("Warning: pauli_factors index out of bounds: l=%d, cb=%d, idx=%d, max=%d\n", l, cb, idx, n_pauli_factors-1);  continue;}
                        
                        // Store forward and backward rates
                        pauli_factors[idx + 0] = coupling_val *        fermi  * 2 * PI;  // Forward
                        pauli_factors[idx + 1] = coupling_val * (1.0 - fermi) * 2 * PI;  // Backward
                        
                        if(verbosity > 3) {printf("generate_fct() l: %d i: %d j: %d cb: %d E_diff: %.6f coupling: %.6f fermi: %.6f factors:[ %.6f, %.6f ]\n", l, c, b, cb, energy_diff, coupling_val, fermi, pauli_factors[idx + 0], pauli_factors[idx + 1]); }
                    }
                }
            }
        }
        
        if(verbosity > 3) printf("PauliSolver::generate_fct() - completed\n");
    }

    /// @brief Adds a real value (fctp) to the matrix element connecting the states bb and aa in the Pauli kernel. 
    /// In addition, adds another real value (fctm) to the diagonal element kern[bb, bb].
    /// @param fctm Value to be added to the diagonal element kern[bb, bb].
    /// @param fctp Value to be added to the matrix element connecting states bb and aa.
    /// @param bb Index of the first state.
    /// @param aa Index of the second state.
    /// @note Modifies the internal kernel matrix.
    void set_matrix_element_pauli(double fctm, double fctp, int bb, int aa) {
        int n = nstates;
        // Only apply matrix updates if indices are within bounds
        // This matches Python's behavior of silently ignoring out-of-bounds indices
        if(bb < n && aa < n) {
            // Add the contributions to the kernel matrix
            kernel[bb*n+bb] += fctm; // diagonal
            kernel[bb*n+aa] += fctp; // off-diagonal
        } else if(verbosity > 2) {
            printf("INFO: Skipping matrix element (n=%d, bb=%d, aa=%d) - indices out of bounds\n", n, bb, aa);
        }
    }

    //inline int index_paulifct        (int l, int i, int j){ return 2*( j + nstates*( i + l*nstates )); }
    inline int index_paulifct(int l, int i) { 
        int idx = 2*( i + ndm1*l);
        // if(verbosity > 3) {
        //     printf("index_paulifct(l=%d, i=%d) = %d (ndm1=%d, array_size=%i)\n",  l, i, idx, ndm1, n_pauli_factors);
        //     // Check if index is out of bounds
        //     if (idx < 0 || idx + 1 >= n_pauli_factors) {printf("ERROR: index_paulifct result %d is out of bounds for array size %i\n", idx, n_pauli_factors);}
        // }
        return idx;
    }


/*
    // Python QmeQ version of generate_fct from qmeq/approach/base/pauli.py 
       - generate_coupling_terms(self, b, bp, bcharge)
       - generate_fct(self)

    def generate_coupling_terms(self, b, bp, bcharge):
        """Generate coupling terms for the Pauli master equation."""
        Approach.generate_coupling_terms(self, b, bp, bcharge)
        paulifct = self.paulifct
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm
        acharge = bcharge-1
        ccharge = bcharge+1
        bb = si.get_ind_dm0(b, b, bcharge)

        # Handle transitions from lower charge states
        for a in statesdm[acharge]:   # Loop over states with charge acharge = bcharge-1
            aa = si.get_ind_dm0(a, a, acharge)
            ba = si.get_ind_dm1(b, a, acharge)
            fctm, fctp = 0, 0
            for l in range(nleads):
                fctm -= paulifct[l, ba, 1]  # Electron leaving
                fctp += paulifct[l, ba, 0]  # Electron entering 
            kh.set_matrix_element_pauli(fctm, fctp, bb, aa)
            if self.verbosity > verb:   print(f"ApproachPauli.generate_coupling_terms() state:{b} other:{a} rate:{fctp:.6f}")
        
        # Handle transitions to higher charge states
        for c in statesdm[ccharge]: # Loop over states with charge ccharge = bcharge+1
            cc = si.get_ind_dm0(c, c, ccharge)
            cb = si.get_ind_dm1(c, b, bcharge)
            fctm, fctp = 0, 0
            for l in range(nleads):
                fctm -= paulifct[l, cb, 0]  # Electron entering
                fctp += paulifct[l, cb, 1]  # Electron leaving
            kh.set_matrix_element_pauli(fctm, fctp, bb, cc)
*/

    void generate_coupling_terms(int b) {

        const int n = nstates;
        const int Q = count_electrons(b);
        const int max_charge = states_by_charge.size() - 1;

        //int bb = b; // Original
        int bb = state_order2[b];
        //int bb = get_ind_dm0(b, b, Q);  // Transform to density matrix index, matching Python approach

        if(verbosity > 3){  printf("PauliSolver::generate_coupling_terms() b: %i Q: %i \n", b, Q );  }

        int n2 = n * n;

        if( Q>0 ){ // Handle transitions from lower charge states (a -> b)
            int Qlower=Q-1;
            if(verbosity > 3){ printf("PauliSolver::generate_coupling_terms() Q-1 states: " );  print_vector( states_by_charge[Qlower].data(), states_by_charge[Qlower].size()); }          // for (int a : states_by_charge[Q-1]) printf("%i ", a); printf("\n");
            
            for (int a : states_by_charge[Qlower]) {
                //if (get_changed_site(b, a) == -1) continue;

                //int aa = a; // Original
                int aa = state_order2[a];
                //int aa = get_ind_dm0(a, a, Qlower);
                int ba = get_ind_dm1(b, a, Qlower);
                
                // if(verbosity > 3) {
                //     printf("INDEX-LOWER: state(b)=%d, other(a)=%d, charge(Q)=%d, aa=%d, ba=%d\n", b, a, Q, aa, ba);
                //     printf("INDEX-LOWER: dictdm[b]=%d, dictdm[a]=%d, shiftlst0[%d]=%d, lenlst[%d]=%d\n",  dictdm[b], dictdm[a], Qlower, shiftlst0[Qlower], Qlower, lenlst[Qlower]);
                // }
                
                double fctm = 0.0, fctp = 0.0;
                for (int l = 0; l < nleads; l++) {
                    //int idx = l * n2 * 2 + b * n * 2 + a * 2;
                    int idx = index_paulifct( l, ba );
                    //if(verbosity > 3) { printf("FACTOR-LOWER: lead=%d, ba=%d, idx=%d, idx+0=%d, idx+1=%d, factor[0]=%.6f, factor[1]=%.6f\n", l, ba, idx, idx, idx+1, pauli_factors[idx + 0], pauli_factors[idx + 1]);}
                    fctm -= pauli_factors[idx + 1];
                    fctp += pauli_factors[idx + 0];
                }
                
                if(verbosity > 3){ printf("set_matrix_element_pauli() LOWER [%i,%i] fctm: %.6f fctp: %.6f    bb: %i aa: %i \n", b, a, fctm, fctp, bb, aa); }
                set_matrix_element_pauli(fctm, fctp, bb, aa );
            }
        }        
        if( Q<states_by_charge.size()-1 ){ // Handle transitions to higher charge states (b -> c) 
            int Qhigher=Q+1;
            if(verbosity > 3){ printf("PauliSolver::generate_coupling_terms() Q+1 states: " );  print_vector( states_by_charge[Qhigher].data(), states_by_charge[Qhigher].size() ); } 
            for (int c : states_by_charge[Qhigher]) {
                //if (get_changed_site(b, c) == -1) continue;

                //int cc = c; // Original
                int cc = state_order2[c];
                //int cc = get_ind_dm0(c, c, Qhigher );
                int cb = get_ind_dm1(c, b, Q       );
                
                // if(verbosity > 3) {
                //     printf("INDEX-HIGHER: state(b)=%d, other(c)=%d, charge(Q)=%d, cc=%d, cb=%d\n", b, c, Q, cc, cb);
                //     printf("INDEX-HIGHER: dictdm[b]=%d, dictdm[c]=%d, shiftlst0[%d]=%d, shiftlst1[%d]=%d\n",  dictdm[b], dictdm[c], Qhigher, shiftlst0[Qhigher], Q, shiftlst1[Q]);
                // }
                
                double fctm = 0.0, fctp = 0.0;
                for (int l = 0; l < nleads; l++) {
                    //int idx = l * n2 * 2 + c * n * 2 + b * 2;
                    int idx = index_paulifct( l, cb );
                    //if(verbosity > 3) { printf("FACTOR-HIGHER: lead=%d, cb=%d, idx=%d, idx+0=%d, idx+1=%d, factor[0]=%.6f, factor[1]=%.6f\n", l, cb, idx, idx, idx+1, pauli_factors[idx + 0], pauli_factors[idx + 1]);}
                    fctm -= pauli_factors[idx + 0];
                    fctp += pauli_factors[idx + 1];
                }
                //int cc = c * n + c;
                
                if(verbosity > 3){ printf("set_matrix_element_pauli() HIGHER [%i,%i] fctm: %.6f fctp: %.6f    bb: %i aa: %i \n", b, c, fctm, fctp, bb, cc); }
                set_matrix_element_pauli( fctm, fctp, bb, cc );
            }
        }
        //if(verbosity > 3) { printf( "generate_coupling_terms() b: %i kernel: \n", b ); print_matrix(kernel, n, n); }

    }

    void normalize_kernel() {
        const int n = nstates;
        // Set first row to all ones (like Python)
        for(int j = 0; j < n; j++) { kernel[j] = 1.0; }        
        // if(verbosity > 3) {
        //     printf("Phase 2 - After normalization\n");
        //     print_matrix(kernel, n, n, "%.6g");
        // }
    }

    
    // Generate kernel matrix
    void generate_kern() {
        if(verbosity > 0) printf("\nPauliSolver::generate_kern() Building kernel matrix...\n");
        // -- set kernel to zero using memset
        memset(kernel, 0, sizeof(double) * nstates * nstates);
        state_order2 = {0,1,2,4,3,5,6,7};
        
        if(verbosity > 1) {
            printf("PauliSolver::generate_kern() starting\n");
            print_lead_params();
            print_state_energies();
            print_tunneling_amplitudes();
            print_states_by_charge();
        }

        const int n = nstates;
        
        // IMPORTANT: Only initialize states if needed - do not regenerate Pauli factors here
        // This avoids the double calculation problem when generate_fct() is called separately
        if(states_by_charge.empty()) {
            if(verbosity > 0) printf("PauliSolver::generate_kern() - initializing states_by_charge\n");
            init_states_by_charge();
        }

        // IMPORTANT: Check if pauli_factors have been calculated already
        // Only regenerate if they haven't been calculated yet
        if(pauli_factors == nullptr || n_pauli_factors == 0) {
            if(verbosity > 0) printf("PauliSolver::generate_kern() - pauli_factors not yet calculated, generating now\n");
            generate_fct();
        } else if(verbosity > 0) {
            printf("PauliSolver::generate_kern() - using pre-calculated pauli_factors\n");
        }

        if(verbosity > 0) {
            printf("PauliSolver::generate_kern().1 kernel generation\n");
            printf("pauli_factors[nlead%i,ndm1=%i,%i]:\n", nleads, ndm1, 2);
            printf("pauli_factors[lead=0]:\n");  print_matrix(pauli_factors       , ndm1, 2, "%16.8f");
            printf("pauli_factors[lead=1]:\n");  print_matrix(pauli_factors+ndm1*2, ndm1, 2, "%16.8f");
            //printf("DO NOT GO ANY FURTHER IN DEBUGGING UNTIL pauli_factors agree with python QmeQ reference \n");
            //exit(0);
        }

        //exit(0);

        if(verbosity > 1){ printf("\n==== PauliSolver::generate_kern().2 goto generate_coupling_terms()\n"); }
        std::fill(kernel, kernel + n * n, 0.0);
        for(int state = 0; state < n; state++) { 
            int b = state_order_inv[state];
            //generate_coupling_terms(b); 
            if(verbosity > 2) { printf("\n---- PauliSolver::generate_kern() -> generate_coupling_terms( istate=%i -> b=%i ) \n", state, b); }
            generate_coupling_terms(b);
        }
        if(verbosity > 1) { 
            printf("\nPauliSolver::generate_kern() final kernel:\n");
            print_matrix(kernel, n, n, "%16.8f");
            printf("===== PauliSolver::generate_kern() DONE \n");
        }

        //exit(0);
        //normalize_kernel();
        //if(verbosity > 0) { print_matrix(kernel, n, n, "Phase 2 - After normalization"); }
    }

    // Solve the kernel matrix equation
    void solve_kern() {
        const int n = nstates;
        
        // Create a copy of kernel matrix since solve() modifies it
        double* kern_copy = new double[n * n];
        std::copy(kernel, kernel + n * n, kern_copy);
        
        // Print the original kernel matrix for debugging
        if(verbosity > 1) {
            printf("PauliSolver::solve_kern() original kernel:\n");
            print_matrix(kernel, n, n, "%18.15f " );
        }
        
        // Apply normalization condition by replacing the first row with all ones
        // This is equivalent to Python's approach where kern[0] = self.norm_vec
        for(int j = 0; j < n; j++) {
            kern_copy[j] = 1.0;
        }
        
        // Set up RHS vector with first element = 1, rest = 0
        // This is equivalent to Python's approach where bvec[0] = 1
        double* rhs = new double[n];
        rhs[0] = 1.0;
        std::fill(rhs + 1, rhs + n, 0.0);
        
        if(verbosity > 1) {
            printf("PauliSolver::solve_kern() modified kernel with normalization row:\n");
            print_matrix(kern_copy, n, n, "%18.15f " );
            printf("PauliSolver::solve_kern() rhs: ");
            print_vector(rhs, n, "%18.15f " );
        }
        
        // Solve the system using Gaussian elimination
        linSolve_gauss(n, kern_copy, rhs, probabilities);
        
        if(verbosity > 1) {
            printf("PauliSolver::solve_kern() probabilities from Gaussian solver: ");
            print_vector(probabilities, n, "%18.15f " );
        }
        
        delete[] kern_copy;
        delete[] rhs;
    }
    
    // Update kernel matrix when parameters have changed
    void updateKernelMatrix() {
        if (!kernel_updated) {
            // Make sure energies and coupling are up to date
            updateStateEnergies();
            updateTunnelingAmplitudes();
            
            // Now generate the kernel matrix
            generate_kern();
            
            kernel_updated = true;
        }
    }
    
    // Solve the master equation with optional parameter updates
    void solve() {
        // Check if any parameters were changed and update as needed
        updateKernelMatrix();
        
        // Solve the kernel matrix equation
        solve_kern();
    }
    
    // Print methods for debugging
    void print_lead_params() const {
        printf("PauliSolver::print_lead_params() nleads: %d\n", nleads);
        for (int l = 0; l < nleads; l++) {
            //printf("  Lead %d: mu=%.6f, temp=%.6f, gamma=%.6f\n", l, leads[l].mu, leads[l].temp, leads[l].gamma);
            printf("  Lead %d: mu=%.6f, temp=%.6f\n", l, leads[l].mu, leads[l].temp);
        }
    }
    
    void print_state_energies() const {
        printf("PauliSolver::print_state_energies() nstates: %d\n", nstates);
        for (int i = 0; i < nstates; i++) {
            printf("  State %d(->%d): Energy=%.6f\n", i, state_order[i], energies[i]);
        }
    }
    
    void print_tunneling_amplitudes() const {
        printf("PauliSolver::print_tunneling_amplitudes() coupling:%p\n", coupling);
        if (!coupling) return;
        int n2 = nstates * nstates;
        for (int l = 0; l < nleads; l++) {
            printf("  Lead %d:\n", l);
            print_matrix(coupling + l * n2, nstates, nstates, "%16.8f");
        }
    }
    
    void print_states_by_charge() const {
        printf("PauliSolver::print_states_by_charge():\n");
        for (int i = 0; i < states_by_charge.size(); i++) {
            printf("  Charge %d: [", i);
            for (int j = 0; j < states_by_charge[i].size(); j++) {
                printf("%d", states_by_charge[i][j]);
                if (j < states_by_charge[i].size() - 1) {
                    printf(", ");
                }
            }
            printf("]\n");
        }
    }

    // Calculate current through a specific lead using the compact structure
    double generate_current(int lead_idx) {
        if(verbosity > 3) printf("\nDEBUG: generate_current() lead: %d this: %p\n", lead_idx, this);
        
        double current = 0.0;
        const int ncharge = states_by_charge.size();
        
        // Calculate current for each charge state transition
        for(int charge = 0; charge < ncharge - 1; charge++) {
            const int charge_higher = charge + 1;
            
            // Handle transitions: lower charge -> higher charge (b -> c)
            for(int b : states_by_charge[charge]) {
                const int bb = state_order2[b];
                
                for(int c : states_by_charge[charge_higher]) {
                    const int cc = state_order2[c];
                    const int cb = get_ind_dm1(c, b, charge);
                    
                    // Get factors from compact structure
                    const int idx = index_paulifct(lead_idx, cb);
                    const double fct_enter = pauli_factors[idx    ]; // Electron entering (b -> c)
                    const double fct_leave = pauli_factors[idx + 1]; // Electron leaving (c -> b)
                    
                    // Calculate current contribution
                    double fct1 =  probabilities[bb] * fct_enter;   // Electron entering: phi0[bb] * paulifct[l, cb, 0]
                    double fct2 = -probabilities[cc] * fct_leave;  // Electron leaving: -phi0[cc] * paulifct[l, cb, 1]
                    double contrib = fct1 + fct2;
                    
                    current += contrib;
                    
                    if(verbosity > 3) { printf("DEBUG: generate_current() lead:%d c:%d b:%d cb:%d fct1:%.6f fct2:%.6f contrib:%.6f\n",  lead_idx, c, b, cb, fct1, fct2, contrib); }
                }
            }
        }
        
        return current;
    }

/*
    def generate_current(self):
        """
        Calculates currents using Pauli master equation approach.

        Parameters
        ----------
        current : array
            (Modifies) Values of the current having nleads entries.
        energy_current : array
            (Modifies) Values of the energy current having nleads entries.
        heat_current : array
            (Modifies) Values of the heat current having nleads entries.
        """
        verb_print_(2,"ApproachPauli.generate_current() Calculating currents...")
        
        debug_print("DEBUG: ApproachPauli.generate_current()")
        phi0, E, paulifct, si = self.phi0, self.qd.Ea, self.paulifct, self.si
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        current = self.current
        energy_current = self.energy_current

        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c in statesdm[ccharge]:
                cc = si.get_ind_dm0(c, c, ccharge)
                for b in statesdm[bcharge]:
                    bb = si.get_ind_dm0(b, b, bcharge)
                    cb = si.get_ind_dm1(c, b, bcharge)
                    for l in range(nleads):
                        fct1 = +phi0[bb]*paulifct[l, cb, 0]
                        fct2 = -phi0[cc]*paulifct[l, cb, 1]
                        current[l] += fct1 + fct2
                        energy_current[l] += -(E[b]-E[c])*(fct1 + fct2)

        self.heat_current[:] = energy_current - current*self.leads.mulst
*/

    // Getter methods
    const double* get_kernel()        const { return kernel; }
    const double* get_probabilities() const { return probabilities; }
    const double* get_energies()      const { return energies; }
    const double* get_rhs()           const { return rhs; }
    const double* get_pauli_factors() const { return pauli_factors; }
};
