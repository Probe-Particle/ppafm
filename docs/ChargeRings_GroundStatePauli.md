# Ground State and Thermal Equilibrium Solvers for PauliSolver

This document outlines the implementation of two new equilibrium solvers for the `PauliSolver` class in `pauli.hpp`. These solvers provide alternative methods to calculate the equilibrium state of the system without solving the full Pauli Master Equation (PME), which can be computationally expensive.

## Table of Contents
1. [Overview of New Functions](#overview-of-new-functions)
2. [Implementation Details](#implementation-details)
   - [Ground State Solver](#ground-state-solver)
   - [Boltzmann Equilibrium Solver](#boltzmann-equilibrium-solver)
3. [Integration with Existing Code](#integration-with-existing-code)
4. [Usage Examples](#usage-examples)
5. [Technical Considerations](#technical-considerations)
6. [Future Extensions](#future-extensions)

## Overview of New Functions

Two new public methods will be added to the `PauliSolver` class:

1. **`solve_equilibrium_ground_state()`**
   - Finds the many-body state with the minimum energy
   - Sets its probability to 1.0
   - Sets all other state probabilities to 0.0
   - Represents a zero-temperature equilibrium

2. **`solve_equilibrium_boltzmann(double temperature, double chemical_potential)`**
   - Calculates thermal equilibrium probabilities using Boltzmann distribution
   - Considers both the many-body state energies and the number of electrons in each state
   - Normalizes probabilities to sum to 1.0
   - Represents a finite-temperature equilibrium

## Implementation Details

### Ground State Solver

```cpp
void PauliSolver::solve_equilibrium_ground_state() {
    // Ensure energies are calculated
    if (nstates == 0 || energies == nullptr || probabilities == nullptr) {
        if (verbosity > 0) {
            printf("Error: Solver not properly initialized.\n");
        }
        return;
    }

    // Find state with minimum energy
    double min_energy = energies[0];
    int min_idx = 0;
    for (int i = 1; i < nstates; ++i) {
        if (energies[i] < min_energy) {
            min_energy = energies[i];
            min_idx = i;
        }
    }

    // Set probabilities
    for (int i = 0; i < nstates; ++i) {
        probabilities[i] = 0.0;
    }
    probabilities[min_idx] = 1.0;

    if (verbosity > 0) {
        printf("Ground state found at index %d with energy %.6f meV\n", 
               min_idx, min_energy);
    }
}
```

### Boltzmann Equilibrium Solver

```cpp
void PauliSolver::solve_equilibrium_boltzmann(double temperature, double chemical_potential) {
    // Input validation
    if (nstates == 0 || energies == nullptr || probabilities == nullptr) {
        if (verbosity > 0) {
            printf("Error: Solver not properly initialized.\n");
        }
        return;
    }
    
    if (temperature <= 0) {
        if (verbosity > 0) {
            printf("Warning: Temperature <= 0, using ground state instead.\n");
        }
        solve_equilibrium_ground_state();
        return;
    }

    double sum_weights = 0.0;
    double kT = KB * temperature;  // KB is Boltzmann constant in meV/K
    
    // Calculate unnormalized Boltzmann weights
    for (int i = 0; i < nstates; ++i) {
        int num_electrons = count_electrons(i);
        double effective_energy = energies[i] - num_electrons * chemical_potential;
        probabilities[i] = exp(-effective_energy / kT);
        sum_weights += probabilities[i];
    }

    // Normalize probabilities
    if (sum_weights > 0) {
        for (int i = 0; i < nstates; ++i) {
            probabilities[i] /= sum_weights;
        }
    } else {
        // Fallback to ground state if all weights are zero (unlikely)
        if (verbosity > 0) {
            printf("Warning: All Boltzmann weights zero, using ground state.\n");
        }
        solve_equilibrium_ground_state();
        return;
    }

    if (verbosity > 0) {
        printf("Boltzmann equilibrium calculated for T=%.6f K, mu=%.6f meV\n", 
               temperature, chemical_potential);
        if (verbosity > 1) {
            printf("State probabilities:\n");
            for (int i = 0; i < nstates; ++i) {
                printf("  State %d: p=%.6f\n", i, probabilities[i]);
            }
        }
    }
}
```

## Integration with Existing Code

### Required Modifications to `pauli.hpp`

1. Add method declarations to the `PauliSolver` class:

```cpp
class PauliSolver {
    // ... existing code ...
    
    // New equilibrium solvers
    void solve_equilibrium_ground_state();
    void solve_equilibrium_boltzmann(double temperature, double chemical_potential);
    
    // ... rest of the class ...
};
```

2. The implementations can be placed either:
   - Inline in the header file (for small functions)
   - In the corresponding `.cpp` file (for larger implementations)

### Dependencies

- Both functions require `calculate_state_energies()` to be called first to populate the `energies` array
- `solve_equilibrium_boltzmann` requires `count_electrons()` to be implemented
- The Boltzmann constant `KB` must be defined (it's already defined as a global constant)

## Usage Examples

### Example 1: Using Ground State Solver

```cpp
// Initialize solver and set parameters
PauliSolver solver(nSingle, nstates, nleads, verbosity);
solver.setHsingle(Hsingle);
solver.setW(W);

// Calculate state energies
solver.calculate_state_energies();

// Find and set ground state
solver.solve_equilibrium_ground_state();

// Get probabilities
double* probs = solver.get_probabilities(nstates);
```

### Example 2: Using Boltzmann Equilibrium Solver

```cpp
// Initialize solver and set parameters
PauliSolver solver(nSingle, nstates, nleads, verbosity);
solver.setHsingle(Hsingle);
solver.setW(W);

// Calculate state energies
solver.calculate_state_energies();

// Set lead parameters (for chemical potential and temperature)
double substrate_temp = 4.2;  // K
double substrate_mu = 0.0;    // meV

// Calculate thermal equilibrium
solver.solve_equilibrium_boltzmann(substrate_temp, substrate_mu);

// Get probabilities
double* probs = solver.get_probabilities(nstates);
```

## Technical Considerations

### Energy Reference
- The chemical potential is referenced to the same energy scale as the single-particle energies in `Hsingle`
- The total energy of a many-body state includes both the single-particle energies and the Coulomb interaction `W`

### Numerical Stability
- For very low temperatures, the Boltzmann weights can underflow
- The implementation includes a fallback to the ground state if all weights become zero

### Performance
- Both solvers are O(N) in the number of states
- The ground state solver is slightly faster as it only needs to find the minimum
- The Boltzmann solver requires additional calculations for the weights and normalization

## Future Extensions

1. **Hybrid Solver**: Combine the two approaches by first finding low-energy states and only calculating Boltzmann weights for those
2. **Grand Canonical Ensemble**: Extend to allow variable particle number
3. **Parallelization**: Implement parallel computation of state energies and weights
4. **Caching**: Cache the results of `calculate_state_energies()` to avoid redundant calculations
5. **Python Bindings**: Add Python wrappers for the new functions