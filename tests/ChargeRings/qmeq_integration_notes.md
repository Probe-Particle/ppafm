# QmeQ Integration Notes

## Overview

This document provides guidance for recalculating the ChargeRings simulation results using QmeQ (Quantum Master Equation for Quantum dot transport calculations) in a standalone script for headless execution on a cluster.

## Current Data Structure Analysis

The current simulation saves the following data in `.npz` files:

- `spos`: Site positions in 3D space
- `STM`: STM current data 
- `Es`: Site energies
- `Ts`: Tunneling rates
- `probs`: Probabilities of different states
- `params_json`: All simulation parameters as a JSON string

Additionally, the parameters are saved separately in a `.json` file for easy access.

## Required QmeQ Inputs

QmeQ requires the following key inputs for transport calculations in quantum dot systems:

1. **Site Energies**: Energy levels of the individual quantum dots (available in `Es`)
2. **Lead Couplings**: Tunneling rates between dots and leads (available in `Ts`)
3. **Coulomb Interactions**: On-site and interdot Coulomb repulsion (available in `params['onSiteCoulomb']`)
4. **Temperature**: Thermal energy (available in `params['Temp']`)
5. **Chemical Potentials**: Bias voltage settings (available in `params['VBias']`)

## Data Sufficiency Analysis

Analyzing the saved data:

- ✅ **Site Energies**: Directly available in `Es`
- ✅ **Lead Couplings**: Available in `Ts` 
- ✅ **Coulomb Interaction**: Available in `params['onSiteCoulomb']`
- ✅ **Temperature**: Available in `params['Temp']`
- ✅ **Bias Voltage**: Available in `params['VBias']`
- ✅ **Geometry**: Site positions available in `spos`

All the necessary parameters appear to be present in the saved data files. The `.npz` file contains the essential data arrays, and the same parameters are stored both in the `params_json` field of the `.npz` file and in a separate `.json` file.

## Implementation Steps

### 1. Create a QmeQ-based Solver Script

```python
#!/usr/bin/env python

import numpy as np
import json
import argparse
import os
import qmeq

def load_data_from_npz(npz_file):
    """Load simulation data from .npz file"""
    data = np.load(npz_file)
    # Extract data arrays
    sim_data = {}
    for key in data.keys():
        sim_data[key] = data[key]
    
    # Extract parameters from JSON string
    if 'params_json' in sim_data:
        sim_data['params'] = json.loads(sim_data['params_json'])
    
    return sim_data

def create_qmeq_system(sim_data):
    """Create a QmeQ system from the simulation data"""
    params = sim_data['params']
    Es = sim_data['Es']
    nsite = int(params['nsite'])
    
    # Create QmeQ system with site energies
    qd_system = qmeq.Builder(
        nsingle=nsite,  # Number of single-particle states
        nleads=2,       # Number of leads (source and drain)
        indexing='lin'  # Linear indexing for many-body states
    )
    
    # Set site energies
    for i in range(nsite):
        qd_system.add_site(i, Es[i])
    
    # Set Coulomb interaction
    U = params['onSiteCoulomb']
    for i in range(nsite):
        qd_system.add_interaction(i, i, U)
    
    # Set lead coupling from Ts
    Ts = sim_data['Ts']
    GammaS = params['GammaS']  # Source lead coupling
    GammaT = params['GammaT']  # Drain lead coupling
    
    # Configure leads with appropriate energy-dependent tunneling
    for i in range(nsite):
        # Coupling to source lead (lead 0)
        qd_system.add_lead_coupling(i, 0, GammaS)
        # Coupling to drain lead (lead 1)
        qd_system.add_lead_coupling(i, 1, GammaT)
    
    # Configure lead properties
    # Lead 0: Source
    # Lead 1: Drain
    qd_system.add_lead(0, 'fermi', params['Temp'], 0.0)  # Source lead chemical potential
    qd_system.add_lead(1, 'fermi', params['Temp'], -params['VBias'])  # Drain lead with bias
    
    return qd_system

def calculate_with_qmeq(qd_system, method='pauli'):
    """Calculate transport using QmeQ"""
    # Create QmeQ solver based on method
    if method == 'pauli':
        # Pauli master equation (similar to current solver)
        solver = qmeq.approach.Pauli(qd_system)
    elif method == 'redfield':
        # Redfield approach (more accurate for coherent effects)
        solver = qmeq.approach.Redfield(qd_system)
    elif method == '1vN':
        # First-order von Neumann approach
        solver = qmeq.approach.FirstOrderVN(qd_system)
    elif method == 'lindblad':
        # Lindblad master equation
        solver = qmeq.approach.Lindblad(qd_system)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Run calculation
    solver.solve()
    
    return solver

def save_qmeq_results(solver, original_data, output_file):
    """Save QmeQ results in a similar format to original data"""
    # Get original data and params
    sim_data = original_data.copy()
    params = sim_data['params']
    
    # Replace relevant data with QmeQ results
    sim_data['current_qmeq'] = solver.current
    sim_data['probs_qmeq'] = solver.phi
    
    # Add QmeQ method information to params
    params['qmeq_method'] = solver.approach
    sim_data['params'] = params
    
    # Update params_json
    sim_data['params_json'] = json.dumps(params)
    
    # Save to npz file
    np.savez(output_file, **sim_data)
    
    # Also save updated params to JSON
    json_file = os.path.splitext(output_file)[0] + '.json'
    with open(json_file, 'w') as f:
        json.dump(params, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Recalculate ChargeRings results using QmeQ')
    parser.add_argument('input_file', help='Path to input .npz file')
    parser.add_argument('--output', help='Path to output .npz file (default: input_file_qmeq.npz)')
    parser.add_argument('--method', choices=['pauli', 'redfield', '1vN', 'lindblad'], 
                        default='pauli', help='QmeQ solution method')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        base, ext = os.path.splitext(args.input_file)
        args.output = f"{base}_qmeq_{args.method}{ext}"
    
    print(f"Loading data from {args.input_file}")
    sim_data = load_data_from_npz(args.input_file)
    
    print(f"Creating QmeQ system")
    qd_system = create_qmeq_system(sim_data)
    
    print(f"Calculating transport using QmeQ ({args.method})")
    solver = calculate_with_qmeq(qd_system, method=args.method)
    
    print(f"Saving results to {args.output}")
    save_qmeq_results(solver, sim_data, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()
```

### 2. Cluster Execution Script

Create a simple bash script for submitting jobs to a cluster:

```bash
#!/bin/bash
#SBATCH --job-name=qmeq_calc
#SBATCH --output=qmeq_calc_%j.out
#SBATCH --error=qmeq_calc_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load required modules (adjust as needed for your cluster)
module load python/3.8

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run calculation
python qmeq_solver.py $1 --method $2
```

### 3. Usage Examples

#### Local Execution

```bash
# Run with default Pauli master equation approach
python qmeq_solver.py simulation_results.npz

# Run with Redfield approach for better accuracy
python qmeq_solver.py simulation_results.npz --method redfield
```

#### Cluster Execution

```bash
# Submit job to cluster
sbatch run_qmeq.sh simulation_results.npz pauli

# Submit multiple jobs with different methods
for method in pauli redfield 1vN lindblad; do
  sbatch run_qmeq.sh simulation_results.npz $method
done
```

## Comparing Results

To compare the results from different solvers:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load original and QmeQ results
orig_data = np.load('simulation_results.npz')
qmeq_data = np.load('simulation_results_qmeq_redfield.npz')

# Compare currents
plt.figure(figsize=(10, 6))
plt.plot(orig_data['STM'], label='Original C++ Pauli')
plt.plot(qmeq_data['current_qmeq'], label='QmeQ Redfield')
plt.xlabel('Position')
plt.ylabel('Current')
plt.legend()
plt.title('Current Comparison')
plt.savefig('current_comparison.png')

# Compare state probabilities
plt.figure(figsize=(10, 6))
for i in range(min(orig_data['probs'].shape[-1], 5)):
    plt.plot(orig_data['probs'][:, i], 'o-', label=f'Orig State {i}')
    plt.plot(qmeq_data['probs_qmeq'][:, i], 'x--', label=f'QmeQ State {i}')
plt.xlabel('State Index')
plt.ylabel('Probability')
plt.legend()
plt.title('State Probability Comparison')
plt.savefig('probability_comparison.png')
```

## Challenges and Considerations

1. **QmeQ Installation**: QmeQ requires specific dependencies. Consider using a Docker container or a predefined environment module on the cluster.

2. **Mapping Between Solvers**: The state indexing between the C++ Pauli solver and QmeQ might differ. Ensure proper mapping of states for meaningful comparisons.

3. **Parameter Conversion**: Some parameters might need conversion between the models (e.g., units, conventions).

4. **Performance Considerations**: More advanced methods like Redfield or 1vN are significantly more computationally expensive than Pauli master equation approaches.

5. **Memory Requirements**: QmeQ can require significant memory for systems with many sites due to the exponential growth of the Hilbert space.
