#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import os
import argparse
from sys import path

# Setup for QmeQ and C++ Pauli solvers
bASAN = True
bQmeQ = True  # Toggle to switch between using QmeQ or not

if bASAN:
    # Get ASan library path
    asan_lib = os.popen('gcc -print-file-name=libasan.so').read().strip()
    print("Preloading ASan library: ", asan_lib)
    # Set LD_PRELOAD environment variable
    os.environ['LD_PRELOAD'] = asan_lib
    os.environ['ASAN_OPTIONS'] = 'detect_leaks=0'

# Add the necessary paths
path.insert(0, '../../pyProbeParticle')
import pauli as psl
from pauli import PauliSolver

# Import QmeQ if enabled
if bQmeQ:
    sys.path.append('/home/prokop/git_SW/qmeq')
    import qmeq
    from qmeq import config
    from qmeq import indexing as qmqsi
    from qmeq.config import verb_print_
    import QmeQ_pauli as qmeq_pauli

def load_scan_data(filename):
    """Load and parse the scan data file"""
    # Read all lines
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Separate header lines (starting with #) from data lines
    header_lines = [line for line in lines if line.startswith('#')]
    data_lines = [line for line in lines if not line.startswith('#')]
    
    # Extract parameters from header - much more elegant approach
    params = {}
    data_columns = {}
    
    section = 'parameters'  # Default section
    
    for line in header_lines:
        # Skip lines that don't contain colons or aren't parameter definitions
        if not ':' in line or not line.startswith('# #'):
            continue
        
        # Remove the '# #' prefix and strip whitespace
        line = line[3:].strip()
        
        # Check for section headers
        if 'Data columns' in line:
            section = 'columns'
            continue
        elif 'Calculation parameters' in line:
            section = 'parameters'
            continue
        
        # Split by colon and extract key-value pair
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        # Handle numeric values
        try:
            value = float(value)
        except ValueError:
            pass  # Keep as string if not a number
        
        # Store in appropriate dictionary based on section
        if section == 'parameters':
            params[key] = value
        elif section == 'columns' and key.isdigit():
            data_columns[int(key)] = value
    
    # Parse data lines
    data = np.array([list(map(float, line.strip().split())) for line in data_lines])
    
    # Create structured data arrays based on data_columns information
    scan_data = {
        'distance': data[:, 0],
        'x': data[:, 1],
        'y': data[:, 2]
    }
    
    # Map column indices to meaningful names using data_columns
    col_map = {}
    for col, desc in data_columns.items():
        # Convert 1-based column index to 0-based for numpy
        col_idx = col - 1
        if col_idx < data.shape[1]:
            # Clean up description for use as dict key
            key = desc.replace('[A]', '').strip()
            col_map[key] = col_idx
    
    # Add data columns based on their descriptions
    for key, col_idx in col_map.items():
        if col_idx < data.shape[1]:
            scan_data[key] = data[:, col_idx]
    
    # Ensure essential columns exist (fallback to default indices if not found)
    for i in range(1, 4):
        esite_key = f'Esite_{i}'
        if esite_key not in scan_data and 3+i-1 < data.shape[1]:
            scan_data[esite_key] = data[:, 3+i-1]
    
    return params, scan_data, data_columns

def prepare_leads_cpp(params):
    """Prepare static inputs for leads based on parameters from data file"""
    # Constants
    NSingle = int(params['nsite'])  # number of impurity states
    NLeads = 2   # number of leads
    
    # Lead parameters
    muS = 0.0    # substrate chemical potential
    muT = 0.0    # tip chemical potential
    Temp = params['temperature']  # temperature from file
    DBand = 1000.0  # lead bandwidth
    
    # Use parameters from file when available, otherwise use defaults
    GammaS = 0.20  # coupling to substrate - default
    GammaT = 0.05  # coupling to tip - default
    
    # Tunneling amplitudes
    VS = np.sqrt(GammaS/np.pi)  # substrate
    VT = np.sqrt(GammaT/np.pi)  # tip
    
    # Position-dependent coefficients (could be extracted from file data)
    coeffT = 0.3  # default value
    
    # Leads
    lead_mu = np.array([muS, muT + params['VBias']   ])
    lead_temp = np.array([Temp, Temp])
    lead_gamma = np.array([GammaS, GammaT])
    
    # Lead Tunneling matrix
    TLeads = np.array([
        [VS, VS, VS],
        [VT, VT*coeffT, VT*coeffT]
    ])
    
    # Return additional parameters for QmeQ
    return TLeads, lead_mu, lead_temp, lead_gamma, NSingle, NLeads, VS, VT, coeffT, muS, muT, DBand

def run_cpp_scan(params, scan_data):
    """Run C++ Pauli simulation for the entire scan - efficient implementation"""
    nsteps = len(scan_data['distance'])
    print(f"\nRunning C++ Pauli simulation...")
    currents = np.zeros(nsteps)
    
    # Get leads parameters once - only use what we need for C++
    TLeads, lead_mu, lead_temp, lead_gamma, NSingle, NLeads, *_ = prepare_leads_cpp(params)
    
    # Create the Pauli solver once
    state_order = [0, 4, 2, 6, 1, 5, 3, 7]
    state_order = np.array(state_order, dtype=np.int32)
    NStates = 2**NSingle
    W = params['onSiteCoulomb']  # inter-site coupling
    t = 0.0      # direct hopping

    print(f"NSingle: {NSingle}, NLeads: {NLeads}, NStates: {NStates}")
    print(f"W: {W}, t: {t}")
    
    # Initialize solver once
    pauli = PauliSolver(NSingle, NLeads, verbosity=verbosity)
    #pauli.set_leads(lead_mu, lead_temp, lead_gamma)
    pauli.set_lead(0, lead_mu[0], lead_temp[0] )
    pauli.set_lead(1, lead_mu[1], lead_temp[1] )
    pauli.set_tunneling(TLeads)
    
    for i in range(nsteps):
        # Get on-site energies from scan data at the specified index
        eps1 = scan_data['Esite_1'][i]*1000.0
        eps2 = scan_data['Esite_2'][i]*1000.0
        eps3 = scan_data['Esite_3'][i]*1000.0
        
        # Prepare single-particle Hamiltonian
        Hsingle = np.array([
            [eps1, t, 0],
            [t, eps2, t],
            [0, t, eps3]
        ])
        
        # Solve and get current
        currents[i] = pauli.solve_hsingle(Hsingle, W, 1, state_order)
        
        if i % 10 == 0:  # Print progress every 10 steps
            print(f"Processed step {i+1}/{nsteps}, Current: {currents[i]}    eps: {eps1} {eps2} {eps3}")
    
    return currents

def run_cpp_scan_2(params, scan_data):
    """Run C++ Pauli simulation for the entire scan using scan_current()"""
    nsteps = len(scan_data['distance'])
    print(f"\nRunning C++ Pauli simulation using scan_current...")
    
    # Get leads parameters once
    TLeads, lead_mu, lead_temp, lead_gamma, NSingle, NLeads, *_ = prepare_leads_cpp(params)
    
    # Create the Pauli solver once
    pauli = PauliSolver(NSingle, NLeads, verbosity=verbosity)
    pauli.set_lead(0, lead_mu[0], lead_temp[0])
    pauli.set_lead(1, lead_mu[1], lead_temp[1])
    pauli.set_tunneling(TLeads)
    
    # Prepare single-particle Hamiltonians using vectorized operations
    eps = np.column_stack((scan_data['Esite_1'], scan_data['Esite_2'], scan_data['Esite_3'])) * 1000.0
    hsingles = np.zeros((nsteps, 3, 3))
    hsingles[:, np.arange(3), np.arange(3)] = eps
    
    # Set up other parameters
    Ws = np.full(nsteps, params['onSiteCoulomb'])
    state_order = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)
    VGates = np.zeros(nsteps)  # Initialize VGates array

    # Run scan
    currents = pauli.scan_current(hsingles=hsingles, Ws=Ws, VGates=VGates, state_order=state_order)
    
    return currents

def run_qmeq_scan(params, scan_data):
    """Run QmeQ Pauli simulation for the entire scan
    
    Uses helper functions from QmeQ_pauli.py for better modularity and to avoid code duplication.
    """
    if not bQmeQ:
        print("QmeQ functionality is disabled. Set bQmeQ=True to enable.")
        return None
        
    nsteps = len(scan_data['distance'])
    print(f"\nRunning QmeQ Pauli simulation...")
    
    # Get all leads parameters including ones needed for QmeQ
    _, _, _, _, NSingle, NLeads, VS, VT, coeffT, muS, muT, DBand = prepare_leads_cpp(params)
    
    # Get inter-site coupling
    W = params['onSiteCoulomb']
    VBias = params['VBias']
    Temp = params['temperature']
    t = 0.0  # direct hopping
    
    print(f"QmeQ: NSingle: {NSingle}, NLeads: {NLeads}")
    print(f"QmeQ: W: {W}, t: {t}, Temp: {Temp}, VBias: {VBias}")
    
    # Initialize currents array
    currents_qmeq = np.zeros(nsteps)
    
    # Process each step
    for i in range(nsteps):
        # Get on-site energies for this position (convert to meV)
        eps1 = scan_data['Esite_1'][i]*1000.0
        eps2 = scan_data['Esite_2'][i]*1000.0
        eps3 = scan_data['Esite_3'][i]*1000.0
        
        # Use helper functions from QmeQ_pauli.py to build Hamiltonian and leads
        hsingle, coulomb = qmeq_pauli.build_hamiltonian(eps1, eps2, eps3, t, W)
        mu_L, Temp_L, TLeads_dict = qmeq_pauli.build_leads(muS, muT+VBias, Temp, VS, [VT, VT*coeffT, VT*coeffT] )
        
        # Run QmeQ solver with low verbosity using the helper function
        qmeq_result = qmeq_pauli.run_QmeQ_solver( NSingle, hsingle, coulomb, NLeads, TLeads_dict, mu_L, Temp_L, DBand, verbosity )
        
        # Extract current from result
        if qmeq_result is not None:
            currents_qmeq[i] = qmeq_result['current']
        else:
            print(f"Error running QmeQ solver at step {i}")
            currents_qmeq[i] = np.nan
        
        # Print progress every 10 steps
        if i % 10 == 0 or i == nsteps-1:
            print(f"Processed QmeQ step {i+1}/{nsteps}, Current: {currents_qmeq[i]}    eps: {eps1} {eps2} {eps3}")
    
    return currents_qmeq


def compare_results(cpp_results, qmeq_results, distance, tol=1e-8, print_every=50):
    """Compare results from C++ and QmeQ solvers"""
    if qmeq_results is None:
        print("QmeQ results not available for comparison.")
        return None
        
    nsteps = len(distance)
    print(f"\nComparing C++ and QmeQ results (tolerance: {tol})...")
    
    diffs = np.abs(cpp_results - qmeq_results)
    max_diff = np.max(diffs)
    avg_diff = np.mean(diffs)
    
    print(f"Maximum difference: {max_diff}")
    print(f"Average difference: {avg_diff}")
    
    # Check if differences are within tolerance
    within_tol = (max_diff <= tol)
    print(f"Results {'match within tolerance' if within_tol else 'DO NOT match within tolerance'}.")
    
    # Print some sample comparisons
    print("\nSample comparisons (position, C++, QmeQ, diff):")
    step = max(1, nsteps // print_every)
    for i in range(0, nsteps, step):
        print(f"{distance[i]:.6f}, {cpp_results[i]:.10f}, {qmeq_results[i]:.10f}, {diffs[i]:.10f}")
    
    return within_tol

def plot_results(scan_data, cpp_currents, qmeq_currents=None, save_path=None):
    """Plot the simulation results"""
    plt.figure(figsize=(12, 10))
    
    # Plot 1: On-site energies
    plt.subplot(3, 1, 1)
    plt.plot(scan_data['distance'], scan_data['Esite_1'], 'r-', label='Esite_1')
    plt.plot(scan_data['distance'], scan_data['Esite_2'], 'g-', label='Esite_2')
    plt.plot(scan_data['distance'], scan_data['Esite_3'], 'b-', label='Esite_3')
    plt.xlabel('Distance [Å]')
    plt.ylabel('On-site Energy')
    plt.title('On-site Energies along Scan Line')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Calculated currents
    plt.subplot(3, 1, 2)
    if cpp_currents is not None:
        plt.plot(scan_data['distance'], cpp_currents, '.-r', label='C++ Current')
    if qmeq_currents is not None:
        plt.plot(scan_data['distance'], qmeq_currents, '.-b', label='QmeQ Current')
    plt.xlabel('Distance [Å]')
    plt.ylabel('Current')
    plt.title('Comparison of Calculated Currents')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Difference between C++ and QmeQ results (if both available)
    if cpp_currents is not None and qmeq_currents is not None:
        plt.subplot(3, 1, 3)
        diff = np.abs(cpp_currents - qmeq_currents)
        plt.plot(scan_data['distance'], diff, 'k-', label='|C++ - QmeQ|')
        plt.xlabel('Distance [Å]')
        plt.ylabel('Absolute Difference')
        plt.yscale('log')  # Use log scale for better visibility of small differences
        plt.title('Difference Between C++ and QmeQ Results')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save to file if path is provided, otherwise try to show
    if save_path:
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path)
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot: {e}")
            print("Try running with --save option to save plots to file instead.")
    
'''
run like this:
python pauli_1D_load.py --sample 20 --solver both --compare
python pauli_1D_load.py --sample 1 --solver both --compare
'''

if __name__ == "__main__":
# Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Load scan data and run Pauli simulation')
    parser.add_argument('filename', nargs='?', default="rings/0.60_line_scan.dat", help='Path to scan data file')
    #parser.add_argument('filename', nargs='?', default="rings/0.60_line_scan_tail.dat", help='Path to scan data file')
    parser.add_argument('--save', '-s', help='Save plot to file instead of displaying')
    parser.add_argument('--sample', '-n', type=int, default=1, help='Sample every n-th point to speed up calculation')
    parser.add_argument('--solver', choices=['cpp', 'qmeq', 'both'], default='cpp', help='Which solver to use (cpp, qmeq, or both)')
    parser.add_argument('--compare', action='store_true',  help='Compare C++ and QmeQ results')
    args = parser.parse_args()
    
    print(f"Loading data from {args.filename}")
    params, scan_data, data_columns = load_scan_data(args.filename)

    # Convert VBias to meV
    params['VBias'] *= 1000.0
    
    # Print extracted parameters
    print("\nExtracted Parameters:")
    print("Calculation parameters: ")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    if data_columns:
        print("Data columns: ")
        for col, desc in sorted(data_columns.items()):
            print(f"{col}: {desc}")
    
    # Sample data if requested
    if args.sample and args.sample > 1:
        print(f"\nSampling every {args.sample}th point from {len(scan_data['distance'])} points")
        for key in scan_data:
            scan_data[key] = scan_data[key][::args.sample]
        print(f"Reduced to {len(scan_data['distance'])} points")
    
    # Run simulations based on solver choice
    cpp_currents = None
    qmeq_currents = None

    verbosity = 0
    
    if args.solver in ['cpp', 'both']:
        cpp_currents1 = None
        #cpp_currents1 = run_cpp_scan  (params, scan_data)  # If I uncomment this, the instabilities disappear
        cpp_currents  = run_cpp_scan_2(params, scan_data) # this works but resulting current suffer from instabilities
        print("cpp_currents.shape ", cpp_currents.shape)
        #print("\nC++ currents:",  cpp_currents)
    
    if args.solver in ['qmeq', 'both'] and bQmeQ:
        qmeq_currents = run_qmeq_scan(params, scan_data)
    elif args.solver == 'qmeq' and not bQmeQ:
        print("QmeQ is not enabled. Set bQmeQ=True at the top of the script to enable it.")
    
    # Compare results if requested and both solvers were run
    if args.compare and cpp_currents is not None and qmeq_currents is not None:
        compare_results(cpp_currents, qmeq_currents, scan_data['distance'])
    
    # Save data to NumPy file
    np_save_file = args.save.replace('.png', '.npz') if args.save else 'pauli_scan_results.npz'
    print(f"\nSaving numerical results to {np_save_file}")
    
    # Prepare dictionary of results
    results_dict = {
        'distance': scan_data['distance'],
        'esite1': scan_data['Esite_1'],
        'esite2': scan_data['Esite_2'],
        'esite3': scan_data['Esite_3']
    }
    
    if cpp_currents is not None:
        results_dict['cpp_current'] = cpp_currents
        
    if qmeq_currents is not None:
        results_dict['qmeq_current'] = qmeq_currents
        
    if 'STM_total' in scan_data:
        results_dict['stm_total'] = scan_data['STM_total']
        
    np.savez(np_save_file, **results_dict)
    
    # Plot results
    print("\nPlotting results...")
    #plot_results(scan_data, cpp_currents, qmeq_currents, save_path=args.save)
    plot_results(scan_data, cpp_currents, cpp_currents1, save_path=args.save)
    
    #return scan_data, cpp_currents, qmeq_currents
    plt.show()
