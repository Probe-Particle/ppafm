#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import copy

# Import from ppafm modules
sys.path.append('../../')
from pyProbeParticle import utils as ut
from pyProbeParticle import pauli

# Import local modules
import pauli_scan
from exp_utils import plot_exp_voltage_line_scan, create_line_coordinates
from monte_carlo_optimizer import MonteCarloOptimizer

def load_experimental_data(filename='exp_rings_data.npz'):
    """
    Load experimental data from npz file
    
    Args:
        filename (str): Path to the experimental data file
        
    Returns:
        tuple: (X, Y, dIdV, I, biases) - Experimental data arrays
    """
    print(f"Loading experimental data from {filename}...")
    try:
        data = np.load(filename)
        # Convert from nm to Å
        X = data['X'] * 10
        Y = data['Y'] * 10
        dIdV = data['dIdV']
        I = data['I']
        biases = data['biases']
        
        # Center coordinates
        cx, cy = data['center_x']*10, data['center_y']*10
        X -= cx
        Y -= cy
        
        print(f"Experimental data loaded successfully.")
        print(f"  Data shapes: X={X.shape}, Y={Y.shape}, I={I.shape}, dIdV={dIdV.shape}, biases={biases.shape}")
        return X, Y, dIdV, I, biases
        
    except FileNotFoundError:
        print(f"ERROR: Could not find experimental data file {filename}")
        return None, None, None, None, None

def extract_experimental_data_along_line(X, Y, I, dIdV, biases, exp_start_point, exp_end_point, pointPerAngstrom=5):
    """
    Extract experimental data along a line
    
    Args:
        X, Y: Experimental X, Y coordinates
        I, dIdV: Experimental current and dI/dV data
        biases: Experimental bias voltages
        exp_start_point: Start point for the line in experimental data
        exp_end_point: End point for the line in experimental data
        pointPerAngstrom: Points per Angstrom for interpolation
        
    Returns:
        tuple: (exp_STM, exp_dIdV, dist, biases) - Extracted experimental data,
        distance array, and bias voltages
    """
    print(f"Extracting experimental data along line from {exp_start_point} to {exp_end_point}...")
    
    # Create line coordinates for experiment and interpolate experimental data
    # For STM current
    exp_STM, dist = plot_exp_voltage_line_scan(
        X, Y, I, biases, 
        exp_start_point, exp_end_point, 
        pointPerAngstrom=pointPerAngstrom,
        ax=None,  # No plotting, just return the data
        cmap='hot'  # Use hot colormap for STM
    )
    
    # For dI/dV
    exp_dIdV, _ = plot_exp_voltage_line_scan(
        X, Y, dIdV, biases, 
        exp_start_point, exp_end_point, 
        pointPerAngstrom=pointPerAngstrom,
        ax=None  # No plotting, just return the data
    )
    
    print(f"Experimental data extracted successfully.")
    print(f"  Extracted data shapes: STM={exp_STM.shape}, dIdV={exp_dIdV.shape}, dist={dist.shape}")
    
    return exp_STM, exp_dIdV, dist, biases

def main():
    # Define simulation parameters
    params = {
            'nsite': 3,            # Number of sites
            'phi0': 0.0,           # Phase angle (radians)
            'phiRot': 1.3,         # Rotation angle (radians)
            'radius': 5.2,         # Ring radius (Å)
            'phi0_ax': 0.2,         # Axial phase angle (radians)
            'VBias': 0.7,          # Bias voltage (V)
            'Rtip': 3.0,           # Tip radius (Å)
            'z_tip': 5.0,          # Tip height (Å)
            'zV0': -1.0,           # Potential offset (V)
            'zVd': 15.0,           # Potential decay length (Å)
            'zQd': 0.0,            # Charge decay length (Å)
            'Q0': 1.0,             # Reference charge (e)
            'Qzz': 10.0,           # Quadrupole moment (e·Å²)
            'Esite': -0.1,         # Site energy (eV)
            'W': 0.02,             # Tunnel coupling (eV)
            'decay': 0.3,          # Decay parameter
            'GammaS': 0.01,        # Source coupling (eV)
            'GammaT': 0.01,        # Tip coupling (eV)
            'Temp': 0.224,         # Temperature (eV)
            'L': 20.0,             # Canvas size (Å)
            'npix': 100            # Number of pixels
    }
    
    # Define the optimization parameters and their ranges
    # Focus on parameters that significantly affect the simulation
    param_ranges = {
        'Esite':  ( -0.2, 0.0 ),      # Site energy
        'z_tip':  (  3.0, 7.0 ),      # Tip height
        'VBias':  (  0.5, 1.0 ),      # Bias voltage
        'Rtip':   (  2.0, 4.0 ),      # Tip radius
        'radius': (  4.0, 6.0 ),      # QD radius
        'phiRot': (  0.8, 1.8 ),      # Rotation angle
        'zV0':    ( -2.0, 0.0 ),      # Potential offset
        'decay':  (  0.1, 0.5 )       # Decay parameter
    }

    # Define experimental data line points (from GUI parameters)
    exp_start_point = (9.72, -6.96)   # From ep1_x, ep1_y
    exp_end_point = (-11.0, 15.0)     # From ep2_x, ep2_y
    
    # Define simulation line points (from GUI parameters)
    sim_start_point = (9.72, -9.96)   # From p1_x, p1_y
    sim_end_point = (-11.0, 12.0)     # From p2_x, p2_y
    
    # Load experimental data
    exp_X, exp_Y, exp_dIdV, exp_I, exp_biases = load_experimental_data()
    if exp_X is None:
        print("Cannot proceed without experimental data.")
        return
    
    # Extract experimental data along the specified line
    exp_STM, exp_dIdV_line, exp_dist, exp_biases = extract_experimental_data_along_line(
        exp_X, exp_Y, exp_I, exp_dIdV, exp_biases, 
        exp_start_point, exp_end_point
    )
    
    # Create dummy x_positions array - not needed since we use the exp_dist instead
    # The optimizer uses the x positions for interpolation
    x_positions = np.linspace(0, exp_dist[-1], len(exp_dist))
    
    # Slightly modify parameters to simulate difference between experiment and initial simulation
    modified_params = params.copy()
    modified_params['Esite'] = params['Esite'] * 1.2
    modified_params['z_tip'] = params['z_tip'] * 0.9
    
    print("\nOptimization parameters:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"{param}: [{min_val:.3f}, {max_val:.3f}]")
    
    # Create figure to visualize the experimental data
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot extracted experimental STM data
    im0 = axs[0].imshow(exp_STM, aspect='auto', origin='lower', cmap='hot',  extent=[0, exp_dist[-1], exp_biases[0], exp_biases[-1]])
    axs[0].set_title('Experimental STM')
    axs[0].set_xlabel('Distance (Å)')
    axs[0].set_ylabel('Voltage (V)')
    plt.colorbar(im0, ax=axs[0])
    
    # Plot extracted experimental dI/dV data
    im1 = axs[1].imshow(exp_dIdV_line, aspect='auto', origin='lower', cmap='bwr',  extent=[0, exp_dist[-1], exp_biases[0], exp_biases[-1]])
    axs[1].set_title('Experimental dI/dV')
    axs[1].set_xlabel('Distance (Å)')
    axs[1].set_ylabel('Voltage (V)')
    plt.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig('experimental_data_extract.png', dpi=150)
    
    # Initialize the optimizer with extracted experimental data
    print("\nInitializing Monte Carlo optimizer...")
    optimizer = MonteCarloOptimizer(
        initial_params=modified_params,
        exp_data=exp_STM,  # Using STM data for optimization
        exp_voltages=exp_biases,
        exp_x=x_positions,
        param_ranges=param_ranges,
        start_point=sim_start_point,  # Using simulation line start/end points
        end_point=sim_end_point,
        nx=50,  # Number of x points
        nV=len(exp_biases)  # Match the number of voltage points from experimental data
    )
    
    # Run optimization
    print("\nRunning optimization...")
    t_start = time.time()
    best_params = optimizer.optimize(
        num_iterations=1000,         # Use 100+ for real optimization
        mutation_strength=0.1,      # Relative parameter change size
        temperature=0.01,           # Initial temperature (enables hill climbing)
        temperature_decay=0.95,     # Temperature decay rate
        early_stop_iterations=20    # Stop if no improvement after this many iterations
    )
    t_end = time.time()
    
    # Print timing and results
    print(f"\nOptimization completed in {t_end - t_start:.2f} seconds")
    print(f"Best distance: {optimizer.best_distance:.6f}")
    
    # Print parameter changes
    print("\nParameter changes:")
    for param in param_ranges.keys():
        initial = modified_params[param]
        optimized = best_params[param]
        diff = optimized - initial
        diff_percent = (diff / initial) * 100 if initial != 0 else float('inf')
        print(f"{param}: {initial:.6f} -> {optimized:.6f} (Δ = {diff:.6f}, {diff_percent:.2f}%)")
    
    # Create comparison plot
    print("\nGenerating comparison plots...")
    comparison_fig = optimizer.plot_comparison()
    progress_fig = optimizer.plot_optimization_progress()
    
    # Save results
    print("\nSaving results...")
    optimizer.save_results("fit_sim_exp_results")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
