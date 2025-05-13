#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Add parent directory to path
sys.path.append('../../')
from pyProbeParticle import utils as ut
from pyProbeParticle import pauli

# Import local modules
import pauli_scan
from monte_carlo_optimizer import MonteCarloOptimizer

# Load experimental data (adjust paths as needed)
def load_experimental_data(filename):
    """Load experimental data from NPZ file"""
    data = np.load(filename)
    return data

# Extract data along a line from experimental image
def extract_line_data(exp_data, extent, start_point, end_point, n_points=100):
    """Extract data along a line from a 2D image"""
    # Unpack extent
    xmin, xmax, ymin, ymax = extent
    
    # Extract coordinates
    x1, y1 = start_point
    x2, y2 = end_point
    
    # Create line points
    t = np.linspace(0, 1, n_points)
    x = x1 + (x2 - x1) * t
    y = y1 + (y2 - y1) * t
    
    # Map to pixel coordinates
    nx, ny = exp_data.shape
    ix = ((x - xmin) / (xmax - xmin) * (nx - 1)).astype(int)
    iy = ((y - ymin) / (ymax - ymin) * (ny - 1)).astype(int)
    
    # Clip to valid range
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    
    # Extract data
    data = np.array([exp_data[iy[i], ix[i]] for i in range(n_points)])
    
    return data, x, y

# Define parameter ranges for optimization
def create_param_ranges(params, variation_fraction=0.2):
    """Create parameter ranges based on initial parameters and variation fraction"""
    param_ranges = {}
    for key, value in params.items():
        # Skip non-numeric parameters
        if not isinstance(value, (int, float)):
            continue
        
        # Skip parameters that should not be varied
        if key in ['npix', 'L', 'nx', 'ny']:
            continue
            
        # Create range
        value_range = abs(value) * variation_fraction
        param_min = value - value_range
        param_max = value + value_range
        param_ranges[key] = (param_min, param_max)
    
    return param_ranges

if __name__ == "__main__":
    # Load parameters from JSON file (or define directly)
    try:
        with open('default_params.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        # Default parameters if file not found
        params = {
            'nsite': 3,
            'radius': 5.2,
            'phiRot': 1.3,
            'phi0_ax': 0.2,
            'VBias': 0.7,
            'Rtip': 3.0,
            'z_tip': 5.0,
            'zV0': -1.0,
            'zVd': 15.0,
            'zQd': 0.0,
            'Q0': 1.0,
            'Qzz': 10.0,
            'Esite': -0.1,
            'W': 0.02,
            'decay': 0.3,
            'GammaS': 0.01,
            'GammaT': 0.01,
            'Temp': 0.224,
            'L': 20.0,
            'npix': 100
        }
    
    # Define the optimization parameters and their ranges
    # Focus on parameters that significantly affect the simulation
    param_ranges = {
        'Esite': (-0.2, 0.0),        # Site energy
        'z_tip': (3.0, 7.0),          # Tip height
        'VBias': (0.5, 1.0),          # Bias voltage
        'Rtip': (2.0, 4.0),           # Tip radius
        'radius': (4.0, 6.0),         # QD radius
        'phiRot': (0.8, 1.8),         # Rotation angle
        'zV0': (-2.0, 0.0),           # Potential offset
        'decay': (0.1, 0.5)           # Decay parameter
    }
    

    # Define experimental data line points (where to extract from experimental data)
    # These values come from the GUI's ep1_x, ep1_y, ep2_x, ep2_y parameters
    exp_start_point = (9.72, -6.96)   # (x, y) coordinates for line start in experimental data
    exp_end_point = (-11.0, 15.0)     # (x, y) coordinates for line end in experimental data
    
    # Define simulation line points (where to simulate along)
    # In the GUI, these would come from p1_x, p1_y, p2_x, p2_y parameters
    sim_start_point = (-5.0, -5.0)    # (x, y) coordinates for simulation line start
    sim_end_point = (5.0, 5.0)        # (x, y) coordinates for simulation line end
    
    # Try to load experimental data
    # For this example, we'll generate synthetic "experimental" data
    # In a real case, you would load actual experimental data using:
    # exp_X, exp_Y, exp_dIdV, exp_I, exp_biases = exp_utils.load_experimental_data()
    # and then extract data along exp_start_point to exp_end_point
    
    print("Generating synthetic experimental data...")
    original_results = pauli_scan.calculate_xV_scan_orb(
        params, exp_start_point, exp_end_point, nx=50, nV=30, Vmin=0.0, Vmax=1.0
    )
    
    # Extract data properly
    # The output of calculate_xV_scan_orb has the following structure:
    # STM, dIdV, Es, Ts, probs, x, Vbiases, spos, rots
    STM_exp = original_results[0]     # STM current (shape should be [nV, nx])
    dIdV_exp = original_results[1]    # dI/dV data (shape should be [nV, nx])
    Es = original_results[2]          # Energies
    x_exp = original_results[5]       # X positions (shape should be [nx])
    voltage_values = original_results[6]   # Voltage values (shape should be [nV])
    
    # Add some noise to create "experimental" data
    np.random.seed(42)  # For reproducibility
    noise_level = 0.2
    # Add noise to STM data instead of dIdV data
    STM_exp = STM_exp + np.random.normal(0, noise_level * np.std(STM_exp), STM_exp.shape)
    
    # Print shape information for debugging
    print(f"Experimental data shapes: STM={STM_exp.shape}, voltages={voltage_values.shape}, x={x_exp.shape}")
    
    # Ensure data has the correct dimensions for Wasserstein distance calculation
    if len(STM_exp.shape) != 2:
        print(f"Warning: Unexpected STM shape {STM_exp.shape}, reshaping...")
        # Try to reshape to [nV, nx]
        nV, nx = len(voltage_values), len(x_exp)
        STM_exp = STM_exp.reshape(nV, nx)
    
    print("Preparing data for optimization...")
    
    # Slightly modify parameters to simulate difference between experiment and initial simulation
    modified_params = params.copy()
    modified_params['Esite'] = params['Esite'] * 1.2
    modified_params['z_tip'] = params['z_tip'] * 0.9
    
    print("\nOptimization parameters:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"{param}: [{min_val:.3f}, {max_val:.3f}]")
    
    # Initialize the optimizer
    print("\nInitializing Monte Carlo optimizer...")
    optimizer = MonteCarloOptimizer(
        initial_params=modified_params,
        exp_data=STM_exp,  # Using STM data instead of dIdV
        exp_voltages=voltage_values,
        exp_x=x_exp,
        param_ranges=param_ranges,
        start_point=sim_start_point,  # Using simulation start point, not experimental
        end_point=sim_end_point,     # Using simulation end point, not experimental
        nx=50,  # Number of x points
        nV=30   # Number of voltage points
    )
    
    # Run optimization
    print("\nRunning optimization...")
    t_start = time.time()
    best_params = optimizer.optimize(
        num_iterations=20,          # Reduced for example, use 100+ for real optimization
        mutation_strength=0.1,      # Relative parameter change size
        temperature=0.01,           # Initial temperature (set to None to only accept improvements)
        temperature_decay=0.9,      # Temperature decay rate
        early_stop_iterations=10    # Stop if no improvement after this many iterations
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
    optimizer.save_results("monte_carlo_results")
    
    # Show plots
    plt.show()
