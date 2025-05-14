#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os

# Import our fitting framework
from PauliFitter import PauliFitter, load_experimental_data, extract_experimental_data_along_line
from distance_metrics import AVAILABLE_METRICS

def plot_optimization_progress(fitter, filename=None):
    """
    Plot the optimization progress over iterations
    
    Parameters
    ----------
    fitter : PauliFitter
        The PauliFitter instance with optimization history
    filename : str
        If provided, save the plot to this file
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot distance vs iteration
    ax1.plot(fitter.history['iterations'], fitter.history['distances'], 'b-')
    ax1.set_yscale('log')
    ax1.set_ylabel('Distance')
    ax1.set_title(f'Optimization Progress - {fitter.metric_name}')
    ax1.grid(True)
    
    # Plot parameter values vs iteration
    param_arrays = {}
    for i, name in enumerate(fitter.param_names):
        param_arrays[name] = np.array([p[i] for p in fitter.history['parameters']])
    
    for i, (name, values) in enumerate(param_arrays.items()):
        # Normalize to [0, 1] range for better visualization
        min_val, max_val = fitter.param_ranges[name]
        norm_values = (values - min_val) / (max_val - min_val)
        ax2.plot(fitter.history['iterations'], norm_values, label=name)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Normalized Parameter Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=150)
    
    return fig


def plot_high_resolution_comparison(exp_data, exp_voltages, exp_x, sim_data, sim_voltages, sim_x, filename=None):
    """
    Plot a comparison between experimental data and high-resolution simulation
    
    Parameters
    ----------
    exp_data : numpy array
        Experimental data array
    exp_voltages, exp_x : numpy arrays
        1D arrays with experimental voltages and x positions
    sim_data : numpy array
        Simulation data array
    sim_voltages, sim_x : numpy arrays
        1D arrays with simulation voltages and x positions
    filename : str
        If provided, save the plot to this file
    """
    # Create extents for the plots
    exp_extent = [0, exp_x[-1], exp_voltages[0], exp_voltages[-1]]
    sim_extent = [0, sim_x[-1], sim_voltages[0], sim_voltages[-1]]
    
    # Create figure with 3 panels in one row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot experimental data
    im1 = ax1.imshow(exp_data, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
    ax1.set_title('Experimental Data')
    ax1.set_xlabel('Distance (Å)')
    ax1.set_ylabel('Voltage (V)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot high-resolution simulation
    im2 = ax2.imshow(sim_data, extent=sim_extent, aspect='auto', origin='lower', cmap='hot')
    ax2.set_title('High-Resolution Simulation')
    ax2.set_xlabel('Distance (Å)')
    ax2.set_ylabel('Voltage (V)')
    plt.colorbar(im2, ax=ax2)
    
    # Calculate and plot difference (Sim - Exp)
    # First interpolate simulation to experimental grid
    from scipy.interpolate import RectBivariateSpline
    
    # Ensure coordinates are strictly increasing
    sort_idx = np.argsort(sim_x)
    sorted_sim_x = sim_x[sort_idx]
    sorted_sim_data = sim_data[:, sort_idx]
    
    interp = RectBivariateSpline(sim_voltages, sorted_sim_x, sorted_sim_data)
    
    # Create meshgrid for interpolation points
    V_grid, X_grid = np.meshgrid(exp_voltages, exp_x, indexing='ij')
    interp_sim_data = interp.ev(V_grid, X_grid)
    
    # Normalize both datasets
    exp_norm = exp_data / np.mean(exp_data)
    sim_norm = interp_sim_data / np.mean(interp_sim_data)
    
    # Calculate difference
    diff = sim_norm - exp_norm
    
    im3 = ax3.imshow(diff, extent=exp_extent, aspect='auto', origin='lower', cmap='bwr')
    ax3.set_title('Difference (Sim - Exp)')
    ax3.set_xlabel('Distance (Å)')
    ax3.set_ylabel('Voltage (V)')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=150)
    
    return fig


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fit Pauli transport simulation to experimental data')
    parser.add_argument('--metric',       type=str, default='wasserstein_v', choices=list(AVAILABLE_METRICS.keys()), help='Distance metric to use for optimization')
    parser.add_argument('--iterations',   type=int, default=100, help='Number of optimization iterations')
    parser.add_argument('--switch-metric', action='store_true', help='Switch metrics after initial optimization')
    parser.add_argument('--switch-to',    type=str, default='wasserstein_combined', choices=list(AVAILABLE_METRICS.keys()), help='Metric to switch to after initial optimization')
    args = parser.parse_args()
    
    # Define simulation parameters
    params = {
        'nsite': 3,            # Number of sites
        'phi0': 0.0,           # Phase angle (radians)
        'phiRot': 1.3,         # Rotation angle (radians)
        'radius': 5.2,         # Ring radius (Å)
        'phi0_ax': 0.2,        # Axial phase angle (radians)
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
    param_ranges = {
        'Esite':  (-0.15, 0.05),      # Site energy
        'z_tip':  (3.0, 5.0),         # Tip height
        'radius': (4.0, 6.0),         # QD radius
        'zV0':    (-2.0, 0.0),        # Potential offset
        'zVd':    (5.0, 20.0),       # Potential decay length
    }

    # Define line points for experimental data and simulation
    exp_start_point = (9.72, -6.96)   # From GUI: ep1_x, ep1_y
    sim_start_point = (9.72, -9.96)   # From GUI: p1_x, p1_y

    exp_end_point   = (-11.0, 15.0)     # From GUI: ep2_x, ep2_y
    sim_end_point   = (-11.0, 12.0)     # From GUI: p2_x, p2_y
    
    # Load experimental data
    exp_X, exp_Y, exp_dIdV, exp_I, exp_biases = load_experimental_data()
    if exp_X is None:
        print("Cannot proceed without experimental data.")
        return
    
    # Extract experimental data along the specified line
    exp_STM, exp_dIdV_line, exp_dist, exp_biases = extract_experimental_data_along_line(
        exp_X, exp_Y, exp_I, exp_dIdV, exp_biases, exp_start_point, exp_end_point
    )
    
    # Create x_positions array for the experimental data
    x_positions = np.linspace(0, exp_dist[-1], len(exp_dist))
    
    # Slightly modify parameters to simulate difference between experiment and initial simulation
    # (for demonstration purposes)
    modified_params = params.copy()
    modified_params['Esite'] = params['Esite'] * 1.2
    modified_params['z_tip'] = params['z_tip'] * 0.9
    
    # Ensure required parameters are present for pauli_scan module
    if 'VBias' not in modified_params:
        max_bias = max(exp_biases)
        modified_params['VBias'] = max_bias
    
    print(f"\nUsing distance metric: {args.metric}")
    
    # Create the PauliFitter instance
    fitter = PauliFitter(
        initial_params = modified_params,
        param_ranges   = param_ranges,
        exp_data       = exp_STM,
        exp_voltages   = exp_biases,
        exp_x          = x_positions,
        sim_start_point= sim_start_point,
        sim_end_point=sim_end_point,
        nx=50,  # Lower resolution for optimization
        metric_name=args.metric
    )
    
    # Run optimization
    print(f"\nRunning optimization with {args.iterations} iterations...")
    t_start = time.time()
    best_params = fitter.optimize(
        num_iterations=args.iterations,
        mutation_strength=0.1,
        temperature=0.01,
        temperature_decay=0.95,
        early_stop_iterations=50
    )
    t_end = time.time()
    
    # Print optimization results
    print(f"\nOptimization completed in {t_end - t_start:.2f} seconds")
    print(f"Best distance ({args.metric}): {fitter.best_distance:.6f}")
    
    # Print parameter changes
    print("\nParameter changes:")
    for param in param_ranges.keys():
        initial = modified_params[param]
        optimized = best_params[param]
        diff = optimized - initial
        diff_percent = (diff / initial) * 100 if initial != 0 else float('inf')
        print(f"{param}: {initial:.6f} -> {optimized:.6f} (Δ = {diff:.6f}, {diff_percent:.2f}%)")
    
    # Switch metric if requested
    if args.switch_metric:
        print(f"\nSwitching metric from {args.metric} to {args.switch_to}")
        fitter.change_metric(args.switch_to)
        print(f"New best distance ({args.switch_to}): {fitter.best_distance:.6f}")
        
        # Optionally run additional optimization with new metric
        print(f"\nRunning additional optimization with {args.switch_to} metric...")
        t_start = time.time()
        fitter.optimize(
            num_iterations=args.iterations // 2,  # Fewer iterations for refinement
            mutation_strength=0.05,  # Smaller mutations for fine-tuning
            temperature=0.005,
            temperature_decay=0.95,
            early_stop_iterations=25
        )
        t_end = time.time()
        
        print(f"Refinement completed in {t_end - t_start:.2f} seconds")
        print(f"Best distance ({args.switch_to}): {fitter.best_distance:.6f}")
    
    # Generate comparison plots with the original optimization metric
    print("\nGenerating comparison plots...")
    
    # Run high-resolution simulation
    print("Running high-resolution simulation with 400x200 points...")
    base_filename = f"pauli_fit_results_{args.metric}"
    saved_files, highres_data = fitter.save_results(base_filename)
    
    # Plot optimization progress
    plot_optimization_progress(fitter, f"{base_filename}_progress.png")
    
    # Plot high-resolution comparison
    x_highres, voltages_highres, STM_highres, _ = highres_data
    plot_high_resolution_comparison(fitter.exp_data, fitter.exp_voltages, fitter.exp_x,   STM_highres, voltages_highres, x_highres,  "pauli_highres_comparison.png")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
