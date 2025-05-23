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
from exp_utils import load_and_extract_experimental_data, visualize_experimental_data
from MonteCarloOptimizer import MonteCarloOptimizer
from fitting_plots import plot_optimization_progress,  plot_comparison
from wasserstein_distance import  wasserstein_2d_grid

# Global verbosity control
verbosity = 1  # 0=quiet, 1=normal, 2=verbose

# Global plotting and saving controls
PLOT_FIGURES = True
SAVE_FIGURES = True
VIEW_FIGURES = True
PLOT_HIGHRES = True
PLOT_PROGRESS = True

# Initialize PauliSolver once
pauli_solver = pauli.PauliSolver(nSingle=3, nleads=2, verbosity=0)

def create_simulation_callback(sim_start_point, sim_end_point, nx, nV, Vmin, Vmax, orbital_2D=None, orbital_lvec=None):
    def run_simulation(params):
        STM, dIdV, Es, Ts, probs, x, voltages, spos, rots = pauli_scan.calculate_xV_scan_orb(
            params, sim_start_point, sim_end_point,
            orbital_2D=orbital_2D, orbital_lvec=orbital_lvec,
            nx=nx, nV=nV, Vmin=Vmin, Vmax=Vmax,
            pauli_solver=pauli_solver
        )
        return STM, dIdV, x, voltages
    return run_simulation

def create_distance_callback(exp_data, exp_voltages, exp_x):
    def calculate_distance(sim_results):
        STM, dIdV, x, voltages = sim_results
        if verbosity > 1:
            print(f"calculate_distance(): exp_data.shape: {exp_data.shape}")
            print(f"  exp_voltages: [{exp_voltages.min():.3f}, {exp_voltages.max():.3f}]")
            print(f"  exp_x: [{exp_x.min():.3f}, {exp_x.max():.3f}]")
            print(f"  sim_data.shape: {STM.shape}")
            print(f"  voltages: [{voltages.min():.3f}, {voltages.max():.3f}]")
            print(f"  x: [{x.min():.3f}, {x.max():.3f}]")
        # Calculate grid spacing
        dy = np.abs(exp_voltages[1] - exp_voltages[0]) if len(exp_voltages) > 1 else 1.0
        dx = np.abs(exp_x[1] - exp_x[0]) if len(exp_x) > 1 else 1.0
        distance = wasserstein_2d_grid(STM, exp_data, dx=dx, dy=dy)
        if verbosity > 0:
            print(f"  Wasserstein distance: {distance:.6f}")
        return distance
    return calculate_distance

def save_results_callback(optimizer, run_dir):

    if PLOT_FIGURES:
        best_params     = optimizer.best_params
        params          = optimizer.initial_params # Assuming initial_params is accessible from optimizer
        param_ranges    = optimizer.param_ranges # Assuming param_ranges is accessible from optimizer
        exp_STM         = globals().get('exp_STM') # Accessing global variables for plotting
        exp_dIdV        = globals().get('exp_dIdV')
        exp_dist        = globals().get('exp_dist')
        exp_biases      = globals().get('exp_biases')
        sim_start_point = globals().get('sim_start_point')
        sim_end_point   = globals().get('sim_end_point')

        # Print timing and results
        # t_start and t_end are not available here, so we skip printing optimization time
        print(f"\nOptimization completed.")
        print(f"Best distance: {optimizer.best_distance:.6f}")
        
        # Print parameter changes
        print("\nParameter changes:")
        for param in param_ranges.keys():
            initial = params[param]
            optimized = best_params[param]
            diff = optimized - initial
            diff_percent = (diff / initial) * 100 if initial != 0 else float('inf')
            print(f"{param}: {initial:.6f} -> {optimized:.6f} (Δ = {diff:.6f}, {diff_percent:.2f}%) ")
        
        # Calculate consistent voltage ranges
        STM, dIdV, x, voltages = optimizer.best_sim_results
        vmin = min(np.min(exp_biases), np.min(voltages))
        vmax = max(np.max(exp_biases), np.max(voltages))
        sim_distance = np.hypot(sim_end_point[0]-sim_start_point[0], sim_end_point[1]-sim_start_point[1])

        # Calculate extents
        exp_extent = [exp_dist[0], exp_dist[-1], exp_biases[0], exp_biases[-1]]
        sim_extent = [x[0], x[-1], voltages[0], voltages[-1]]
        
        # Ensure 2D arrays for plotting
        if optimizer.best_sim_results[1].ndim == 3:
            sim_dIdV = optimizer.best_sim_results[1][0]
        else:
            sim_dIdV = optimizer.best_sim_results[1]

        # Save comparison figure

        fig_comp = plot_comparison( exp_STM=exp_STM, exp_dIdV=exp_dIdV, sim_STM=optimizer.best_sim_results[0], sim_dIdV=sim_dIdV,  exp_extent=exp_extent, sim_extent=sim_extent, ylim=[voltages[0], voltages[-1]] )
        
        if PLOT_PROGRESS:
            fig_prog = plot_optimization_progress(optimizer)
        
        if PLOT_HIGHRES:
            # Run high-resolution simulation with optimized parameters
            print("\n Figure 3: High-resolution simulation with optimized parameters")
            highres_sim_cb  = create_simulation_callback(sim_start_point, sim_end_point, nx=200, nV=100, Vmin=min(exp_biases), Vmax=max(exp_biases))
            highres_results = highres_sim_cb(best_params)
            STM_highres, dIdV_highres, x_highres, voltages_highres = highres_results
            sim_extent_highres  = [x_highres[0], x_highres[-1], voltages_highres[0], voltages_highres[-1]]
            highres_fig = plot_comparison( exp_STM=exp_STM, exp_dIdV=exp_dIdV, exp_extent=exp_extent, sim_STM=STM_highres, sim_dIdV=dIdV_highres, sim_extent=sim_extent_highres,  ylim=[voltages_highres[0], voltages_highres[-1]], scale_dIdV=3.0 )

        if SAVE_FIGURES:
            fig_comp.savefig(run_dir/"comparison.png", dpi=300)
            fig_prog.savefig(run_dir/"progress.png", dpi=300)
            highres_fig.savefig(run_dir/"comparison_hires.png", dpi=300)

        if VIEW_FIGURES:
            plt.show()
        else:
            plt.close('all')


if __name__ == "__main__":
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
        'Esite':  ( -0.15,0.05 ),      # Site energy
        'z_tip':  (  3.0, 70 ),      # Tip height
        'radius': (  2.0, 4.0 ),      # QD radius
        'zV0':    ( -4.0, 0.5 ),      # Potential offset
        'zVd':    (  5.0, 20.0 ),     # Potential decay length
    }

    # Define experimental data line points (from GUI parameters)
    exp_start_point = (9.72, -6.96)  # From ep1_x, ep1_y
    exp_end_point   = (-11.0, 15.0)  # From ep2_x, ep2_y
    
    # Define simulation line points (from GUI parameters)
    global sim_start_point, sim_end_point  # Make these accessible to plotting function
    sim_start_point = (9.72, -9.96)  # From p1_x, p1_y
    sim_end_point   = (-11.0, 12.0)  # From p2_x, p2_y
    
    # Load and extract experimental data
    exp_STM, exp_dIdV, exp_dist, exp_biases = load_and_extract_experimental_data(start_point=exp_start_point, end_point=exp_end_point)
            
    print("\nOptimization parameters:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"{param}: [{min_val:.3f}, {max_val:.3f}]")
    
    # Create figure to visualize the experimental data
    exp_fig = visualize_experimental_data(exp_STM, exp_dIdV, exp_dist, exp_biases)
    #plt.show()
    
    # Create the callback functions
    exp_x_coords_for_fitting = exp_dist
    simulation_cb = create_simulation_callback( sim_start_point=sim_start_point, sim_end_point=sim_end_point, nx=len(exp_dist), nV=len(exp_biases), Vmin=min(exp_biases), Vmax=max(exp_biases) )
    distance_cb   = create_distance_callback( exp_data=exp_STM, exp_voltages=exp_biases, exp_x=exp_x_coords_for_fitting )
    
    # Initialize the general Monte Carlo optimizer
    print("\nInitializing Monte Carlo optimizer...")
    optimizer = MonteCarloOptimizer( initial_params=params, param_ranges=param_ranges, simulation_callback=simulation_cb, distance_callback=distance_cb, result_dir="fitting_results" )
    optimizer.metadata.update({"exp_points":[exp_start_point, exp_end_point], "sim_points":[sim_start_point, sim_end_point]})

    # Run optimization
    print("\nRunning optimization...")
    t_start = time.time()
    best_params = optimizer.optimize( num_iterations=1000, mutation_strength=0.1, temperature=0.01, temperature_decay=0.95, early_stop_iterations=20, save_callback=save_results_callback )
    t_end = time.time()
    
    # The rest of the script is now handled by the save_results_callback
    # No need for explicit plotting or saving here.
    
    print(f"\nOptimization finished in {t_end - t_start:.2f} seconds.")
    print("Results saved to: fitting_results/")
