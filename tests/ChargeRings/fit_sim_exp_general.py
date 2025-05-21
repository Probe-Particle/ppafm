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
from MonteCarloOptimizer import MonteCarloOptimizer
from scipy.interpolate import RectBivariateSpline
from fitting_plots import plot_optimization_progress, plot_parameter_correlations

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

# Define the callbacks for the General Monte Carlo Optimizer

def create_simulation_callback(sim_start_point, sim_end_point, nx=50, nV=None, orbital_2D=None, orbital_lvec=None):
    """
    Create a callback function for running simulations
    
    Args:
        sim_start_point (tuple): Start point for simulation line
        sim_end_point (tuple): End point for simulation line
        nx (int): Number of points along the x axis
        nV (int, optional): Number of voltage points
        orbital_2D, orbital_lvec: Optional orbital data
        
    Returns:
        callable: Simulation callback function
    """
    pauli_solver = pauli.PauliSolver(nSingle=3, nleads=2, verbosity=0)
    
    def run_simulation(params):
        """
        Run a simulation with the given parameters
        
        Args:
            params (dict): Simulation parameters
            
        Returns:
            tuple: (STM, dIdV, voltages, x) - Simulation results
        """
        # Get voltage range from exp_biases if available from the outer scope
        # This ensures the simulation voltage range matches the experimental range
        try:
            # Try to access exp_biases from outer scope
            Vmin = exp_biases[0]  # Min voltage from experimental data
            Vmax = exp_biases[-1]  # Max voltage from experimental data
            print(f"Using experimental voltage range: [{Vmin:.3f}V, {Vmax:.3f}V]")
        except (NameError, IndexError):
            # Fallback to parameter-based range if experimental data not available
            Vmin = 0.0
            Vmax = params.get('VBias', 1.0)
            print(f"Using parameter-based voltage range: [{Vmin:.3f}V, {Vmax:.3f}V]")
        
        # Run the simulation
        STM, dIdV, Es, Ts, probs, x, voltages, spos, rots = pauli_scan.calculate_xV_scan_orb(
            params, 
            sim_start_point, 
            sim_end_point,
            orbital_2D=orbital_2D,
            orbital_lvec=orbital_lvec,
            pauli_solver=pauli_solver,
            nx=nx, 
            nV=nV,
            Vmin=Vmin,
            Vmax=Vmax,
            bLegend=False  # No need for legends in optimization
        )
        
        return STM, dIdV, voltages, x
    
    return run_simulation

def create_distance_callback(exp_data, exp_voltages, exp_x):
    """
    Create a callback function for calculating distance between simulation and experiment
    
    Args:
        exp_data (np.ndarray): Experimental data
        exp_voltages (np.ndarray): Experimental voltage values
        exp_x (np.ndarray): Experimental x positions
        
    Returns:
        tuple: (distance_callback, interpolate_function) - Functions for calculating distance and interpolating
    """
    from wasserstein_distance import wasserstein_1d_grid
    
    def wasserstein_2d_xV(img1, img2, dx=1.0, dy=1.0):
        """
        Calculate 2D Wasserstein distance between two images along V direction.
        
        Args:
            img1, img2: Images to compare (shape [nV, nx])
            dx, dy: Grid spacing
            
        Returns:
            float: Distance metric
        """
        # Make sure the images have the same shape
        if img1.shape != img2.shape:
            raise ValueError(f"Images must have the same shape: {img1.shape} vs {img2.shape}")
        
        # Calculate 1D Wasserstein for each column (V direction/y-axis)
        distances = []
        for i in range(img1.shape[1]):  # Loop over x positions
            col1 = img1[:, i]  # Get column from first image (voltage profile)
            col2 = img2[:, i]  # Get column from second image (voltage profile)
            dist = wasserstein_1d_grid(col1, col2, dy)
            distances.append(dist)
        
        # Return average distance across all columns
        return np.mean(distances)
    
    def interpolate_simulation_to_exp_grid(sim_data, sim_voltages, sim_x):
        """
        Interpolate simulation data to match experimental grid.
        
        Args:
            sim_data (np.ndarray): Simulation data
            sim_voltages (np.ndarray): Voltage values for simulation
            sim_x (np.ndarray): X positions for simulation
            
        Returns:
            np.ndarray: Interpolated simulation data matching experimental grid
        """
        # Sort the simulation data to ensure coordinates are strictly increasing
        # (required by RectBivariateSpline when grid=True)
        volt_idx = np.argsort(sim_voltages)
        sorted_voltages = sim_voltages[volt_idx]
        sorted_sim_data = sim_data[volt_idx, :]
        
        x_idx = np.argsort(sim_x)
        sorted_x = sim_x[x_idx]
        sorted_sim_data = sorted_sim_data[:, x_idx]
        
        # Create interpolation function with sorted data
        # Important: Note the order here - in RectBivariateSpline(y, x, z), the y coordinate is first
        # In our case, voltage is the y-coordinate (vertical axis in typical plots)
        interp = RectBivariateSpline(sorted_voltages, sorted_x, sorted_sim_data)
        
        # Create grid of points to evaluate at
        vv, xx = np.meshgrid(exp_voltages, exp_x, indexing='ij')
        
        # Evaluate at each point (doesn't require strictly increasing coordinates)
        # Must use ev(y, x) to match the exp_utils.interpolate_3d_plane_fast convention
        interpolated = interp.ev(vv, xx)
        
        return interpolated

    def calculate_distance(sim_results):
        """
        Calculate the distance between simulation results and experimental data
        
        Args:
            sim_results (tuple): (STM, dIdV, voltages, x) from simulation
            
        Returns:
            float: Distance metric (lower is better)
        """
        STM, dIdV, voltages, x = sim_results
        
        # Use STM current data for comparison
        sim_data = STM
        
        # Interpolate simulation data to match experimental grid
        sim_data_interp = interpolate_simulation_to_exp_grid(sim_data, voltages, x)
        
        # Calculate 2D Wasserstein distance
        # Use the spacing from the experimental data for consistency
        dv = np.abs(exp_voltages[1] - exp_voltages[0]) if len(exp_voltages) > 1 else 1.0
        dx = np.abs(exp_x[1] - exp_x[0]) if len(exp_x) > 1 else 1.0
        
        distance = wasserstein_2d_xV(sim_data_interp, exp_data, dx=dx, dy=dv)
        
        return distance
    
    # Return both the distance calculation function and the interpolation function
    # so they can be accessed separately
    return calculate_distance, interpolate_simulation_to_exp_grid

def create_comparison_plot(optimizer, exp_data, exp_voltages, exp_x, figsize=(15, 10)):
    """
    Create a comparison plot between experimental and optimized simulation data
    
    Args:
        optimizer (MonteCarloOptimizer): The optimizer instance
        exp_data (np.ndarray): Experimental data
        exp_voltages (np.ndarray): Experimental voltage values
        exp_x (np.ndarray): Experimental x positions
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Unpack best simulation results
    STM, dIdV, voltages, x = optimizer.best_sim_results
    
    # Calculate distance between points for extent
    x1, y1 = sim_start_point
    x2, y2 = sim_end_point
    dist = np.hypot(x2-x1, y2-y1)
    
    # Create extent for both plots
    sim_extent = [0, dist, min(voltages), max(voltages)]
    exp_extent = [0, max(exp_x), min(exp_voltages), max(exp_voltages)]
    
    # Plot experimental data - Use hot colormap for STM current data
    im0 = axs[0, 0].imshow(exp_data, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
    axs[0, 0].set_title('Experimental STM')
    axs[0, 0].set_xlabel('Distance (Å)')
    axs[0, 0].set_ylabel('Voltage (V)')
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Plot best simulation - Use hot colormap for STM current data
    im1 = axs[0, 1].imshow(STM, extent=sim_extent, aspect='auto', origin='lower', cmap='hot')
    axs[0, 1].set_title('Best Simulation STM')
    axs[0, 1].set_xlabel('Distance (Å)')
    axs[0, 1].set_ylabel('Voltage (V)')
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Access the interpolation function stored in the optimizer object
    interpolate_func = optimizer.interpolate_func
    
    # Create helper function to use interpolation function
    def get_interpolated_sim(sim_results):
        STM, dIdV, voltages, x = sim_results
        return interpolate_func(STM, voltages, x)
    
    # Interpolate simulation to match experimental grid
    interp_sim = get_interpolated_sim(optimizer.best_sim_results)
        
    # Plot difference
    diff = interp_sim - exp_data
    im2 = axs[1, 0].imshow(diff, extent=exp_extent, aspect='auto', origin='lower', cmap='bwr')
    axs[1, 0].set_title('Difference (Sim - Exp)')
    axs[1, 0].set_xlabel('Distance (Å)')
    axs[1, 0].set_ylabel('Voltage (V)')
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Plot optimization progress
    axs[1, 1].plot(range(1, len(optimizer.distance_history) + 1), optimizer.distance_history, 'b-')
    axs[1, 1].set_title('Optimization Progress')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Distance (lower is better)')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

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
    
    # Load experimental data
    exp_X, exp_Y, exp_dIdV, exp_I, exp_biases = load_experimental_data()
    if exp_X is None:
        print("Cannot proceed without experimental data.")
        return
    
    # Extract experimental data along the specified line
    exp_STM, exp_dIdV_line, exp_dist, exp_biases = extract_experimental_data_along_line( exp_X, exp_Y, exp_I, exp_dIdV, exp_biases, exp_start_point, exp_end_point )
    
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
    im0 = axs[0].imshow(exp_STM, aspect='auto', origin='lower', cmap='hot', extent=[0, exp_dist[-1], exp_biases[0], exp_biases[-1]])
    axs[0].set_title('Experimental STM')
    axs[0].set_xlabel('Distance (Å)')
    axs[0].set_ylabel('Voltage (V)')
    plt.colorbar(im0, ax=axs[0])
    
    # Plot extracted experimental dI/dV data
    im1 = axs[1].imshow(exp_dIdV_line, aspect='auto', origin='lower', cmap='bwr', extent=[0, exp_dist[-1], exp_biases[0], exp_biases[-1]])
    axs[1].set_title('Experimental dI/dV')
    axs[1].set_xlabel('Distance (Å)')
    axs[1].set_ylabel('Voltage (V)')
    plt.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig('experimental_data_extract.png', dpi=150)
    
    # Create the callback functions
    simulation_cb = create_simulation_callback( sim_start_point=sim_start_point, sim_end_point=sim_end_point, nx=50, nV=len(exp_biases) )
    
    # Get both distance calculation and interpolation functions
    distance_cb, interpolate_func = create_distance_callback( exp_data=exp_STM, exp_voltages=exp_biases, exp_x=x_positions)
    
    # Initialize the general Monte Carlo optimizer
    print("\nInitializing Monte Carlo optimizer...")
    optimizer = MonteCarloOptimizer( initial_params=modified_params, param_ranges=param_ranges, simulation_callback=simulation_cb, distance_callback=distance_cb )
    
    # Store the interpolation function for later use
    optimizer.interpolate_func = interpolate_func
    
    # Run optimization
    print("\nRunning optimization...")
    t_start = time.time()
    best_params = optimizer.optimize(
        num_iterations=1000,        # Number of iterations
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
    comparison_fig = create_comparison_plot(
        optimizer=optimizer,
        exp_data=exp_STM,
        exp_voltages=exp_biases,
        exp_x=x_positions
    )
    progress_fig = plot_optimization_progress(optimizer)
    param_corr_fig = plot_parameter_correlations(optimizer)
    
    # Run high-resolution simulation with optimized parameters
    print("\nRunning high-resolution simulation with optimized parameters...")
    nx_highres = 400  # 8x the original resolution along x-axis
    nV_highres = 200  # High resolution along voltage axis as well
    
    # Create a high-resolution simulation callback with explicit voltage range
    # Define a custom function that captures the voltage range from experimental data
    def create_highres_simulation_callback():
        # Force the simulation to use the experimental voltage range
        V_min = min(exp_biases)
        V_max = max(exp_biases)
        print(f"Using experimental voltage range for high-res: [{V_min:.3f}V, {V_max:.3f}V]")
        
        # Use the original callback function but explicitly pass voltage range
        def run_highres_simulation(params):
            # Run the simulation with experimental voltage range
            STM, dIdV, Es, Ts, probs, x, voltages, spos, rots = pauli_scan.calculate_xV_scan_orb(
                params, 
                sim_start_point, 
                sim_end_point,
                nx=nx_highres, 
                nV=nV_highres,
                Vmin=V_min,
                Vmax=V_max,
                bLegend=False
            )
            return STM, dIdV, voltages, x
            
        return run_highres_simulation
    
    # Use our custom callback that enforces the experimental voltage range
    highres_sim_cb = create_highres_simulation_callback()
    
    # Run the high-resolution simulation
    highres_results = highres_sim_cb(best_params)
    STM_highres, dIdV_highres, voltages_highres, x_highres = highres_results
    
    # Create high-resolution comparison plot
    fig_highres, axs_highres = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate distance for extent
    x1, y1 = sim_start_point
    x2, y2 = sim_end_point
    dist = np.hypot(x2-x1, y2-y1)
    
    # Create extents for plots
    sim_extent = [0, dist, min(voltages_highres), max(voltages_highres)]
    exp_extent = [0, max(x_positions), min(exp_biases), max(exp_biases)]
    
    # Plot experimental data
    im0 = axs_highres[0, 0].imshow(exp_STM, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
    axs_highres[0, 0].set_title('Experimental STM', fontsize=14)
    axs_highres[0, 0].set_xlabel('Distance (Å)', fontsize=12)
    axs_highres[0, 0].set_ylabel('Voltage (V)', fontsize=12)
    plt.colorbar(im0, ax=axs_highres[0, 0])
    
    # Plot high-resolution simulation
    im1 = axs_highres[0, 1].imshow(STM_highres, extent=sim_extent, aspect='auto', origin='lower', cmap='hot')
    axs_highres[0, 1].set_title('High-Resolution Simulation STM (4x)', fontsize=14)
    axs_highres[0, 1].set_xlabel('Distance (Å)', fontsize=12)
    axs_highres[0, 1].set_ylabel('Voltage (V)', fontsize=12)
    plt.colorbar(im1, ax=axs_highres[0, 1])
    
    # Use the interpolation function stored in the optimizer object
    # This avoids trying to access it through closure which was causing problems
    
    # Create helper function to use the interpolation function
    def get_interpolated_sim_highres(sim_results):
        STM, dIdV, voltages, x = sim_results
        return optimizer.interpolate_func(STM, voltages, x)
    
    # Interpolate high-resolution simulation to match experimental grid
    interp_sim_highres = get_interpolated_sim_highres(highres_results)
    
    # Plot difference
    diff_highres = interp_sim_highres - exp_STM
    im2 = axs_highres[1, 0].imshow(diff_highres, extent=exp_extent, aspect='auto', origin='lower', cmap='bwr')
    axs_highres[1, 0].set_title('Difference (Sim - Exp)', fontsize=14)
    axs_highres[1, 0].set_xlabel('Distance (Å)', fontsize=12)
    axs_highres[1, 0].set_ylabel('Voltage (V)', fontsize=12)
    plt.colorbar(im2, ax=axs_highres[1, 0])
    
    # Plot linecuts at different voltages
    voltage_indices = [int(len(voltages_highres) * p) for p in [0.25, 0.5, 0.75]]
    for i, v_idx in enumerate(voltage_indices):
        v_val = voltages_highres[v_idx]
        axs_highres[1, 1].plot(x_highres, STM_highres[v_idx, :], label=f'V = {v_val:.2f}V')
    
    axs_highres[1, 1].set_title('STM Linecuts at Different Voltages', fontsize=14)
    axs_highres[1, 1].set_xlabel('Distance (Å)', fontsize=12)
    axs_highres[1, 1].set_ylabel('STM Current (arb. units)', fontsize=12)
    axs_highres[1, 1].legend()
    axs_highres[1, 1].grid(True)
    
    plt.tight_layout()
    highres_file = "fit_sim_exp_general_results_highres.png"
    fig_highres.savefig(highres_file, dpi=200)
    
    # Save results
    print("\nSaving results...")
    params_file = "fit_sim_exp_general_results_params.json"
    progress_file = "fit_sim_exp_general_results_progress.png"
    comparison_file = "fit_sim_exp_general_results_comparison.png"
    
    # Save parameter file
    import json
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Save figures
    progress_fig.savefig(progress_file, dpi=150)
    comparison_fig.savefig(comparison_file, dpi=150)
    
    print(f"Results saved to {params_file}, {progress_file}, {comparison_file}, {highres_file}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
