#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import copy
import json
from scipy.interpolate import RectBivariateSpline

# Import from ppafm modules
sys.path.append('../../')
from pyProbeParticle import utils as ut
from pyProbeParticle import pauli

# Import local modules
import pauli_scan
from exp_utils import plot_exp_voltage_line_scan, create_line_coordinates
from MonteCarloOptimizer import MonteCarloOptimizer
from distance_metrics import get_metric_function, AVAILABLE_METRICS

# =====================================================================
# Data Loading and Extraction Functions
# =====================================================================

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

# =====================================================================
# Simulation Callback Functions
# =====================================================================

def create_simulation_callback(sim_start_point, sim_end_point, nx=50, nV=None,  Vmin=None, Vmax=None, orbital_2D=None, orbital_lvec=None):
    """
    Create a callback function for running simulations
    
    Args:
        sim_start_point (tuple): Start point for simulation line
        sim_end_point (tuple): End point for simulation line
        nx (int): Number of points along the x axis
        nV (int, optional): Number of voltage points
        Vmin, Vmax (float, optional): Voltage range
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
        # Use provided voltage range if specified
        v_min = Vmin
        v_max = Vmax
        
        if v_min is None or v_max is None:
            # Try to access exp_biases from outer scope if not provided
            try:
                # Try to access exp_biases from outer scope
                v_min = exp_biases[0]  # Min voltage from experimental data
                v_max = exp_biases[-1]  # Max voltage from experimental data
                print(f"Using experimental voltage range: [{v_min:.3f}V, {v_max:.3f}V]")
            except (NameError, IndexError):
                # Fallback to parameter-based range if experimental data not available
                v_min = 0.0
                v_max = params.get('VBias', 1.0)
                print(f"Using parameter-based voltage range: [{v_min:.3f}V, {v_max:.3f}V]")
        
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
            Vmin=v_min,
            Vmax=v_max,
            bLegend=False  # No need for legends in optimization
        )
        
        return STM, dIdV, voltages, x
    
    return run_simulation

# =====================================================================
# Distance Callback Functions
# =====================================================================

def create_distance_callback(exp_data, exp_voltages, exp_x, metric_name='wasserstein_v'):
    """
    Create a callback function for calculating distance between simulation and experiment
    
    Args:
        exp_data (np.ndarray): Experimental data
        exp_voltages (np.ndarray): Experimental voltage values
        exp_x (np.ndarray): Experimental x positions
        metric_name (str): Name of the distance metric to use
        
    Returns:
        tuple: (distance_callback, interpolate_function) - Functions for calculating distance and interpolating
    """
    # Get the distance metric function
    distance_metric = get_metric_function(metric_name)
    
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
        # Check if shapes are compatible for interpolation
        if sim_data.shape[0] < 4 or sim_data.shape[1] < 4:
            raise ValueError(f"Simulation data shape {sim_data.shape} too small for interpolation")
            
        # Ensure voltages are strictly increasing
        if not np.all(np.diff(sim_voltages) > 0):
            # Sort voltages and corresponding data if needed
            sort_idx = np.argsort(sim_voltages)
            sim_voltages = sim_voltages[sort_idx]
            sim_data = sim_data[sort_idx, :]
            
        # Ensure x positions are strictly increasing
        if not np.all(np.diff(sim_x) > 0):
            sort_idx = np.argsort(sim_x)
            sim_x = sim_x[sort_idx]
            sim_data = sim_data[:, sort_idx]
            
        # Create interpolator
        interp_func = RectBivariateSpline(
            sim_voltages, sim_x, sim_data, 
            kx=min(3, len(sim_voltages)-1), 
            ky=min(3, len(sim_x)-1)
        )
        
        # Interpolate onto experimental grid
        # This creates a 2D array of shape (len(exp_voltages), len(exp_x))
        interp_data = interp_func(exp_voltages, exp_x)
        
        return interp_data
    
    def calculate_distance(sim_results):
        """
        Calculate the distance between simulation results and experimental data
        
        Args:
            sim_results (tuple): (STM, dIdV, voltages, x) from simulation
            
        Returns:
            float: Distance metric (lower is better)
        """
        # Unpack simulation results
        STM, dIdV, voltages, x = sim_results
        
        try:
            # Interpolate simulation data to match experimental grid
            interp_sim = interpolate_simulation_to_exp_grid(STM, voltages, x)
            
            # Calculate distance using the selected metric
            distance = distance_metric(interp_sim, exp_data)
            return distance
            
        except Exception as e:
            print(f"Error calculating distance: {e}")
            # Return a large distance to reject this simulation
            return float('inf')
    
    return calculate_distance, interpolate_simulation_to_exp_grid

# =====================================================================
# Visualization Functions
# =====================================================================

def create_comparison_plot(optimizer, exp_data, exp_voltages, exp_x, sim_start_point, sim_end_point, figsize=(15, 10)):
    """
    Create a comparison plot between experimental and optimized simulation data
    
    Args:
        optimizer (MonteCarloOptimizer): The optimizer instance
        exp_data (np.ndarray): Experimental data
        exp_voltages (np.ndarray): Experimental voltage values
        exp_x (np.ndarray): Experimental x positions
        sim_start_point (tuple): Simulation start point
        sim_end_point (tuple): Simulation end point
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Get simulation results with best parameters
    sim_results = optimizer.simulation_callback(optimizer.best_params)
    STM, dIdV, voltages, x = sim_results
    
    # Interpolate simulation to match experimental grid for comparison
    def get_interpolated_sim(sim_results):
        STM, dIdV, voltages, x = sim_results
        # We'll use the interpolate function that's part of the distance callback
        return optimizer.interpolate_func(STM, voltages, x)
    
    # Get interpolated simulation data
    interp_sim = get_interpolated_sim(sim_results)
    
    # Calculate distance for extent
    x1, y1 = sim_start_point
    x2, y2 = sim_end_point
    dist = np.hypot(x2-x1, y2-y1)
    
    # Create extents for plots
    sim_extent = [0, dist, min(voltages), max(voltages)]
    exp_extent = [0, max(exp_x), min(exp_voltages), max(exp_voltages)]
    
    # Plot experimental data
    im0 = axs[0, 0].imshow(exp_data, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
    axs[0, 0].set_title('Experimental STM')
    axs[0, 0].set_xlabel('Distance (Å)')
    axs[0, 0].set_ylabel('Voltage (V)')
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Plot raw simulation data
    im1 = axs[0, 1].imshow(STM, extent=sim_extent, aspect='auto', origin='lower', cmap='hot')
    axs[0, 1].set_title('Simulation STM (Best Parameters)')
    axs[0, 1].set_xlabel('Distance (Å)')
    axs[0, 1].set_ylabel('Voltage (V)')
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot interpolated simulation data
    im2 = axs[1, 0].imshow(interp_sim, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
    axs[1, 0].set_title('Interpolated Simulation STM')
    axs[1, 0].set_xlabel('Distance (Å)')
    axs[1, 0].set_ylabel('Voltage (V)')
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Plot difference
    diff = interp_sim - exp_data
    im3 = axs[1, 1].imshow(diff, extent=exp_extent, aspect='auto', origin='lower', cmap='bwr')
    axs[1, 1].set_title('Difference (Sim - Exp)')
    axs[1, 1].set_xlabel('Distance (Å)')
    axs[1, 1].set_ylabel('Voltage (V)')
    plt.colorbar(im3, ax=axs[1, 1])
    
    plt.tight_layout()
    return fig

# =====================================================================
# PauliFitter Class
# =====================================================================

class PauliFitter(MonteCarloOptimizer):
    """
    Specialized optimization class for fitting Pauli transport simulations to experimental data.
    
    This class inherits from MonteCarloOptimizer and provides a higher-level interface for
    setting up and running optimizations specifically for Pauli transport simulations.
    """
    
    def __init__(self, initial_params, param_ranges, 
                 exp_data, exp_voltages, exp_x,
                 sim_start_point, sim_end_point, 
                 nx=50, nV=None, Vmin=None, Vmax=None,
                 metric_name='wasserstein_v',
                 orbital_2D=None, orbital_lvec=None):
        """
        Initialize the Pauli fitter.
        
        Args:
            initial_params (dict): Initial simulation parameters
            param_ranges (dict): Dictionary mapping parameter names to (min, max) tuples
            exp_data (np.ndarray): Experimental data
            exp_voltages (np.ndarray): Experimental voltage values
            exp_x (np.ndarray): Experimental x positions
            sim_start_point (tuple): Start point for simulation line
            sim_end_point (tuple): End point for simulation line
            nx (int): Number of points along the x axis
            nV (int, optional): Number of voltage points
            Vmin, Vmax (float, optional): Voltage range
            metric_name (str): Name of the distance metric to use
            orbital_2D, orbital_lvec: Optional orbital data
        """
        # Store simulation parameters
        self.sim_start_point = sim_start_point
        self.sim_end_point = sim_end_point
        self.nx   = nx
        self.nV   = nV   if nV is not None else len(exp_voltages)
        self.Vmin = Vmin if Vmin is not None else min(exp_voltages)
        self.Vmax = Vmax if Vmax is not None else max(exp_voltages)
        self.orbital_2D = orbital_2D
        self.orbital_lvec = orbital_lvec
        
        # Store experimental data
        self.exp_data = exp_data
        self.exp_voltages = exp_voltages
        self.exp_x = exp_x
        
        # Store metric name
        self.metric_name = metric_name
        
        # Create simulation callback
        simulation_cb = create_simulation_callback(
            sim_start_point=sim_start_point,
            sim_end_point=sim_end_point,
            nx=nx,
            nV=self.nV,
            Vmin=self.Vmin,
            Vmax=self.Vmax,
            orbital_2D=orbital_2D,
            orbital_lvec=orbital_lvec
        )
        
        # Create distance callback
        distance_cb, interpolate_func = create_distance_callback(
            exp_data=exp_data,
            exp_voltages=exp_voltages,
            exp_x=exp_x,
            metric_name=metric_name
        )
        
        # Store interpolation function for later use
        self.interpolate_func = interpolate_func
        
        # Initialize parent class
        super().__init__(
            initial_params=initial_params,
            param_ranges=param_ranges,
            simulation_callback=simulation_cb,
            distance_callback=distance_cb
        )
    
    def change_metric(self, metric_name):
        """
        Change the distance metric used for optimization.
        
        Args:
            metric_name (str): Name of the new distance metric to use
            
        Returns:
            None
        """
        if metric_name not in AVAILABLE_METRICS:
            valid_metrics = list(AVAILABLE_METRICS.keys())
            raise ValueError(f"Unknown metric '{metric_name}'. Valid options are: {valid_metrics}")
        
        print(f"Changing distance metric from '{self.metric_name}' to '{metric_name}'")
        self.metric_name = metric_name
        
        # Create new distance callback with the new metric
        distance_cb, interpolate_func = create_distance_callback(
            exp_data=self.exp_data,
            exp_voltages=self.exp_voltages,
            exp_x=self.exp_x,
            metric_name=metric_name
        )
        
        # Update distance callback and interpolation function
        self.distance_callback = distance_cb
        self.interpolate_func = interpolate_func
        
        # Recalculate best distance
        self.best_sim_results = self.simulation_callback(self.best_params)
        self.best_distance = self.distance_callback(self.best_sim_results)
        print(f"New best distance with '{metric_name}' metric: {self.best_distance}")
    
    def high_resolution_simulation(self, nx_highres=400, nV_highres=200):
        """
        Run a high-resolution simulation with the best parameters.
        
        Args:
            nx_highres (int): Number of points along the x axis for high-res simulation
            nV_highres (int): Number of voltage points for high-res simulation
            
        Returns:
            tuple: (STM, dIdV, voltages, x) - High-resolution simulation results
        """
        print(f"Running high-resolution simulation with {nx_highres}x{nV_highres} points...")
        
        # Create a high-resolution simulation callback
        highres_sim_cb = create_simulation_callback(
            sim_start_point=self.sim_start_point,
            sim_end_point=self.sim_end_point,
            nx=nx_highres,
            nV=nV_highres,
            Vmin=self.Vmin,
            Vmax=self.Vmax,
            orbital_2D=self.orbital_2D,
            orbital_lvec=self.orbital_lvec
        )
        
        # Run the high-resolution simulation with the best parameters
        highres_results = highres_sim_cb(self.best_params)
        
        return highres_results
    
    def plot_high_resolution_comparison(self, nx_highres=400, nV_highres=200, figsize=(15, 12)):
        """
        Generate and plot high-resolution simulation results compared to experimental data.
        
        Args:
            nx_highres (int): Number of points along the x axis for high-res simulation
            nV_highres (int): Number of voltage points for high-res simulation
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure with high-resolution comparison
        """
        # Run high-resolution simulation
        highres_results = self.high_resolution_simulation(nx_highres, nV_highres)
        STM_highres, dIdV_highres, voltages_highres, x_highres = highres_results
        
        # Create figure for high-resolution comparison
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Calculate distance for extent
        x1, y1 = self.sim_start_point
        x2, y2 = self.sim_end_point
        dist = np.hypot(x2-x1, y2-y1)
        
        # Create extents for plots
        sim_extent = [0, dist, min(voltages_highres), max(voltages_highres)]
        exp_extent = [0, max(self.exp_x), min(self.exp_voltages), max(self.exp_voltages)]
        
        # Plot experimental data
        im0 = axs[0, 0].imshow(self.exp_data, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
        axs[0, 0].set_title('Experimental STM')
        axs[0, 0].set_xlabel('Distance (Å)')
        axs[0, 0].set_ylabel('Voltage (V)')
        plt.colorbar(im0, ax=axs[0, 0])
        
        # Plot high-resolution simulation
        im1 = axs[0, 1].imshow(STM_highres, extent=sim_extent, aspect='auto', origin='lower', cmap='hot')
        axs[0, 1].set_title(f'High-Resolution Simulation STM ({nx_highres}x{nV_highres})')
        axs[0, 1].set_xlabel('Distance (Å)')
        axs[0, 1].set_ylabel('Voltage (V)')
        plt.colorbar(im1, ax=axs[0, 1])
        
        # Create helper function to interpolate high-res results to experimental grid
        def interpolate_highres_to_exp(sim_results):
            STM, dIdV, voltages, x = sim_results
            return self.interpolate_func(STM, voltages, x)
        
        # Interpolate high-resolution simulation to match experimental grid
        interp_highres = interpolate_highres_to_exp(highres_results)
        
        # Plot interpolated high-resolution simulation
        im2 = axs[1, 0].imshow(interp_highres, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
        axs[1, 0].set_title('Interpolated High-Res Simulation')
        axs[1, 0].set_xlabel('Distance (Å)')
        axs[1, 0].set_ylabel('Voltage (V)')
        plt.colorbar(im2, ax=axs[1, 0])
        
        # Plot difference
        diff_highres = interp_highres - self.exp_data
        im3 = axs[1, 1].imshow(diff_highres, extent=exp_extent, aspect='auto', origin='lower', cmap='bwr')
        axs[1, 1].set_title('Difference (Sim - Exp)')
        axs[1, 1].set_xlabel('Distance (Å)')
        axs[1, 1].set_ylabel('Voltage (V)')
        plt.colorbar(im3, ax=axs[1, 1])
        
        plt.tight_layout()
        
        # Save the figure
        highres_file = "pauli_highres_comparison.png"
        fig.savefig(highres_file, dpi=200)
        print(f"High-resolution comparison saved to '{highres_file}'")
        
        return fig
    
    def save_results(self, base_filename):
        """
        Save optimization results to files.
        
        Args:
            base_filename (str): Base filename (without extension)
            
        Returns:
            list: List of saved filenames
        """
        # Call parent class method first
        saved_files = super().save_results(base_filename)
        
        # Save high-resolution simulation results
        highres_results = self.high_resolution_simulation()
        STM_highres, dIdV_highres, voltages_highres, x_highres = highres_results
        
        # Save high-resolution data to NPZ
        highres_file = f"{base_filename}_highres.npz"
        np.savez(highres_file,STM=STM_highres,dIdV=dIdV_highres,voltages=voltages_highres,x=x_highres,params=self.best_params)
        saved_files.append(highres_file)
        
        # Save high-resolution comparison plot
        highres_plot_file = f"{base_filename}_highres.png"
        self.plot_high_resolution_comparison().savefig(highres_plot_file, dpi=200)
        saved_files.append(highres_plot_file)
        
        print(f"Additional high-resolution results saved to {', '.join(saved_files[-2:])}")
        
        return saved_files
