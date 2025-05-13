#!/usr/bin/python

import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import os
import sys

# Import from ppafm modules
sys.path.append('../../')
from pyProbeParticle import utils as ut
from pyProbeParticle import pauli

# Import local modules
import pauli_scan
from wasserstein_distance import wasserstein_1d_grid


class MonteCarloOptimizer:
    """
    Monte Carlo optimizer for matching experimental and simulated STM images.
    
    Uses Wasserstein distance (Earth Mover's Distance) as a metric to evaluate
    similarity between experimental and simulated data.
    """
    
    def __init__(self, initial_params, exp_data, exp_voltages, exp_x, 
                 param_ranges, start_point, end_point, 
                 orbital_2D=None, orbital_lvec=None, pauli_solver=None,
                 nx=100, nV=100, Vmin=0.0, Vmax=None):
        """
        Initialize the Monte Carlo optimizer.
        
        Args:
            initial_params (dict): Initial simulation parameters
            exp_data (np.ndarray): Experimental data (2D array with dimensions [nV, nx])
            exp_voltages (np.ndarray): Voltage values for experimental data
            exp_x (np.ndarray): X positions for experimental data
            param_ranges (dict): Dictionary with parameter ranges as {param_name: (min, max)}
            start_point (tuple): Start point (x, y) for the scan line
            end_point (tuple): End point (x, y) for the scan line
            orbital_2D (np.ndarray, optional): 2D orbital data
            orbital_lvec (np.ndarray, optional): Lattice vectors for orbital data
            pauli_solver (object, optional): Pauli solver instance
            nx (int): Number of x points for simulation
            nV (int): Number of V points for simulation
            Vmin (float): Minimum voltage for simulation
            Vmax (float, optional): Maximum voltage for simulation
        """
        self.current_params = copy.deepcopy(initial_params)
        self.best_params = copy.deepcopy(initial_params)
        self.param_ranges = param_ranges
        self.start_point = start_point
        self.end_point = end_point
        self.orbital_2D = orbital_2D
        self.orbital_lvec = orbital_lvec
        self.pauli_solver = pauli_solver
        self.nx = nx
        self.nV = nV
        self.Vmin = Vmin
        self.Vmax = Vmax if Vmax is not None else initial_params.get('VBias', 1.0)
        
        # Store experimental data - ensure coordinates are sorted for interpolation
        self.exp_data = exp_data
        
        # Sort voltage values and data if needed
        volt_idx = np.argsort(exp_voltages)
        self.exp_voltages = exp_voltages[volt_idx]
        self.exp_data = exp_data[volt_idx, :] if len(volt_idx) > 1 and not np.array_equal(volt_idx, np.arange(len(volt_idx))) else exp_data
        
        # Sort x positions and data if needed
        x_idx = np.argsort(exp_x)
        self.exp_x = exp_x[x_idx]
        self.exp_data = self.exp_data[:, x_idx] if len(x_idx) > 1 and not np.array_equal(x_idx, np.arange(len(x_idx))) else self.exp_data
        
        # Initialize tracking variables
        self.iteration_history = []
        self.distance_history = []
        self.accepted_changes = 0
        self.current_iteration = 0
        
        # Run initial simulation and calculate initial distance
        self.best_sim_results = self.run_simulation(self.best_params)
        self.best_distance = self.calculate_distance(self.best_sim_results, self.exp_data)
        self.distance_history.append(self.best_distance)
        
        print(f"Initialized with distance: {self.best_distance}")
        
    def run_simulation(self, params):
        """
        Run simulation with the given parameters.
        
        Args:
            params (dict): Simulation parameters
            
        Returns:
            tuple: (STM, dIdV, voltages, x_positions) - Simulation results
        """
        # Call pauli_scan.calculate_xV_scan_orb to get simulation results
        STM, dIdV, Es, Ts, probs, x, voltages, spos, rots = pauli_scan.calculate_xV_scan_orb(
            params, 
            self.start_point, 
            self.end_point,
            orbital_2D=self.orbital_2D,
            orbital_lvec=self.orbital_lvec,
            pauli_solver=self.pauli_solver,
            nx=self.nx, 
            nV=self.nV,
            Vmin=self.Vmin,
            Vmax=self.Vmax,
            bLegend=False  # No need for legends in optimization
        )
        
        return STM, dIdV, voltages, x
    
    def wasserstein_2d_xV(self, img1, img2, dx=1.0, dy=1.0):
        """
        Calculate 2D Wasserstein distance between two images along V direction.
        
        Args:
            img1 (np.ndarray): First image (2D array with dimensions [nV, nx])
            img2 (np.ndarray): Second image (2D array with dimensions [nV, nx])
            dx (float): Grid spacing in x direction
            dy (float): Grid spacing in voltage direction
            
        Returns:
            float: Combined Wasserstein distance
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
    
    def interpolate_simulation_to_exp_grid(self, sim_data, sim_voltages, sim_x):
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
        vv, xx = np.meshgrid(self.exp_voltages, self.exp_x, indexing='ij')
        
        # Evaluate at each point (doesn't require strictly increasing coordinates)
        # Must use ev(y, x) to match the exp_utils.interpolate_3d_plane_fast convention
        interpolated = interp.ev(vv, xx)
        
        return interpolated
    
    def calculate_distance(self, sim_results, exp_data):
        """
        Calculate distance between simulation and experimental data.
        
        Args:
            sim_results (tuple): (STM, dIdV, voltages, x) from simulation
            exp_data (np.ndarray): Experimental data
            
        Returns:
            float: Distance metric (lower is better)
        """
        STM, dIdV, voltages, x = sim_results
        
        # Use STM current data instead of dIdV for comparison
        sim_data = STM
        
        # Interpolate simulation data to match experimental grid
        sim_data_interp = self.interpolate_simulation_to_exp_grid(sim_data, voltages, x)
            
        # Calculate 2D Wasserstein distance
        # Use the spacing from the experimental data for consistency
        dv = np.abs(self.exp_voltages[1] - self.exp_voltages[0]) if len(self.exp_voltages) > 1 else 1.0
        dx = np.abs(self.exp_x[1] - self.exp_x[0]) if len(self.exp_x) > 1 else 1.0
        
        distance = self.wasserstein_2d_xV(sim_data_interp, exp_data, dx=dx, dy=dv)
        
        return distance
    
    def mutate_params(self, params, mutation_strength=0.1):
        """
        Make random changes to parameters within allowed ranges.
        
        Args:
            params (dict): Current parameters
            mutation_strength (float): Relative size of parameter changes (0.0-1.0)
            
        Returns:
            dict: New parameters with random mutations
        """
        new_params = copy.deepcopy(params)
        
        # Randomly select a parameter to mutate
        param_names = list(self.param_ranges.keys())
        param_to_mutate = np.random.choice(param_names)
        
        # Get range for the selected parameter
        param_min, param_max = self.param_ranges[param_to_mutate]
        param_range = param_max - param_min
        
        # Current value
        current_value = params[param_to_mutate]
        
        # Calculate mutation
        mutation = np.random.normal(0, mutation_strength * param_range)
        new_value = current_value + mutation
        
        # Ensure new value is within allowed range
        new_value = max(param_min, min(param_max, new_value))
        
        # Update parameter
        new_params[param_to_mutate] = new_value
        
        return new_params, param_to_mutate, current_value, new_value
    
    def optimize(self, num_iterations=100, mutation_strength=0.1, 
                 temperature=None, temperature_decay=0.95,
                 early_stop_iterations=None, min_improvement=1e-6,
                 callback=None):
        """
        Run Monte Carlo optimization.
        
        Args:
            num_iterations (int): Maximum number of iterations
            mutation_strength (float): Relative size of parameter changes (0.0-1.0)
            temperature (float, optional): Initial temperature for simulated annealing
                                        If None, only accept improvements
            temperature_decay (float): Rate at which temperature decreases
            early_stop_iterations (int, optional): Stop if no improvement after this many iterations
            min_improvement (float): Minimum improvement to consider significant
            callback (function, optional): Function to call after each iteration
            
        Returns:
            dict: Optimized parameters
        """
        current_temp = temperature
        no_improvement_count = 0
        start_time = time.time()
        
        for i in range(num_iterations):
            self.current_iteration = i + 1
            
            # Mutate parameters
            new_params, param_name, old_value, new_value = self.mutate_params(
                self.current_params, mutation_strength)
            
            # Run simulation with new parameters
            sim_results = self.run_simulation(new_params)
            
            # Calculate distance
            distance = self.calculate_distance(sim_results, self.exp_data)
            
            # Determine whether to accept the new parameters
            accept = False
            
            if distance < self.best_distance - min_improvement:  # Improvement
                accept = True
                self.best_params = copy.deepcopy(new_params)
                self.best_distance = distance
                self.best_sim_results = sim_results
                no_improvement_count = 0
                print(f"Iteration {i+1}: New best distance {distance:.6f}")
                print(f"  Changed {param_name}: {old_value:.6f} -> {new_value:.6f}")
            elif temperature is not None:  # Simulated annealing with temperature
                # Calculate acceptance probability
                delta = distance - self.best_distance
                if delta < 0:  # Improvement
                    p_accept = 1.0
                else:  # Worse, but may still accept based on temperature
                    p_accept = np.exp(-delta / current_temp) if current_temp > 0 else 0.0
                    
                # Accept with calculated probability
                if np.random.random() < p_accept:
                    accept = True
                    print(f"Iteration {i+1}: Accepted worse solution with p={p_accept:.4f}")
                    print(f"  Distance: {self.best_distance:.6f} -> {distance:.6f}")
                    print(f"  Changed {param_name}: {old_value:.6f} -> {new_value:.6f}")
                    no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:  # No temperature, only accept improvements
                no_improvement_count += 1
            
            # Update current parameters if accepted
            if accept:
                self.current_params = copy.deepcopy(new_params)
                self.accepted_changes += 1
            
            # Update history
            self.iteration_history.append({
                'iteration': i+1,
                'params': copy.deepcopy(new_params if accept else self.current_params),
                'distance': distance,
                'accepted': accept,
                'param_changed': param_name,
                'old_value': old_value,
                'new_value': new_value,
                'best_distance': self.best_distance
            })
            self.distance_history.append(self.best_distance)
            
            # Update temperature
            if temperature is not None:
                current_temp *= temperature_decay
            
            # Call callback if provided
            if callback is not None:
                callback(self, i)
                
            # Check early stopping
            if early_stop_iterations and no_improvement_count >= early_stop_iterations:
                print(f"Stopping early after {i+1} iterations with no improvement")
                break
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Optimization completed in {elapsed:.2f} seconds")
        print(f"Iterations: {self.current_iteration}, Accepted changes: {self.accepted_changes}")
        print(f"Best distance: {self.best_distance}")
        
        return self.best_params
    
    def plot_optimization_progress(self, figsize=(10, 6)):
        """
        Plot optimization progress.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        iterations = list(range(1, len(self.distance_history) + 1))
        ax.plot(iterations, self.distance_history, 'b-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Distance (lower is better)')
        ax.set_title('Optimization Progress')
        ax.grid(True)
        
        return fig
    
    def plot_comparison(self, figsize=(15, 10)):
        """
        Plot comparison between experimental and best simulation data.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Unpack best simulation results
        STM, dIdV, voltages, x = self.best_sim_results
        
        # Calculate distance between points for extent
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        dist = np.hypot(x2-x1, y2-y1)
        
        # Create extent for both plots
        sim_extent = [0, dist, self.Vmin, self.Vmax]
        exp_extent = [0, dist, min(self.exp_voltages), max(self.exp_voltages)]
        
        # Plot experimental data
        im0 = axs[0, 0].imshow(self.exp_data, extent=exp_extent, aspect='auto', origin='lower', cmap='bwr')
        axs[0, 0].set_title('Experimental dI/dV')
        axs[0, 0].set_xlabel('Distance (\u00c5)')
        axs[0, 0].set_ylabel('Voltage (V)')
        plt.colorbar(im0, ax=axs[0, 0])
        
        # Plot best simulation
        im1 = axs[0, 1].imshow(dIdV, extent=sim_extent, aspect='auto', origin='lower', cmap='bwr')
        axs[0, 1].set_title('Best Simulation dI/dV')
        axs[0, 1].set_xlabel('Distance (\u00c5)')
        axs[0, 1].set_ylabel('Voltage (V)')
        plt.colorbar(im1, ax=axs[0, 1])
        
        # Interpolate simulation to match experimental grid
        interp_sim = self.interpolate_simulation_to_exp_grid(dIdV, voltages, x)
            
        # Plot difference
        diff = interp_sim - self.exp_data
        im2 = axs[1, 0].imshow(diff, extent=exp_extent, aspect='auto', origin='lower', cmap='bwr')
        axs[1, 0].set_title('Difference (Sim - Exp)')
        axs[1, 0].set_xlabel('Distance (\u00c5)')
        axs[1, 0].set_ylabel('Voltage (V)')
        plt.colorbar(im2, ax=axs[1, 0])
        
        # Plot optimization progress
        axs[1, 1].plot(range(1, len(self.distance_history) + 1), self.distance_history, 'b-')
        axs[1, 1].set_title('Optimization Progress')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Distance (lower is better)')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, base_filename):
        """
        Save optimization results to files.
        
        Args:
            base_filename (str): Base filename (without extension)
            
        Returns:
            list: List of saved filenames
        """
        saved_files = []
        
        # Save best parameters to JSON
        param_file = f"{base_filename}_params.json"
        import json
        with open(param_file, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        saved_files.append(param_file)
        
        # Save optimization history to NPZ
        history_file = f"{base_filename}_history.npz"
        np.savez(history_file, 
                 distance_history=np.array(self.distance_history),
                 iterations=np.arange(1, len(self.distance_history) + 1))
        saved_files.append(history_file)
        
        # Save comparison figure
        fig_file = f"{base_filename}_comparison.png"
        self.plot_comparison().savefig(fig_file, dpi=150)
        saved_files.append(fig_file)
        
        # Save progress figure
        prog_file = f"{base_filename}_progress.png"
        self.plot_optimization_progress().savefig(prog_file, dpi=150)
        saved_files.append(prog_file)
        
        print(f"Results saved to {', '.join(saved_files)}")
        
        return saved_files


# Usage example
if __name__ == "__main__":
    print("Monte Carlo Optimizer module - import and use the MonteCarloOptimizer class")
    print("Example usage:")
    print("""
    # Initialize optimizer
    optimizer = MonteCarloOptimizer(
        initial_params=params, 
        exp_data=exp_data,
        exp_voltages=voltages,
        exp_x=x_positions,
        param_ranges=param_ranges,
        start_point=(x1, y1),
        end_point=(x2, y2)
    )

    # Run optimization
    optimized_params = optimizer.optimize(num_iterations=100)
    
    # Plot results
    optimizer.plot_comparison()
    plt.show()
    """)
