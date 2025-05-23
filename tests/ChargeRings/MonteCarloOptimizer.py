#!/usr/bin/python

import numpy as np
import time
import copy
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class FitDatasetManager:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.summary_file = self.base_dir/"summary.jsonl"
        
    def create_run_dir(self):
        base_name = datetime.now().strftime("%Y_%m_%d")
        existing = [d for d in self.base_dir.glob(f"{base_name}_*") if d.is_dir()]
        next_num = max([int(d.name.split("_")[-1]) for d in existing] + [0]) + 1
        run_dir = self.base_dir/f"{base_name}_{next_num:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def record_run(self, optimizer):
        run_dir = self.create_run_dir()
        with open(run_dir/"params.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "error": optimizer.best_distance,
                "parameters": optimizer.best_params,
                "metadata": optimizer.metadata
            }, f, indent=4)
        with open(self.summary_file, "a") as f:
            f.write(json.dumps({"run_dir":run_dir.name,"error":optimizer.best_distance,"params":optimizer.best_params,"timestamp":datetime.now().isoformat()}) + "\n")
        return run_dir

class MonteCarloOptimizer:
    """
    General-purpose Monte Carlo optimizer that can optimize any set of parameters.
    
    This optimizer uses a simulated annealing approach to optimize parameters. It
    allows for parameter-specific mutation ranges and uses callbacks for all
    application-specific logic.
    """
    
    def __init__(self, 
            initial_params, 
            param_ranges,
            simulation_callback,
            distance_callback,
            mutation_factors=None,
            copy_params_func=None,
            result_dir=None
        ):
        """
        Initialize the Monte Carlo optimizer.
        
        Args:
            initial_params (dict or any): Initial parameters to optimize
            param_ranges (dict): Dictionary mapping parameter names to (min, max) tuples
            simulation_callback (callable): Function to run simulation with parameters
                function signature: simulation_callback(params) -> simulation_results
            distance_callback (callable): Function to calculate distance between simulation
                and reference data
                function signature: distance_callback(simulation_results) -> float
            mutation_factors (dict, optional): Relative mutation sizes for each parameter
            copy_params_func (callable, optional): Custom function to deep copy parameters
                if a simple copy.deepcopy() is not sufficient
            result_dir (str, optional): Directory to save results to
        """
        self.initial_params = copy.deepcopy(initial_params)
        self.current_params = copy.deepcopy(initial_params)
        self.best_params = copy.deepcopy(initial_params)
        self.param_ranges = param_ranges
        
        # Store callbacks
        self.simulation_callback = simulation_callback
        self.distance_callback = distance_callback
        
        # Use custom copy function if provided, otherwise use deepcopy
        self.copy_params_func = copy_params_func or (lambda p: copy.deepcopy(p))
        
        # Set default mutation factors if not provided
        if mutation_factors is None:
            self.mutation_factors = {}
            for param in param_ranges:
                min_val, max_val = param_ranges[param]
                # Default mutation factor is 10% of the parameter range
                self.mutation_factors[param] = 0.1 * (max_val - min_val)
        else:
            self.mutation_factors = mutation_factors
        
        # Initialize tracking variables
        self.iteration_history = []
        self.distance_history = []
        # Track parameter values per iteration for external plotting
        self.parameters_history = [self.copy_params_func(self.current_params)]
        self.accepted_changes = 0
        self.current_iteration = 0
        
        # Run initial simulation and calculate initial distance
        print("Running initial simulation...")
        self.best_sim_results = self.simulation_callback(self.best_params)
        self.best_distance = self.distance_callback(self.best_sim_results)
        self.distance_history.append(self.best_distance)
        
        print(f"Initialized with distance: {self.best_distance}")
        
        self.dataset_manager = FitDatasetManager(result_dir) if result_dir else None
        self.metadata = {}

    def mutate_params(self, params, mutation_strength=0.1):
        """
        Make random changes to parameters within allowed ranges.
        
        Args:
            params (dict or any): Current parameters
            mutation_strength (float): Global scaling factor for mutation size
            
        Returns:
            tuple: (new_params, param_name, old_value, new_value)
        """
        # Create a copy of the parameters
        new_params = self.copy_params_func(params)
        
        # Select a random parameter to mutate
        param_name = np.random.choice(list(self.param_ranges.keys()))
        min_val, max_val = self.param_ranges[param_name]
        
        # Get current value
        if isinstance(params, dict):
            old_value = params[param_name]
        else:
            # For non-dict objects, assume they have attribute access
            old_value = getattr(params, param_name)
        
        # Calculate mutation size based on parameter-specific factor
        param_factor = self.mutation_factors.get(param_name, 0.1 * (max_val - min_val))
        mutation_size = mutation_strength * param_factor
        
        # Generate new value with normal distribution around current value
        new_value = old_value + np.random.normal(0, mutation_size)
        
        # Clip to allowed range
        new_value = np.clip(new_value, min_val, max_val)
        
        # Update parameter
        if isinstance(new_params, dict):
            new_params[param_name] = new_value
        else:
            # For non-dict objects, assume they have attribute access
            setattr(new_params, param_name, new_value)
        
        return new_params, param_name, old_value, new_value
    
    def optimize(self, num_iterations=100, mutation_strength=0.1, temperature=0.01, temperature_decay=0.95, early_stop_iterations=None, min_improvement=1e-6, progress_callback=None, save_callback=None):
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
            progress_callback (callable, optional): Function to call after each iteration
                function signature: progress_callback(optimizer, iteration)
            save_callback (callable, optional): Function to call after optimization
                function signature: save_callback(optimizer, run_dir)
            
        Returns:
            dict or any: Optimized parameters
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
            sim_results = self.simulation_callback(new_params)
            
            # Calculate distance
            distance = self.distance_callback(sim_results)
            
            # Determine whether to accept the new parameters
            accept = False
            
            if distance < self.best_distance - min_improvement:  # Improvement
                accept = True
                self.best_params = self.copy_params_func(new_params)
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
                self.current_params = self.copy_params_func(new_params)
                self.accepted_changes += 1
            
            # Update history
            self.iteration_history.append({
                'iteration': i+1,
                'params': self.copy_params_func(new_params if accept else self.current_params),
                'distance': distance,
                'accepted': accept,
                'param_changed': param_name,
                'old_value': old_value,
                'new_value': new_value,
                'best_distance': self.best_distance
            })
            self.distance_history.append(self.best_distance)
            # record current parameters
            self.parameters_history.append(self.copy_params_func(self.current_params))
            
            # Update temperature
            if temperature is not None:
                current_temp *= temperature_decay
            
            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(self, i)
                
            # Check early stopping
            if early_stop_iterations and no_improvement_count >= early_stop_iterations:
                print(f"Stopping early after {i+1} iterations with no improvement")
                break
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Optimization completed in {elapsed:.2f} seconds")
        print(f"Iterations: {self.current_iteration}, Accepted changes: {self.accepted_changes}")
        print(f"Best distance: {self.best_distance}")
        
        if self.dataset_manager:
            run_dir = self.dataset_manager.record_run(self)
            if save_callback: 
                save_callback(self, run_dir)

        return self.best_params
    
    def save_results(self, base_filename):
        """
        Save optimization results to files.
        
        Args:
            base_filename (str): Base filename (without extension)
            
        Returns:
            list: List of saved filenames
        """
        saved_files = []
        
        # Save best parameters to JSON if they're a dictionary
        if isinstance(self.best_params, dict):
            param_file = f"{base_filename}_params.json"
            with open(param_file, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            saved_files.append(param_file)
        
        # Save optimization history to NPZ
        history_file = f"{base_filename}_history.npz"
        np.savez(history_file,  distance_history=np.array(self.distance_history),  iterations=np.arange(1, len(self.distance_history) + 1))
        saved_files.append(history_file)
        
        # Note: progress figure can be generated externally using fitting_plots.plot_optimization_progress
        print(f"Results saved to {', '.join(saved_files)}")
        
        return saved_files

    @property
    def history(self):
        """
        Returns optimization history dict for plotting.
        Keys: 'iterations', 'distances', 'parameters'.
        """
        return {
            'iterations': list(range(1, len(self.distance_history) + 1)),
            'distances': self.distance_history,
            'parameters': self.parameters_history
        }
