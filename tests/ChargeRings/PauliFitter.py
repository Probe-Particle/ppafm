import numpy as np
import time
import json
import sys
import os
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import differential_evolution

# Add parent directory to path so we can import ppafm modules
sys.path.append('../../')
from pyProbeParticle import utils as ut
from pyProbeParticle import pauli

# Import distance metrics
from distance_metrics import get_metric_function

# =====================================================================
# Utility Functions for Data Loading & Processing
# =====================================================================

def load_experimental_data(data_path='exp_rings_data.npz'):
    """
    Load experimental data from NPZ file
    
    Parameters
    ----------
    data_path : str
        Path to the experimental data NPZ file
        
    Returns
    -------
    X, Y, dIdV, I, biases : numpy arrays
        Experimental data arrays
    """
    print(f"Loading experimental data from {data_path}...")
    try:
        data = np.load(data_path)
        # Convert from nm to Å and center coordinates
        X = data['X'] * 10
        Y = data['Y'] * 10
        center_x = data['center_x'] * 10
        center_y = data['center_y'] * 10
        X -= center_x
        Y -= center_y
        dIdV = data['dIdV']
        I = data['I'] 
        biases = data['biases']
        
        print("Experimental data loaded successfully.")
        print(f"  Data shapes: X={X.shape}, Y={Y.shape}, I={I.shape}, dIdV={dIdV.shape}, biases={biases.shape}")
        
        return X, Y, dIdV, I, biases
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return None, None, None, None, None


def extract_experimental_data_along_line(X, Y, I, dIdV, biases, start_point, end_point, num_points=150):
    """
    Extract experimental data along a line defined by start and end points
    
    Parameters
    ----------
    X, Y : numpy arrays
        2D arrays containing the X and Y coordinates
    I, dIdV : numpy arrays
        3D arrays containing the current and differential conductance data
    biases : numpy array
        1D array containing the bias voltages
    start_point, end_point : tuple
        (x, y) coordinates of the line start and end points
    num_points : int
        Number of points to sample along the line
        
    Returns
    -------
    STM, dIdV_line, dist, biases : numpy arrays
        Extracted data arrays along the line
    """
    print(f"Extracting experimental data along line from {start_point} to {end_point}...")
    
    try:
        # Unpack points
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Calculate distance
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Create points along the line
        t = np.linspace(0, 1, num_points)
        x_line = x1 + t * (x2 - x1)
        y_line = y1 + t * (y2 - y1)
        
        # Calculate distances along the line
        dists = np.sqrt((x_line - x1)**2 + (y_line - y1)**2)
        
        # Extract data along the line for each bias voltage
        STM = np.zeros((len(biases), num_points))
        dIdV_line = np.zeros((len(biases), num_points))
        
        # Use the exported exp_utils function that properly calls the right interpolation function
        from exp_utils import plot_exp_voltage_line_scan, create_line_coordinates
        
        # Calculate the total distance for pointPerAngstrom
        total_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        points_per_angstrom = num_points / total_distance
        
        # Extract STM data along the line
        STM, dists = plot_exp_voltage_line_scan(
            X, Y, I, biases, 
            start_point, end_point, 
            pointPerAngstrom=points_per_angstrom,
            ax=None  # No plotting, just return the data
        )
        
        # Extract dIdV data along the line
        dIdV_line, _ = plot_exp_voltage_line_scan(
            X, Y, dIdV, biases, 
            start_point, end_point, 
            pointPerAngstrom=points_per_angstrom,
            ax=None  # No plotting, just return the data
        )
        
        print("Experimental data extracted successfully.")
        print(f"  Extracted data shapes: STM={STM.shape}, dIdV={dIdV_line.shape}, dist={dists.shape}")
        
        return STM, dIdV_line, dists, biases
    except Exception as e:
        print(f"Error extracting experimental data: {e}")
        return None, None, None, None


# =====================================================================
# Simulation Callback Functions
# =====================================================================

def create_simulation_callback(sim_start_point, sim_end_point, nx=50, nV=None, Vmin=None, Vmax=None, orbital_2D=None, orbital_lvec=None):
    """
    Create a callback function for running simulations
    
    Parameters
    ----------
    sim_start_point, sim_end_point : tuple
        (x, y) coordinates of the line start and end points for simulation
    nx : int
        Number of points along the x-axis for simulation
    nV : int
        Number of voltage points for simulation
    Vmin, Vmax : float
        Minimum and maximum voltage values for simulation
    orbital_2D : numpy array
        2D orbital data for simulation
    orbital_lvec : numpy array
        Lattice vectors for simulation
        
    Returns
    -------
    simulate : function
        Callback function for running simulations
    """
    
    def simulate(params):
        """
        Run a simulation with the given parameters and return the results
        
        Parameters
        ----------
        params : dict
            Dictionary of simulation parameters
            
        Returns
        -------
        x_positions, voltages, STM : numpy arrays
            Simulation results
        """
        # Import the correct pauli scan module to avoid circular import
        from pauli_scan import calculate_xV_scan_orb
        
        # Calculate the positions along the line
        x1, y1 = sim_start_point
        x2, y2 = sim_end_point
        
        # Set up voltage range
        v_range = None
        if Vmin is not None and Vmax is not None:
            v_range = (Vmin, Vmax, nV if nV is not None else 100)
            
        # Convert parameter names to match what the pauli_scan module expects
        pauli_params = params.copy()
        
        # Ensure we have a 'VBias' parameter (used by the original code)
        if 'VBias' not in pauli_params and v_range is not None:
            pauli_params['VBias'] = v_range[1]  # Use Vmax as VBias if not present
        
        # Set default values for required parameters if not specified
        v_min = v_range[0] if v_range else 0.0
        v_max = v_range[1] if v_range else pauli_params.get('VBias', 1.0)
        n_v = v_range[2] if v_range and len(v_range) > 2 else 100  # Default to 100 voltage points
        
        # Run simulation - function returns many values but we only need a few
        results = calculate_xV_scan_orb(
            params=pauli_params,
            start_point=[x1, y1],
            end_point=[x2, y2],
            orbital_2D=orbital_2D,
            orbital_lvec=orbital_lvec,
            nx=nx,
            Vmin=v_min,
            Vmax=v_max,
            nV=n_v
        )
        
        # Unpack only what we need (the function returns 9 values)
        # STM, dIdV, Es, Ts, probs, x, Vbiases, spos, rots = results
        STM = results[0]  # Current data
        x_positions = results[5]  # x positions
        voltages = results[6]  # Voltage values
        
        return x_positions, voltages, STM
    
    return simulate


def interpolate_simulation_to_experiment(sim_x, sim_voltages, sim_data, exp_x, exp_voltages):
    """
    Interpolate simulation data to match experimental data dimensions
    
    Parameters
    ----------
    sim_x, sim_voltages : numpy arrays
        1D arrays with simulation x positions and voltages
    sim_data : numpy array
        2D array with simulation data
    exp_x, exp_voltages : numpy arrays
        1D arrays with experimental x positions and voltages
        
    Returns
    -------
    interp_sim_data : numpy array
        Interpolated simulation data to match experimental dimensions
    """
    # Ensure the arrays are sorted for interpolation
    if not np.all(np.diff(sim_voltages) > 0):
        # Sort the voltages and reorder the data accordingly
        sort_indices = np.argsort(sim_voltages)
        sim_voltages = sim_voltages[sort_indices]
        sim_data = sim_data[sort_indices]
    
    if not np.all(np.diff(sim_x) > 0):
        # Sort the x positions and reorder the data accordingly
        sort_indices = np.argsort(sim_x)
        sim_x = sim_x[sort_indices]
        sim_data = sim_data[:, sort_indices]
    
    # Create interpolator
    interp = RectBivariateSpline(sim_voltages, sim_x, sim_data)
    
    # Interpolate to experimental grid
    interp_sim_data = interp(exp_voltages, exp_x)
    
    return interp_sim_data


def create_distance_callback(exp_data, exp_voltages, exp_x, simulation_callback, metric_function, initial_params):
    """
    Create a callback function for calculating the distance between simulation and experiment
    
    Parameters
    ----------
    exp_data : numpy array
        Experimental data array
    exp_voltages, exp_x : numpy arrays
        1D arrays with experimental voltages and x positions
    simulation_callback : function
        Callback function for running simulations
    metric_function : function
        Function to calculate the distance between simulation and experiment
    initial_params : dict
        Initial parameters to use as a base for optimization
        
    Returns
    -------
    calculate_distance : function
        Callback function for calculating the distance
    """
    
    def calculate_distance(param_list, param_names):
        """
        Calculate the distance between simulation and experiment
        
        Parameters
        ----------
        param_list : list
            List of parameter values
        param_names : list
            List of parameter names
            
        Returns
        -------
        distance : float
            Distance between simulation and experiment
        """
        # Start with the original initial_params as a base to maintain all required parameters
        params = initial_params.copy()
        
        # Update only the parameters we're optimizing
        for name, value in zip(param_names, param_list):
            params[name] = value
        
        # Run simulation
        sim_x, sim_voltages, sim_data = simulation_callback(params)
        
        # Interpolate simulation data to match experimental dimensions
        interp_sim_data = interpolate_simulation_to_experiment(
            sim_x, sim_voltages, sim_data, exp_x, exp_voltages
        )
        
        # Calculate distance using the provided metric function
        distance = metric_function(exp_data, interp_sim_data)
        
        return distance, interp_sim_data
    
    return calculate_distance


# =====================================================================
# PauliFitter Class
# =====================================================================

class PauliFitter:
    """
    Class for fitting Pauli transport simulations to experimental data
    """
    
    def __init__(self, initial_params, param_ranges, exp_data, exp_voltages, exp_x,
                 sim_start_point, sim_end_point, nx=50, nV=None, Vmin=None, Vmax=None,
                 orbital_2D=None, orbital_lvec=None, metric_name='wasserstein_v'):
        """
        Initialize the PauliFitter
        
        Parameters
        ----------
        initial_params : dict
            Dictionary of initial parameter values
        param_ranges : dict
            Dictionary of parameter ranges (min, max)
        exp_data : numpy array
            Experimental data array
        exp_voltages, exp_x : numpy arrays
            1D arrays with experimental voltages and x positions
        sim_start_point, sim_end_point : tuple
            (x, y) coordinates of the line start and end points for simulation
        nx : int
            Number of points along the x-axis for simulation
        nV : int
            Number of voltage points for simulation
        Vmin, Vmax : float
            Minimum and maximum voltage values for simulation
        orbital_2D : numpy array
            2D orbital data for simulation
        orbital_lvec : numpy array
            Lattice vectors for simulation
        metric_name : str
            Name of the distance metric to use
        """
        # Store experimental data
        self.exp_data = exp_data
        self.exp_voltages = exp_voltages
        self.exp_x = exp_x
        
        # Store simulation parameters
        self.sim_start_point = sim_start_point
        self.sim_end_point = sim_end_point
        self.nx   = nx
        self.nV   = nV   if nV is not None else len(exp_voltages)
        self.Vmin = Vmin if Vmin is not None else min(exp_voltages)
        self.Vmax = Vmax if Vmax is not None else max(exp_voltages)
        self.orbital_2D = orbital_2D
        self.orbital_lvec = orbital_lvec
        
        # Store optimization parameters
        self.initial_params = initial_params.copy()
        self.param_ranges = param_ranges.copy()
        self.param_names = list(param_ranges.keys())
        self.best_params = self.initial_params.copy()
        self.best_distance = float('inf')
        self.best_sim_data = None
        
        # Create simulation callback
        self.simulation_callback = create_simulation_callback(
            sim_start_point, sim_end_point, nx, nV, Vmin, Vmax, orbital_2D, orbital_lvec
        )
        
        # Set up metric
        self.metric_name = metric_name
        self.metric_function = get_metric_function(metric_name)
        
        # Create distance callback
        self.distance_callback = create_distance_callback(
            exp_data, exp_voltages, exp_x, self.simulation_callback, self.metric_function, self.initial_params
        )
        
        # Initialize optimization history
        self.history = {
            'iterations': [],
            'distances': [],
            'accepted': [],
            'parameters': []
        }
    
    def change_metric(self, new_metric_name):
        """
        Change the distance metric used for optimization
        
        Parameters
        ----------
        new_metric_name : str
            Name of the new distance metric to use
            
        Returns
        -------
        None
        """
        self.metric_name = new_metric_name
        self.metric_function = get_metric_function(new_metric_name)
        
        # Update distance callback
        self.distance_callback = create_distance_callback(
            self.exp_data, self.exp_voltages, self.exp_x,
            self.simulation_callback, self.metric_function, self.initial_params
        )
        
        # Recalculate best distance with new metric
        params_list = [self.best_params[name] for name in self.param_names]
        new_distance, _ = self.distance_callback(params_list, self.param_names)
        self.best_distance = new_distance
    
    def optimize(self, num_iterations=100, mutation_strength=0.1, temperature=0.01,
                temperature_decay=0.95, early_stop_iterations=50):
        """
        Run simulated annealing optimization
        
        Parameters
        ----------
        num_iterations : int
            Number of optimization iterations
        mutation_strength : float
            Strength of mutations in simulated annealing
        temperature : float
            Initial temperature for simulated annealing
        temperature_decay : float
            Temperature decay factor
        early_stop_iterations : int
            Number of iterations with no improvement before early stopping
            
        Returns
        -------
        best_params : dict
            Dictionary of best parameter values found
        """
        # Start the timer for performance tracking
        t_start = time.time()
        
        # Initialize optimization
        current_params = self.best_params.copy()
        current_params_list = [current_params[name] for name in self.param_names]
        current_distance, current_sim_data = self.distance_callback(current_params_list, self.param_names)
        
        if current_distance < self.best_distance:
            self.best_distance = current_distance
            self.best_params = current_params.copy()
            self.best_sim_data = current_sim_data
        
        # Initialize history
        self.history['iterations'].append(0)
        self.history['distances'].append(current_distance)
        self.history['accepted'].append(True)
        self.history['parameters'].append(current_params_list.copy())
        
        # Run optimization
        no_improvement_count = 0
        accepted_count = 0
        current_temp = temperature
        
        # We're including iteration 0 in the history, so we start from 1
        for i in range(1, num_iterations + 1):
            # Create a new candidate by perturbing the current parameters
            candidate_params = current_params.copy()
            
            # Choose a random parameter to perturb
            param_to_perturb = np.random.choice(self.param_names)
            param_range = self.param_ranges[param_to_perturb]
            param_width = param_range[1] - param_range[0]
            
            # Apply mutation
            perturbation = np.random.normal(0, mutation_strength * param_width)
            candidate_params[param_to_perturb] += perturbation
            
            # Ensure the parameter stays within bounds
            candidate_params[param_to_perturb] = np.clip(
                candidate_params[param_to_perturb], param_range[0], param_range[1]
            )
            
            # Calculate distance for the candidate
            candidate_params_list = [candidate_params[name] for name in self.param_names]
            candidate_distance, candidate_sim_data = self.distance_callback(candidate_params_list, self.param_names)
            
            # Decide whether to accept the candidate
            accept = False
            if candidate_distance < current_distance:
                accept = True
            else:
                # Calculate acceptance probability using the Metropolis criterion
                delta = candidate_distance - current_distance
                acceptance_prob = np.exp(-delta / current_temp)
                accept = np.random.random() < acceptance_prob
            
            # Update current parameters if the candidate is accepted
            if accept:
                current_params = candidate_params.copy()
                current_params_list = candidate_params_list.copy()
                current_distance = candidate_distance
                current_sim_data = candidate_sim_data
                accepted_count += 1
                
                # Update best parameters if this is the best solution so far
                if current_distance < self.best_distance:
                    self.best_distance = current_distance
                    self.best_params = current_params.copy()
                    self.best_sim_data = current_sim_data
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Update history
            self.history['iterations'].append(i)
            self.history['distances'].append(current_distance)
            self.history['accepted'].append(accept)
            self.history['parameters'].append(current_params_list.copy())
            
            # Update temperature
            current_temp *= temperature_decay
            
            # Early stopping
            if no_improvement_count >= early_stop_iterations:
                print(f"Stopping early after {no_improvement_count} iterations with no improvement")
                break
        
        print(f"Optimization completed in {time.time() - t_start:.2f} seconds")
        print(f"Iterations: {i}, Accepted changes: {accepted_count}")
        print(f"Best distance: {self.best_distance}")
        
        return self.best_params
    
    def run_high_resolution_simulation(self, nx_highres=400, nV_highres=200):
        """
        Run a high-resolution simulation with the best parameters
        
        Parameters
        ----------
        nx_highres : int
            Number of x positions for the high-resolution simulation
        nV_highres : int
            Number of voltage points for the high-resolution simulation
            
        Returns
        -------
        x_highres, voltages_highres, STM_highres, dIdV_highres : numpy arrays
            High-resolution simulation results
        """
        print(f"Running high-resolution simulation with {nx_highres}x{nV_highres} points...")
        
        # Create a high-resolution simulation callback
        highres_callback = create_simulation_callback(
            self.sim_start_point, self.sim_end_point,
            nx=nx_highres, nV=nV_highres,
            Vmin=self.Vmin, Vmax=self.Vmax,
            orbital_2D=self.orbital_2D, orbital_lvec=self.orbital_lvec
        )
        
        # Run high-resolution simulation with best parameters
        x_highres, voltages_highres, STM_highres = highres_callback(self.best_params)
        
        # Calculate dIdV (just a placeholder, actual dIdV calculation would be done elsewhere)
        dIdV_highres = np.zeros_like(STM_highres)  # Placeholder
        
        return x_highres, voltages_highres, STM_highres, dIdV_highres
    
    def save_results(self, base_filename=None):
        """
        Save optimization results to files
        
        Parameters
        ----------
        base_filename : str
            Base filename for saved files
            
        Returns
        -------
        saved_files : list
            List of saved files
        """
        # Create base filename if not provided
        if base_filename is None:
            base_filename = f"pauli_fit_results_{self.metric_name}"
        
        saved_files = []
        
        # Save parameters
        params_file = f"{base_filename}_params.json"
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        saved_files.append(params_file)
        
        # Save optimization history
        history_file = f"{base_filename}_history.npz"
        np.savez(history_file,
                 iterations=np.array(self.history['iterations']),
                 distances=np.array(self.history['distances']),
                 accepted=np.array(self.history['accepted']),
                 parameters=np.array(self.history['parameters']))
        saved_files.append(history_file)
        
        # Run high-resolution simulation
        x_highres, voltages_highres, STM_highres, dIdV_highres = self.run_high_resolution_simulation()
        
        # Save high-resolution data to NPZ
        highres_file = f"{base_filename}_highres.npz"
        np.savez(highres_file,STM=STM_highres,dIdV=dIdV_highres,voltages=voltages_highres,x=x_highres,params=self.best_params)
        saved_files.append(highres_file)
        
        return saved_files, (x_highres, voltages_highres, STM_highres, dIdV_highres)
