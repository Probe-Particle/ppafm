# Monte Carlo Optimization for STM Image Matching

## Overview

This document outlines a plan for implementing an automatic parameter optimization system for matching experimental and simulated STM images using a Monte Carlo approach. The system will use the Wasserstein distance (Earth Mover's Distance) as a metric to evaluate the similarity between experimental and simulated data.

## Key Files and Functions

### Existing Code References

1. **Wasserstein Distance Implementation**
   - File: `/home/prokop/git/ppafm/tests/ChargeRings/wasserstein_distance.py`
   - Function: `wasserstein_1d_grid(ys1, ys2, dx)` - Computes 1D Wasserstein distance on regular grid

2. **Experimental Data Extraction**
   - File: `/home/prokop/git/ppafm/tests/ChargeRings/CombinedChargeRingsGUI_v5.py`
   - Functions:
     - `calculate_1d_scan(self, start_point, end_point, pointPerAngstrom=5)` - Creates a 1D scan between two points
     - `plot_voltage_line_scan_exp(self, start, end, pointPerAngstrom=5)` - Plots voltage scan data along a line

3. **Simulation Functions**
   - File: `/home/prokop/git/ppafm/tests/ChargeRings/pauli_scan.py`
   - Functions:
     - `calculate_xV_scan_orb(params, start_point, end_point, ...)` - Calculates voltage scan along a line
     - `sweep_param_xV(params, scan_params, ...)` - Shows how to vary parameters for simulation

## Implementation Plan

### 1. Create a `MonteCarloOptimizer` class

```python
class MonteCarloOptimizer:
    def __init__(self, initial_params, experimental_data, param_ranges, ...):
        # Initialize with parameters, experimental data, and parameter ranges
```

### 2. Key Methods

#### Data Extraction and Preparation

```python
def extract_experimental_data(self, start_point, end_point):
    # Extract experimental data along a line
```

#### Simulation

```python
def run_simulation(self, params):
    # Run simulation with given parameters
```

#### Metric Calculation

```python
def calculate_2d_wasserstein(self, sim_data, exp_data):
    # Calculate Wasserstein distance for 2D images
```

#### Monte Carlo Optimization

```python
def optimize(self, num_iterations, temperature=1.0):
    # Run Monte Carlo optimization
```

### 3. Wasserstein Distance for 2D Images

The 2D Wasserstein distance will be calculated by:
1. Computing 1D Wasserstein along each column (V-direction) using the existing `wasserstein_1d_grid` function
2. Summing or averaging these distances to get a total measure of similarity

```python
def wasserstein_2d_xV(img1, img2, dx, dy):
    # Apply 1D Wasserstein along V-direction (y-axis) for each x position
    # Return combined metric
```

## Parameter Optimization Strategy

1. **Parameter Selection**
   - Identify key parameters that significantly affect the simulation results
   - Define reasonable ranges for each parameter

2. **Monte Carlo Algorithm**
   - Start with initial parameter set
   - For each iteration:
     - Make random changes to parameters within their ranges
     - Run simulation with new parameters
     - Calculate Wasserstein distance between simulation and experiment
     - Accept or reject based on distance reduction (always accept better solutions)
     - Optionally implement temperature for simulated annealing variant

3. **Convergence Criteria**
   - Maximum number of iterations
   - Minimum improvement threshold
   - Time limit

## Implementation Details

### Variables to Track

- `current_params`: Current best parameter set
- `current_distance`: Current best distance metric
- `iteration_history`: Track optimization progress
- `accepted_changes`: Counter for accepted parameter changes

### Parallelization Considerations

- Potential for parallel simulation runs to speed up optimization
- Parameter sweeps across multiple dimensions

## Usage Example

```python
# Initialize optimizer
optimizer = MonteCarloOptimizer(
    initial_params=params, 
    experimental_data=exp_data,
    param_ranges=param_ranges,
    start_point=(x1, y1),
    end_point=(x2, y2)
)

# Run optimization
optimized_params = optimizer.optimize(num_iterations=1000)
```

## Visualization

- Plot optimization progress (distance vs. iteration)
- Comparison of experimental and optimized simulation data
- Parameter evolution during optimization
