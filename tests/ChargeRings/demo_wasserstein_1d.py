# demo_wasserstein_1d.py
import numpy as np
from interactive_plotter_1D import InteractivePlotter1D
from wasserstein_distance import wasserstein_1d_general, wasserstein_1d_grid

# --- 1. Define Common X-Coordinates ---
# This is the domain on which your y=f(x) signals will be sampled.
# Must be a regular grid if use_grid_optimized_wd=True in plotter.
x_sample_points = np.linspace(-12, 22, 600)

# --- 2. Define Model Functions ---
# These functions take x_sample_points and a params dictionary, 
# and return raw (unnormalized) y-values (signal intensities).
# The plotter will handle normalization. y-values should be non-negative.

def gaussian_signal_model(xs, params):
    """Generates a Gaussian peak y=f(x)."""
    center = params.get('center', 0)
    sigma = params.get('sigma', 1)
    amplitude = params.get('amplitude', 1)
    sigma = max(sigma, 1e-6) 
    ys = amplitude * np.exp(-((xs - center)**2) / (2 * sigma**2))
    return ys # Raw y-values, plotter normalizes

def multi_feature_signal_model(xs, params):
    """A signal y=f(x) with multiple features: two Gaussians and a step."""
    p1_c = params.get('peak1_center', 0)
    p1_s = max(params.get('peak1_sigma', 1), 1e-6)
    p1_a = params.get('peak1_amp', 1)
    peak1_ys = p1_a * np.exp(-((xs - p1_c)**2) / (2 * p1_s**2))

    p2_c = params.get('peak2_center', 5)
    p2_s = max(params.get('peak2_sigma', 0.5), 1e-6)
    p2_a = params.get('peak2_amp', 0.7)
    peak2_ys = p2_a * np.exp(-((xs - p2_c)**2) / (2 * p2_s**2))
    
    step_e = params.get('step_edge', -5)
    step_h = params.get('step_height', 0.3)
    step_ys = np.zeros_like(xs)
    step_ys[xs >= step_e] = step_h
    
    combined_ys = peak1_ys + peak2_ys + step_ys
    return combined_ys # Raw y-values

def wasserstein_metric_callback(ref_ys, interactive_ys, x_coords):
    """Callback function that calculates Wasserstein distance"""
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    if not np.allclose(np.diff(x_coords), dx):
        # Use general W1 if grid is irregular
        wd = wasserstein_1d_general(x_coords, ref_ys, x_coords, interactive_ys)
    else:
        # Use optimized grid version if grid is regular
        wd = wasserstein_1d_grid(ref_ys, interactive_ys, dx)
    return wd, f'Wasserstein Distance (W1): {wd:.4f}'

# --- 3. Define Parameters for the REFERENCE Signal y_ref = f_ref(x) ---
reference_signal_function = gaussian_signal_model 
reference_signal_parameters = {
    'center': -2,
    'sigma': 1.5,
    'amplitude': 1.0
}

# --- 4. Define Parameters and Slider Configurations for the INTERACTIVE Signal y_int = f_int(x) ---
interactive_signal_function = multi_feature_signal_model
# Format: param_name: (min_val, max_val, init_val, step_val)
interactive_signal_parameter_configs = {
    'peak1_center': (-10, 10, 0, 0.1),
    'peak1_sigma':  (0.2, 4, 1.2, 0.05),
    'peak1_amp':    (0.0, 2, 0.8, 0.05),
    'peak2_center': (-8, 18, 8, 0.1),
    'peak2_sigma':  (0.2, 3, 0.6, 0.05),
    'peak2_amp':    (0.0, 1.5, 0.5, 0.05),
    'step_edge':    (-10, 5, -4, 0.2),
    'step_height':  (0.0, 1, 0.1, 0.02)
}

# --- 5. Create and Show the Plotter ---
if __name__ == '__main__':
    plotter = InteractivePlotter1D(
        x_coords_domain=x_sample_points,
        reference_model_func=reference_signal_function,
        reference_params=reference_signal_parameters,
        interactive_model_func=interactive_signal_function,
        interactive_param_configs=interactive_signal_parameter_configs,
        metric_callback=wasserstein_metric_callback,
        dist_title='1D Signals (Normalized y-values treated as mass)',
        cdf_title='Cumulative Distribution Functions (CDFs)'
    )
    plotter.show()

    # --- Optional: Example of using general W1 distance with explicit (xs, ys) pairs ---
    print("\n--- Example of using general W1 distance with explicit (xs, ys) pairs ---")
    # Distribution 1: sampled at specific x-points
    xs_data1 = np.array([-1, 0, 0.5, 1.2], dtype=float)
    ys_data1 = np.array([0.1, 0.5, 0.3, 0.1], dtype=float) # Intensities at xs_data1
    
    # Distribution 2: sampled at different x-points
    xs_data2 = np.array([-0.5, 0.8, 1.5, 2.0], dtype=float)
    ys_data2 = np.array([0.2, 0.2, 0.4, 0.2], dtype=float) # Intensities at xs_data2
    
    from wasserstein_distance import wasserstein_1d_general # Correct import
    dist_general_example = wasserstein_1d_general(xs_data1, ys_data1, xs_data2, ys_data2)
    if not np.isnan(dist_general_example):
        print(f"General W1 distance for sparse (xs,ys) example: {dist_general_example:.4f}")
    else:
        print("General W1 distance calculation failed for sparse (xs,ys) example.")