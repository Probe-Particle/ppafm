#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means
from skimage.restoration import estimate_sigma
from skimage.restoration import denoise_tv_chambolle


import plot_utils as pu


# Functions for experimental data handling

def load_experimental_data(filename='exp_rings_data.npz'):
    """Load experimental data from npz file
    
    Args:
        filename: Path to npz file with experimental data
        
    Returns:
        Tuple of (exp_X, exp_Y, exp_dIdV, exp_I, exp_biases)
    """
    data = np.load(filename)
    # Convert from nm to Å (1 nm = 10 Å)
    exp_X = data['X'] * 10
    exp_Y = data['Y'] * 10
    exp_dIdV = data['dIdV']
    exp_I = data['I']
    exp_biases = data['biases']
    center_x = data['center_x'] * 10  # Convert to Å
    center_y = data['center_y'] * 10  # Convert to Å
    
    # Center the coordinates
    exp_X -= center_x
    exp_Y -= center_y
    
    return exp_X, exp_Y, exp_dIdV, exp_I, exp_biases


def interpolate_3d_plane_slow(xs, ys, zs, vals, line_points):
    """Interpolate 3D data along a 2D line using LinearNDInterpolator (slow but works for any grid)
    
    Args:
        xs: X coordinates (shape: [nz, ny, nx] or [nz, n])
        ys: Y coordinates (shape: [nz, ny, nx] or [nz, n])
        zs: Z values (bias voltages, shape: [nz])
        vals: Data values (shape: [nz, ny, nx] or [nz, n])
        line_points: Points along line to interpolate (shape: [npoints, 2])
        
    Returns:
        Interpolated values along line (shape: [nz, npoints])
    """
    # We'll process each voltage independently
    npoints = len(line_points)
    exp_didv = np.zeros((len(zs), npoints))
    for i in range(len(zs)):
        exp_x = xs[i]
        exp_y = ys[i]
        points = np.column_stack((exp_x.flatten(), exp_y.flatten()))
        exp_data = vals[i]
        values = exp_data.flatten()
        # Create interpolator
        interp = LinearNDInterpolator(points, values)
        # Evaluate at all points along the line at once
        exp_didv[i,:] = interp(line_points)  # LinearNDInterpolator takes points, not separate x,y arrays
    return exp_didv


def interpolate_3d_plane_fast(xs, ys, zs, vals, line_points):
    """Simple fast interpolation assuming data is on a regular grid
    
    Args:
        xs: X coordinates (shape: [nz, ny, nx])
        ys: Y coordinates (shape: [nz, ny, nx])
        zs: Z values (bias voltages, shape: [nz])
        vals: Data values (shape: [nz, ny, nx])
        line_points: Points along line to interpolate (shape: [npoints, 2])
        
    Returns:
        Interpolated values along line (shape: [nz, npoints])
    """
    npoints = len(line_points)
    x_coords = line_points[:, 0]  # Extract x coordinates from points
    y_coords = line_points[:, 1]  # Extract y coordinates from points
    exp_didv = np.zeros((len(zs), npoints))

    # Get min/max of x and y from the coordinate arrays
    ny, nx = vals[0].shape  
    x_min, x_max = np.min(xs[0]), np.max(xs[0])
    y_min, y_max = np.min(ys[0]), np.max(ys[0])
    
    # Create evenly spaced grid coordinates matching the data dimensions
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)

    # Process each voltage independently
    for i in range(len(zs)):
        interp = RectBivariateSpline(y_grid, x_grid, vals[i])
        exp_didv[i,:] = interp.ev(y_coords, x_coords)
    
    return exp_didv


def create_line_coordinates(start_point, end_point, npoints=None, points_per_angstrom=5):
    """Create line coordinates between two points
    
    Args:
        start_point: Starting point (x, y)
        end_point: Ending point (x, y)
        npoints: Number of points, if None calculated based on distance
        points_per_angstrom: Points per Angstrom if npoints is None
        
    Returns:
        Tuple of (x, y, distance) arrays
    """
    x1, y1 = start_point
    x2, y2 = end_point
    
    # Calculate distance
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # Calculate number of points if not provided
    if npoints is None:
        npoints = max(100, int(dist * points_per_angstrom))
    
    # Create line coordinates
    x = np.linspace(x1, x2, npoints)
    y = np.linspace(y1, y2, npoints)
    distance = np.sqrt((x - x[0])**2 + (y - y[0])**2)
    
    return x, y, distance


def resample_to_simulation_grid(data, src_extent, target_size=100, target_extent=(-20, 20, -20, 20)):
    """Resample data to match simulation grid and extent
    
    Args:
        data: Source data array
        src_extent: Source extent [xmin, xmax, ymin, ymax]
        target_size: Size of target grid (assumed square)
        target_extent: Target extent [xmin, xmax, ymin, ymax]
    
    Returns:
        Resampled data array matching simulation grid
    """
    # Create source coordinate grids
    x_src = np.linspace(src_extent[0], src_extent[1], data.shape[1])
    y_src = np.linspace(src_extent[2], src_extent[3], data.shape[0])
    
    # Create target coordinate grids
    x_target = np.linspace(target_extent[0], target_extent[1], target_size)
    y_target = np.linspace(target_extent[2], target_extent[3], target_size)
    
    # Create interpolator
    interpolator = RectBivariateSpline(y_src, x_src, data)
    resampled = interpolator(y_target, x_target, grid=True).reshape(len(y_target), len(x_target))
    
    # Create mask for points outside source extent
    xx, yy = np.meshgrid(x_target, y_target)
    mask = ((xx < src_extent[0]) | (xx > src_extent[1]) | 
            (yy < src_extent[2]) | (yy > src_extent[3]))
    
    # Set points outside source extent to zero
    resampled[mask] = 0
    
    return resampled


def create_overlay_image(exp_data, sim_data, exp_extent, sim_extent):
    """Create RGB overlay of experimental and simulation data
    
    Args:
        exp_data: Experimental dI/dV data
        sim_data: Simulated charge data
        exp_extent: Extent of experimental data [xmin, xmax, ymin, ymax]
        sim_extent: Extent of simulation data [xmin, xmax, ymin, ymax]
        
    Returns:
        Tuple of (rgb_image, extent)
    """
    # Resample experimental data to simulation grid
    exp_resampled = resample_to_simulation_grid(
        exp_data, 
        exp_extent,
        target_size=sim_data.shape[0],
        target_extent=sim_extent
    )
    
    # Normalize experimental data to [0,1] range for red channel
    exp_norm = (exp_resampled - np.min(exp_resampled)) / (np.max(exp_resampled) - np.min(exp_resampled))
    
    # Normalize simulation data to [0,1] range for green channel
    sim_norm = (sim_data - np.min(sim_data)) / (np.max(sim_data) - np.min(sim_data))
    
    # Create RGB image (Red: experimental, Green: simulation, Blue: zeros)
    rgb_image = np.zeros((*sim_data.shape, 3))
    rgb_image[..., 0] = exp_norm  # Red channel: experimental
    rgb_image[..., 1] = sim_norm  # Green channel: simulation
    
    return rgb_image, sim_extent


def get_exp_voltage_line_scan(X, Y, data, biases, start, end, pointPerAngstrom=5):
    """Get experimental dI/dV along a line scan for different voltages"""
    x, y, dist = create_line_coordinates(start, end, points_per_angstrom=pointPerAngstrom)
    points  = np.column_stack((x, y))
    data_1d = interpolate_3d_plane_fast(X, Y, biases, data, points)
    return data_1d, dist, x, y

def plot_exp_voltage_line_scan(X, Y, data, biases, start, end, ax=None, title='', ylims=None, pointPerAngstrom=5, cmap='bwr'):
    """Plot experimental dI/dV along a line scan for different voltages
    
    Args:
        X: Experimental X coordinates
        Y: Experimental Y coordinates
        data: Experimental dI/dV data
        biases: Experimental bias voltages
        start: Start point for experimental line
        end: End point for experimental line
        ax: Matplotlib axis for plotting (required)
        title: Optional suffix for plot title
        pointPerAngstrom: Points per Angstrom for interpolation
        
    Returns:
        Tuple of (exp_didv, exp_distance) - interpolated data and distance array
    """
    # Create line coordinates for experiment
    # x, y, dist = create_line_coordinates( start, end, points_per_angstrom=pointPerAngstrom)
    # T0      = time.perf_counter()
    # points  = np.column_stack((x, y))
    # data_1d = interpolate_3d_plane_fast(X, Y, biases, data, points)
    # print(f"plot_exp_voltage_line_scan(): {time.perf_counter() - T0:.5f} [s]")

    data_1d, dist, x, y = get_exp_voltage_line_scan(X, Y, data, biases, start, end, pointPerAngstrom=pointPerAngstrom)
    
    # Plot experimental dI/dV if axis is provided
    if ax is not None:
        #print("Creating experimental plot...")
        # Ensure biases are sorted for extent to avoid ValueError in imshow
        sorted_biases = np.sort(biases)
        extent = [dist[0], dist[-1], sorted_biases[0], sorted_biases[-1]]
        plot_title = 'Experimental dI/dV'
        if title:
            plot_title += f' {title}'
        ax = pu.plot_imshow(ax, data_1d, title=plot_title, extent=extent, cmap=cmap, xlabel='Distance (Å)', ylabel='Bias Voltage (V)', aspect='auto')
        im = ax.images[-1]  # Get the image from the axis
        # Ensure voltage axis spans from zero to max for comparison
        if ylims is not None:
            ax.set_ylim(ylims)
        # Return the image handle separately
        return im, (data_1d, dist)
    
    return data_1d, dist


def plot_experimental_data(exp_X, exp_Y, exp_dIdV, exp_I, exp_biases, idx, params=None, sim_data=None, ellipse_params=None,  ax_didv=None, ax_current=None, ax_overlay=None, draw_exp_scan_line_func=None, cmap_STM='inferno', cmap_dIdV='bwr'):
    """Plot experimental data
    
    Args:
        exp_X: Experimental X coordinates
        exp_Y: Experimental Y coordinates
        exp_dIdV: Experimental dI/dV data
        exp_I: Experimental current data
        exp_biases: Experimental bias voltages
        idx: Index into bias voltages to plot
        params: Dictionary of parameters (optional, for plot extents)
        sim_data: Simulation data for overlay (optional)
        ellipse_params: Parameters for ring/ellipse visualization (optional)
        axes: List of 3 matplotlib axes [ax_didv, ax_current, ax_overlay] (optional)
        draw_exp_scan_line_func: Function to draw scan line on axes (optional)
        
    Returns:
        Tuple of (ax_didv, ax_current, ax_overlay) axes
    """
    import matplotlib.pyplot as plt
    
    # # Create axes if not provided
    # if axes is None or len(axes) < 3:
    #     fig, axes_new = plt.subplots(1, 3, figsize=(15, 5))
    #     ax_didv, ax_current, ax_overlay = axes_new
    # else:
    #     ax_didv, ax_current, ax_overlay = axes
    #     for ax in [ax_didv, ax_current, ax_overlay]:
    #         ax.clear()
    
    # Get plot extents
    xmin, xmax = np.min(exp_X[0]), np.max(exp_X[0])
    ymin, ymax = np.min(exp_Y[0]), np.max(exp_Y[0])
    exp_extent = [xmin, xmax, ymin, ymax]
    
    # Get simulation extent if params provided
    if params is not None and 'L' in params:
        L = params['L']
        sim_extent = [-L, L, -L, L]
    else:
        # Default to experimental extent
        sim_extent = exp_extent
    
    if ax_didv is not None:
        # Plot dI/dV
        ax_didv.clear()
        maxval = np.max(np.abs(exp_dIdV[idx]))
        ax_didv = pu.plot_imshow(ax_didv, exp_dIdV[idx], title=f'Exp. dI/dV at {exp_biases[idx]:.3f} V', extent=exp_extent, cmap=cmap_dIdV, vmin=-maxval, vmax=maxval, xlabel='X [Å]', ylabel='Y [Å]')
        
        # Plot ellipses if parameters provided
        if ellipse_params is not None:plot_ellipses(ax_didv, ellipse_params)
        # Draw scan line if function provided
        if draw_exp_scan_line_func is not None:  draw_exp_scan_line_func(ax_didv)
    
    if ax_current is not None:
        # Plot Current
        ax_current.clear()
        ax_current = pu.plot_imshow(ax_current, exp_I[idx], title=f'Exp. Current at {exp_biases[idx]:.3f} V', extent=exp_extent, cmap=cmap_STM, vmin=0.0, vmax=600.0, xlabel='X [Å]', ylabel='Y [Å]')
    
    # Create and plot overlay if simulation data available
    if ax_overlay is not None:
        if sim_data is not None:
            # Create RGB overlay
            rgb_overlay, extent = create_overlay_image(exp_dIdV[idx], sim_data, exp_extent, sim_extent)
            
            ax_overlay.clear()
            ax_overlay.imshow(rgb_overlay, aspect='equal', origin='lower', extent=extent)
            ax_overlay.set_title('Overlay (Red: Exp, Green: Sim)')
            ax_overlay.set_xlabel('X [Å]')
            ax_overlay.set_ylabel('Y [Å]')
        else:
            ax_overlay.set_title('No simulation data for overlay')
            ax_overlay.grid(True)
    
    return ax_didv, ax_current, ax_overlay


def plot_ellipses(ax, params):
    """Plot ellipses for quantum dot visualization
    
    Args:
        ax: Matplotlib axis to plot on
        params: Dictionary with parameters for ellipse visualization
    """
    if 'nsite' not in params or 'radius' not in params:
        return
    
    nsite = params['nsite']
    radius = params['radius']
    phiRot = params.get('phiRot', 0)
    R_major = params.get('R_major', 1.0)
    R_minor = params.get('R_minor', 1.0)
    phi0_ax = params.get('phi0_ax', 0)
    
    # Number of points for ellipse
    n = 100
    
    for i in range(nsite):
        # Calculate quantum dot position
        phi = phiRot + i * 2 * np.pi / nsite
        dir_x = np.cos(phi)
        dir_y = np.sin(phi)
        qd_pos_x = radius * dir_x
        qd_pos_y = radius * dir_y
        
        # Calculate ellipse orientation
        phi_ax = phi + phi0_ax
        t = np.linspace(0, 2*np.pi, n)
        
        # Create ellipse in local coordinates
        x_local = R_major * np.cos(t)
        y_local = R_minor * np.sin(t)
        
        # Rotate ellipse
        x_rot = x_local * np.cos(phi_ax) - y_local * np.sin(phi_ax)
        y_rot = x_local * np.sin(phi_ax) + y_local * np.cos(phi_ax)
        
        # Translate to quantum dot position
        x = x_rot + qd_pos_x
        y = y_rot + qd_pos_y
        
        # Plot ellipse
        ax.plot(x, y, ':', color='white', alpha=0.8, linewidth=1)
        
        # Plot center point
        ax.plot(qd_pos_x, qd_pos_y, '+', color='white', markersize=5)


def load_and_extract_experimental_data(filename='exp_rings_data.npz', start_point=None, end_point=None, pointPerAngstrom=5, verbosity=1):
    """
    Load and extract experimental data along specified line
    Returns: (exp_STM, exp_dIdV, exp_dist, exp_biases)
    """
    print(f"Loading experimental data from {filename}..." if verbosity>0 else "", end='')
    data = np.load(filename)
    
    # Process 3D experimental data
    X = data['X'] * 10  # nm to Å
    Y = data['Y'] * 10
    dIdV = data['dIdV']
    I = data['I']
    biases = data['biases']
    
    # Center coordinates
    cx, cy = data['center_x']*10, data['center_y']*10
    X -= cx
    Y -= cy
        
    exp_STM, exp_dist, exp_x, exp_biases = get_exp_voltage_line_scan(X, Y, I,    biases, start_point, end_point, pointPerAngstrom=pointPerAngstrom)
    exp_dIdV, _ , _, _                   = get_exp_voltage_line_scan(X, Y, dIdV, biases, start_point, end_point, pointPerAngstrom=pointPerAngstrom)
    
    if verbosity>0:
        print(f"Loaded experimental data. Shapes: STM={exp_STM.shape}, dIdV={exp_dIdV.shape}")
    
    return exp_STM, exp_dIdV, exp_dist, biases

def plot2d(data, ax=None, extent=None, title=None, xlabel='Distance [Å]', ylabel='Bias [V]', bCbar=True, cmap='hot'):
    if ax is None: fig, ax = plt.subplots()
    ax = pu.plot_imshow(ax, data, title=title, extent=extent, cmap=cmap, xlabel=xlabel, ylabel=ylabel, bGrid=False)
    if bCbar:
        im = ax.images[-1]  # Get the image from the axis
        ax.figure.colorbar(im, ax=ax, label='Current [nA]')
    return ax.images[-1]

def visualize_experimental_data(exp_STM, exp_dIdV, exp_dist, exp_biases):
    """
    Create figure to visualize experimental data
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot2d(exp_STM,  ax=axes[0], extent=[exp_dist[0], exp_dist[-1], exp_biases[0], exp_biases[-1]], title='Experimental STM', xlabel='Distance [Å]', ylabel='Bias [V]', bCbar=True)
    plot2d(exp_dIdV, ax=axes[1], extent=[exp_dist[0], exp_dist[-1], exp_biases[0], exp_biases[-1]], title='Experimental dIdV', xlabel='Distance [Å]', ylabel='Bias [V]', bCbar=True, cmap='bwr')
    plt.tight_layout()
    return fig


# --- Denoising Functions ---

def denoise_gaussian(data_3d, sigma=1.0):
    """Applies a Gaussian filter for denoising.

    Args:
        data_3d (np.ndarray): The 3D data array to denoise.
        sigma (float): Standard deviation for Gaussian kernel. Larger values mean more smoothing.

    Returns:
        np.ndarray: Denoised 3D data array.
    """
    print(f"Applying Gaussian filter with sigma={sigma}")
    return gaussian_filter(data_3d, sigma=sigma)



def denoise_nl_means(data_3d, h_factor=0.8, patch_size=5, patch_distance=7, fast_mode=True):
    """Applies Non-Local Means denoising.

    Args:
        data_3d (np.ndarray): The 3D data array to denoise.
        h_factor (float): Factor to determine denoising strength 'h'.
                          h = h_factor * estimated_noise_std.
        patch_size (int): Size of the square patch extracted around each voxel.
        patch_distance (int): Maximum distance within which to search for similar patches.
        fast_mode (bool): If True, uses a faster, approximate algorithm.

    Returns:
        np.ndarray: Denoised 3D data array.
    """
    from skimage.restoration import denoise_nl_means
    from skimage.restoration import estimate_sigma

    # Estimate noise standard deviation using a robust method
    # For 3D data, estimate_sigma can be applied to slices or the whole volume
    # For a single 2D slice expanded to (1, H, W), we can estimate sigma on the 2D plane
    if data_3d.shape[0] == 1: # If it's effectively a 2D image
        noise_level = estimate_sigma(data_3d[0,:,:], average_sigmas=True)
    else:
        noise_level = estimate_sigma(data_3d, average_sigmas=True)

    if noise_level == 0:
        noise_level = 1e-6 # Avoid division by zero if data is constant

    print(f"Estimated noise_level: {noise_level:.4f}")
    h = h_factor * noise_level
    print(f"Applying Non-Local Means filter with h={h:.3f}, patch_size={patch_size}, patch_distance={patch_distance}")
    return denoise_nl_means(
        data_3d,
        h=h,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=fast_mode,
        preserve_range=True # Important to keep data range consistent
    )

def apply_image_processing_method(data_3d, method_name, method_params):
    """
    Applies a specified image processing method to 3D data.

    Args:
        data_3d (np.ndarray): Input 3D data array (can be 1xHxW for 2D images).
        method_name (str): Name of the method to apply ('nl_means', 'tv_denoising').
        method_params (dict): Dictionary of parameters specific to the chosen method.

    Returns:
        np.ndarray: Processed 3D data array.
    """
    # Handle 1xHxW input by processing as 2D and expanding back if necessary
    is_2d_input = (data_3d.shape[0] == 1)
    if is_2d_input:
        data_2d = data_3d[0, :, :]
    else:
        data_2d = None # Not used for true 3D data

    processed_data = None

    if method_name == 'nl_means':
        # NL-means still uses h_factor and noise_level estimation
        h_factor = method_params.get('h_factor', 0.1)
        patch_size = method_params.get('patch_size', 7)
        patch_distance = method_params.get('patch_distance', 21)
        fast_mode = method_params.get('fast_mode', True)

        # Re-use the existing denoise_nl_means logic for noise estimation
        # This part is a bit redundant but keeps the original logic intact for NL-means
        if is_2d_input:
            noise_level = estimate_sigma(data_2d, average_sigmas=True)
        else:
            noise_level = estimate_sigma(data_3d, average_sigmas=True)

        if noise_level == 0:
            noise_level = 1e-6

        h = h_factor * noise_level
        print(f"Applying Non-Local Means filter with h={h:.3f}, patch_size={patch_size}, patch_distance={patch_distance}")
        processed_data = denoise_nl_means(
            data_3d, # Pass 3D data to original denoise_nl_means
            h_factor=h_factor, # Pass h_factor to original denoise_nl_means
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=fast_mode
        ) # The original denoise_nl_means will handle preserve_range

    elif method_name == 'tv_denoising':
        weight = method_params.get('weight', 1.0) # Denoising weight. The larger weight, the more denoising (at the expense of fidelity).
        print(f"Applying Total Variation Denoising with weight={weight:.3f}")
        if is_2d_input:
            processed_data = denoise_tv_chambolle(data_2d, weight=weight)
        else:
            processed_data = denoise_tv_chambolle(data_3d, weight=weight)

    else:
        raise ValueError(f"Unknown image processing method: {method_name}")

    # Ensure output is 3D if input was 3D (even if 1xHxW)
    if is_2d_input and processed_data.ndim == 2: # If it was 1xHxW input and output is 2D
        return np.expand_dims(processed_data, axis=0)
    else:
        return processed_data

# Main test function
if __name__ == "__main__":
    # Example usage
    exp_X, exp_Y, exp_dIdV, exp_I, exp_biases = load_experimental_data()
    
    # Display info
    print(f"Loaded experimental data:")
    print(f"  X shape: {exp_X.shape}")
    print(f"  Y shape: {exp_Y.shape}")
    print(f"  dIdV shape: {exp_dIdV.shape}")
    print(f"  I shape: {exp_I.shape}")
    print(f"  Biases: {exp_biases.shape}, range: {exp_biases[0]:.3f} to {exp_biases[-1]:.3f} V")
    
    # Plot example for one voltage
    idx = len(exp_biases) // 2
    params = {
        'L': 20.0,
        'nsite': 3,
        'radius': 5.2,
        'phiRot': 0.8,
        'R_major': 8.0,
        'R_minor': 10.0,
        'phi0_ax': 0.2,
    }
    
    # Plot experimental data
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    plot_experimental_data(exp_X, exp_Y, exp_dIdV, exp_I, exp_biases, idx, params, axes=[ax1, ax2, ax3])
    plt.tight_layout()
    plt.show()
    
    # Example of line scan
    exp_start_point = (-10.0, 10.0)
    exp_end_point = (10.0, -10.0)
    fig, ax1, ax2 = plot_voltage_line_scan(exp_X, exp_Y, exp_dIdV, exp_biases, exp_start_point, exp_end_point)
    plt.tight_layout()
    plt.show()
