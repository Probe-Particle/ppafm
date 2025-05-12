#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline

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
    x, y, dist = create_line_coordinates( start, end, points_per_angstrom=pointPerAngstrom)
    nps = len(x)
    
    # Interpolate experimental data
    #print("Interpolating experimental data...")
    T0 = time.perf_counter()
    points = np.column_stack((x, y))
    data_1d = interpolate_3d_plane_fast(X, Y, biases, data, points)
    print(f"plot_exp_voltage_line_scan(): {time.perf_counter() - T0:.5f} [s]")
    
    # Plot experimental dI/dV if axis is provided
    if ax is not None:
        #print("Creating experimental plot...")
        if cmap == 'bwr':
            vmax = np.max(np.abs(data_1d))
            vmin = -vmax
        else:
            vmax = None
            vmin = None
        extent = [dist[0], dist[-1], biases[0], biases[-1]]
        im = ax.imshow(data_1d, aspect='auto', origin='lower', cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, interpolation='nearest')
        # Ensure voltage axis spans from zero to max for comparison
        if ylims is not None:
            ax.set_ylim(ylims)
        title = 'Experimental dI/dV'
        if title:
            title += f' {title}'
        ax.set_title(title)
        ax.set_xlabel('Distance (Å)')
        ax.set_ylabel('Bias Voltage (V)')
        # Return the image handle separately
        return im, (data_1d, dist)
    
    return data_1d, dist


def plot_experimental_data(exp_X, exp_Y, exp_dIdV, exp_I, exp_biases, idx, params=None, sim_data=None, ellipse_params=None,  ax_didv=None, ax_current=None, ax_overlay=None, draw_exp_scan_line_func=None):
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
        im1 = ax_didv.imshow(exp_dIdV[idx], aspect='equal', origin='lower', 
                            cmap='seismic', vmin=-maxval, vmax=maxval, extent=exp_extent)
        ax_didv.set_title(f'Exp. dI/dV at {exp_biases[idx]:.3f} V')
        ax_didv.set_xlabel('X [Å]')
        ax_didv.set_ylabel('Y [Å]')
        
        # Plot ellipses if parameters provided
        if ellipse_params is not None:
            plot_ellipses(ax_didv, ellipse_params)
        
        # Draw scan line if function provided
        if draw_exp_scan_line_func is not None:
            draw_exp_scan_line_func(ax_didv)
    
    if ax_current is not None:
        # Plot Current
        ax_current.clear()
        im2 = ax_current.imshow(exp_I[idx], aspect='equal', origin='lower',  cmap='inferno', vmin=0.0, vmax=600.0, extent=exp_extent)
        ax_current.set_title(f'Exp. Current at {exp_biases[idx]:.3f} V')
        ax_current.set_xlabel('X [Å]')
        ax_current.set_ylabel('Y [Å]')
    
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
