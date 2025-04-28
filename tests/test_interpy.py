import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
#import pyProbeParticle.interpy as interpy
from pyProbeParticle.InterpolatorRBF     import InterpolatorRBF
from pyProbeParticle.InterpolatorKriging import InterpolatorKriging

def check_interp_accuracy( interp, data_vals, points ):    
    if interp.update_weights(data_vals):
        y_samp = interp.evaluate(points)
        print("Original | Interpolated | Difference")
        for i in range(ndata):
            print(f"{data_vals[i]:<9.4f}| {y_samp[i]:<13.4f}| {np.abs(data_vals[i] - y_samp[i]):.4e}")
    else:
        print("Failed to update weights for RBF.")

'''
if __name__ == "__main__":
    # Example Data Points (Geometry)
    data_points = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5],
        [2, 0], [2, 1], [0, 2], [1, 2], [2, 2],
        [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]
    ])
    ndata = data_points.shape[0]

    # Define RBF/Covariance support radius
    # Make it large enough for some points to be neighbors, but smaller than the domain extent
    R_basis = 1.1 # e.g., slightly larger than diagonal of unit square cells

    data_vals1 = np.sin(   data_points[:, 0] * np.pi) * np.cos(data_points[:, 1] * np.pi) # Example values 1
    data_vals2 = np.exp(-((data_points[:, 0]-1.0)**2 + (data_points[:, 1]-1.0)**2) / 0.5) # Example values 2 (Gaussian peak)

    rbf     = InterpolatorRBF(data_points, R_basis)
    kriging = InterpolatorKriging(data_points, R_basis)

    #nx,ny = 10,10
    #x_grid, y_grid = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 2, ny))
    #query_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    check_interp( rbf, data_vals1,     data_points )
    check_interp( rbf, data_vals2,     data_points )

    check_interp( kriging, data_vals1, data_points )
    check_interp( kriging, data_vals2, data_points )



'''


# --- New plotting function ---
def plot_interpolator_grid(interpolator, data_points, data_vals, grid_bounds, nx=50, ny=50, title=None, ax=None):
    """
    Samples the interpolator on a 2D grid and plots the result.

    Args:
        interpolator: An initialized InterpolatorRBF or InterpolatorKriging object.
        data_points: (N, 2) array of original data point locations.
        data_vals: (N,) array of original data values.
        grid_bounds: Tuple (xmin, xmax, ymin, ymax) for the grid.
        nx, ny: Resolution of the grid.
        title: Optional title for the plot.
        ax: Optional matplotlib Axes object to plot on.
    """
    print(f"\n--- Plotting grid for {type(interpolator).__name__} ---")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure # Get figure from axes

    # 1. Update interpolator weights/coefficients with the current data values
    if not interpolator.update_weights(data_vals):
        print(f"ERROR: Failed to update weights for {title or type(interpolator).__name__}. Skipping plot.")
        ax.set_title(f"{title or type(interpolator).__name__}\n(Weight Update Failed)")
        return

    # 2. Create the grid
    xmin, xmax, ymin, ymax = grid_bounds
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    Xs, Ys = np.meshgrid(xs, ys)
    query_points = np.vstack([Xs.ravel(), Ys.ravel()]).T

    # 3. Evaluate the interpolator on the grid
    interpolated_grid_vals = interpolator.evaluate(query_points)

    if interpolated_grid_vals is None:
        print(f"ERROR: Evaluation failed for {title or type(interpolator).__name__}. Skipping plot.")
        ax.set_title(f"{title or type(interpolator).__name__}\n(Evaluation Failed)")
        return

    # 4. Reshape the results for plotting
    Z = interpolated_grid_vals.reshape((ny, nx))

    # 5. Plot the interpolated grid
    extent = (xmin, xmax, ymin, ymax)
    im = ax.imshow(Z, origin='lower', extent=extent, aspect='auto', cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Interpolated Value')

    # 6. Plot the original data points
    scatter = ax.scatter(data_points[:, 0], data_points[:, 1], c=data_vals, cmap='viridis', edgecolors='k', s=50, label='Data Points')
    # Add a second colorbar for the scatter points if desired, or just use edgecolors
    # fig.colorbar(scatter, ax=ax, label='Original Data Value') # Can be confusing with imshow cmap
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{type(interpolator).__name__} Interpolation")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)


if __name__ == "__main__":
    # Example Data Points (Geometry)
    data_points = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5],
        [2, 0], [2, 1], [0, 2], [1, 2], [2, 2],
        [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]
    ])
    ndata = data_points.shape[0]

    # Define RBF/Covariance support radius
    R_basis = 1.1 # e.g., slightly larger than diagonal of unit square cells

    # Example data values
    data_vals1 = np.sin(   data_points[:, 0] * np.pi) * np.cos(data_points[:, 1] * np.pi) # Example values 1
    data_vals2 = np.exp(-((data_points[:, 0]-1.0)**2 + (data_points[:, 1]-1.0)**2) / 0.5) # Example values 2 (Gaussian peak)

    # --- Initialize Interpolators ---
    rbf     = InterpolatorRBF(data_points, R_basis)
    kriging = InterpolatorKriging(data_points, R_basis)

    # --- Check Accuracy at Data Points (Optional but good practice) ---
    check_interp_accuracy( rbf, data_vals1, data_points )
    check_interp_accuracy( rbf, data_vals2, data_points )
    check_interp_accuracy( kriging, data_vals1, data_points )
    check_interp_accuracy( kriging, data_vals2, data_points )

    # --- Plotting ---
    grid_bounds = ( -0.5, 2.5, -0.5, 2.5 ) # Define plot area slightly larger than data
    nx_plot, ny_plot = 100, 100           # Define grid resolution for plots

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel() # Flatten axes array for easy indexing
    plot_interpolator_grid(rbf,     data_points, data_vals1, grid_bounds, nx=nx_plot, ny=ny_plot, title="RBF Interpolation (sin*cos data)",      ax=axes[0])
    plot_interpolator_grid(rbf,     data_points, data_vals2, grid_bounds, nx=nx_plot, ny=ny_plot, title="RBF Interpolation (Gaussian data)",     ax=axes[1])
    plot_interpolator_grid(kriging, data_points, data_vals1, grid_bounds, nx=nx_plot, ny=ny_plot, title="Kriging Interpolation (sin*cos data)",  ax=axes[2])
    plot_interpolator_grid(kriging, data_points, data_vals2, grid_bounds, nx=nx_plot, ny=ny_plot, title="Kriging Interpolation (Gaussian data)", ax=axes[3])

    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.show()




   



