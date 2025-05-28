import numpy as np
from mayavi import mlab
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from exp_utils import load_experimental_data

def plot_mayavi_isosurface(filename='exp_rings_data.npz'):
    # Load experimental data
    exp_X, exp_Y, exp_dIdV, exp_I, exp_biases = load_experimental_data(filename=filename)

    # Mayavi expects data in (Z, Y, X) order for mlab.contour3d if we want Z to be bias
    # Our exp_dIdV is (n_biases, n_y, n_x)
    # So, we need to transpose it to (n_x, n_y, n_biases) if we want X, Y, Bias as axes
    # Or, we can keep it as (n_biases, n_y, n_x) and map axes accordingly.
    
    # Let's assume exp_dIdV is (V, Y, X) and we want to plot it as (X, Y, V)
    # So, we need to reorder dimensions to (X, Y, V) for Mayavi's mlab.contour3d
    # This means transposing exp_dIdV from (n_biases, n_y, n_x) to (n_x, n_y, n_biases)
    
    # First, create a 3D grid for the data values. Mayavi's contour3d can take a direct numpy array.
    # However, the coordinates need to be set up correctly.
    
    # Get unique X and Y coordinates
    unique_x = np.unique(exp_X)
    unique_y = np.unique(exp_Y)

    # Create a 3D grid for the data, ensuring the order is (X, Y, Bias)
    # exp_dIdV is (n_biases, n_y, n_x)
    # We want to reshape it to (n_x, n_y, n_biases)
    
    # Mayavi's mlab.contour3d expects data in (Z, Y, X) order if you don't provide explicit coordinates.
    # If we want X, Y, Bias, then the data array should be (Bias, Y, X) and then we map axes.
    # Or, we can transpose it to (X, Y, Bias) and then plot.
    
    # Let's keep exp_dIdV as (n_biases, n_y, n_x) and define the extent for mlab.contour3d
    # mlab.contour3d(data, extent=[xmin, xmax, ymin, ymax, zmin, zmax])
    
    # Define the extent for the plot
    xmin, xmax = np.min(unique_x), np.max(unique_x)
    ymin, ymax = np.min(unique_y), np.max(unique_y)
    zmin, zmax = np.min(exp_biases), np.max(exp_biases)

    # Apply a scaling factor to the Z-axis (Bias) to adjust aspect ratio.
    # Increase this value to make the Z-axis appear more stretched (less flattened).
    # Decrease this value to make the Z-axis appear more compressed (more flattened).
    z_scale_factor = 80.0 # Adjusted from 40.0
    zmin_scaled = zmin * z_scale_factor
    zmax_scaled = zmax * z_scale_factor

    # Define Isosurface Thresholds
    positive_cutoff = 0.5 * np.max(exp_dIdV[exp_dIdV > 0]) if np.any(exp_dIdV > 0) else 0.1
    negative_cutoff = 0.5 * np.min(exp_dIdV[exp_dIdV < 0]) if np.any(exp_dIdV < 0) else -0.1

    # Fallback if no positive/negative values exist
    if not np.any(exp_dIdV > 0): positive_cutoff = np.max(exp_dIdV) * 0.8
    if not np.any(exp_dIdV < 0): negative_cutoff = np.min(exp_dIdV) * 0.8

    isosurface_values = [negative_cutoff, positive_cutoff]

    # Create the Mayavi Figure
    mlab.figure(size=(800, 600), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

    # Use mlab.contour3d directly on the numpy array
    # The data is (n_biases, n_y, n_x). Mayavi will treat the first dimension as Z.
    # So, Z-axis will be Bias, Y-axis will be Y, X-axis will be X.
    # We need to transpose to (n_x, n_y, n_biases) if we want X, Y, Bias as axes.
    # Let's transpose exp_dIdV to (n_x, n_y, n_biases) to match X, Y, Bias axes.
    # This assumes exp_X is (ny, nx) and exp_Y is (ny, nx)
    
    # To get data in (X, Y, Bias) order, we need to reorder exp_dIdV
    # exp_dIdV.shape is (n_biases, n_y, n_x)
    # We want new_data.shape to be (n_x, n_y, n_biases)
    # So, new_data[x_idx, y_idx, bias_idx] = exp_dIdV[bias_idx, y_idx, x_idx]
    
    # Create a new array with the desired shape
    n_biases, n_y, n_x = exp_dIdV.shape
    data_mayavi = np.zeros((n_x, n_y, n_biases))
    for v_idx in range(n_biases):
        for y_idx in range(n_y):
            for x_idx in range(n_x):
                data_mayavi[x_idx, y_idx, v_idx] = exp_dIdV[v_idx, y_idx, x_idx]

    # Now, the extent should match the new data_mayavi dimensions
    # extent=[xmin, xmax, ymin, ymax, zmin, zmax] -> [X_min, X_max, Y_min, Y_Max, Bias_min_scaled, Bias_max_scaled]
    extent_mayavi = [xmin, xmax, ymin, ymax, zmin_scaled, zmax_scaled]

    mlab.contour3d(data_mayavi, contours=isosurface_values, colormap='RdBu', opacity=0.5, extent=extent_mayavi)

    # Add Enhancements
    mlab.colorbar(title='dIdV Value', orientation='vertical')
    mlab.axes(xlabel='X [Å]', ylabel='Y [Å]', zlabel='Bias [V] (Scaled)')
    mlab.title("Mayavi Isosurface Plot of dIdV Data (Positive and Negative)")

    # Display the Plot
    mlab.show()

if __name__ == '__main__':
    plot_mayavi_isosurface()
