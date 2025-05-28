import numpy as np
from mayavi import mlab
import sys
import os
import vtk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from exp_utils import load_experimental_data, denoise_gaussian, denoise_nl_means

def plot_mayavi_volume(filename='exp_rings_data.npz', denoise_method='gaussian', denoise_params=None):
    # Load experimental data
    exp_X, exp_Y, exp_dIdV, exp_I, exp_biases = load_experimental_data(filename=filename)

    # Prepare data for Mayavi (transpose to X, Y, Bias)
    n_biases, n_y, n_x = exp_dIdV.shape
    data_mayavi = np.zeros((n_x, n_y, n_biases))
    for v_idx in range(n_biases):
        for y_idx in range(n_y):
            for x_idx in range(n_x):
                data_mayavi[x_idx, y_idx, v_idx] = exp_dIdV[v_idx, y_idx, x_idx]

    # Apply denoising if specified
    if denoise_method == 'gaussian':
        data_mayavi = denoise_gaussian(data_mayavi, **(denoise_params or {}))
    elif denoise_method == 'nl_means':
        data_mayavi = denoise_nl_means(data_mayavi, **(denoise_params or {}))
    elif denoise_method != 'none':
        print(f"Warning: Unknown denoise_method '{denoise_method}'. No denoising applied.")

    # Define the extent for the plot
    unique_x = np.unique(exp_X)
    unique_y = np.unique(exp_Y)
    xmin, xmax = np.min(unique_x), np.max(unique_x)
    ymin, ymax = np.min(unique_y), np.max(unique_y)
    zmin, zmax = np.min(exp_biases), np.max(exp_biases)

    # Apply a scaling factor to the Z-axis (Bias) to adjust aspect ratio.
    # Increase this value to make the Z-axis appear more stretched (less flattened).
    # Decrease this value to make the Z-axis appear more compressed (more flattened).
    z_scale_factor = 8.0 # Adjusted from 40.0

    # Use original zmin, zmax for extent, as scaling will be applied to the actor
    extent_mayavi = [xmin, xmax, ymin, ymax, zmin, zmax]

    # Create the Mayavi Figure
    mlab.figure(size=(800, 600), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

    # Create a Mayavi data source
    src = mlab.pipeline.scalar_field(data_mayavi, extent=extent_mayavi)

    # Create the volume module from the source
    vol_module = mlab.pipeline.volume(src)

    # Apply scaling to the volume actor's Z-axis
    vol_module.actors[0].scale = [1.0, 1.0, z_scale_factor]

    # Access the volume property object
    vol_property = vol_module.volume_property

    # Define volumetric cutoffs (via Opacity)
    data_min = np.min(data_mayavi)
    data_max = np.max(data_mayavi)

    # Example: Make values near zero transparent, and extreme values visible
    positive_visible_threshold = 0.1 * data_max if data_max > 0 else 0.05
    negative_visible_threshold = 0.1 * data_min if data_min < 0 else -0.05

    # Configure Opacity Transfer Function (Piecewise Function)
    otf = vtk.vtkPiecewiseFunction()
    # Add points as (value, opacity) pairs
    otf.AddPoint(data_min, 1.0) # Fully opaque at min
    otf.AddPoint(negative_visible_threshold, 0.0) # Fully transparent at negative threshold
    otf.AddPoint(positive_visible_threshold, 0.0) # Fully transparent at positive threshold
    otf.AddPoint(data_max, 1.0) # Fully opaque at max
    vol_property.set_scalar_opacity(otf)

    # Configure Color Transfer Function
    ctf = vtk.vtkColorTransferFunction()
    # Use a diverging colormap: blue for negative, red for positive, grey for zero
    ctf.AddRGBPoint(data_min, 0.0, 0.0, 1.0) # Blue for min (most negative)
    ctf.AddRGBPoint(negative_visible_threshold, 0.5, 0.5, 0.5) # Grey near negative cutoff
    ctf.AddRGBPoint(0.0, 0.5, 0.5, 0.5) # Explicitly map 0 to grey
    ctf.AddRGBPoint(positive_visible_threshold, 0.5, 0.5, 0.5) # Grey near positive cutoff
    ctf.AddRGBPoint(data_max, 1.0, 0.0, 0.0) # Red for max (most positive)
    vol_property.set_color(ctf)

    # Add Enhancements
    mlab.axes(xlabel='X [Å]', ylabel='Y [Å]', zlabel='Bias [V] (Scaled)')
    mlab.title("Mayavi Volume Rendering of dIdV Data (Fog Effect)")

    # Display the Plot
    mlab.show()

if __name__ == '__main__':
    #plot_mayavi_volume( denoise_method='gaussian', denoise_params={'sigma': 1.5 })
    plot_mayavi_volume( )
