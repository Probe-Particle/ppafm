import os
import numpy as np
from mayavi import mlab
import sys
import vtk
from traits.api import HasTraits, Range, Instance, Enum, on_trait_change, Property, cached_property
from traitsui.api import View, Item, Group
from mayavi.core.scene import Scene
from mayavi.modules.volume import Volume
import matplotlib.cm as cm
import matplotlib.colors as colors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from exp_utils import load_experimental_data, denoise_gaussian, denoise_nl_means


class VolumeViewer(HasTraits):
    # Define traits for interactive parameters
    positive_threshold = Range(low='data_min', high='data_max', value=0.1, label='Positive Threshold')
    negative_threshold = Range(low='data_min', high='data_max', value=-0.1, label='Negative Threshold')
    z_scale_factor = Range(1.0, 100.0, 8.0, editor=None, label='Z-Axis Scale')
    colormap_name = Enum('RdBu', 'viridis', 'plasma', 'magma', 'cividis', label='Colormap')

    # Internal storage for Mayavi objects
    scene = Instance(Scene)
    vol_module = Instance(Volume)
    data_mayavi = Instance(np.ndarray)
    data_min = Property(depends_on='data_mayavi')
    data_max = Property(depends_on='data_mayavi')

    view = View(
        Group(
            Item('positive_threshold'),
            Item('negative_threshold'),
            Item('z_scale_factor'),
            Item('colormap_name'),
            show_border=True,
            label='Volume Rendering Controls'
        ),
        resizable=True,
        title='Mayavi Volume Viewer'
    )

    def _get_data_min(self):
        return np.min(self.data_mayavi)

    def _get_data_max(self):
        return np.max(self.data_mayavi)

    def __init__(self, filename='exp_rings_data.npz', denoise_method='none', denoise_params=None):
        super(VolumeViewer, self).__init__()
        # Initialize scene and vol_module to None
        self.scene = None
        self.vol_module = None

        # Load experimental data
        exp_X, exp_Y, exp_dIdV, exp_I, exp_biases = load_experimental_data(filename=filename)

        # Prepare data for Mayavi (transpose to X, Y, Bias)
        n_biases, n_y, n_x = exp_dIdV.shape
        data_mayavi = np.zeros((n_x, n_y, n_biases))
        for v_idx in range(n_biases):
            for y_idx in range(n_y):
                for x_idx in range(n_x):
                    data_mayavi[x_idx, y_idx, v_idx] = exp_dIdV[v_idx, y_idx, x_idx]

        # Store data for Mayavi
        self.data_mayavi = data_mayavi

        # Set initial thresholds based on data range
        self.positive_threshold = 0.1 * self.data_max if self.data_max > 0 else 0.05
        self.negative_threshold = 0.1 * self.data_min if self.data_min < 0 else -0.05

        # Apply denoising if specified
        if denoise_method == 'gaussian':
            self.data_mayavi = denoise_gaussian(data_mayavi, **(denoise_params or {}))
        elif denoise_method == 'nl_means':
            self.data_mayavi = denoise_nl_means(data_mayavi, **(denoise_params or {}))
        elif denoise_method != 'none':
            print(f"Warning: Unknown denoise_method '{denoise_method}'. No denoising applied.")
        else:
            self.data_mayavi = data_mayavi

        # Define the extent for the plot
        unique_x = np.unique(exp_X)
        unique_y = np.unique(exp_Y)
        xmin, xmax = np.min(unique_x), np.max(unique_x)
        ymin, ymax = np.min(unique_y), np.max(unique_y)
        zmin, zmax = np.min(exp_biases), np.max(exp_biases)

        # Use original zmin, zmax for extent, as scaling will be applied to the actor
        extent_mayavi = [xmin, xmax, ymin, ymax, zmin, zmax]

        # Create the Mayavi Figure
        self.scene = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

        # Create a Mayavi data source
        src = mlab.pipeline.scalar_field(self.data_mayavi, extent=extent_mayavi)

        # Create the volume module from the source
        self.vol_module = mlab.pipeline.volume(src)

        # Initial plot update
        self._update_plot()

        # Add Enhancements
        mlab.axes(xlabel='X [Å]', ylabel='Y [Å]', zlabel='Bias [V] (Scaled)')
        mlab.title("Mayavi Volume Rendering of dIdV Data (Fog Effect)")

    @on_trait_change('positive_threshold,negative_threshold,z_scale_factor,colormap_name')
    def _update_plot(self):
        if not self.vol_module:
            return

        # Apply scaling to the volume actor's Z-axis
        self.vol_module.actors[0].scale = [1.0, 1.0, self.z_scale_factor]

        # Access the volume property object
        vol_property = self.vol_module.volume_property

        # Configure Opacity Transfer Function (Piecewise Function)
        otf = vtk.vtkPiecewiseFunction()
        otf.AddPoint(self.data_min, 1.0)
        otf.AddPoint(self.negative_threshold, 0.0)
        otf.AddPoint(self.positive_threshold, 0.0)
        otf.AddPoint(self.data_max, 1.0)
        vol_property.set_scalar_opacity(otf)

        # Configure Color Transfer Function
        ctf = vtk.vtkColorTransferFunction()
        if self.colormap_name == 'RdBu':
            ctf.AddRGBPoint(self.data_min, 0.0, 0.0, 1.0)
            ctf.AddRGBPoint(self.negative_threshold, 0.5, 0.5, 0.5)
            ctf.AddRGBPoint(0.0, 0.5, 0.5, 0.5)
            ctf.AddRGBPoint(self.positive_threshold, 0.5, 0.5, 0.5)
            ctf.AddRGBPoint(self.data_max, 1.0, 0.0, 0.0)
        else:
            # Use matplotlib colormaps for other selections
            cmap = cm.get_cmap(self.colormap_name)
            # Sample the colormap at 256 points
            for i in range(256):
                x = float(i) / 255.0
                r, g, b, a = cmap(x) # Get RGBA from matplotlib colormap
                # Map x (0-1) to data range (data_min to data_max)
                data_val = self.data_min + x * (self.data_max - self.data_min)
                ctf.AddRGBPoint(data_val, r, g, b)

        vol_property.set_color(ctf)


if __name__ == '__main__':
    # Example usage:
    # To run with default settings (gaussian denoising):
    # viewer = VolumeViewer()
    # To run with no denoising:
    # viewer = VolumeViewer(denoise_method='none')
    # To run with Non-Local Means denoising:
    # viewer = VolumeViewer(denoise_method='nl_means', denoise_params={'h': 0.05, 'templateWindowSize': 7, 'searchWindowSize': 21})

    viewer = VolumeViewer(denoise_method='none')
    viewer.configure_traits()
    mlab.show()
