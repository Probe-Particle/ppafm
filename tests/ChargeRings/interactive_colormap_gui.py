import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
import json
import colormaps # Import to register custom colormaps

class ColormapGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Diverging Colormap")
        self.setGeometry(100, 100, 1000, 700)

        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QtWidgets.QHBoxLayout(self.main_widget)

        # Matplotlib canvas
        self.fig = plt.Figure(figsize=(8, 6)) # Back to a more square-ish size
        self.canvas = FigureCanvas(self.fig)
        self.ax_image = self.fig.add_subplot(111) # Only one main subplot
        self.layout.addWidget(self.canvas)

        # Control panel
        self.control_panel = QtWidgets.QScrollArea()
        self.control_panel.setWidgetResizable(True)
        self.control_widget = QtWidgets.QWidget()
        self.control_panel.setWidget(self.control_widget)
        self.control_layout = QtWidgets.QVBoxLayout(self.control_widget)
        self.layout.addWidget(self.control_panel)

        self.colorbar_instance = None # Initialize colorbar instance
        self.param_widgets = {}
        self.init_params()
        self.create_controls()
        self.generate_sample_data()
        self.update_plot()

    def init_params(self):
        # Default colors (RGB values from 0-1)
        self.params = {
            'min_r': {'value': 0.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2},
            'min_g': {'value': 0.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2},
            'min_b': {'value': 1.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2}, # Blue

            'center_r': {'value': 1.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2},
            'center_g': {'value': 1.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2},
            'center_b': {'value': 1.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2}, # White

            'max_r': {'value': 1.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2},
            'max_g': {'value': 0.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2},
            'max_b': {'value': 0.0, 'range': (0.0, 1.0), 'step': 0.01, 'decimals': 2}, # Red

            'gamma_neg_r': {'value': 1.0, 'range': (0.1, 5.0), 'step': 0.1, 'decimals': 1},
            'gamma_neg_g': {'value': 1.0, 'range': (0.1, 5.0), 'step': 0.1, 'decimals': 1},
            'gamma_neg_b': {'value': 1.0, 'range': (0.1, 5.0), 'step': 0.1, 'decimals': 1},

            'gamma_pos_r': {'value': 1.0, 'range': (0.1, 5.0), 'step': 0.1, 'decimals': 1},
            'gamma_pos_g': {'value': 1.0, 'range': (0.1, 5.0), 'step': 0.1, 'decimals': 1},
            'gamma_pos_b': {'value': 1.0, 'range': (0.1, 5.0), 'step': 0.1, 'decimals': 1},
        }

    def create_controls(self):
        self.add_group_box("Min Color", ['min_r', 'min_g', 'min_b'])
        self.add_group_box("Center Color", ['center_r', 'center_g', 'center_b'])
        self.add_group_box("Max Color", ['max_r', 'max_g', 'max_b'])
        self.add_group_box("Negative Gamma (min to center)", ['gamma_neg_r', 'gamma_neg_g', 'gamma_neg_b'])
        self.add_group_box("Positive Gamma (center to max)", ['gamma_pos_r', 'gamma_pos_g', 'gamma_pos_b'])

        # --- Predefined Colormap Controls ---
        predefined_group = QtWidgets.QGroupBox("Predefined Colormap Comparison")
        predefined_layout = QtWidgets.QFormLayout(predefined_group)

        self.cmap_predefined_options = ['bwr', 'bwr_r', 'seismic', 'seismic_r', 'PiYG', 'PiYG_r', 'PRGn', 'PRGn_r', 'RdBu', 'RdBu_r',  'vanimo', 'vanimo_r','coolwarm', 'coolwarm_r', 'PuRdR-w-BuGn', 'BuGnR-w-PuRd' ]
        self.predefined_cmap_combo = QtWidgets.QComboBox()
        self.predefined_cmap_combo.addItems(self.cmap_predefined_options)
        self.predefined_cmap_combo.setCurrentText('bwr')
        self.predefined_cmap_combo.currentIndexChanged.connect(self.on_param_change)
        predefined_layout.addRow("Colormap:", self.predefined_cmap_combo)

        self.use_predefined_checkbox = QtWidgets.QCheckBox("Use Predefined for Image")
        self.use_predefined_checkbox.setChecked(False)
        self.use_predefined_checkbox.stateChanged.connect(self.on_param_change)
        predefined_layout.addRow(self.use_predefined_checkbox)
        self.control_layout.addWidget(predefined_group)

        self.auto_update_checkbox = QtWidgets.QCheckBox("Auto Update")
        self.auto_update_checkbox.setChecked(True)
        self.control_layout.addWidget(self.auto_update_checkbox)

        update_button = QtWidgets.QPushButton("Update Plot")
        update_button.clicked.connect(self.update_plot)
        self.control_layout.addWidget(update_button)

        # Add Save/Load/Export buttons
        io_layout = QtWidgets.QHBoxLayout()

        save_params_button = QtWidgets.QPushButton("Save Params")
        save_params_button.clicked.connect(self.save_params)
        io_layout.addWidget(save_params_button)

        load_params_button = QtWidgets.QPushButton("Load Params")
        load_params_button.clicked.connect(self.load_params)
        io_layout.addWidget(load_params_button)
        self.control_layout.addLayout(io_layout)

        export_cmap_button = QtWidgets.QPushButton("Export Colormap (.npy)")
        export_cmap_button.clicked.connect(self.export_colormap)
        self.control_layout.addWidget(export_cmap_button)

        self.control_layout.addStretch(1) # Push everything to the top

    def add_group_box(self, title, param_keys):
        group_box = QtWidgets.QGroupBox(title)
        group_layout = QtWidgets.QFormLayout(group_box)
        for key in param_keys:
            spec = self.params[key]
            spin_box = QtWidgets.QDoubleSpinBox()
            spin_box.setRange(spec['range'][0], spec['range'][1])
            spin_box.setValue(spec['value'])
            spin_box.setSingleStep(spec['step'])
            spin_box.setDecimals(spec['decimals'])
            spin_box.valueChanged.connect(self.on_param_change)
            group_layout.addRow(key.replace('_', ' ').title(), spin_box)
            self.param_widgets[key] = spin_box
        self.control_layout.addWidget(group_box)

    def on_param_change(self):
        if self.auto_update_checkbox.isChecked():
            self.update_plot()

    def get_current_params(self):
        current_params = {}
        for key, widget in self.param_widgets.items():
            current_params[key] = widget.value()
        return current_params

    def generate_sample_data(self):
        # Data from colormaps.py example
        w = 0.05
        xs = np.linspace(-1, 1, 256)
        X, Y = np.meshgrid(xs, xs)
        self.data = np.exp(-(Y/w)**2) * X \
                  + np.exp(-((Y-0.75)/w)**2) * X \
                  + np.exp(-((Y+0.75)/w)**2) * X \
                  + Y*0.5
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        self.data_center = 0.0 # Assuming diverging around zero

    def create_custom_colormap(self, params):
        # Extract colors
        color_min    = np.array([params['min_r'],   params['min_g'],   params['min_b']])
        color_center = np.array([params['center_r'],params['center_g'],params['center_b']])
        color_max    = np.array([params['max_r'],   params['max_g'],   params['max_b']])

        # Extract gammas
        gamma_neg    = np.array([params['gamma_neg_r'], params['gamma_neg_g'], params['gamma_neg_b']])
        gamma_pos    = np.array([params['gamma_pos_r'], params['gamma_pos_g'], params['gamma_pos_b']])

        # Create a lookup table for the colormap
        n_steps = 256
        cmap_array = np.zeros((n_steps, 3))

        # Create a linear space for the colormap's internal values (0 to 1)
        linear_steps = np.linspace(0, 1, n_steps)

        for i in range(n_steps):
            val_norm = linear_steps[i] # This is the normalized value for the colormap

            if val_norm < 0.5: # Negative part (from min to center)
                # To be symmetric, the "zero" of the interpolation ramp must be at the center.
                # We define an interpolation factor `t` that is 0 at the center and 1 at the minimum.
                t = (0.5 - val_norm) / 0.5
                f = t ** gamma_neg
                # As f goes from 0 (at center) to 1 (at min), we interpolate from color_center to color_min.
                cmap_array[i, :] = color_center + (color_min - color_center) * f
            else: # Positive part (from center to max)
                # Map val_norm from [0.5, 1] to [0, 1] for gamma interpolation
                x_interp = (val_norm - 0.5) / 0.5
                f = x_interp ** gamma_pos
                cmap_array[i, :] = color_center + (color_max - color_center) * f
        
        # Ensure RGB values are clipped to [0, 1]
        cmap_array = np.clip(cmap_array, 0, 1)

        return ListedColormap(cmap_array, name='custom_diverging'), cmap_array # Return both colormap object and raw RGB values

    def update_plot(self):
        current_params = self.get_current_params()
        custom_cmap, cmap_rgb_values = self.create_custom_colormap(current_params)

        self.ax_image.clear()

        # --- Get predefined colormap data ---
        predefined_cmap_name = self.predefined_cmap_combo.currentText()
        try:
            predefined_cmap_obj = plt.get_cmap(predefined_cmap_name)
            predefined_rgb_values = predefined_cmap_obj(np.linspace(0, 1, 256))[:, :3]
        except ValueError:
            print(f"Warning: Colormap '{predefined_cmap_name}' not found. Using 'bwr'.")
            predefined_cmap_obj = plt.get_cmap('bwr')
            predefined_rgb_values = predefined_cmap_obj(np.linspace(0, 1, 256))[:, :3]

        # --- Determine which colormap to use for the image ---
        display_cmap = predefined_cmap_obj if self.use_predefined_checkbox.isChecked() else custom_cmap


        # Remove existing colorbar if it exists
        if self.colorbar_instance is not None:
            self.colorbar_instance.remove()

        # Find the maximum absolute deviation from the center
        max_abs_dev = max(abs(self.data_min - self.data_center), abs(self.data_max - self.data_center))
        
        # Set vmin and vmax for imshow to be symmetric around data_center
        vmin_imshow = self.data_center - max_abs_dev
        vmax_imshow = self.data_center + max_abs_dev

        # Plot the image
        im = self.ax_image.imshow(self.data, cmap=display_cmap, origin='lower', vmin=vmin_imshow, vmax=vmax_imshow, extent=[0, 1, 0, 1]) # Set extent to 0-1
        self.colorbar_instance = self.fig.colorbar(im, ax=self.ax_image)
        self.ax_image.set_title("Custom Diverging Colormap")

        # --- Plot both custom and predefined RGB curves over the imshow plot ---
        x_vals = np.linspace(0, 1, cmap_rgb_values.shape[0])

        # Plot custom curves (solid lines)
        self.ax_image.plot(x_vals, cmap_rgb_values[:, 0], color='red',   linewidth=2, alpha=0.9, label='Custom R')
        self.ax_image.plot(x_vals, cmap_rgb_values[:, 1], color='green', linewidth=2, alpha=0.9, label='Custom G')
        self.ax_image.plot(x_vals, cmap_rgb_values[:, 2], color='blue',  linewidth=2, alpha=0.9, label='Custom B')

        # Plot predefined curves (dotted lines)
        self.ax_image.plot(x_vals, predefined_rgb_values[:, 0], color='red',   linestyle=':', linewidth=2, alpha=0.7, label=f'"{predefined_cmap_name}" R')
        self.ax_image.plot(x_vals, predefined_rgb_values[:, 1], color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'"{predefined_cmap_name}" G')
        self.ax_image.plot(x_vals, predefined_rgb_values[:, 2], color='blue',  linestyle=':', linewidth=2, alpha=0.7, label=f'"{predefined_cmap_name}" B')
        self.ax_image.legend(loc='upper left', fontsize='x-small')

        # self.fig.tight_layout() # tight_layout does not work well with add_axes
        self.canvas.draw_idle()

    def closeEvent(self, event):
        plt.close(self.fig) # Close the matplotlib figure when the window is closed
        event.accept()

    def save_params(self):
        """Save current color and gamma parameters to a JSON file."""
        params_to_save = self.get_current_params()

        # Open file dialog
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON Files (*.json)")

        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            try:
                with open(filename, 'w') as f:
                    json.dump(params_to_save, f, indent=4)
                print(f"Parameters saved to {filename}")
            except Exception as e:
                print(f"Error saving parameters: {e}")

    def load_params(self):
        """Load color and gamma parameters from a JSON file."""
        # Open file dialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON Files (*.json)")

        if filename:
            try:
                with open(filename, 'r') as f:
                    loaded_params = json.load(f)

                # Set widget values from loaded params
                for key, value in loaded_params.items():
                    if key in self.param_widgets:
                        self.param_widgets[key].setValue(value)

                print(f"Parameters loaded from {filename}")
                self.update_plot() # Update plot with new parameters
            except Exception as e:
                print(f"Error loading parameters: {e}")

    def export_colormap(self):
        """Export the generated colormap array to a .npy file."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Colormap", "", "NumPy Array (*.npy)")
        if filename:
            current_params = self.get_current_params()
            _, cmap_rgb_values = self.create_custom_colormap(current_params)
            np.save(filename, cmap_rgb_values)
            print(f"Colormap array saved to {filename}")

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = ColormapGUI()
    window.show()
    app.exec_()