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
            'zero_shift': {'value': 0.0, 'range': (-1e6, 1e6), 'step': 0.1, 'decimals': 4},  # Large range for flexibility
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
        self.n_steps = 31 # Default number of colormap steps

    def create_controls(self):
        # Add zero shift control at the top
        self.add_group_box("Data Shift", ['zero_shift'])
        
        # Create a single group box for all color parameters
        color_group = QtWidgets.QGroupBox("Colormap Parameters")
        grid_layout = QtWidgets.QGridLayout(color_group)
        
        # Add column headers
        grid_layout.addWidget(QtWidgets.QLabel("R"), 0, 1, QtCore.Qt.AlignCenter)
        grid_layout.addWidget(QtWidgets.QLabel("G"), 0, 2, QtCore.Qt.AlignCenter)
        grid_layout.addWidget(QtWidgets.QLabel("B"), 0, 3, QtCore.Qt.AlignCenter)
        
        # Add rows with controls
        params = [
            ('min', 'Min'),
            ('center', 'Center'),
            ('max', 'Max'),
            ('gamma_pos', 'Gamma +'),
            ('gamma_neg', 'Gamma -')
        ]
        
        for row, (prefix, label) in enumerate(params, start=1):
            grid_layout.addWidget(QtWidgets.QLabel(label), row, 0)
            
            for col, color in enumerate(['r', 'g', 'b'], start=1):
                key = f'{prefix}_{color}'
                spin_box = QtWidgets.QDoubleSpinBox()
                spin_box.setRange(self.params[key]['range'][0], self.params[key]['range'][1])
                spin_box.setValue(self.params[key]['value'])
                spin_box.setSingleStep(self.params[key]['step'])
                spin_box.setDecimals(self.params[key]['decimals'])
                spin_box.valueChanged.connect(self.on_param_change)
                grid_layout.addWidget(spin_box, row, col)
                self.param_widgets[key] = spin_box
        
        self.control_layout.addWidget(color_group)
        
        # Add flip colormap button
        flip_button = QtWidgets.QPushButton("Flip Colormap")
        flip_button.clicked.connect(self.flip_colormap)
        self.control_layout.addWidget(flip_button)

        # Number of colormap steps
        nsteps_layout = QtWidgets.QFormLayout()
        self.nsteps_spinbox = QtWidgets.QSpinBox()
        self.nsteps_spinbox.setRange(2, 512)
        self.nsteps_spinbox.setValue(self.n_steps)
        self.nsteps_spinbox.valueChanged.connect(self.on_nsteps_change)
        nsteps_layout.addRow("Colormap Steps:", self.nsteps_spinbox)
        self.control_layout.addLayout(nsteps_layout)

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

        # --- Data Loading Controls ---
        data_loading_group = QtWidgets.QGroupBox("Data Loading")
        data_loading_layout = QtWidgets.QFormLayout(data_loading_group)

        self.data_key_input = QtWidgets.QLineEdit("dIdV")
        data_loading_layout.addRow("Array Name:", self.data_key_input)

        load_data_button = QtWidgets.QPushButton("Load Data (.npz)")
        load_data_button.clicked.connect(self.load_data_from_npz)
        data_loading_layout.addRow(load_data_button)

        self.control_layout.addWidget(data_loading_group)

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

        export_txt_btn = QtWidgets.QPushButton("Export Colormap Values (TXT)")
        export_txt_btn.clicked.connect(self.export_colormap_txt)
        self.control_layout.addWidget(export_txt_btn)

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

    def on_nsteps_change(self, value):
        self.n_steps = value
        self.on_param_change()

    def on_param_change(self):
        if self.auto_update_checkbox.isChecked():
            self.update_plot()

    def get_current_params(self):
        current_params = {}
        for key, widget in self.param_widgets.items():
            current_params[key] = widget.value()
        return current_params

    def generate_sample_data(self):
        """Generate example data for visualization."""
        # Data from colormaps.py example
        w = 0.05
        xs = np.linspace(-1, 1, 256)
        X, Y = np.meshgrid(xs, xs)
        self.data = np.exp(-(Y/w)**2) * X \
                  + np.exp(-((Y-0.75)/w)**2) * X \
                  + np.exp(-((Y+0.75)/w)**2) * X \
                  + Y*0.5
        self.current_slice = self.data  # Store as current slice for plotting
        print("Generated sample data")

    def create_custom_colormap(self, params):
        """
        Create custom diverging colormap using the new colormaps.py functions.
        Returns both the smooth colormap and the coarse RGB values for plotting curves.
        """
        # Extract parameters in format expected by generate_diverging_colormap
        # min_rgb = 
        # center_rgb = np.array([params['center_r'], params['center_g'], params['center_b']])
        # max_rgb = np.array([params['max_r'], params['max_g'], params['max_b']])
        # gamma_neg_rgb = np.array([params['gamma_neg_r'], params['gamma_neg_g'], params['gamma_neg_b']])
        # gamma_pos_rgb = np.array([params['gamma_pos_r'], params['gamma_pos_g'], params['gamma_pos_b']])
        
        # Generate colormap using the new function
        coarse_cmap_array = colormaps.generate_diverging_colormap(
            min       = np.array([params['min_r'],       params['min_g'],       params['min_b']]),
            center    = np.array([params['center_r'],    params['center_g'],    params['center_b']]),
            max       = np.array([params['max_r'],       params['max_g'],       params['max_b']]),
            gamma_neg = np.array([params['gamma_neg_r'], params['gamma_neg_g'], params['gamma_neg_b']]),
            gamma_pos = np.array([params['gamma_pos_r'], params['gamma_pos_g'], params['gamma_pos_b']]),
            n_steps=self.n_steps
        )
        smooth_cmap_array, cmap = colormaps.resample_colormap(coarse_cmap_array)
        return cmap, coarse_cmap_array, smooth_cmap_array

    def update_plot(self):
        """Update the plot with current data slice and colormap settings."""
        if not hasattr(self, 'current_slice') or self.current_slice is None:
            return
            
        # Apply zero shift to data
        zero_shift = self.param_widgets['zero_shift'].value() if 'zero_shift' in self.param_widgets else 0.0
        shifted_data = self.current_slice + zero_shift
        
        current_params = self.get_current_params()
        custom_cmap, cmap_rgb_values,_ = self.create_custom_colormap(current_params)
        print("cmap_rgb_values.shape ", cmap_rgb_values.shape )
        print("custom_cmap ", custom_cmap)

        self.ax_image.clear()

        # Get predefined colormap data
        predefined_cmap_name = self.predefined_cmap_combo.currentText()
        try:
            predefined_cmap_obj = plt.get_cmap(predefined_cmap_name)
            predefined_rgb_values = predefined_cmap_obj(np.linspace(0, 1, self.n_steps))[:, :3]
        except ValueError:
            print(f"Warning: Colormap '{predefined_cmap_name}' not found. Using 'bwr'.")
            predefined_cmap_obj = plt.get_cmap('bwr')
            predefined_rgb_values = predefined_cmap_obj(np.linspace(0, 1, self.n_steps))[:, :3]

        #print( "custom_cmap type", type(custom_cmap), "len", len(custom_cmap) )

        # Determine which colormap to use
        display_cmap = predefined_cmap_obj if self.use_predefined_checkbox.isChecked() else custom_cmap

        # Remove existing colorbar if it exists
        if self.colorbar_instance is not None:
            self.colorbar_instance.remove()

        # Find the maximum absolute deviation from the center
        data_min = np.min(shifted_data)
        data_max = np.max(shifted_data)
        data_center = 0.0 # Center is now at the shifted zero
        max_abs_dev = max(abs(data_min - data_center), abs(data_max - data_center))

        # Set vmin and vmax for imshow to be symmetric around data_center
        vmin_imshow = data_center - max_abs_dev
        vmax_imshow = data_center + max_abs_dev

        # Plot the image
        im = self.ax_image.imshow(shifted_data, cmap=display_cmap, origin='lower', vmin=vmin_imshow, vmax=vmax_imshow, extent=[0, 1, 0, 1])
        self.colorbar_instance = self.fig.colorbar(im, ax=self.ax_image)
        
        # Add slice info to title if we have 3D data
        title = "Custom Diverging Colormap"
        if hasattr(self, 'data') and self.data.ndim == 3:
            title += f" (Slice axis={self.slice_axis}, idx={self.slice_idx})"
        if zero_shift != 0:
            title += f" [Zero shift: {zero_shift:.3f}]"
        self.ax_image.set_title(title)

        # Plot both custom and predefined RGB curves over the imshow plot
        x_vals = np.linspace(0, 1, cmap_rgb_values.shape[0])

        # Plot custom curves (solid lines with markers to show control points)
        self.ax_image.plot(x_vals, cmap_rgb_values[:, 0], 'o-', color='red',   linewidth=2, alpha=0.9, label='Custom R')
        self.ax_image.plot(x_vals, cmap_rgb_values[:, 1], 'o-', color='green', linewidth=2,  alpha=0.9, label='Custom G')
        self.ax_image.plot(x_vals, cmap_rgb_values[:, 2], 'o-', color='blue',  linewidth=2, alpha=0.9, label='Custom B')

        # Plot predefined curves (dotted lines)
        self.ax_image.plot(x_vals, predefined_rgb_values[:, 0], color='red',   linestyle=':', linewidth=2, alpha=0.7, label=f'"{predefined_cmap_name}" R')
        self.ax_image.plot(x_vals, predefined_rgb_values[:, 1], color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'"{predefined_cmap_name}" G')
        self.ax_image.plot(x_vals, predefined_rgb_values[:, 2], color='blue',  linestyle=':', linewidth=2, alpha=0.7, label=f'"{predefined_cmap_name}" B')
        self.ax_image.legend(loc='upper left', fontsize='x-small')

        self.canvas.draw_idle()

    def closeEvent(self, event):
        plt.close(self.fig) # Close the matplotlib figure when the window is closed
        event.accept()

    def load_data_from_npz(self):
        """Load a 2D or 3D array from a .npz file."""
        array_name = self.data_key_input.text()
        if not array_name:
            print("Warning: Please specify an array name to load.")
            return

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Data", "", "NumPy NPZ Files (*.npz)")

        if filename:
            try:
                with np.load(filename) as data_file:
                    if array_name in data_file:
                        self.data = data_file[array_name]
                        if self.data.ndim == 3:
                            self.slice_axis = 0
                            self.slice_idx = 0
                            self.slice_max = self.data.shape[0] - 1
                            self.update_slice_controls()
                            self.current_slice = np.take(self.data, self.slice_idx, axis=self.slice_axis)
                        elif self.data.ndim == 2:
                            self.current_slice = self.data
                        else:
                            raise ValueError(f"Array '{array_name}' must be 2D or 3D.")
                        print(f"Successfully loaded array '{array_name}' from {filename}")
                        self.update_plot()
                    else:
                        raise KeyError(f"Array '{array_name}' not found in {filename}. Available keys: {list(data_file.keys())}")
            except Exception as e:
                print(f"Error loading data: {e}")

    def save_params(self):
        """Save current color and gamma parameters to a JSON file."""
        params_to_save = self.get_current_params()
        params_to_save['n_steps'] = self.n_steps

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

                # Set n_steps first if it exists, remove it from dict
                self.n_steps = loaded_params.pop('n_steps', 31)
                self.nsteps_spinbox.setValue(self.n_steps)

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
            _, cmap_rgb_values,_ = self.create_custom_colormap(current_params)
            np.save(filename, cmap_rgb_values)
            print(f"Colormap array saved to {filename}")

    def export_colormap_txt(self):
        """Export the current colormap RGB values to a text file with parameter header."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Colormap Values", "", "Text Files (*.txt)")
        if filename:
            current_params = self.get_current_params()
            _,_, cmap_rgb_values = self.create_custom_colormap(current_params)
            
            # Create parameter header
            param_keys = ['min_r', 'min_g', 'min_b', 
                         'center_r', 'center_g', 'center_b',
                         'max_r', 'max_g', 'max_b',
                         'gamma_neg_r', 'gamma_neg_g', 'gamma_neg_b',
                         'gamma_pos_r', 'gamma_pos_g', 'gamma_pos_b']
            param_str = ' '.join(f"{k}={current_params[k]:.4f}" for k in param_keys)            
            # Save with 2-line header
            np.savetxt(filename, cmap_rgb_values, fmt='%.6f', header=f"# {param_str}\n# R\tG\tB", delimiter='\t', comments='')
            print(f"Colormap values saved to {filename}")

    def update_slice_controls(self):
        """Add UI controls for selecting slice axis and index."""
        # Add slice controls
        slice_controls_group = QtWidgets.QGroupBox("Slice Controls")
        slice_controls_layout = QtWidgets.QFormLayout(slice_controls_group)

        self.slice_axis_combo = QtWidgets.QComboBox()
        self.slice_axis_combo.addItems(['0', '1', '2'])
        self.slice_axis_combo.currentTextChanged.connect(self.on_slice_axis_change)
        slice_controls_layout.addRow("Slice Axis:", self.slice_axis_combo)

        self.slice_idx_spinbox = QtWidgets.QSpinBox()
        self.slice_idx_spinbox.setRange(0, self.slice_max)
        self.slice_idx_spinbox.setValue(self.slice_idx)
        self.slice_idx_spinbox.valueChanged.connect(self.on_slice_idx_change)
        slice_controls_layout.addRow("Slice Index:", self.slice_idx_spinbox)

        self.control_layout.addWidget(slice_controls_group)

    def on_slice_axis_change(self, axis):
        """Handle slice axis change."""
        self.slice_axis = int(axis)
        self.slice_max = self.data.shape[self.slice_axis] - 1
        self.slice_idx_spinbox.setRange(0, self.slice_max)
        self.current_slice = np.take(self.data, self.slice_idx, axis=self.slice_axis)
        self.update_plot()

    def on_slice_idx_change(self, idx):
        """Handle slice index change."""
        self.slice_idx = idx
        self.current_slice = np.take(self.data, self.slice_idx, axis=self.slice_axis)
        self.update_plot()

    def flip_colormap(self):
        """Flip the colormap by swapping min/max colors and gamma values."""
        # Swap min and max colors
        for color in ['r', 'g', 'b']:
            min_val = self.param_widgets[f'min_{color}'].value()
            max_val = self.param_widgets[f'max_{color}'].value()
            self.param_widgets[f'min_{color}'].setValue(max_val)
            self.param_widgets[f'max_{color}'].setValue(min_val)
        
        # Swap negative and positive gamma values
        for color in ['r', 'g', 'b']:
            neg_gamma = self.param_widgets[f'gamma_neg_{color}'].value()
            pos_gamma = self.param_widgets[f'gamma_pos_{color}'].value()
            self.param_widgets[f'gamma_neg_{color}'].setValue(pos_gamma)
            self.param_widgets[f'gamma_pos_{color}'].setValue(neg_gamma)
        
        # Update the plot
        self.update_plot()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = ColormapGUI()
    window.show()
    app.exec_()