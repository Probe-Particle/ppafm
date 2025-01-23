#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from GUITemplate import GUITemplate
from charge_rings_core import calculate_tip_potential, calculate_qdot_system
from charge_rings_plotting import plot_tip_potential, plot_qdot_system

class ApplicationWindow(GUITemplate):
    def __init__(self):
        # First call parent constructor
        super().__init__("Combined Charge Rings GUI v4")
        
        # Then set parameter specifications
        self.param_specs = {
            # Tip Parameters
            'VBias':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.0, 2.0),   'value': 1.0, 'step': 0.1},
            'Rtip':          {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 5.0),   'value': 1.0, 'step': 0.5},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 20.0),  'value': 2.0, 'step': 0.5},
            
            # System Parameters
            'cCouling':      {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.02, 'step': 0.01, 'decimals': 3},
            'temperature':   {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 10.0, 'step': 1.0},
            'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            
            # Mirror Parameters
            'zV0':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': -2.5, 'step': 0.1},
            'zQd':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': 0.0,  'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 5.0, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0,'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-1.0, 1.0), 'value': -0.1,'step': 0.01},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0), 'value': 0.0, 'step': 0.5},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 50.0),  'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (50, 500),    'value': 100,  'step': 50},
            'decay':         {'group': 'Visualization',     'widget': 'double', 'range': (0.1, 2.0),   'value': 0.3,  'step': 0.1,   'decimals': 2},
            'dQ':            {'group': 'Visualization',     'widget': 'double', 'range': (0.001, 0.1), 'value': 0.02, 'step': 0.001, 'decimals': 3},
            
            # Experimental Data
            'exp_slice':     {'group': 'Experimental Data', 'widget': 'int',    'range': (0, 13),     'value': 0,    'step': 1},
        }
        
        self.create_gui()
        
        # Load experimental data
        self.load_experimental_data()
        
        # Setup matplotlib figure with 3x3 layout
        self.fig = Figure(figsize=(15, 15))
        self.canvas = FigureCanvas(self.fig)
        self.main_widget.layout().insertWidget(0, self.canvas)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(331)  # 1D Potential
        self.ax2 = self.fig.add_subplot(332)  # Tip Potential
        self.ax3 = self.fig.add_subplot(333)  # Site Potential
        self.ax4 = self.fig.add_subplot(334)  # Energies
        self.ax5 = self.fig.add_subplot(335)  # Total Charge
        self.ax6 = self.fig.add_subplot(336)  # STM
        self.ax7 = self.fig.add_subplot(338)  # Experimental dI/dV
        self.ax8 = self.fig.add_subplot(339)  # Experimental Current
        self.ax9 = self.fig.add_subplot(337)  # Additional plot if needed
        
        self.run()
        
    def load_experimental_data(self):
        """Load experimental data from npz file"""
        data = np.load('exp_rings_data.npz')
        # Convert from nm to Å (1 nm = 10 Å)
        self.exp_X = data['X'] * 10
        self.exp_Y = data['Y'] * 10
        self.exp_dIdV = data['dIdV']
        self.exp_I = data['I']
        self.exp_biases = data['biases']
        center_x = data['center_x'] * 10  # Convert to Å
        center_y = data['center_y'] * 10  # Convert to Å
        
        # Center the coordinates
        self.exp_X -= center_x
        self.exp_Y -= center_y
        
        # Update exp_slice range based on actual data size
        self.param_specs['exp_slice']['range'] = (0, len(self.exp_biases) - 1)
        self.param_specs['exp_slice']['value'] = len(self.exp_biases) // 2
        
        # Set initial voltage index to middle
        self.exp_idx = len(self.exp_biases) // 2

    def create_overlay_image(self, exp_data, sim_data, exp_extent, sim_extent):
        """Create RGB overlay of experimental and simulation data
        
        Args:
            exp_data: Experimental dI/dV data
            sim_data: Simulated charge data
            exp_extent: Extent of experimental data [xmin, xmax, ymin, ymax]
            sim_extent: Extent of simulation data [xmin, xmax, ymin, ymax]
        """
        # Normalize experimental data to [0,1] range for red channel
        exp_norm = (exp_data - np.min(exp_data)) / (np.max(exp_data) - np.min(exp_data))
        
        # Normalize simulation data to [0,1] range for green channel
        sim_norm = (sim_data - np.min(sim_data)) / (np.max(sim_data) - np.min(sim_data))
        
        # Interpolate simulation data to match experimental data grid
        x_exp = np.linspace(exp_extent[0], exp_extent[1], exp_data.shape[1])
        y_exp = np.linspace(exp_extent[2], exp_extent[3], exp_data.shape[0])
        x_sim = np.linspace(sim_extent[0], sim_extent[1], sim_data.shape[1])
        y_sim = np.linspace(sim_extent[2], sim_extent[3], sim_data.shape[0])
        
        from scipy.interpolate import interp2d
        sim_interp = interp2d(x_sim, y_sim, sim_norm)
        sim_resampled = sim_interp(x_exp, y_exp)
        
        # Create RGB image (Red: experimental, Green: simulation, Blue: zeros)
        rgb_image = np.zeros((*exp_data.shape, 3))
        rgb_image[..., 0] = exp_norm  # Red channel: experimental
        rgb_image[..., 1] = sim_resampled  # Green channel: simulation
        
        return rgb_image, exp_extent

    def plot_experimental_data(self):
        """Plot experimental data in the bottom row"""
        # Get current slice from GUI
        params = self.get_param_values()
        self.exp_idx = params['exp_slice']
        
        # Get plot extents
        xmin, xmax = np.min(self.exp_X[0]), np.max(self.exp_X[0])
        ymin, ymax = np.min(self.exp_Y[0]), np.max(self.exp_Y[0])
        exp_extent = [xmin, xmax, ymin, ymax]
        
        # Clear axes
        self.ax7.clear()
        self.ax8.clear()
        self.ax9.clear()
        
        # Plot dI/dV
        maxval = np.max(np.abs(self.exp_dIdV[self.exp_idx]))
        self.ax7.imshow(self.exp_dIdV[self.exp_idx], aspect='equal', 
                       cmap='seismic', vmin=-maxval, vmax=maxval,
                       extent=exp_extent)
        self.ax7.set_title(f'Exp. dI/dV at {self.exp_biases[self.exp_idx]:.3f} V')
        self.ax7.set_xlabel('X [Å]')
        self.ax7.set_ylabel('Y [Å]')
        
        # Plot Current
        self.ax8.imshow(self.exp_I[self.exp_idx], aspect='equal',
                       cmap='inferno', vmin=0.0, vmax=600.0,
                       extent=exp_extent)
        self.ax8.set_title(f'Exp. Current at {self.exp_biases[self.exp_idx]:.3f} V')
        self.ax8.set_xlabel('X [Å]')
        self.ax8.set_ylabel('Y [Å]')
        
        # Create and plot overlay
        # Get simulation data from ax5 (total charge plot)
        sim_image = self.ax5.images[0] if self.ax5.images else None
        if sim_image:
            sim_data = sim_image.get_array()
            sim_extent = sim_image.get_extent()
            
            # Create RGB overlay
            rgb_overlay, extent = self.create_overlay_image(
                self.exp_dIdV[self.exp_idx], 
                sim_data,
                exp_extent,
                sim_extent
            )
            
            # Plot overlay
            self.ax9.imshow(rgb_overlay, aspect='equal', extent=extent)
            self.ax9.set_title('Overlay (Red: Exp, Green: Sim)')
            self.ax9.set_xlabel('X [Å]')
            self.ax9.set_ylabel('Y [Å]')
        else:
            self.ax9.set_title('Run simulation first')
            self.ax9.grid(True)

    def run(self):
        """Main calculation and plotting function"""
        params = self.get_param_values()
        
        # Calculate tip potential and quantum dot system
        tip_data = calculate_tip_potential(**params)
        qdot_data = calculate_qdot_system(**params)
        
        # Plot results
        plot_tip_potential(self.ax1, self.ax2, self.ax3, **tip_data, **params)
        plot_qdot_system(self.ax4, self.ax5, self.ax6, **qdot_data, **params)
        
        # Plot experimental data
        self.plot_experimental_data()
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
