#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple, Union
from enum import Enum, auto

from GUITemplate import GUITemplate, PlotConfig, PlotType, PlotManager, PlotType
from charge_rings_core import calculate_tip_potential, calculate_qdot_system
from charge_rings_plotting import plot_tip_potential, plot_qdot_system, plot_ellipses

class ApplicationWindow(GUITemplate):
    def __init__(self):
        super().__init__("Combined Charge Rings GUI v6")
        
        # Parameter specifications
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
            
            # Ellipse Parameters
            'R_major':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (1.0, 10.0),   'value': 3.0, 'step': 0.1},
            'R_minor':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (1.0, 10.0),   'value': 2.0, 'step': 0.1},
            'phi0_ax':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (-3.14, 3.14), 'value': 0.0, 'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-1.0, 1.0),   'value': -0.1,'step': 0.01},
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
        
        # Initialize plot manager
        self.plot_manager = PlotManager(self.fig)
        
        # Configure plots
        self.plot_manager.add_plot('potential_1d',   PlotConfig( ax=self.ax1, title="1D Potential (z=0)",   plot_type=PlotType.MULTILINE,  xlabel="x [Å]", ylabel="V [V]",                 grid=True ))
        self.plot_manager.add_plot('tip_potential',  PlotConfig( ax=self.ax2, title="Tip Potential",        plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="z [Å]", cmap='bwr',     grid=True, clim=(-1.0, 1.0) ))
        self.plot_manager.add_plot('site_potential', PlotConfig( ax=self.ax3, title="Site Potential",       plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="z [Å]", cmap='seismic', grid=True, clim=(-1.0, 1.0) ))
        self.plot_manager.add_plot('energies',       PlotConfig( ax=self.ax4, title="Site Energies",        plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="y [Å]", cmap='seismic', grid=True, clim=(-1.0, 1.0) ))
        self.plot_manager.add_plot('total_charge',   PlotConfig( ax=self.ax5, title="Total Charge",         plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="y [Å]", cmap='seismic', grid=True, clim=(-1.0, 1.0) ))
        self.plot_manager.add_plot('stm',            PlotConfig( ax=self.ax6, title="STM",                  plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="y [Å]", cmap='inferno', grid=True, clim=(0.0, 1.0) ))
        self.plot_manager.add_plot('exp_didv',       PlotConfig( ax=self.ax7, title="Experimental dI/dV",   plot_type=PlotType.IMAGE, xlabel="X [Å]", ylabel="Y [Å]", cmap='seismic', grid=True, clim=(-1.0, 1.0) ))
        self.plot_manager.add_plot('exp_current',    PlotConfig( ax=self.ax8, title="Experimental Current", plot_type=PlotType.IMAGE, xlabel="X [Å]", ylabel="Y [Å]", cmap='inferno', grid=True, clim=(0.0, 600.0) ))
        
        self.load_experimental_data()
        self.plot_manager.initialize_plots()
        
        self.run()  # Call run() method
    
    def load_experimental_data(self):
        """Load experimental data from npz file"""
        data_path = os.path.join(os.path.dirname(__file__), 'exp_rings_data.npz')
        try:
            data = np.load(data_path)
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
            
            # Calculate experimental data extent
            self.exp_extent = [
                np.min(self.exp_X), np.max(self.exp_X),
                np.min(self.exp_Y), np.max(self.exp_Y)
            ]
            
            # Update exp_slice range based on actual data size
            self.param_specs['exp_slice']['range'] = (0, len(self.exp_biases) - 1)
            self.param_specs['exp_slice']['value'] = len(self.exp_biases) // 2
            
            # Set initial voltage index to middle
            self.exp_idx = len(self.exp_biases) // 2
            
        except FileNotFoundError:
            print(f"Warning: Experimental data file not found at {data_path}")
            self.exp_X = None
            self.exp_Y = None
            self.exp_dIdV = None
            self.exp_I = None
            self.exp_biases = None

    def resample_to_simulation_grid(self, data, src_extent, target_size=100, target_extent=(-20, 20, -20, 20)):
        """Resample data to match simulation grid and extent"""
        # Create source coordinate grids
        x_src = np.linspace(src_extent[0], src_extent[1], data.shape[1])
        y_src = np.linspace(src_extent[2], src_extent[3], data.shape[0])
        
        # Create target coordinate grid
        x_target = np.linspace(target_extent[0], target_extent[1], target_size)
        y_target = np.linspace(target_extent[2], target_extent[3], target_size)
        
        # Create interpolator
        from scipy.interpolate import interp2d
        interpolator = interp2d(x_src, y_src, data)
        
        # Resample data to target grid
        resampled = interpolator(x_target, y_target)
        
        # Create mask for points outside source extent
        xx, yy = np.meshgrid(x_target, y_target)
        mask = ((xx < src_extent[0]) | (xx > src_extent[1]) | 
                (yy < src_extent[2]) | (yy > src_extent[3]))
        
        # Set points outside source extent to zero
        resampled[mask] = 0
        
        return resampled

    def create_overlay_image(self, exp_data, sim_data, exp_extent, sim_extent):
        """Create RGB overlay of experimental and simulation data"""
        # Resample experimental data to simulation grid
        exp_resampled = self.resample_to_simulation_grid(
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

    def run(self):
        """Main calculation and plot update routine"""
        params = self.get_param_values()
        
        # Calculate tip potential
        tip_data = calculate_tip_potential(**params)
        Vtip = tip_data['Vtip']
        Esites = tip_data['Esites']
        V1d = tip_data['V1d']
        
        # Calculate quantum dot system
        qdot_data = calculate_qdot_system(**params)
        Es = qdot_data['Es'][:,:,0]  # Take first layer for 2D plot
        total_charge = qdot_data['total_charge'][:,:,0]  # Take first layer for 2D plot
        STM = qdot_data['STM'][:,:,0]  # Take first layer for 2D plot
        
        # Update plots with new data
        self.plot_manager.restore_backgrounds()
        
        # Update 1D potential plot with all three lines
        x_coords = np.linspace(-params['L'], params['L'], len(V1d))
        self.plot_manager.update_plot('potential_1d', [
            (x_coords, V1d),
            (x_coords, (V1d + params['Esite']) * params['VBias']),
            (x_coords, np.full_like(x_coords, params['VBias']))
        ])
        
        # Update image plots
        extent = [-params['L'], params['L'], -params['L'], params['L']]
        self.plot_manager.update_plot('tip_potential',  Vtip,         extent=extent)
        self.plot_manager.update_plot('site_potential', Esites,       extent=extent)
        self.plot_manager.update_plot('energies',       Es,           extent=extent)
        self.plot_manager.update_plot('total_charge',   total_charge, extent=extent)
        self.plot_manager.update_plot('stm',            STM,          extent=extent)
        
        # Update experimental plots if data is available
        if hasattr(self, 'exp_dIdV') and self.exp_dIdV is not None:
            self.exp_idx = params['exp_slice']
            
            # Update experimental plots
            self.plot_manager.update_plot('exp_didv', self.exp_dIdV[self.exp_idx], extent=self.exp_extent)
            self.plot_manager.update_plot('exp_current', self.exp_I[self.exp_idx], extent=self.exp_extent)
            
            # Create and update overlay
            rgb_overlay, overlay_extent = self.create_overlay_image( self.exp_dIdV[self.exp_idx], total_charge, self.exp_extent, extent)
            self.ax9.imshow(rgb_overlay, extent=overlay_extent)
            self.ax9.set_title('Overlay (Red: Exp, Green: Sim)')
            self.ax9.set_xlabel('X [Å]')
            self.ax9.set_ylabel('Y [Å]')
            self.ax9.draw_artist(self.ax9.images[0])
        
        # Plot ellipses on relevant plots
        for ax in [self.ax5, self.ax7]:
            plot_ellipses(ax, **params)
            for artist in ax.lines:
                ax.draw_artist(artist)
        
        # Perform final blitting update
        self.plot_manager.blit()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
