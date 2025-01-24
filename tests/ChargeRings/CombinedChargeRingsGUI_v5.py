#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
        
        # Setup matplotlib figure with 3x3 layout and enable blitting
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

        # Initialize plot artists with dummy data
        dummy_data = np.zeros((100, 100))
        self.line1d, = self.ax1.plot([], [], label='V_tip')
        self.line1d_esite, = self.ax1.plot([], [], label='V_tip + E_site')
        self.line1d_vbias, = self.ax1.plot([], [], label='VBias')
        self.im2 = self.ax2.imshow(dummy_data, animated=True)
        self.im3 = self.ax3.imshow(dummy_data, animated=True)
        self.im4 = self.ax4.imshow(dummy_data, animated=True)
        self.im5 = self.ax5.imshow(dummy_data, animated=True)
        self.im6 = self.ax6.imshow(dummy_data, animated=True)
        self.im7 = self.ax7.imshow(dummy_data, animated=True)
        self.im8 = self.ax8.imshow(dummy_data, animated=True)
        self.im9 = self.ax9.imshow(dummy_data, animated=True)
        
        # Initialize plot artists with dummy data
        dummy_data = np.zeros((100, 100))
        self.line1d, = self.ax1.plot([], [], label='V_tip')
        self.line1d_esite, = self.ax1.plot([], [], label='V_tip + E_site')
        self.line1d_vbias, = self.ax1.plot([], [], label='VBias')
        self.im2 = self.ax2.imshow(dummy_data, animated=True)
        self.im3 = self.ax3.imshow(dummy_data, animated=True)
        self.im4 = self.ax4.imshow(dummy_data, animated=True)
        self.im5 = self.ax5.imshow(dummy_data, animated=True)
        self.im6 = self.ax6.imshow(dummy_data, animated=True)
        self.im7 = self.ax7.imshow(dummy_data, animated=True)
        self.im8 = self.ax8.imshow(dummy_data, animated=True)
        self.im9 = self.ax9.imshow(dummy_data, animated=True)

        # Load experimental data after all plot elements are initialized
        self.load_experimental_data()
        
        # Store background for blitting
        self.fig.canvas.draw()
        self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in 
                          [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, 
                           self.ax6, self.ax7, self.ax8, self.ax9]]
        
        # Enable interactive mode
        plt.ion()
        
        self.run()
        
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
            
            # Initialize experimental plots
            self.im7.set_cmap('seismic')
            self.im8.set_cmap('inferno')
            self.im9.set_cmap('jet')
            
        except FileNotFoundError:
            print(f"Warning: Experimental data file not found at {data_path}")

    def resample_to_simulation_grid(self, data, src_extent, target_size=100, target_extent=(-20, 20, -20, 20)):
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
        """Create RGB overlay of experimental and simulation data
        
        Args:
            exp_data: Experimental dI/dV data
            sim_data: Simulated charge data
            exp_extent: Extent of experimental data [xmin, xmax, ymin, ymax]
            sim_extent: Extent of simulation data [xmin, xmax, ymin, ymax]
        """
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
        rgb_image[..., 1] = sim_norm*0  # Green channel: simulation
        
        return rgb_image, sim_extent

    def plot_ellipses(self, ax, params, x_scale=1.0, y_scale=1.0, x_offset=0.0, y_offset=0.0):
        """Plot ellipses for each quantum dot site with optional scaling and offset
        
        Args:
            ax: matplotlib axis to plot on
            params: dictionary of parameters
        """
        nsite   = params['nsite']
        radius  = params['radius']
        phiRot  = params['phiRot']
        R_major = params['R_major']
        R_minor = params['R_minor']
        phi0_ax = params['phi0_ax']
        
        # Number of points for ellipse
        n = 100
        
        for i in range(nsite):
            # Calculate quantum dot position
            phi = phiRot + i * 2 * np.pi / nsite
            dir_x = np.cos(phi)
            dir_y = np.sin(phi)
            qd_pos_x = dir_x * radius
            qd_pos_y = dir_y * radius
            
            # Calculate ellipse points
            phi_ax = phi0_ax + phi
            t = np.linspace(0, 2*np.pi, n)
            
            # Create ellipse in local coordinates
            x_local = R_major * np.cos(t)
            y_local = R_minor * np.sin(t)
            
            # Rotate ellipse
            x_rot = x_local * np.cos(phi_ax) - y_local * np.sin(phi_ax)
            y_rot = x_local * np.sin(phi_ax) + y_local * np.cos(phi_ax)
            
            # Translate, scale and center to quantum dot position
            x = (x_rot * x_scale) + (qd_pos_x * x_scale)
            y = (y_rot * y_scale) + (qd_pos_y * y_scale)
            
            # Plot ellipse
            ax.plot(x, y, ':', color='white', alpha=0.8, linewidth=1)
            
            # Plot center point
            ax.plot(qd_pos_x, qd_pos_y, '+', color='white', markersize=5)

    def plot_experimental_data(self):
        """Plot experimental data in the bottom row"""
        # Get parameters
        params = self.get_param_values()
        self.exp_idx = params['exp_slice']
        L = params['L']
        
        # Get plot extents
        xmin, xmax = np.min(self.exp_X[0]), np.max(self.exp_X[0])
        ymin, ymax = np.min(self.exp_Y[0]), np.max(self.exp_Y[0])
        exp_extent = [xmin, xmax, ymin, ymax]
        sim_extent = [-L, L, -L, L]
        
        # Clear axes
        self.ax7.clear()
        self.ax8.clear()
        self.ax9.clear()
        
        # Plot dI/dV
        maxval = np.max(np.abs(self.exp_dIdV[self.exp_idx]))
        self.ax7.imshow(self.exp_dIdV[self.exp_idx], aspect='equal',  cmap='seismic', vmin=-maxval, vmax=maxval, extent=exp_extent)
        self.ax7.set_title(f'Exp. dI/dV at {self.exp_biases[self.exp_idx]:.3f} V')
        self.ax7.set_xlabel('X [Å]')
        self.ax7.set_ylabel('Y [Å]')
        
        # Plot ellipses on dI/dV plot
        self.plot_ellipses(self.ax7, params)
        
        # Plot Current
        self.ax8.imshow(self.exp_I[self.exp_idx], aspect='equal',  cmap='inferno', vmin=0.0, vmax=600.0,  extent=exp_extent)
        self.ax8.set_title(f'Exp. Current at {self.exp_biases[self.exp_idx]:.3f} V')
        self.ax8.set_xlabel('X [Å]')
        self.ax8.set_ylabel('Y [Å]')
        
        # Create and plot overlay
        # Get simulation data from ax5 (total charge plot)
        sim_image = self.ax5.images[0] if self.ax5.images else None
        if sim_image:
            sim_data = sim_image.get_array()
            
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
        
        # Restore backgrounds
        for ax, bg in zip([self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, 
                          self.ax6, self.ax7, self.ax8, self.ax9], 
                         self.backgrounds):
            self.fig.canvas.restore_region(bg)
        
        # Update 1D potential plot with correct parameter handling
        x_coords = np.linspace(-params['L'], params['L'], len(tip_data['V1d']))
        self.line1d.set_data(x_coords, tip_data['V1d'])
        self.line1d_esite.set_data(x_coords, (tip_data['V1d'] + params['Esite']) * params['VBias'])
        self.line1d_vbias.set_data(x_coords, np.full_like(x_coords, params['VBias']))
        
        # Update plot limits and labels
        self.ax1.set_xlim(-params['L'], params['L'])
        self.ax1.set_ylim(-params['VBias']*1.1, params['VBias']*1.1)
        self.ax1.set_title("1D Potential (z=0)")
        self.ax1.set_xlabel("x [Å]")
        self.ax1.set_ylabel("V [V]")
        self.ax1.grid(True)
        
        # Draw artists
        self.ax1.draw_artist(self.line1d)
        self.ax1.draw_artist(self.line1d_esite)
        self.ax1.draw_artist(self.line1d_vbias)
        
        # Update legend
        self.ax1.legend(['V_tip', 'V_tip + E_site', 'VBias'], loc='upper right')
        
        # Update image plots with proper colormaps and scaling
        extent = [-params['L'], params['L'], -params['L'], params['L']]
        
        # Tip Potential (bwr colormap, symmetric scaling)
        self.im2.set_data(tip_data['Vtip'])
        self.im2.set_extent(extent)
        self.im2.set_cmap('bwr')
        self.im2.set_clim(-params['VBias'], params['VBias'])
        self.ax2.draw_artist(self.im2)
        
        # Site Potential (bwr colormap, symmetric scaling)
        self.im3.set_data(tip_data['Esites'])
        self.im3.set_extent(extent)
        self.im3.set_cmap('bwr')
        self.im3.set_clim(-params['VBias'], params['VBias'])
        self.ax3.draw_artist(self.im3)
        
        # Energies (bwr colormap, symmetric scaling)
        Eplot = np.max(qdot_data['Es'], axis=2)
        vmax = np.abs(Eplot).max()
        self.im4.set_data(Eplot)
        self.im4.set_extent(extent)
        self.im4.set_cmap('bwr')
        self.im4.set_clim(-vmax, vmax)
        self.ax4.draw_artist(self.im4)
        
        # Total Charge (bwr colormap with proper scaling)
        total_charge = qdot_data['total_charge'].reshape(qdot_data['total_charge'].shape[0], -1)
        charge_max = np.abs(total_charge).max()
        self.im5.set_data(total_charge)
        self.im5.set_extent(extent)
        self.im5.set_cmap('bwr')
        self.im5.set_clim(-charge_max, charge_max)
        self.ax5.set_title("Total Charge")
        self.ax5.set_xlabel("x [Å]")
        self.ax5.set_ylabel("y [Å]")
        self.ax5.draw_artist(self.im5)
        
        # STM (gray colormap with proper scaling)
        stm_max = np.abs(qdot_data['STM']).max()
        self.im6.set_data(qdot_data['STM'])
        self.im6.set_extent(extent)
        self.im6.set_cmap('gray')
        self.im6.set_clim(0, stm_max)
        self.ax6.set_title("STM")
        self.ax6.set_xlabel("x [Å]")
        self.ax6.set_ylabel("y [Å]")
        self.ax6.draw_artist(self.im6)

        # Clear and redraw ellipses on total charge plot
        for artist in self.ax5.lines[:]:
            artist.remove()
        self.plot_ellipses(self.ax5, params)
        
        # Clear and redraw ellipses on experimental plot
        if hasattr(self, 'exp_dIdV') and self.exp_dIdV is not None:
            # Clear previous ellipses
            for artist in self.ax7.lines[:]:
                artist.remove()
            
            # Use same extent as simulated plot
            sim_extent = [-params['L'], params['L'], -params['L'], params['L']]
            
            # Plot ellipses using simulated coordinates
            self.plot_ellipses(self.ax7, params)
            
            # Set same extent as simulated plot
            self.ax7.set_xlim(sim_extent[:2])
            self.ax7.set_ylim(sim_extent[2:])
            self.ax7.set_aspect('equal')
            
        # Redraw ellipses
        for ax in [self.ax5, self.ax7]:
            for artist in ax.lines:
                ax.draw_artist(artist)
        
        # Plot ellipses on both charge and experimental plots
        self.plot_ellipses(self.ax5, params)
        if hasattr(self, 'exp_dIdV') and self.exp_dIdV is not None:
            self.plot_ellipses(self.ax7, params)
        
        # Update experimental data plots if available
        if hasattr(self, 'exp_dIdV') and self.exp_dIdV is not None:
            self.exp_idx = params['exp_slice']
            
            # Update dI/dV plot with proper colormap and scaling
            maxval = np.max(np.abs(self.exp_dIdV[self.exp_idx]))
            self.im7.set_data(self.exp_dIdV[self.exp_idx])
            self.im7.set_extent(self.exp_extent)
            self.im7.set_cmap('seismic')
            self.im7.set_clim(-maxval, maxval)
            self.ax7.set_title(f'Exp. dI/dV at {self.exp_biases[self.exp_idx]:.3f} V')
            self.ax7.set_xlabel('X [Å]')
            self.ax7.set_ylabel('Y [Å]')
            self.ax7.draw_artist(self.im7)
            
            # Update current plot with proper colormap
            self.im8.set_data(self.exp_I[self.exp_idx])
            self.im8.set_extent(self.exp_extent)
            self.im8.set_cmap('inferno')
            self.im8.set_clim(0.0, 600.0)
            self.ax8.set_title(f'Exp. Current at {self.exp_biases[self.exp_idx]:.3f} V')
            self.ax8.set_xlabel('X [Å]')
            self.ax8.set_ylabel('Y [Å]')
            self.ax8.draw_artist(self.im8)
            
            # Create and update overlay
            rgb_overlay, overlay_extent = self.create_overlay_image(
                self.exp_dIdV[self.exp_idx],
                total_charge,
                self.exp_extent,
                extent
            )
            self.im9.set_data(rgb_overlay)
            self.im9.set_extent(overlay_extent)
            self.ax9.set_title('Overlay (Red: Exp, Green: Sim)')
            self.ax9.set_xlabel('X [Å]')
            self.ax9.set_ylabel('Y [Å]')
            self.ax9.draw_artist(self.im9)
        else:
            # Clear experimental plots if no data
            self.im7.set_data(np.zeros((100, 100)))
            self.im8.set_data(np.zeros((100, 100)))
            self.im9.set_data(np.zeros((100, 100)))
            self.ax7.draw_artist(self.im7)
            self.ax8.draw_artist(self.im8)
            self.ax9.draw_artist(self.im9)
        
        # Blit the changes to the screen
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
