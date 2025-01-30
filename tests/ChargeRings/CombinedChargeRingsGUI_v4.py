#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
import matplotlib.pyplot as plt
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
            'VBias':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.0, 2.0),   'value': 0.2, 'step': 0.1},
            'Rtip':          {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 5.0),   'value': 2.5, 'step': 0.5},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 20.0),  'value': 2.0, 'step': 0.5},
            
            # System Parameters
            'cCouling':      {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.02, 'step': 0.01, 'decimals': 3},
            'temperature':   {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 3.0, 'step': 1.0},
            'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            
            # Mirror Parameters
            'zV0':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': -2.5, 'step': 0.1},
            'zQd':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value':  0.0,  'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 6.0, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.3,'step': 0.1},
            
            # Ellipse Parameters
            'R_major':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (1.0, 10.0),   'value': 8.0, 'step': 0.1},
            'R_minor':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (1.0, 10.0),   'value': 10.0, 'step': 0.1},
            'phi0_ax':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (-3.14, 3.14), 'value': 0.2, 'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-1.0, 1.0),   'value': -0.01,'step': 0.01},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0), 'value': 0.0, 'step': 0.5},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 50.0),  'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (50, 500),    'value': 100,  'step': 50},
            'decay':         {'group': 'Visualization',     'widget': 'double', 'range': (0.1, 2.0),   'value': 0.3,  'step': 0.1,   'decimals': 2},
            'dQ':            {'group': 'Visualization',     'widget': 'double', 'range': (0.001, 0.1), 'value': 0.02, 'step': 0.001, 'decimals': 3},
            
            # Experimental Data
            'exp_slice':     {'group': 'Experimental Data', 'widget': 'int',    'range': (0, 13),     'value': 8,    'step': 1},
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
        
        # Initialize click-and-drag variables
        self.clicking = False
        self.start_point = None
        self.line_artist = None
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
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

    def plot_ellipses(self, ax, params):
        """Plot ellipses for each quantum dot site
        
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
            
            # Translate to quantum dot position
            x = x_rot + qd_pos_x
            y = y_rot + qd_pos_y
            
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
        
        # Plot results
        plot_tip_potential(self.ax1, self.ax2, self.ax3, **tip_data, **params)
        plot_qdot_system(self.ax4, self.ax5, self.ax6, **qdot_data, **params)
        
        # Plot ellipses on total charge plot
        self.plot_ellipses(self.ax5, params)
        
        # Plot experimental data
        self.plot_experimental_data()
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse button press event"""
        if event.inaxes == self.ax4:  # Check if click is in Energies plot
            self.clicking = True
            self.start_point = (event.xdata, event.ydata)
            # Create temporary line
            if self.line_artist:
                self.line_artist.remove()
            self.line_artist = self.ax4.plot([event.xdata], [event.ydata], 'r--')[0]
            self.canvas.draw()

    def on_mouse_motion(self, event):
        """Handle mouse motion event"""
        if self.clicking and event.inaxes == self.ax4 and self.start_point:
            # Update temporary line
            if self.line_artist:
                self.line_artist.set_data(
                    [self.start_point[0], event.xdata],
                    [self.start_point[1], event.ydata]
                )
                self.canvas.draw()

    def on_mouse_release(self, event):
        """Handle mouse button release event"""
        if self.clicking and event.inaxes == self.ax4 and self.start_point:
            end_point = (event.xdata, event.ydata)
            self.clicking = False
            # Calculate and plot 1D scan
            self.calculate_1d_scan(self.start_point, end_point)
            self.start_point = None
            if self.line_artist:
                self.line_artist.remove()
                self.line_artist = None
            self.canvas.draw()

    def calculate_1d_scan(self, start_point, end_point):
        """Calculate and plot 1D scan between two points"""
        # Get current data from plots
        energy_data = self.ax4.images[0].get_array() if self.ax4.images else None
        charge_data = self.ax5.images[0].get_array() if self.ax5.images else None
        stm_data    = self.ax6.images[0].get_array() if self.ax6.images else None
        
        if not all([energy_data is not None, charge_data is not None, stm_data is not None]):
            print("No data available for 1D scan")
            return
            
        params = self.get_param_values()
        L    = params['L']
        npix = params['npix']
        
        # Convert plot coordinates to pixel coordinates
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Convert to pixel coordinates
        px1 = int((x1 + L) * (npix - 1) / (2 * L))
        py1 = int((y1 + L) * (npix - 1) / (2 * L))
        px2 = int((x2 + L) * (npix - 1) / (2 * L))
        py2 = int((y2 + L) * (npix - 1) / (2 * L))
        
        # Create line coordinates using Bresenham's algorithm
        num_points = int(np.sqrt((px2 - px1)**2 + (py2 - py1)**2))
        x = np.linspace(px1, px2, num_points)
        y = np.linspace(py1, py2, num_points)
        x = x.astype(int)
        y = y.astype(int)
        
        # Extract values along the line
        energy_profile = energy_data[y, x]
        charge_profile = charge_data[y, x]
        stm_profile = stm_data[y, x]
        
        # Calculate real space coordinates for plotting
        real_x = np.linspace(-L, L, npix)[x]
        real_y = np.linspace(-L, L, npix)[y]
        distance = np.sqrt((real_x - real_x[0])**2 + (real_y - real_y[0])**2)
        
        # Create new figure for 1D scan
        scan_fig = plt.figure(figsize=(10, 8))
        ax1 = scan_fig.add_subplot(311)
        ax2 = scan_fig.add_subplot(312)
        ax3 = scan_fig.add_subplot(313)
        
        # Plot profiles
        ax1.plot(distance, energy_profile, 'b-', label='Energy')
        ax1.set_ylabel('Energy')
        ax1.legend()
        
        ax2.plot(distance, charge_profile, 'r-', label='Charge')
        ax2.set_ylabel('Charge')
        ax2.legend()
        
        ax3.plot(distance, stm_profile, 'g-', label='STM')
        ax3.set_ylabel('STM')
        ax3.set_xlabel('Distance [Å]')
        ax3.legend()
        
        scan_fig.tight_layout()
        plt.show()
        
        # Save data to file
        save_data = np.column_stack((distance, energy_profile, charge_profile, stm_profile))
        header   = 'Distance[A] Energy Charge STM\nLine scan from ({:.2f}, {:.2f}) to ({:.2f}, {:.2f})'.format( real_x[0], real_y[0], real_x[-1], real_y[-1] )
        filename = 'line_scan_{:.1f}_{:.1f}_to_{:.1f}_{:.1f}.dat'.format( real_x[0], real_y[0], real_x[-1], real_y[-1] )
        np.savetxt(filename, save_data, header=header)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
