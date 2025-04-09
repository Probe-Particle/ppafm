#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import LinearNDInterpolator

from GUITemplate import GUITemplate
from charge_rings_core import calculate_tip_potential, calculate_qdot_system, makeCircle, compute_site_energies, compute_site_tunelling, occupancy_FermiDirac
from charge_rings_plotting import plot_tip_potential, plot_qdot_system

import data_line 

class ApplicationWindow(GUITemplate):
    def __init__(self):
        # First call parent constructor
        super().__init__("Combined Charge Rings GUI v4")
        
        # {'VBias': 0.2, 'Rtip': 2.5, 'z_tip': 2.0, 'cCouling': 0.02, 'temperature': 3.0, 'onSiteCoulomb': 3.0, 'zV0': -3.3, 'zQd': 0.0, 'nsite': 3.0, 'radius': 5.2, 'phiRot': 0.79, 'R_major': 8.0, 'R_minor': 10.0, 'phi0_ax': 0.2, 'Esite': -0.04, 'Q0': 1.0, 'Qzz': 0.0, 'L': 20.0, 'npix': 100.0, 'decay': 0.3, 'dQ': 0.02, 'exp_slice': 10.0}
        # Then set parameter specifications
        self.param_specs = {
            # Tip Parameters
            'VBias':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.0, 2.0),   'value': 0.2, 'step': 0.1},
            'Rtip':          {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 5.0),   'value': 2.5, 'step': 0.5},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 20.0),  'value': 2.0, 'step': 0.5},
            
            # System Parameters
            'cCouling':      {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.02, 'step': 0.001, 'decimals': 3},
            'temperature':   {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 3.0,  'step': 1.0},
            'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            
            # Mirror Parameters
            'zV0':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': -3.3, 'step': 0.1},
            'zQd':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value':  0.0, 'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 5.2, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': 0.8,'step': 0.1},
            
            # Ellipse Parameters
            'R_major':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (1.0, 10.0),   'value': 8.0, 'step': 0.1},
            'R_minor':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (1.0, 10.0),   'value': 10.0, 'step': 0.1},
            'phi0_ax':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (-3.14, 3.14), 'value': 0.2, 'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-1.0, 1.0),   'value': -0.04,'step': 0.002, 'decimals': 3},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0), 'value': 0.0, 'step': 0.5},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 50.0),  'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (50, 500),    'value': 100,  'step': 50},
            'decay':         {'group': 'Visualization',     'widget': 'double', 'range': (0.1, 2.0),   'value': 0.3,  'step': 0.1,   'decimals': 2},
            'dQ':            {'group': 'Visualization',     'widget': 'double', 'range': (0.001, 0.1), 'value': 0.02, 'step': 0.001, 'decimals': 3},


            # 1D scan end points
            # 'p1_x':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': -6.5, 'step': 0.5},
            # 'p1_y':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': 10.0, 'step': 0.5},
            # 'p2_x':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value':  6.5, 'step': 0.5},
            # 'p2_y':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': -10.0, 'step': 0.5},

# 1.50000000e+01 1.50000000e+01
            'p1_x':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': 15.0, 'step': 0.5},
            'p1_y':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': 15.0, 'step': 0.5},
            'p2_x':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': -15.0, 'step': 0.5},
            'p2_y':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': -15.0, 'step': 0.5},
            
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
        
        # Initialize scan line visualization
        self.scan_line_artist = None
        
        # Create scan button layout
        scan_layout = QtWidgets.QHBoxLayout()
        self.layout0.addLayout(scan_layout)
        
        # Add Run 1D Scan button
        btn_run_scan = QtWidgets.QPushButton('Run 1D Scan (p1-p2)')
        btn_run_scan.setToolTip('Run 1D scan between p1 and p2 points defined in parameters')
        btn_run_scan.clicked.connect(self.run_1d_scan_p1p2)
        scan_layout.addWidget(btn_run_scan)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)


        ref_datline_fname = '/home/prokop/git/ppafm/tests/ChargeRings/Vlado/input/0.20_line_scan.dat'
        self.ref_params, self.ref_columns, self.ref_data_line = data_line.read_dat_file(ref_datline_fname); 
        print( "ref_params ", self.ref_params); 
        print( "ref_columns ", self.ref_columns); 
        print( "ref_data_line ", self.ref_data_line)
        #exit()
        
        # Plot reference data path
        self.draw_reference_line(self.ax4)
        
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
            rgb_overlay, extent = self.create_overlay_image( self.exp_dIdV[self.exp_idx], sim_data, exp_extent, sim_extent )
            
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
        tip_data  = calculate_tip_potential(**params)
        qdot_data = calculate_qdot_system(**params)
        
        # Plot results
        plot_tip_potential(self.ax1, self.ax2, self.ax3, **tip_data, **params)
        plot_qdot_system(self.ax4, self.ax5, self.ax6, **qdot_data, **params)
        self.draw_scan_line(self.ax4)
        self.draw_reference_line(self.ax4)
        
        # Plot ellipses on total charge plot
        self.plot_ellipses(self.ax5, params)
        
        # Plot experimental data
        self.plot_experimental_data()
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse button press event"""
        if event.inaxes in [self.ax4, self.ax7]:  # Energy plot or experimental dI/dV
            self.clicking = True
            self.start_point = (event.xdata, event.ydata)
            print(f"Mouse press in {'energy plot' if event.inaxes == self.ax4 else 'experimental plot'} at {self.start_point}")

    def on_mouse_motion(self, event):
        """Handle mouse motion event"""
        if self.clicking and event.inaxes in [self.ax4, self.ax7]:
            # Remove old line if it exists
            if self.line_artist:
                self.line_artist.remove()
            
            # Draw line from start point to current point
            self.line_artist = event.inaxes.plot([self.start_point[0], event.xdata],
                                               [self.start_point[1], event.ydata], 'r-')[0]
            self.canvas.draw()

    def on_mouse_release(self, event):
        """Handle mouse button release event"""
        if not self.clicking or event.inaxes is None:
            return
            
        end_point = (event.xdata, event.ydata)
        self.clicking = False
        
        print(f"Mouse release in subplot {event.inaxes}")
        print(f"Start point: {self.start_point}")
        print(f"End point: {end_point}")
        
        if event.inaxes == self.ax4:  # Energy plot (2,1)
            print("Processing energy plot line scan")
            self.calculate_1d_scan(self.start_point, end_point)
        elif event.inaxes == self.ax7:  # Experimental dI/dV plot (3,2)
            print("Processing experimental plot voltage scan")
            try:
                self.plot_voltage_line_scan(self.start_point, end_point)
            except Exception as e:
                print(f"Error in plot_voltage_line_scan: {str(e)}")
                import traceback
                traceback.print_exc()
        
        self.start_point = None
        if self.line_artist:
            self.line_artist.remove()
            self.line_artist = None
        self.canvas.draw()

    def calculate_1d_scan(self, start_point, end_point, pointPerAngstrom=5 ):
        """Calculate 1D scan between two points"""
        params = self.get_param_values()
        L = params['L']
        nsite = params['nsite']
        
        # Create line coordinates in real space (no rounding)
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Calculate number of points based on distance
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        npoints = max(100, int(dist * pointPerAngstrom))
        
        # Create line coordinates
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        distance = np.sqrt((x - x[0])**2 + (y - y[0])**2)
        
        # Create positions array for calculations
        pTips = np.zeros((npoints, 3))
        pTips[:,0] = x
        pTips[:,1] = y
        pTips[:,2] = params['z_tip'] + params['Rtip']
        
        # Calculate site positions
        spos, phis = makeCircle(n=nsite, R=params['radius'], phi0=params['phiRot'])
        spos[:,2] = params['zQd']
        
        # Calculate energies and charges for each site
        Esite_arr = np.full(nsite, params['Esite'])
        Es = compute_site_energies(pTips, spos, VBias=params['VBias'], Rtip=params['Rtip'], zV0=params['zV0'], E0s=Esite_arr)
        
        # Calculate tunneling and charges for each site
        Ts = compute_site_tunelling(pTips, spos, beta=params['decay'], Amp=1.0)
        Qs = np.zeros(Es.shape)
        Is = np.zeros(Es.shape)
        for i in range(nsite):
            Qs[:,i] = occupancy_FermiDirac(Es[:,i], params['temperature'])
            Is[:,i] = Ts[:,i] * (1-Qs[:,i])
        
        Qtot = np.sum(Qs, axis=1)
        STM  = np.sum(Is, axis=1)
        
        # Plot results
        self.plot_1d_scan_results(distance, Es, Qs, Is, Qtot, STM, nsite)
        
        # Save data to file
        self.save_1d_scan_data(params, distance, x, y, Es, Qs, Is, Qtot, STM, nsite, x1, y1, x2, y2)

    def plot_1d_scan_results(self, distance, Es, Qs, Is, Qtot, STM, nsite):
        """Plot results of 1D scan"""
        # Create new figure for 1D scan
        scan_fig = plt.figure(figsize=(10, 12))
        ax1 = scan_fig.add_subplot(311)
        ax2 = scan_fig.add_subplot(312)
        ax3 = scan_fig.add_subplot(313)

        bRef = False
        if hasattr(self, 'ref_data_line'):
            if self.ref_data_line is not None:
                bRef = True
        
        # Plot individual site energies
        for i in range(nsite):
            ax1.plot(distance, Es[:,i], '-', label=f'Site {i+1}')
            
            # Plot reference energies if available
            if bRef:
                icol = self.ref_columns[f'Esite_{i+1}']
                ax1.plot( self.ref_data_line[:,0],  self.ref_data_line[:,icol], '--', alpha=0.7, label=f'Ref Site {i+1}' )
                
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot individual site charges
        for i in range(nsite):
            ax2.plot(distance, Qs[:,i], '-', label=f'Site {i+1}')
        ax2.plot(distance, Qtot, 'k-', label='Total', linewidth=2)
        ax2.set_ylabel('Charge')
        ax2.legend()
        ax2.grid(True)
        
        # Plot individual site currents
        for i in range(nsite):
            ax3.plot(distance, Is[:,i], '-', label=f'Site {i+1}')
            
            # Plot reference tunneling if available
            if bRef:
                icol = self.ref_columns[f'Tsite_{i+1}']
                ax3.plot( self.ref_data_line[:,0],  self.ref_data_line[:,icol], '--', alpha=0.7, label=f'Ref Site {i+1}' )
                
        ax3.plot(distance, STM, 'k-', label='Total', linewidth=2)
        ax3.set_ylabel('Current')
        ax3.legend()
        ax3.grid(True)
        
        scan_fig.tight_layout()
        plt.show()
    
    def save_1d_scan_data(self, params, distance, x, y, Es, Qs, Is, Qtot, STM, nsite, x1, y1, x2, y2):
        """Save 1D scan data to file"""
        # Prepare header with parameters
        param_header = "# Calculation parameters:\n"
        for key, value in params.items():
            param_header += f"# {key}: {value}\n"
        
        # Add column description
        param_header += "#\n# Data columns:\n"
        param_header += "# 1: Distance[A]\n"
        param_header += "# 2: x[A]\n"
        param_header += "# 3: y[A]\n"
        for i in range(nsite):
            param_header += f"# {i+4}: Esite_{i+1}\n"
        for i in range(nsite):
            param_header += f"# {i+4+nsite}: Qsite_{i+1}\n"
        for i in range(nsite):
            param_header += f"# {i+4+2*nsite}: Isite_{i+1}\n"
        param_header += f"# {4+3*nsite}: Qtotal\n"
        param_header += f"# {5+3*nsite}: STM_total\n"
        
        # Add line coordinates
        param_header += f"\n# Line scan from ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})"
        
        # Combine all data
        save_data = np.column_stack([distance, x, y] + 
                                  [Es[:,i] for i in range(nsite)] + 
                                  [Qs[:,i] for i in range(nsite)] + 
                                  [Is[:,i] for i in range(nsite)] + 
                                  [Qtot, STM])
        
        filename = 'line_scan_{:.1f}_{:.1f}_to_{:.1f}_{:.1f}.dat'.format(x1, y1, x2, y2)
        np.savetxt(filename, save_data, header=param_header)
        print(f"Data saved to {filename}")

    def plot_voltage_line_scan(self, start_point, end_point, pointPerAngstrom=5):
        """Plot simulated charge and experimental dI/dV along a line scan for different voltages"""
        print(f"Starting voltage line scan from {start_point} to {end_point}")
        
        # Create new figure
        fig = Figure(figsize=(12, 5))
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(121)  # Simulated charge
        ax2 = fig.add_subplot(122)  # Experimental dI/dV
        
        # Create line coordinates in real space
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Calculate number of points based on distance
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        npoints = max(100, int(dist * pointPerAngstrom))
        
        # Create line coordinates
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        distance = np.sqrt((x - x[0])**2 + (y - y[0])**2)
        line_points = np.column_stack((x, y))
        
        # Create arrays to store results
        sim_charge = np.zeros((len(self.exp_biases), npoints))
        exp_didv = np.zeros((len(self.exp_biases), npoints))
        
        # Get current parameters
        params = self.get_param_values()
        nsite = params['nsite']
        
        # Calculate site positions (constant for all voltages)
        spos, phis = makeCircle(n=nsite, R=params['radius'], phi0=params['phiRot'])
        spos[:,2] = params['zQd']
        Esite_arr = np.full(nsite, params['Esite'])
        
        # Create positions array for calculations
        pTips = np.zeros((npoints, 3))
        pTips[:,0] = x
        pTips[:,1] = y
        pTips[:,2] = params['z_tip'] + params['Rtip']
        
        # Calculate tunneling (constant for all voltages)
        Ts = compute_site_tunelling(pTips, spos, beta=params['decay'], Amp=1.0)
        
        # Pre-calculate experimental interpolators for each voltage slice
        interpolators = []
        for i in range(len(self.exp_biases)):
            exp_data = self.exp_dIdV[i]
            exp_x = self.exp_X[i]
            exp_y = self.exp_Y[i]
            points = np.column_stack((exp_x.flatten(), exp_y.flatten()))
            values = exp_data.flatten()
            interp = LinearNDInterpolator(points, values)
            interpolators.append(interp)
        
        # Loop over voltages
        print("Processing voltages...")
        for i, vbias in enumerate(self.exp_biases):
            # Update VBias in parameters
            params['VBias'] = vbias
            
            # Calculate energies and charges for each site
            Es = compute_site_energies(pTips, spos, VBias=vbias, Rtip=params['Rtip'], 
                                     zV0=params['zV0'], E0s=Esite_arr)
            
            # Calculate charges for each site
            Qs = np.zeros(Es.shape)
            for j in range(nsite):
                Qs[:,j] = occupancy_FermiDirac(Es[:,j], params['temperature'])
            
            # Store total charge for this voltage
            sim_charge[i,:] = np.sum(Qs, axis=1)
            
            # Interpolate experimental data using pre-calculated interpolator
            exp_didv[i,:] = interpolators[i](line_points)
            
            # Fill NaN values with nearest values if any exist
            nan_mask = np.isnan(exp_didv[i,:])
            if np.any(nan_mask):
                valid_mask = ~np.isnan(exp_didv[i,:])
                if np.any(valid_mask):
                    # Use simple 1D interpolation for NaN filling
                    valid_indices = np.where(valid_mask)[0]
                    nan_indices = np.where(nan_mask)[0]
                    exp_didv[i,nan_indices] = np.interp(nan_indices, valid_indices, exp_didv[i,valid_indices])
        
        print("Creating plots...")
        # Plot simulated charge
        im1 = ax1.imshow(sim_charge, aspect='auto', origin='lower', 
                        extent=[0, distance[-1], self.exp_biases[0], self.exp_biases[-1]])
        ax1.set_title('Simulated Charge')
        ax1.set_xlabel('Distance (Å)')
        ax1.set_ylabel('Bias Voltage (V)')
        fig.colorbar(im1, ax=ax1, label='Charge')
        
        # Plot experimental dI/dV
        im2 = ax2.imshow(exp_didv, aspect='auto', origin='lower',
                        extent=[0, distance[-1], self.exp_biases[0], self.exp_biases[-1]])
        ax2.set_title('Experimental dI/dV')
        ax2.set_xlabel('Distance (Å)')
        ax2.set_ylabel('Bias Voltage (V)')
        fig.colorbar(im2, ax=ax2, label='dI/dV')
        
        # Adjust layout and show
        fig.tight_layout()
        
        # Create a new window to display the plot
        window = QtWidgets.QMainWindow()
        window.setCentralWidget(canvas)
        window.resize(1200, 500)
        window.show()
        
        # Keep a reference to prevent garbage collection
        self._voltage_scan_window = window

    def draw_scan_line(self, ax):
        """Draw line between p1 and p2 points in the Energies panel"""
        if self.scan_line_artist:
            self.scan_line_artist.remove()
            
        params = self.get_param_values()
        p1 = (params['p1_x'], params['p1_y'])
        p2 = (params['p2_x'], params['p2_y'])
        
        self.scan_line_artist, = ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], 
            'r-', linewidth=2, alpha=0.7
        )
        self.canvas.draw()
    
    def draw_reference_line(self, ax):
        """Plot reference data path from loaded file"""
        if not hasattr(self, 'ref_data_line') or self.ref_data_line is None:
            return
            
        # Get column indices
        ix = self.ref_columns['x[A]']
        iy = self.ref_columns['y[A]']
        
        # Plot the path
        ax.plot(
            self.ref_data_line[:,ix], self.ref_data_line[:,iy], 
            'b-', linewidth=1, alpha=0.7
        )
        ax.plot(
            [self.ref_data_line[0,ix], self.ref_data_line[-1,ix]],
            [self.ref_data_line[0,iy], self.ref_data_line[-1,iy]],
            'bo', markersize=5, alpha=0.7
        )
        self.canvas.draw()

    def run_1d_scan_p1p2(self):
        """Run 1D scan using p1 and p2 points from parameters"""
        params = self.get_param_values()
        p1 = (params['p1_x'], params['p1_y'])
        p2 = (params['p2_x'], params['p2_y'])
        
        self.calculate_1d_scan(p1, p2)

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
