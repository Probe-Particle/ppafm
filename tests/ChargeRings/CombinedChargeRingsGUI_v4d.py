#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline

from GUITemplate import GUITemplate
import data_line 

sys.path.append('../../')
from pyProbeParticle import utils as ut
from pyProbeParticle import pauli
#from pauli import run_pauli_scan # Import the high-level scan function

import plot_utils as pu
import pauli_scan 
import exp_utils

verbosity = 0

class ApplicationWindow(GUITemplate):
    def __init__(self):
        # First call parent constructor
        super().__init__("Combined Charge Rings GUI v4")
        
        # {'VBias': 0.2, 'Rtip': 2.5, 'z_tip': 2.0, 'cCouling': 0.02, 'temperature': 3.0, 'onSiteCoulomb': 3.0, 'zV0': -3.3, 'zQd': 0.0, 'nsite': 3.0, 'radius': 5.2, 'phiRot': 0.79, 'R_major': 8.0, 'R_minor': 10.0, 'phi0_ax': 0.2, 'Esite': -0.04, 'Q0': 1.0, 'Qzz': 0.0, 'L': 20.0, 'npix': 100.0, 'decay': 0.3, 'dQ': 0.02, 'exp_slice': 10.0}
        # Then set parameter specifications
        self.param_specs = {
            # Tip Parameters
            'VBias':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.0, 10.0),   'value': 0.2, 'step': 0.1},
            'Rtip':          {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 10.0),   'value': 10.5, 'step': 0.5},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 20.0),   'value': 2.0, 'step': 0.5},
            
            # System Parameters
            'W':             {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.03, 'step': 0.001, 'decimals': 3},
            'GammaS':        {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.01, 'step': 0.001, 'decimals': 3},
            'GammaT':        {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.01, 'step': 0.001, 'decimals': 3},
            'Temp':          {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 0.224,  'step': 0.01},
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
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-1.0, 1.0),   'value': -0.20,'step': 0.002, 'decimals': 3},
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

            'p1_x':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': 15.0, 'step': 0.5},
            'p1_y':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': 15.0, 'step': 0.5},
            'p2_x':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': -15.0, 'step': 0.5},
            'p2_y':          {'group': 'scan',     'widget': 'double', 'range': (-20.0, 20.0),  'value': -15.0, 'step': 0.5},
            
            # Experimental Data
            'exp_slice':     {'group': 'Experimental Data', 'widget': 'int',    'range': (0, 13),     'value': 8,    'step': 1},
            'ep1_x':         {'group': 'Experimental Data', 'widget': 'double', 'range': (-20.0, 20.0),  'value':  9.72, 'step': 0.5},
            'ep1_y':         {'group': 'Experimental Data', 'widget': 'double', 'range': (-20.0, 20.0),  'value': -6.96, 'step': 0.5},
            'ep2_x':         {'group': 'Experimental Data', 'widget': 'double', 'range': (-20.0, 20.0),  'value': -11.0, 'step': 0.5},
            'ep2_y':         {'group': 'Experimental Data', 'widget': 'double', 'range': (-20.0, 20.0),  'value':  15.0, 'step': 0.5},
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
        self.ax5 = self.fig.add_subplot(335)  # dI/dV map
        self.ax6 = self.fig.add_subplot(336)  # Pauli Current
        self.ax7 = self.fig.add_subplot(337)  # Experimental dI/dV
        self.ax8 = self.fig.add_subplot(338)  # Experimental Current
        self.ax9 = self.fig.add_subplot(339)  # Tunneling
        
        # Initialize click-and-drag variables
        self.clicking = False
        self.start_point = None
        self.line_artist = None
        
        # Initialize scan line visualization
        self.scan_line_artist = None
        self.exp_scan_line_artist = None
        
        # Create scan button layout
        scan_layout = QtWidgets.QHBoxLayout()
        self.layout0.addLayout(scan_layout)
        
        # Add Run 1D Scan button
        btn_run_scan = QtWidgets.QPushButton('Run 1D Scan (p1-p2)')
        btn_run_scan.setToolTip('Run 1D scan between p1 and p2 points defined in parameters')
        btn_run_scan.clicked.connect(self.run_1d_scan_p1p2)
        scan_layout.addWidget(btn_run_scan)
        
        # Add Run Voltage Scan button
        btn_voltage_scan = QtWidgets.QPushButton('Run Voltage Scan (ep1-ep2)')
        btn_voltage_scan.setToolTip('Run voltage scan between ep1 and ep2 points defined in parameters')
        btn_voltage_scan.clicked.connect(self.run_voltage_scan_ep1ep2)
        scan_layout.addWidget(btn_voltage_scan)
        
        # Add Run Voltage Scan button for p1-p2
        btn_sim_voltage = QtWidgets.QPushButton('Run Voltage Scan (p1-p2)')
        btn_sim_voltage.setToolTip('Run voltage scan between p1 and p2 points defined in parameters')
        btn_sim_voltage.clicked.connect(self.run_voltage_scan_p1p2)
        scan_layout.addWidget(btn_sim_voltage)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)


        #ref_datline_fname = '/home/prokop/git/ppafm/tests/ChargeRings/Vlado/input/0.20_line_scan.dat'
        ref_datline_fname = './Vlado/input/0.20_line_scan.dat'
        self.ref_params, self.ref_columns, self.ref_data_line = data_line.read_dat_file(ref_datline_fname); 
        #print( "ref_params ", self.ref_params); 
        #print( "ref_columns ", self.ref_columns); 
        #print( "ref_data_line ", self.ref_data_line)
        #exit()
        
        # Plot reference data path
        self.draw_reference_line(self.ax4)

        self.pauli_solver = pauli.PauliSolver( nSingle=3, nleads=2, verbosity=verbosity )
        
        self.run()
        
    def load_experimental_data(self):
        """Load experimental data from npz file"""
        data = np.load('exp_rings_data.npz')
        # Convert from nm to Å (1 nm = 10 Å)
        self.exp_X      = data['X'] * 10
        self.exp_Y      = data['Y'] * 10
        self.exp_dIdV   = data['dIdV']
        self.exp_I      = data['I']
        self.exp_biases = data['biases']
        center_x        = data['center_x'] * 10  # Convert to Å
        center_y        = data['center_y'] * 10  # Convert to Å
        
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
        interpolator = RectBivariateSpline(y_src, x_src, data)
        resampled = interpolator(y_target, x_target, grid=True).reshape(len(y_target), len(x_target))
        
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
        # Use exp_utils function for creating overlay image
        return exp_utils.create_overlay_image(
            exp_data, 
            sim_data, 
            exp_extent,
            sim_extent,
            target_size=sim_data.shape[0]
        )

    def plot_ellipses(self, ax, params):
        """Plot ellipses for each quantum dot site
        
        Args:
            ax: matplotlib axis to plot on
            params: dictionary of parameters
        """
        # Use exp_utils function for plotting ellipses
        exp_utils.plot_ellipses(ax, params)

    def plot_experimental_data(self):
        """Plot experimental data in the bottom row"""
        # Get parameters
        params = self.get_param_values()
        self.exp_idx = params['exp_slice']
        #L = params['L']
        # Get plot extents
        #xmin, xmax = np.min(self.exp_X[0]), np.max(self.exp_X[0])
        #ymin, ymax = np.min(self.exp_Y[0]), np.max(self.exp_Y[0])
        #exp_extent = [xmin, xmax, ymin, ymax]
        #sim_extent = [-L, L, -L, L]
        
        # Create a wrapper for our draw_exp_scan_line method
        def draw_scan_line_wrapper(ax):
            self.draw_exp_scan_line(ax)
        
        # Get simulation data if available
        sim_image = self.ax5.images[0] if self.ax5.images else None
        sim_data = sim_image.get_array() if sim_image else None
        
        # Use exp_utils to plot the experimental data
        exp_utils.plot_experimental_data(
            self.exp_X, self.exp_Y, self.exp_dIdV, self.exp_I, self.exp_biases,
            self.exp_idx, params, sim_data, params,
            #axes=[self.ax7, self.ax8, self.ax9],
            #axes=[None, self.ax8, self.ax9],
            ax_current=self.ax8, ax_didv=self.ax9,
            draw_exp_scan_line_func=draw_scan_line_wrapper
        )

    def draw_exp_scan_line(self, ax):
        """Draw line between ep1 and ep2 points in the Experimental panel"""
        if self.exp_scan_line_artist:
            self.exp_scan_line_artist.remove()
        params = self.get_param_values()
        ep1 = (params['ep1_x'], params['ep1_y'])
        ep2 = (params['ep2_x'], params['ep2_y'])
        self.exp_scan_line_artist, = ax.plot(
            [ep1[0], ep2[0]], [ep1[1], ep2[1]], 
            'r-', linewidth=2, alpha=0.7
        )
        self.canvas.draw()

    def run(self):
        """Main calculation and plotting function"""
        params = self.get_param_values()
        self.ax1.cla(); self.ax2.cla(); self.ax3.cla() 
        self.ax4.cla(); self.ax5.cla(); self.ax6.cla()
        # Run scans with descriptive axis names
        pauli_scan.scan_xV(params,                    ax_V2d=self.ax1, ax_Vtip=self.ax2, ax_Esite=self.ax3)  # ax1=Esite(x,V), ax2=Vtip, ax3=Esite
        pauli_scan.scan_xy(params, self.pauli_solver, ax_Etot=self.ax4, ax_Ttot=self.ax7, ax_STM=self.ax5, ax_dIdV=self.ax6)  # ax4=Etot, ax9=Ttot, ax6=STM, ax5=dIdV
        self.draw_scan_line(self.ax4)
        self.draw_reference_line(self.ax4)
        self.plot_ellipses(self.ax9, params)
        self.plot_experimental_data()

        self.canvas.draw()
    
    def calculate_1d_scan(self, start_point, end_point, pointPerAngstrom=5 ):
        params = self.get_param_values()
        distance, Es, Ts, STM, x, y, x1, y1, x2, y2 = pauli_scan.calculate_1d_scan(  params, start_point, end_point, pointPerAngstrom )
        nsite = int(params['nsite'])
        ref_data_line = getattr(self, 'ref_data_line', None)
        ref_columns   = getattr(self, 'ref_columns', None)
        pauli_scan.plot_1d_scan_results( distance, Es, Ts, STM, nsite, ref_data_line, ref_columns )
        pauli_scan.save_1d_scan_data   ( params, distance, x, y, Es, Ts, STM, nsite, x1, y1, x2, y2 )

    def plot_voltage_line_scan_exp(self, start, end, pointPerAngstrom=5):
        """Plot simulated charge and experimental dI/dV along a line scan for different voltages"""
        params = self.get_param_values()
        
        # For simulation, use p1,p2 instead of ep1,ep2
        sim_start = (params['p1_x'], params['p1_y'])
        sim_end   = (params['p2_x'], params['p2_y'])
        
        print(f"Starting voltage line scan: Experiment: {start} to {end}, Simulation: {sim_start} to {sim_end}")
        
        # Create new figure for displaying in a Qt window
        fig = Figure(figsize=(12, 5))
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(121)  # Simulated charge
        ax2 = fig.add_subplot(122)  # Experimental dI/dV
        
        dist = ((sim_end[0]-sim_start[0])**2 + (sim_end[1]-sim_start[1])**2)**0.5
        sim_npoints = max(100, int(dist * pointPerAngstrom))
        # perform voltage scan
        Vbiases = self.exp_biases
        _, _, _, STM, _ = pauli_scan.calculate_xV_scan(params,sim_start,sim_end, ax_STM=ax1, nx=sim_npoints,nV=100,Vmin=0,Vmax=Vbiases[-1],bLegend=False )
        # style plot
        ax1.set_title('Simulated current (p1-p2)')
        ax1.set_xlabel('Distance (Å)')
        ax1.set_ylabel('Bias Voltage (V)')
        im1 = ax1.get_images()[0]
        fig.colorbar(im1, ax=ax1, label='Charge')
        
        # === Handle the experimental part (ep1,ep2) ===
        im2, (exp_didv, exp_distance) = exp_utils.plot_exp_voltage_line_scan(  
            self.exp_X, self.exp_Y, self.exp_dIdV, self.exp_biases, start, end, ax=ax2,title='(ep1-ep2)', pointPerAngstrom=pointPerAngstrom, ylims=(0, Vbiases[-1])
        )
        fig.colorbar(im2, ax=ax2, label='dI/dV')
        fig.tight_layout()
        canvas.draw()
        
        # Create a new window to display the plot
        window = QtWidgets.QMainWindow()
        window.setCentralWidget(canvas)
        window.resize(1200, 500)
        window.show()
        # Keep reference to prevent garbage collection
        self._exp_voltage_scan_window = window

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
        ax.plot(  self.ref_data_line[:,ix], self.ref_data_line[:,iy],   'b-', linewidth=1, alpha=0.7   )
        ax.plot( [self.ref_data_line[0,ix], self.ref_data_line[-1,ix]], [self.ref_data_line[0,iy], self.ref_data_line[-1,iy]], 'bo', markersize=5, alpha=0.7)
        self.canvas.draw()

    def run_1d_scan_p1p2(self):
        """Run 1D scan using p1 and p2 points from parameters"""
        params = self.get_param_values()
        p1 = (params['p1_x'], params['p1_y'])
        p2 = (params['p2_x'], params['p2_y'])
        
        self.calculate_1d_scan(p1, p2)
        
    def run_voltage_scan_ep1ep2(self):
        """Wrapper to invoke experimental voltage scan and open new window"""
        params = self.get_param_values()
        exp_start = (params['ep1_x'], params['ep1_y'])
        exp_end = (params['ep2_x'], params['ep2_y'])
        # Call plotting method
        self.plot_voltage_line_scan_exp(exp_start, exp_end)

    def run_voltage_scan_p1p2(self):
        """Run voltage-dependent scan along p1-p2 and plot Emax, STM, dI/dV"""
        params = self.get_param_values()
        start = (params['p1_x'], params['p1_y'])
        end = (params['p2_x'], params['p2_y'])
        # New figure window
        fig = Figure(figsize=(15, 5))
        canvas = FigureCanvas(fig)
        axE = fig.add_subplot(131)
        axS = fig.add_subplot(132)
        axD = fig.add_subplot(133)
        #axI = fig.add_subplot(144)
        # Perform scan
        x, V, Emax, STM, dIdV = pauli_scan.calculate_xV_scan( params, start, end, ax_Emax=axE, ax_STM=axS, ax_dIdV=axD, Vmin=0.0, Vmax=0.6, nx=100, nV=100)
        #pointPerAngstrom=5
        #distance, Es, Ts, STM_1d, x_1d, y, x1, y1, x2, y2 = pauli_scan.calculate_1d_scan(  params, start, end, pointPerAngstrom )
        #axI.plot( x, STM[-1,:], 'r-', label='I[-1]' )
        #axI.plot( x, STM[ 0,:], 'b-', label='I[0 ]' )
        #axI.plot( x_1d, STM_1d, 'g-', label='I_1d' )
        #axI.legend()

        fig.tight_layout()
        # Display in new Qt window
        window = QtWidgets.QMainWindow()
        window.setCentralWidget(canvas)
        #window.resize(1200, 500)
        window.show()
        # Keep reference to prevent garbage collection
        self._sim_voltage_scan_window = window

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
            self.line_artist = event.inaxes.plot([self.start_point[0], event.xdata], [self.start_point[1], event.ydata], 'r-')[0]
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
                self.plot_voltage_line_scan_exp(self.start_point, end_point)
            except Exception as e:
                print(f"Error in plot_voltage_line_scan_exp: {str(e)}")
                import traceback
                traceback.print_exc()
        
        self.start_point = None
        if self.line_artist:
            self.line_artist.remove()
            self.line_artist = None
        self.canvas.draw()
if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
