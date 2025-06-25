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
import orbital_utils
import json, os
import numpy as _np

verbosity = 0

kBoltz = 8.617333262e-5 # eV/K

class ApplicationWindow(GUITemplate):
    def __init__(self):
        # First call parent constructor
        super().__init__("Combined Charge Rings GUI v4")
        
        self.orbital_2D = None
        self.orbital_lvec = None
        
        # {'VBias': 0.2, 'Rtip': 2.5, 'z_tip': 2.0, 'cCouling': 0.02, 'temperature': 3.0, 'onSiteCoulomb': 3.0, 'zV0': -3.3, 'zQd': 0.0, 'nsite': 3.0, 'radius': 5.2, 'phiRot': 0.79, 'R_major': 8.0, 'R_minor': 10.0, 'phi0_ax': 0.2, 'Esite': -0.04, 'Q0': 1.0, 'Qzz': 0.0, 'L': 20.0, 'npix': 100.0, 'decay': 0.3, 'dQ': 0.02, 'exp_slice': 10.0}
        # Then set parameter specifications
        self.param_specs = {

            #'nsite':         {'group': 'Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 5.2, 'step': 0.5},
            'phiRot':        {'group': 'Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.3,'step': 0.1},
            'phi0_ax':       {'group': 'Geometry',     'widget': 'double', 'range': (-3.14, 3.14), 'value': 0.2, 'step': 0.1},

            # Tip Parameters
            'VBias':         {'group': 'Electrostatic Field', 'widget': 'double', 'range': (0.0, 10.0),   'value':  0.70, 'step': 0.02},
            'Rtip':          {'group': 'Electrostatic Field', 'widget': 'double', 'range': (0.5, 10.0),   'value':  3.0,  'step': 0.5},
            'z_tip':         {'group': 'Electrostatic Field', 'widget': 'double', 'range': (0.5, 20.0),   'value':  5.0,  'step': 0.5},
            'zV0':           {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0,  'step': 0.1},
            'zVd':           {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-5.0, 50.0),  'value':  15.0, 'step': 0.1},
            'zQd':           {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-5.0, 5.0),   'value':  0.0,  'step': 0.1},
            'Q0':            {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0,   'step': 0.1},
            'Qzz':           {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-20.0, 20.0), 'value': 10.0,  'step': 0.5},

            'Esite':         {'group': 'Transport Solver',  'widget': 'double', 'range': (-1.0, 1.0),   'value': -0.100, 'step': 0.002, 'decimals': 3 },
            'W':             {'group': 'Transport Solver',  'widget': 'double', 'range': (0.0, 1.0),    'value': 0.05,   'step': 0.001, 'decimals': 3 },
            'Temp':          {'group': 'Transport Solver',  'widget': 'double', 'range': (0.0, 100.0),  'value': 3.0,   'step': 0.05,  'decimals': 2 },
            'decay':         {'group': 'Transport Solver',  'widget': 'double', 'range': (0.1, 2.0),    'value': 0.3,    'step': 0.1,   'decimals': 2 },
            'GammaS':        {'group': 'Transport Solver',  'widget': 'double', 'range': (0.0, 1.0),    'value': 0.01,   'step': 0.001, 'decimals': 3, 'fidget': False },
            'GammaT':        {'group': 'Transport Solver',  'widget': 'double', 'range': (0.0, 1.0),    'value': 0.01,   'step': 0.001, 'decimals': 3, 'fidget': False },
            #'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,    'step': 0.1  },
                        
            # Barrier
            'Et0':          {'group': 'Barrier', 'widget': 'double', 'range': (0.0, 10.0),   'value':  0.2,    'step': 0.01  }, # E0 base height of tunelling barrier
            'wt':           {'group': 'Barrier', 'widget': 'double', 'range': (0.0, 20.0),   'value':  8.0,    'step': 0.1  }, # Amp Amplitude or tunelling barrirer modification
            'At':           {'group': 'Barrier', 'widget': 'double', 'range': (-10.0, 10.0), 'value': -0.1,    'step': 0.01  }, # w Gaussain width for tunelling barrirer modification
            'c_orb':        {'group': 'Barrier', 'widget': 'double', 'range': (0.0, 1.0),    'value':  1.0,    'step': 0.0001, 'decimals': 4  }, # c_orb weight for orbital tunneling
            'T0':           {'group': 'Barrier', 'widget': 'double', 'range': (0.0, 1000.0), 'value':  1.0,    'step': 0.0001, 'decimals': 4  }, # c_orb weight for orbital tunneling

            # Visualization
            'L':             {'group': 'Visualization', 'widget': 'double', 'range': (5.0, 50.0),   'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization', 'widget': 'int',    'range': (50, 500),     'value': 200,  'step': 50},
            'dQ':            {'group': 'Visualization', 'widget': 'double', 'range': (0.001, 0.1),  'value': 0.02, 'step': 0.001, 'decimals': 3},
            #'R_major':       {'group': 'Visualization', 'widget': 'double', 'range': (1.0, 10.0),   'value': 8.0,  'step': 0.1},
            #'R_minor':       {'group': 'Visualization', 'widget': 'double', 'range': (1.0, 10.0),   'value': 10.0, 'step': 0.1},
            
            # simulation end-points for 1D scan
            'p1_x':         {'group': 'Data Cuts', 'widget': 'double', 'range': (-20.0, 20.0),  'value':  9.72, 'step': 0.5,'fidget': False},
            'p1_y':         {'group': 'Data Cuts', 'widget': 'double', 'range': (-20.0, 20.0),  'value': -9.96, 'step': 0.5,'fidget': False},
            'p2_x':         {'group': 'Data Cuts', 'widget': 'double', 'range': (-20.0, 20.0),  'value': -11.0, 'step': 0.5,'fidget': False},
            'p2_y':         {'group': 'Data Cuts', 'widget': 'double', 'range': (-20.0, 20.0),  'value':  12.0, 'step': 0.5,'fidget': False},
            
            # Experimental end-points for 1D scan 
            'exp_slice':     {'group': 'Data Cuts', 'widget': 'int',    'range': (0, 13),        'value': 10,    'step': 1},
            'ep1_x':         {'group': 'Data Cuts', 'widget': 'double', 'range': (-20.0, 20.0),  'value':  9.72, 'step': 0.5,'fidget': False},
            'ep1_y':         {'group': 'Data Cuts', 'widget': 'double', 'range': (-20.0, 20.0),  'value': -6.96, 'step': 0.5,'fidget': False},
            'ep2_x':         {'group': 'Data Cuts', 'widget': 'double', 'range': (-20.0, 20.0),  'value': -11.0, 'step': 0.5,'fidget': False},
            'ep2_y':         {'group': 'Data Cuts', 'widget': 'double', 'range': (-20.0, 20.0),  'value':  15.0, 'step': 0.5,'fidget': False},
        }
        
        # Track probability figures for window management
        self.prob_figs = {}

        
        self.create_gui()
        # Add Save All button next to Save/Load
        btnSaveAll = QtWidgets.QPushButton("Save All")
        btnSaveAll.clicked.connect(self.save_everything)
        self.hbSaveLoad.addWidget(btnSaveAll)
        
        # Load experimental data
        self.load_experimental_data()
        

        # Setup matplotlib figure with 3x3 layout
        self.fig = Figure(figsize=(15, 10))
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
        
        # Define control checkboxes
        self.cbShowProbs = QtWidgets.QCheckBox('probabilities')
        self.cbShowProbs.stateChanged.connect(self.run)
        self.cbMirror    = QtWidgets.QCheckBox('Mirror')
        self.cbMirror.setChecked(True)
        self.cbMirror.stateChanged.connect(self.run)
        self.cbRamp      = QtWidgets.QCheckBox('Ramp')
        self.cbRamp.setChecked(True)
        self.cbRamp.stateChanged.connect(self.run)
        self.cbPlotXV    = QtWidgets.QCheckBox('Compare in xV')
        self.cbPlotXV.setChecked(False)
        self.cbPlotXV.stateChanged.connect(self.run)
        
        # Controls row: checkboxes for probabilities, Mirror, Ramp, Compare in xV
        controls_layout = QtWidgets.QHBoxLayout()
        self.layout0.addLayout(controls_layout)
        controls_layout.addWidget(self.cbShowProbs)
        controls_layout.addWidget(self.cbMirror)
        controls_layout.addWidget(self.cbRamp)
        controls_layout.addWidget(self.cbPlotXV)
        # Checkbox for site/state energies
        self.cbShowEnergies = QtWidgets.QCheckBox('Energies')
        self.cbShowEnergies.stateChanged.connect(self.run)
        controls_layout.addWidget(self.cbShowEnergies)

        # Create orbital input layout
        orbital_layout = QtWidgets.QHBoxLayout()
        self.layout0.addLayout(orbital_layout)
        orbital_layout.addWidget(QtWidgets.QLabel("Orbital file:"))
        self.leOrbitalFile = QtWidgets.QLineEdit()
        self.leOrbitalFile.setText("QD.cub")
        orbital_layout.addWidget(self.leOrbitalFile)
        btnLoadOrb = QtWidgets.QPushButton("Load Orbital")
        btnLoadOrb.clicked.connect(self.load_orbital_file)
        orbital_layout.addWidget(btnLoadOrb)
        self.cbUseOrbital = QtWidgets.QCheckBox("Orbital hopping")
        #self.cbUseOrbital.setChecked(True)
        self.cbUseOrbital.setChecked(False)
        self.load_orbital_file()
        orbital_layout.addWidget(self.cbUseOrbital)
        self.cbUseOrbital.stateChanged.connect(self.run)

        # LinSolver options
        solver_layout = QtWidgets.QHBoxLayout()
        self.layout0.addLayout(solver_layout)
        solver_layout.addWidget(QtWidgets.QLabel("LinSolver:"))
        self.comboLinSolver = QtWidgets.QComboBox()
        self.comboLinSolver.addItem("Gass")
        self.comboLinSolver.addItem("SVD")
        #self.comboLinSolver.addItem("LAPACK")
        solver_layout.addWidget(self.comboLinSolver)
        solver_layout.addWidget(QtWidgets.QLabel("MaxIter:"))
        self.sbMaxIter = QtWidgets.QSpinBox()
        self.sbMaxIter.setRange(1, 10000)
        self.sbMaxIter.setValue(50)
        solver_layout.addWidget(self.sbMaxIter)
        solver_layout.addWidget(QtWidgets.QLabel("Tol:"))
        self.dsTol = QtWidgets.QDoubleSpinBox()
        self.dsTol.setDecimals(12)
        self.dsTol.setSingleStep(1e-12)
        self.dsTol.setRange(0.0, 1.0)
        self.dsTol.setValue(10e-12)
        solver_layout.addWidget(self.dsTol)
        # Connect solver settings changes
        self.comboLinSolver.currentIndexChanged.connect(self.update_lin_solver)
        self.sbMaxIter.valueChanged.connect(self.update_lin_solver)
        self.dsTol.valueChanged.connect(self.update_lin_solver)

        # Colormap selection
        self.cmap_dIdV_options = ['bwr', 'bwr_r', 'seismic', 'seismic_r', 'PiYG', 'PiYG_r', 'PRGn', 'PRGn_r', 'RdBu', 'RdBu_r',  'vanimo', 'vanimo_r','coolwarm', 'coolwarm_r', 'PuRdR-w-BuGn', 'BuGnR-w-PuRd' ]
        self.cmap_STM_options  = ['hot', 'afmhot', 'gnuplot2', 'seismic', 'inferno', 'viridis', 'plasma', 'magma']
        
        # Create colormap selection widgets
        self.cmap_dIdV_combo = QtWidgets.QComboBox()
        self.cmap_dIdV_combo.addItems(self.cmap_dIdV_options)
        self.cmap_dIdV_combo.setCurrentText(pauli_scan.cmap_dIdV)
        self.cmap_dIdV_combo.currentTextChanged.connect(self.on_cmap_changed)
        
        self.cmap_STM_combo = QtWidgets.QComboBox()
        self.cmap_STM_combo.addItems(self.cmap_STM_options)
        self.cmap_STM_combo.setCurrentText(pauli_scan.cmap_STM)
        self.cmap_STM_combo.currentTextChanged.connect(self.on_cmap_changed)
        
        # Add to layout
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_layout.addWidget(QtWidgets.QLabel("dIdV cmap:"))
        cmap_layout.addWidget(self.cmap_dIdV_combo)
        cmap_layout.addWidget(QtWidgets.QLabel("STM cmap:"))
        cmap_layout.addWidget(self.cmap_STM_combo)
        self.layout0.addLayout(cmap_layout)

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        # Plot reference data path
        self.draw_reference_line(self.ax4)

        self.pauli_solver = pauli.PauliSolver( nSingle=3, nleads=2, verbosity=verbosity )
        # apply initial linear solver settings
        self.update_lin_solver()
        self.run()

    def load_experimental_data(self):
        """Load experimental data from npz file"""
        try:
            data = np.load('exp_rings_data.npz')
            # Convert from nm to Å
            self.exp_X      = data['X'] * 10
            self.exp_Y      = data['Y'] * 10
            self.exp_dIdV   = data['dIdV']
            self.exp_I      = data['I']
            self.exp_biases = data['biases']
            # Center coordinates
            cx, cy = data['center_x']*10, data['center_y']*10
            self.exp_X -= cx; self.exp_Y -= cy
            # Update exp_slice range and initial index
            self.param_specs['exp_slice']['range'] = (0, len(self.exp_biases)-1)
            self.param_specs['exp_slice']['value'] = len(self.exp_biases)//2
            self.exp_idx = len(self.exp_biases)//2
            self.bExpLoaded = True
        except FileNotFoundError:
            print("Warning: experimental data file not found; disabling experimental plots.")
            self.exp_X = self.exp_Y = self.exp_dIdV = self.exp_I = None
            self.exp_biases = None
            self.bExpLoaded = False

    def load_orbital_file(self):
        filename = self.leOrbitalFile.text()
        print(f"Loading orbital from {filename}")
        if not os.path.exists(filename): 
            print(f"ERROR: Could not find orbital file {filename}"); 
            self.cbUseOrbital.setChecked(False)
            self.orbital_2D   = None
            self.orbital_lvec = None
            return
        orbital_data, orbital_lvec = orbital_utils.load_orbital(filename)
        orbital_2D = np.transpose(orbital_data,(2,1,0))
        orbital_2D = np.sum(orbital_2D[:,:,orbital_2D.shape[2]//2:],axis=2)
        self.orbital_2D = np.ascontiguousarray(orbital_2D,dtype=np.float64)
        self.orbital_lvec = orbital_lvec
        print(f"Loaded orbital from {filename}")

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
        if not self.bExpLoaded:
            return
        # Get parameters
        params = self.get_param_values()
        self.exp_idx = params['exp_slice']        
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
            ax_current=self.ax8, ax_didv=self.ax9,
            draw_exp_scan_line_func=draw_scan_line_wrapper,
            cmap_STM=pauli_scan.cmap_STM, cmap_dIdV=pauli_scan.cmap_dIdV
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

    def getOrbIfChecked(self):
        if self.orbital_2D is None:
            self.cbUseOrbital.setChecked(False)
        if self.cbUseOrbital.isChecked():
            return self.orbital_2D, self.orbital_lvec
        else:
            return None, None

    def get_param_values(self):
        """Override parent method to include checkbox states"""
        params = super().get_param_values()
        params['bMirror'] = self.cbMirror.isChecked()
        params['bRamp'] = self.cbRamp.isChecked()
        params['nsite'] = 3
        return params

    def run(self):
        """Main calculation and plotting function"""
        params = self.get_param_values()
        T_eV = params['Temp']*kBoltz
        self.pauli_solver.set_lead(0, 0.0, T_eV) # for lead 0 (substrate)
        self.pauli_solver.set_lead(1, 0.0, T_eV) # for lead 1 (tip)
        #self.pauli_solver.print_lead_params()
        self.pauli_solver.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12 )
        pauli.bValidateProbabilities = False

        self.ax1.cla(); self.ax2.cla(); self.ax3.cla() 
        self.ax4.cla(); self.ax5.cla(); self.ax6.cla()
        
        # Prepare probabilities window
        if self.cbShowProbs.isChecked():
            figp = plt.figure(figsize=(4*3, 2*3))
            self.manage_prob_window(figp, 'scanXY')
        else:
            figp = None
        # Prepare energies window
        if self.cbShowEnergies.isChecked():
            # number of site energies maps
            n = int(params['nsite'])
            rows = (n+1)//2
            figE = plt.figure(figsize=(4*n, 3*rows))
            self.manage_prob_window(figE, 'Energies')
        else:
            figE = None

        #bOmp = True   # Seems that currently it is not working
        bOmp = False
        
        pauli_scan.scan_tipField_xV(params, ax_Esite=self.ax1, ax_xV=self.ax2, ax_I2d=self.ax3, Woffsets=[0.0, -params['W'], -params['W']*2.0], bLegend=False)
        #pauli_scan.scan_tipField_xV(params, ax_Esite=self.ax1, ax_xV=self.ax2, ax_I2d=self.ax3, Woffsets=[0.0, params['W'], params['W']*2.0])
        # Determine mode: XY plane or xV line comparison
        if self.cbPlotXV.isChecked():
            Vmax = params['VBias']
            if self.exp_biases is not None:
                Vmax = self.exp_biases[-1]
            # xV simulation + experimental line comparison in main axes
            # clear target axes
            self.ax5.cla(); self.ax6.cla(); self.ax8.cla(); self.ax9.cla()
            # simulation along p1-p2
            sim_start = (params['p1_x'], params['p1_y'])
            sim_end   = (params['p2_x'], params['p2_y'])
            dist = np.hypot(sim_end[0]-sim_start[0], sim_end[1]-sim_start[1])
            orbital_2D, orbital_lvec = self.getOrbIfChecked()
            # plot sim current & dIdV on ax5 (STM) and ax6 (dIdV)
            STM, dIdV, Es, Ts, probs, stateEs, x, Vbiases, spos, rots = pauli_scan.calculate_xV_scan_orb(
                params, sim_start, sim_end,
                orbital_2D=orbital_2D, orbital_lvec=orbital_lvec, pauli_solver=self.pauli_solver,
                ax_Emax=None, ax_STM=self.ax5, ax_dIdV=self.ax6,
                nx=100, nV=100, Vmin=0.0, Vmax=Vmax,
                fig_probs=figp, bOmp=bOmp
            )
            self.ax5.set_title('Sim STM (xV)'); self.ax6.set_title('Sim dI/dV (xV)')
            
            # plot site energies maps if requested
            if figE:
                state_order_labels = pauli.make_state_order(params['nsite'])
                labels = pauli.make_state_labels(state_order_labels)
                pauli_scan.plot_state_maps(stateEs, extent=[0, dist, 0, Vmax], fig=figE, labels=labels, map_type='energy')
                
                # line plot of state energies at maximum voltage
                figCut = plt.figure()
                axCut = figCut.add_subplot(1,1,1)
                state_order = pauli.make_state_order(params['nsite'])
                labels = pauli.make_state_labels(state_order)
                for idx in range(stateEs.shape[2]):
                    axCut.plot(x, stateEs[-1,:,idx], label=labels[idx])
                axCut.set_xlabel('Distance')
                axCut.set_ylabel('Energy [eV]')
                axCut.set_title(f'Energy vs Distance at V={Vmax}')
                axCut.legend()
                self.manage_prob_window(figCut, 'xV Cut')
            # experimental line scans on ax8 (I) and ax9 (dIdV)
            exp_start = (params['ep1_x'], params['ep1_y']); exp_end = (params['ep2_x'], params['ep2_y'])
            if self.bExpLoaded:
                exp_utils.plot_exp_voltage_line_scan(self.exp_X, self.exp_Y, self.exp_I,    self.exp_biases, exp_start, exp_end, ax=self.ax8, ylims=(0, Vmax), cmap=pauli_scan.cmap_STM   )
                exp_utils.plot_exp_voltage_line_scan(self.exp_X, self.exp_Y, self.exp_dIdV, self.exp_biases, exp_start, exp_end, ax=self.ax9, ylims=(0, Vmax), cmap=pauli_scan.cmap_dIdV  )
            else:
                self.ax8.text(0.5,0.5,"No experimental data",ha='center',transform=self.ax8.transAxes)
                self.ax9.text(0.5,0.5,"No experimental data",ha='center',transform=self.ax9.transAxes)
            self.ax8.set_title('Exp I (xV)'); self.ax9.set_title('Exp dI/dV (xV)')
        else:
            # original XY plane simulation + experimental overlay
            orbital_2D, orbital_lvec = self.getOrbIfChecked()
            STM, dIdV, Es, Ts, probs, stateEs, spos, rots = pauli_scan.scan_xy_orb(
                params, orbital_2D=orbital_2D, orbital_lvec=orbital_lvec, pauli_solver=self.pauli_solver,
                ax_Etot=self.ax4, ax_Ttot=self.ax7, ax_STM=self.ax5, ax_dIdV=self.ax6, fig_probs=figp, bOmp=bOmp
            )
            # plot site energies maps if requested
            if figE:
                L = params['L']
                extent = [-L/2, L/2, -L/2, L/2]
                pauli_scan.plot_state_probabilities(stateEs, extent=extent, fig=figE, aspect='equal')
            self.draw_scan_line(self.ax4); self.draw_reference_line(self.ax4); self.plot_ellipses(self.ax9, params)
            self.plot_experimental_data()
            for i,rot in enumerate(rots):
                x, y = spos[i][0], spos[i][1]
                self.ax4.plot([x, x+rot[0][0]], [y, y+rot[0][1]])
        self.fig.tight_layout()
        self.canvas.draw()
        return STM, dIdV, Es, Ts, probs, stateEs, spos, rots
    
    def calculate_1d_scan(self, start_point, end_point, pointPerAngstrom=5 ):
        params = self.get_param_values()
        distance, Es, Ts, STM, x, y, x1, y1, x2, y2, probs = pauli_scan.calculate_1d_scan( params, start_point, end_point, pointPerAngstrom, pauli_solver=self.pauli_solver )
        nsite = int(params['nsite'])
        ref_data_line = getattr(self, 'ref_data_line', None)
        ref_columns   = getattr(self, 'ref_columns', None)
        fig=pauli_scan.plot_1d_scan_results( distance, Es, Ts, STM, nsite, probs=probs, ref_data_line=ref_data_line, ref_columns=ref_columns )
        fig.canvas.draw()
        self.manage_prob_window(fig, '1dScan')
        pauli_scan.save_1d_scan_data   ( params, distance, x, y, Es, Ts, STM, nsite, x1, y1, x2, y2 )
        return distance, Es, Ts, STM, x, y, x1, y1, x2, y2

    def plot_voltage_line_scan_exp(self, start, end, pointPerAngstrom=5):
        """Plot simulated charge and experimental dI/dV along a line scan for different voltages"""
        if not self.bExpLoaded:
            print("Experimental data not loaded; aborting voltage line scan.")
            return
        params = self.get_param_values()
        
        # For simulation, use p1,p2 instead of ep1,ep2
        sim_start = (params['p1_x'], params['p1_y'])
        sim_end   = (params['p2_x'], params['p2_y'])
        
        print(f"Starting voltage line scan: Experiment: {start} to {end}, Simulation: {sim_start} to {sim_end}")
        
        #make subplots
        fig, ((ax_sim_I, ax_exp_I), ( ax_sim_dIdV, ax_exp_dIdV)) = plt.subplots(2, 2)
        canvas = FigureCanvas(fig)
        
        # compute distances and perform voltage scan for simulation
        dist = ((sim_end[0]-sim_start[0])**2 + (sim_end[1]-sim_start[1])**2)**0.5
        sim_npoints = max(100, int(dist * pointPerAngstrom))
        Vbiases = self.exp_biases

        orbital_2D, orbital_lvec = self.getOrbIfChecked()
        _, _, _, STM, sim_dIdV, probs_arr = pauli_scan.calculate_xV_scan_orb(
            params, sim_start, sim_end,
            orbital_2D=orbital_2D, orbital_lvec=orbital_lvec,
            ax_Emax=None, ax_STM=None, ax_dIdV=None,
            nx=sim_npoints, nV=200, Vmin=0.0, Vmax=Vbiases[-1],
            fig_probs=None, pauli_solver=self.pauli_solver
        )
        extent_sim = [0, dist, 0, Vbiases[-1]]
        im1 = ax_sim_I.imshow(STM, aspect='auto', origin='lower', extent=extent_sim, cmap='hot')
        ax_sim_I.axhline( Vbiases[0], ls='--', c='g')
        ax_sim_I.set_title('Simulated current (STM)')

        vmax = np.max(np.abs(sim_dIdV))*0.1; vmin = -vmax
        im2 = ax_sim_dIdV.imshow(sim_dIdV, aspect='auto', origin='lower', extent=extent_sim, cmap='bwr', vmin=vmin, vmax=vmax)
        ax_sim_dIdV.axhline( Vbiases[0], ls='--', c='g')
        ax_sim_dIdV.set_title('Simulated dI/dV')

        # experimental dI/dV plot
        im3, (exp_didv, _)  = exp_utils.plot_exp_voltage_line_scan(self.exp_X, self.exp_Y, self.exp_I   , self.exp_biases, start, end, ax=ax_exp_I,    ylims=(0, Vbiases[-1]), cmap=pauli_scan.cmap_STM, pointPerAngstrom=pointPerAngstrom)
        im4, (exp_didv, _)  = exp_utils.plot_exp_voltage_line_scan(self.exp_X, self.exp_Y, self.exp_dIdV, self.exp_biases, start, end, ax=ax_exp_dIdV, ylims=(0, Vbiases[-1]), pointPerAngstrom=pointPerAngstrom)
        fig.tight_layout()
        canvas.draw()
        # display in Qt window
        window = QtWidgets.QMainWindow()
        window.setCentralWidget(canvas)
        window.show()
        # Keep reference to prevent garbage collection
        self._exp_voltage_scan_window = window
        # Plot probabilities in dedicated window if requested
        if self.cbShowProbs.isChecked():
            # New probability figure
            figp = plt.figure()
            nsite = probs_arr.shape[2]
            axs = pauli_scan.make_grid_axes(figp, nsite)
            # Plot exp voltage probabilities
            pauli_scan.plot_state_probabilities(probs_arr, extent=[0, dist, 0, Vbiases[-1]], axs=axs[:nsite], fig=figp)
            # Show/manage window
            self.manage_prob_window(figp, 'expVoltage')
            figp.canvas.draw()

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
    
    def manage_prob_window(self, fig, key):
        """Parent probability window to main GUI and bring to front."""
        # Store reference
        self.prob_figs[key] = fig
        # Parent to main window (Qt)
        win = fig.canvas.manager.window
        win.setParent(self)
        win.setWindowFlags(QtCore.Qt.Window)
        # Bring to front
        win.show()
        win.raise_()
        win.activateWindow()

    def save_everything(self):
        """Save PNG, JSON, and NPZ data with same base filename"""
        params = self.get_param_values()
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save All", "", "Base name (*.npz)")
        if not fname:
            return
        base = os.path.splitext(fname)[0]
        # Save params JSON
        with open(base + '.json', 'w') as f:
            json.dump(params, f, indent=4)
        # Save figure PNG
        STM, dIdV, Es, Ts, probs, spos, rots = self.run()
        self.canvas.figure.savefig(base + '.png')
        data = { 
            'spos': spos, 'STM': STM, 'Es': Es, 'Ts': Ts, 'probs': probs,
            'params_json': json.dumps(params)
        }
        # Save NPZ
        np.savez(base + '.npz', **data)

    def update_lin_solver(self):
        """Update linear solver settings on the pauli solver"""
        mode = self.comboLinSolver.currentIndex() + 1
        #print("update_lin_solver() mode", mode)
        maxIter = self.sbMaxIter.value()
        tol = self.dsTol.value()
        self.pauli_solver.setLinSolver(mode, maxIter, tol)
        if self.cbAutoUpdate.isChecked():
            self.run()

    def on_cmap_changed(self):
        """Handle colormap selection changes"""
        pauli_scan.cmap_dIdV = self.cmap_dIdV_combo.currentText()
        pauli_scan.cmap_STM  = self.cmap_STM_combo.currentText()
        if hasattr(self, 'ax5') and hasattr(self, 'ax8') and hasattr(self, 'ax9'):
            self.run()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
