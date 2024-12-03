#!/usr/bin/python

import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pyProbeParticle import ChargeRings as cr
from pyProbeParticle import photo
import orbital_utils
from ScanWindow1D import ScanWindow1D

class PlotPanel2D(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.scan_line = None
        self.scan_window = None
        self.start_point = None
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(16, 4.5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Create subplots with optimized spacing
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.25)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        self.setMinimumWidth(1000)
    
    def update_plot(self, Q_total, I_1, dIdQ, spos, extent):
        """Update all plots with new data"""
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.25)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])
        
        # Plot results with proper orientation
        im1 = self.ax1.imshow(Q_total, origin="lower", extent=extent)
        self.ax1.set_title("Total Charge")
        cb1 = self.fig.colorbar(im1, ax=self.ax1, use_gridspec=True, fraction=0.046, pad=0.04)
        
        im2 = self.ax2.imshow(I_1, origin="lower", extent=extent)
        self.ax2.set_title("STM")
        cb2 = self.fig.colorbar(im2, ax=self.ax2, use_gridspec=True, fraction=0.046, pad=0.04)
        
        im3 = self.ax3.imshow(dIdQ, origin="lower", extent=extent)
        self.ax3.set_title("dI/dQ")
        cb3 = self.fig.colorbar(im3, ax=self.ax3, use_gridspec=True, fraction=0.046, pad=0.04)
        
        # Add site markers
        for i, (x, y) in enumerate(zip(spos[:,0], spos[:,1])):
            self.ax1.plot(x, y, 'o', color=f'C{i}')
            self.ax2.plot(x, y, 'o', color=f'C{i}')
            self.ax3.plot(x, y, 'o', color=f'C{i}')
        
        # Set consistent aspect ratio and limits
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_aspect('equal')
            ax.set_xlabel('x [Å]')
            ax.set_ylabel('y [Å]')
        
        # Apply tight layout
        self.fig.tight_layout()
        self.canvas.draw()
    
    def on_mouse_press(self, event):
        if event.inaxes == self.ax1 and event.button == 1:  # Left click on charge plot
            self.start_point = (event.xdata, event.ydata)
            if self.scan_line:
                self.scan_line.remove()
                self.scan_line = None
            self.canvas.draw()
    
    def on_mouse_motion(self, event):
        if event.inaxes and self.start_point and event.button == 1:
            if self.scan_line:
                self.scan_line.remove()
            self.scan_line = self.ax1.plot([self.start_point[0], event.xdata],
                                         [self.start_point[1], event.ydata],
                                         'r--')[0]
            self.canvas.draw()
    
    def on_mouse_release(self, event):
        if event.inaxes and self.start_point and event.button == 1:
            end_point = (event.xdata, event.ydata)
            self.perform_line_scan(self.start_point, end_point)
            self.start_point = None
    
    def perform_line_scan(self, start_point, end_point):
        # Create scan window if not exists
        if not self.scan_window:
            self.scan_window = ScanWindow1D(self, start_point, end_point)
        
        # Get scan parameters from parent
        params = self.parent.get_all_params()
        
        # Generate points along the scan line
        npoints = 100
        x = np.linspace(start_point[0], end_point[0], npoints)
        y = np.linspace(start_point[1], end_point[1], npoints)
        scan_points = np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)
        
        # Perform scan calculations
        charges = []
        currents = []
        energies_list = []
        occupations_list = []
        
        for xi, yi in zip(x, y):
            # Calculate system state at this point
            result = self.parent.calculate_at_point(xi, yi, params)
            charges.append(result['charge'])
            currents.append(result['current'])
            energies_list.append(result['energies'])
            occupations_list.append(result['occupations'])
        
        # Convert to arrays
        charges = np.array(charges)
        currents = np.array(currents)
        energies = np.array(energies_list)
        occupations = np.array(occupations_list)
        
        # Update scan window
        self.scan_window.update_plot(scan_points, charges, currents, energies, occupations)
        self.scan_window.show()

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("ChargeRings Orbital GUI")
        
        # Create main widget
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        # Initialize parameters and UI
        self.init_ui()
        
        # Initialize orbital data
        self.orbital_data = None
        self.orbital_lvec = None
        
        try:
            self.orbital_data, self.orbital_lvec = orbital_utils.load_orbital("QD.cub")
            self.update_plot()
        except Exception as e:
            print("Exception while trying to load default orbital file: ", str(e))
    
    def create_parameter_widget(self, param_name, spec):
        if spec['widget'] == 'double':
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(spec['range'][0], spec['range'][1])
            widget.setValue(spec['value'])
            widget.setSingleStep(spec['step'])
            if 'decimals' in spec:
                widget.setDecimals(spec['decimals'])
        elif spec['widget'] == 'int':
            widget = QtWidgets.QSpinBox()
            widget.setRange(spec['range'][0], spec['range'][1])
            widget.setValue(spec['value'])
        
        widget.valueChanged.connect(self.update_plot)
        return widget
    
    def load_orbital_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Orbital File', '', 'Cube Files (*.cub)')
        if fname:
            try:
                self.orbital_data, self.orbital_lvec = orbital_utils.load_orbital(fname)
                self.update_plot()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load orbital file: {str(e)}")
    
    def get_param(self, name):
        return self.param_specs[name]['widget'].value()
    
    def update_plot(self):
        if self.orbital_data is None:
            return
            
        # Check if auto-update is enabled
        if not self.auto_update_checkbox.isChecked():
            return
        
        # Get parameters
        nsite   = self.get_param('nsite')
        R       = self.get_param('radius')
        phiRot  = self.get_param('phiRot')
        Q0      = self.get_param('Q0')
        Qzz     = self.get_param('Qzz')
        Esite   = self.get_param('Esite')
        z_tip   = self.get_param('z_tip')
        Q_tip   = self.get_param('Q_tip')
        V_Bias  = self.get_param('V_Bias')
        E_Fermi = self.get_param('E_Fermi')
        decay   = self.get_param('decay')
        T       = self.get_param('temperature')
        L       = self.get_param('L')
        npix    = self.get_param('npix')
        
        # Setup geometry
        phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
        spos = np.zeros((nsite, 3))
        spos[:,0] = np.cos(phis)*R
        spos[:,1] = np.sin(phis)*R
        angles = phis + phiRot
        
        # Setup multipoles
        mpols = np.zeros((nsite, 10))
        mpols[:,4] = Qzz
        mpols[:,0] = Q0
        
        # Initialize ChargeRings parameters
        rots = cr.makeRotMats(angles, nsite)
        cr.initRingParams(spos, [Esite]*nsite, rot=rots, MultiPoles=mpols,  E_Fermi=E_Fermi, cCouling=self.get_param('cCouling'),   temperature=T, onSiteCoulomb=self.get_param('onSiteCoulomb'))
        
        # Set up configuration basis for solver_type=2
        confstrs = ["000","001","010","100","110","101","011","111"]
        confs = cr.confsFromStrings(confstrs)
        cr.setSiteConfBasis(confs)
        
        # Prepare calculation grid
        x = np.linspace(-L/2, L/2, npix)
        y = np.linspace(-L/2, L/2, npix)
        X, Y = np.meshgrid(x, y)
        ps = np.zeros((len(x) * len(y), 3))
        ps[:,0] = X.flatten()
        ps[:,1] = Y.flatten()
        ps[:,2] = z_tip
        
        # Get physical dimensions from lattice vectors
        Lx = abs(self.orbital_lvec[1,0])
        Ly = abs(self.orbital_lvec[2,1])
        
        # Setup canvas - match reference implementation exactly
        Lcanv = 60.0  # Fixed canvas size from reference
        dCanv = 0.2   # Fixed grid spacing from reference
        ncanv = int(np.ceil(Lcanv/dCanv))
        canvas_shape = (ncanv, ncanv)
        canvas_dd = np.array([dCanv, dCanv])
        
        # Create tip wavefunction
        tipWf, _ = photo.makeTipField(canvas_shape, canvas_dd, z0=z_tip, beta=decay, bSTM=True)
        
        # Calculate STM maps
        crop_center = (canvas_shape[0]//2, canvas_shape[1]//2)
        crop_size = (canvas_shape[0]//4, canvas_shape[1]//4)  # Back to //4 as per reference
        
        # Calculate physical dimensions of center region
        center_Lx = 2 * crop_size[0] * canvas_dd[0]
        center_Ly = 2 * crop_size[1] * canvas_dd[1]
        
        # Process orbital data - match reference implementation
        orbital_2D = np.transpose(self.orbital_data, (2, 1, 0))
        orbital_2D = np.sum(orbital_2D[:, :, orbital_2D.shape[2]//2:], axis=2)
        orbital_2D = np.ascontiguousarray(orbital_2D, dtype=np.float64)
        
        # Create grid for the center region
        x = np.linspace(-center_Lx/2, center_Lx/2, 2*crop_size[0])
        y = np.linspace(-center_Ly/2, center_Ly/2, 2*crop_size[1])
        X, Y = np.meshgrid(x, y, indexing='xy')
        ps = np.zeros((len(x) * len(y), 3))
        ps[:,0] = X.flatten()
        ps[:,1] = Y.flatten()
        ps[:,2] = z_tip
        
        # Calculate charges and energies
        Qtips = np.ones(len(ps)) * Q_tip
        Q_1, Es_1, _ = cr.solveSiteOccupancies(ps, Qtips, bEsite=True, solver_type=2)
        
        if self.use_orbital_checkbox.isChecked():
            # Calculate STM maps with orbital convolution
            I_1, M_sum, M2_sum, site_coef_maps = orbital_utils.calculate_stm_maps(
                orbital_2D, self.orbital_lvec, spos, angles, canvas_dd, canvas_shape,
                tipWf, ps, Es_1, E_Fermi, V_Bias, decay, T, crop_center, crop_size
            )
        else:
            # Calculate STM maps without orbital convolution - use only site energies
            I_1 = np.zeros((2*crop_size[1], 2*crop_size[0]))
            for i in range(len(spos)):
                # Calculate coefficient c_i = rho_i rho_j [ f_i - f_j ]
                c_i = cr.calculate_site_current(ps, spos[i], Es_1[:,i], E_Fermi + V_Bias, E_Fermi, decay=decay, T=T)
                c_i = c_i.reshape((2*crop_size[1], 2*crop_size[0]))
                I_1 += c_i  # Direct sum without M_i^2 factor
        
        # Calculate dI/dV
        dQ = 0.004
        if self.use_orbital_checkbox.isChecked():
            dIdQ = orbital_utils.calculate_didv( orbital_2D, self.orbital_lvec, spos, angles, canvas_dd, canvas_shape,tipWf, ps, Q_tip, dQ, E_Fermi, V_Bias, decay, T, crop_center, crop_size )[0]
        else:
            # Calculate dI/dV without orbital convolution
            Q_2, Es_2, _ = cr.solveSiteOccupancies(ps, Qtips + dQ, bEsite=True, solver_type=2)
            I_2 = np.zeros((2*crop_size[1], 2*crop_size[0]))
            for i in range(len(spos)):
                c_i = cr.calculate_site_current(ps, spos[i], Es_2[:,i], E_Fermi + V_Bias, E_Fermi, decay=decay, T=T)
                c_i = c_i.reshape((2*crop_size[1], 2*crop_size[0]))
                I_2 += c_i
            dIdQ = (I_2 - I_1) / dQ
        
        # Reshape charge for plotting
        Q_1 = Q_1.reshape((-1, nsite))
        Q_1 = Q_1.reshape((2*crop_size[0], 2*crop_size[1], nsite))  # Note: x,y order
        Q_total = np.sum(Q_1, axis=2)
        
        # Update plot
        self.main_widget.findChild(PlotPanel2D).update_plot(Q_total, I_1, dIdQ, spos, [-center_Lx/2, center_Lx/2, -center_Ly/2, center_Ly/2])
    
    def save_figure(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Figure', '', 'PNG Files (*.png)')
        if fname:
            self.main_widget.findChild(PlotPanel2D).fig.savefig(fname, dpi=300)

    def calculate_at_point(self, x, y, params):
        """Calculate system state at a single point"""
        # Extract parameters
        nsite = params['nsite']
        R = params['radius']
        phiRot = params['phiRot']
        Q0 = params['Q0']
        Qzz = params['Qzz']
        Esite = params['Esite']
        Q_tip = params['Q_tip']
        V_Bias = params['V_Bias']
        E_Fermi = params['E_Fermi']
        decay = params['decay']
        T = params['temperature']
        
        # Setup geometry
        phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
        spos = np.zeros((nsite, 3))
        spos[:,0] = R * np.cos(phis)
        spos[:,1] = R * np.sin(phis)
        
        angles = np.ones(nsite) * phiRot
        
        # Setup multipoles
        mpols = np.zeros((nsite, 10))
        mpols[:,0] = Q0
        mpols[:,4] = Qzz  # Qxx
        mpols[:,9] = Qzz  # Qyy
        mpols[:,5] = -2*Qzz  # Qzz
        
        # Initialize ChargeRings parameters
        rots = cr.makeRotMats(angles, nsite)
        cr.initRingParams(spos, [Esite]*nsite, rot=rots, MultiPoles=mpols,  E_Fermi=E_Fermi, cCouling=params['cCouling'], temperature=T, onSiteCoulomb=params['onSiteCoulomb'])
        
        # Set up configuration basis
        confstrs = ["000","001","010","100","110","101","011","111"]
        confs = cr.confsFromStrings(confstrs)
        cr.setSiteConfBasis(confs)
        
        # Calculate at single point
        p = np.array([[x, y, params['z_tip']]])
        Qtips = np.array([Q_tip])
        
        # Calculate charges and energies
        Q_1, Es_1, _ = cr.solveSiteOccupancies(p, Qtips, bEsite=True, solver_type=2)
        
        # Calculate occupations from energies using safe exponential
        def safe_exp(x):
            """Safe exponential function that handles overflow"""
            with np.errstate(over='ignore'):
                return np.exp(np.clip(x, -700, 700))
        
        occ_1 = np.zeros_like(Es_1)
        kT = T * 8.617333262e-5  # Convert temperature to eV
        for i in range(Es_1.shape[1]):
            occ_1[:,i] = 1.0 / (1.0 + safe_exp((Es_1[:,i] - E_Fermi) / kT))
        
        # Calculate current
        current = 0
        if self.use_orbital_checkbox.isChecked() and self.orbital_data is not None:
            # TODO: Implement orbital convolution for single point
            pass
        else:
            for i in range(len(spos)):
                c_i = cr.calculate_site_current(p, spos[i], Es_1[:,i], 
                                              E_Fermi + V_Bias, E_Fermi, 
                                              decay=decay, T=T)
                current += c_i[0]
        
        return {
            'charge': np.sum(Q_1[0]),
            'current': current,
            'energies': Es_1[0],
            'occupations': occ_1[0]
        }

    def get_all_params(self):
        params = {
            'nsite': self.get_param('nsite'),
            'radius': self.get_param('radius'),
            'phiRot': self.get_param('phiRot'),
            'Q0': self.get_param('Q0'),
            'Qzz': self.get_param('Qzz'),
            'Esite': self.get_param('Esite'),
            'z_tip': self.get_param('z_tip'),
            'Q_tip': self.get_param('Q_tip'),
            'V_Bias': self.get_param('V_Bias'),
            'E_Fermi': self.get_param('E_Fermi'),
            'decay': self.get_param('decay'),
            'temperature': self.get_param('temperature'),
            'cCouling': self.get_param('cCouling'),
            'onSiteCoulomb': self.get_param('onSiteCoulomb'),
        }
        return params

    def init_ui(self):
        # Define parameter specifications
        self.param_specs = {
            # Tip Parameters
            'Q_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (-2.0, 2.0),  'value': 0.13, 'step': 0.01},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (1.0, 20.0),  'value': 5.0,  'step': 0.5},
            'V_Bias':        {'group': 'Tip Parameters',    'widget': 'double', 'range': (-2.0, 2.0),  'value': 1.0,  'step': 0.1},
            
            # System Parameters
            'cCouling':      {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.01, 'step': 0.001, 'decimals': 3},
            'temperature':   {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 2.0,  'step': 1.0},
            'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            'decay':         {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 2.0),   'value': 0.5,  'step': 0.1},
            'E_Fermi':       {'group': 'System Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': 0.0,  'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),      'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),  'value': 7.0, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0),'value': 0.1+np.pi/2, 'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0),'value': -0.2,'step': 0.1},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0),'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0),'value': 15.0,'step': 0.5},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 100.0), 'value': 20.0,'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (10, 1000),   'value': 400},
        }
        
        # Initialize the GUI layout
        layout = QtWidgets.QHBoxLayout(self.main_widget)
        
        # Create control panel layout
        control_panel = QtWidgets.QWidget()
        control_panel.setMaximumWidth(350)  # Limit control panel width
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setSpacing(5)  # Reduce spacing between widgets
        
        # Add parameter groups
        for group_name in sorted(set(spec['group'] for spec in self.param_specs.values())):
            group_box = QtWidgets.QGroupBox(group_name)
            group_layout = QtWidgets.QFormLayout()
            group_layout.setSpacing(5)  # Reduce spacing in form layout
            group_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
            
            for param_name, spec in self.param_specs.items():
                if spec['group'] == group_name:
                    widget = self.create_parameter_widget(param_name, spec)
                    self.param_specs[param_name]['widget'] = widget
                    group_layout.addRow(param_name, widget)
            
            group_box.setLayout(group_layout)
            control_layout.addWidget(group_box)
        
        # Add checkboxes for orbital convolution and auto-update
        checkbox_group = QtWidgets.QGroupBox("Options")
        checkbox_layout = QtWidgets.QVBoxLayout()
        checkbox_layout.setSpacing(5)  # Reduce spacing
        checkbox_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        self.use_orbital_checkbox = QtWidgets.QCheckBox("Use Orbital Convolution")
        self.use_orbital_checkbox.setChecked(True)
        self.use_orbital_checkbox.stateChanged.connect(self.update_plot)
        checkbox_layout.addWidget(self.use_orbital_checkbox)
        
        self.auto_update_checkbox = QtWidgets.QCheckBox("Auto Update")
        self.auto_update_checkbox.setChecked(True)
        checkbox_layout.addWidget(self.auto_update_checkbox)
        
        checkbox_group.setLayout(checkbox_layout)
        control_layout.addWidget(checkbox_group)
        
        # Add buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(5)  # Reduce spacing
        
        update_button = QtWidgets.QPushButton("Update Plot")
        update_button.clicked.connect(self.update_plot)
        button_layout.addWidget(update_button)
        
        save_button = QtWidgets.QPushButton("Save Figure")
        save_button.clicked.connect(self.save_figure)
        button_layout.addWidget(save_button)
        
        control_layout.addLayout(button_layout)
        
        # Add control panel to main layout with fixed width
        layout.addWidget(control_panel)
        
        # Create plot panel
        plot_panel = PlotPanel2D(self)
        layout.addWidget(plot_panel, stretch=2)
        
        # Set the main widget
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
