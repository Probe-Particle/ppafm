#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pyProbeParticle import ChargeRings as chr
from pyProbeParticle import photo
from orbital_utils import load_orbital, calculate_stm_maps, calculate_didv

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("ChargeRings Orbital GUI")
        self.main_widget = QtWidgets.QWidget(self)
        
        # Define parameter specifications
        self.param_specs = {
            # Tip Parameters
            'Q_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (-2.0, 2.0),  'value': 0.13, 'step': 0.01},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (1.0, 20.0),  'value': 5.0, 'step': 0.5},
            'V_Bias':        {'group': 'Tip Parameters',    'widget': 'double', 'range': (-2.0, 2.0),  'value': 1.0, 'step': 0.1},
            
            # System Parameters
            'cCouling':      {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.02, 'step': 0.01, 'decimals': 3},
            'temperature':   {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 2.0, 'step': 0.1},
            'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            'decay':         {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 2.0),   'value': 0.5,  'step': 0.1},
            'E_Fermi':      {'group': 'System Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': 0.0,  'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 7.0, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': 0.1 + np.pi/2, 'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': -0.2,'step': 0.1},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0), 'value': 0.0, 'step': 0.5},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 100.0),  'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (10, 1000),    'value': 400},
        }
        
        # Initialize the GUI layout
        layout = QtWidgets.QHBoxLayout(self.main_widget)
        
        # Create parameter panel
        param_panel = QtWidgets.QWidget()
        param_layout = QtWidgets.QVBoxLayout(param_panel)
        
        # Create parameter widgets grouped by category
        self.param_widgets = {}
        current_group = None
        group_box = None
        group_layout = None
        
        for param_name, spec in self.param_specs.items():
            if spec['group'] != current_group:
                if group_box is not None:
                    param_layout.addWidget(group_box)
                current_group = spec['group']
                group_box = QtWidgets.QGroupBox(current_group)
                group_layout = QtWidgets.QFormLayout(group_box)
            
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
            
            self.param_widgets[param_name] = widget
            group_layout.addRow(param_name, widget)
            
            # Connect value changed signal
            widget.valueChanged.connect(self.update_plot)
        
        if group_box is not None:
            param_layout.addWidget(group_box)
        
        # Add orbital file selection
        file_group = QtWidgets.QGroupBox("Orbital File")
        file_layout = QtWidgets.QVBoxLayout(file_group)
        
        self.file_label = QtWidgets.QLabel("Current file: None")
        file_layout.addWidget(self.file_label)
        
        load_button = QtWidgets.QPushButton("Load Orbital File")
        load_button.clicked.connect(self.load_orbital_file)
        file_layout.addWidget(load_button)
        
        param_layout.addWidget(file_group)
        
        # Add update button
        update_button = QtWidgets.QPushButton("Update Plot")
        update_button.clicked.connect(self.update_plot)
        param_layout.addWidget(update_button)
        
        # Add parameter panel to main layout
        layout.addWidget(param_panel)
        
        # Create plot panel
        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        
        self.fig.tight_layout()
        
        # Add plot panel to main layout
        layout.addWidget(plot_panel)
        
        # Set the main widget
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        # Initialize orbital data
        self.orbital_data = None
        self.orbital_lvec = None
        
        self.orbital_data, self.orbital_lvec = load_orbital("QD.cub")
        self.file_label.setText("Current file: QD.cub")
        self.update_plot()

        # Try to load default orbital file
        # try:
        #     self.orbital_data, self.orbital_lvec = load_orbital("QD.cub")
        #     self.file_label.setText("Current file: QD.cub")
        #     self.update_plot()
        # except Exception as e:
        #     print("Exaption while trying to run GUI calculation with self.update_plot()\n, Probable reason : No default orbital file found. Please load one.")
        #     print(e)
    
    def load_orbital_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Orbital File', '', 'Cube Files (*.cub)')
        if fname:
            try:
                self.orbital_data, self.orbital_lvec = load_orbital(fname)
                self.file_label.setText(f"Current file: {os.path.basename(fname)}")
                self.update_plot()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load orbital file: {str(e)}")
    
    def get_param(self, name):
        return self.param_widgets[name].value()
    
    def update_plot(self):
        if self.orbital_data is None:
            return
        
        # Get parameters
        nsite = self.get_param('nsite')
        R = self.get_param('radius')
        phiRot = self.get_param('phiRot')
        Q0 = self.get_param('Q0')
        Qzz = self.get_param('Qzz')
        Esite = self.get_param('Esite')
        z_tip = self.get_param('z_tip')
        Q_tip = self.get_param('Q_tip')
        V_Bias = self.get_param('V_Bias')
        E_Fermi = self.get_param('E_Fermi')
        decay = self.get_param('decay')
        T = self.get_param('temperature')
        L = self.get_param('L')
        npix = self.get_param('npix')
        
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
        rots = chr.makeRotMats(angles, nsite)
        chr.initRingParams(spos, [Esite]*nsite, rot=rots, MultiPoles=mpols, 
                          E_Fermi=E_Fermi, cCouling=self.get_param('cCouling'), 
                          temperature=T, onSiteCoulomb=self.get_param('onSiteCoulomb'))
        
        # Set up configuration basis for solver_type=2
        confstrs = ["000","001","010","100","110","101","011","111"]
        confs = chr.confsFromStrings(confstrs)
        chr.setSiteConfBasis(confs)
        
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
        Q_1, Es_1, _ = chr.solveSiteOccupancies(ps, Qtips, bEsite=True, solver_type=2)
        
        # Calculate STM maps
        I_1, M_sum, M2_sum, site_coef_maps = calculate_stm_maps(
            orbital_2D, self.orbital_lvec, spos, angles, canvas_dd, canvas_shape,
            tipWf, ps, Es_1, E_Fermi, V_Bias, decay, T, crop_center, crop_size
        )
        
        # Calculate dI/dV
        dQ = 0.004
        dIdQ = calculate_didv(
            orbital_2D, self.orbital_lvec, spos, angles, canvas_dd, canvas_shape,
            tipWf, ps, Q_tip, dQ, E_Fermi, V_Bias, decay, T, crop_center, crop_size
        )[0]
        
        # Reshape charge for plotting
        Q_1 = Q_1.reshape((-1, nsite))
        Q_1 = Q_1.reshape((2*crop_size[0], 2*crop_size[1], nsite))  # Note: x,y order
        Q_total = np.sum(Q_1, axis=2)
        
        # Clear previous plots and colorbars
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            if hasattr(ax, 'colorbar'):
                ax.colorbar.remove()
        
        # Plot results with proper orientation
        extent = [-center_Lx/2, center_Lx/2, -center_Ly/2, center_Ly/2]
        
        im1 = self.ax1.imshow(Q_total, origin="lower", extent=extent)  # Removed .T
        self.ax1.plot(spos[:,0], spos[:,1], 'og')
        self.fig.colorbar(im1, ax=self.ax1)
        self.ax1.set_title("Total Charge")
        
        im2 = self.ax2.imshow(I_1, origin="lower", extent=extent)  # Removed .T
        self.ax2.plot(spos[:,0], spos[:,1], 'og')
        self.fig.colorbar(im2, ax=self.ax2)
        self.ax2.set_title("STM")
        
        im3 = self.ax3.imshow(dIdQ, origin="lower", extent=extent)  # Removed .T
        self.ax3.plot(spos[:,0], spos[:,1], 'og')
        self.fig.colorbar(im3, ax=self.ax3)
        self.ax3.set_title("dI/dQ")
        
        # Adjust layout to prevent overlapping
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
