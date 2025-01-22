#!/usr/bin/python

import sys
import os
import json
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pyProbeParticle import ChargeRings as chr

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("ChargeRings GUI")
        self.main_widget = QtWidgets.QWidget(self)
        
        # Define parameter specifications
        self.param_specs = {
            # Tip Parameters
            'Q_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (-2.0, 2.0),  'value': 0.6, 'step': 0.1},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (1.0, 20.0),  'value': 6.0, 'step': 0.5},
            
            # System Parameters
            'cCouling':      {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.02, 'step': 0.01, 'decimals': 3},
            'temperature':   {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 10.0, 'step': 1.0},
            'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 5.0, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0,'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0,'step': 0.1},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0), 'value': 0.0, 'step': 0.5},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 50.0),  'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (50, 500),    'value': 200,  'step': 50},
            'decay':         {'group': 'Visualization',     'widget': 'double', 'range': (0.1, 2.0),   'value': 0.7,  'step': 0.1,   'decimals': 2},
            'dQ':            {'group': 'Visualization',     'widget': 'double', 'range': (0.001, 0.1), 'value': 0.02, 'step': 0.001, 'decimals': 3},
        }
        
        # Dictionary to store widget references
        self.param_widgets = {}
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        # --- Main Layout
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        
        # --- Matplotlib Canvas
        self.fig = Figure(figsize=(15, 5))
        self.canvas = FigureCanvas(self.fig)
        l00.addWidget(self.canvas)
        
        # --- Control Panel
        l0 = QtWidgets.QVBoxLayout(); l00.addLayout(l0)
        
        # Create widgets for each parameter group
        current_group = None
        current_layout = None
        
        for param_name, spec in self.param_specs.items():
            # Create new group if needed
            if spec['group'] != current_group:
                current_group = spec['group']
                gb = QtWidgets.QGroupBox(current_group); l0.addWidget(gb)
                current_layout = QtWidgets.QVBoxLayout(gb)
            
            # Create widget layout
            hb = QtWidgets.QHBoxLayout(); current_layout.addLayout(hb)
            hb.addWidget(QtWidgets.QLabel(f"{param_name}:"))
            
            # Create appropriate widget type
            if spec['widget'] == 'double':
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(*spec['range'])
                widget.setValue(spec['value'])
                widget.setSingleStep(spec['step'])
                if 'decimals' in spec:
                    widget.setDecimals(spec['decimals'])
            elif spec['widget'] == 'int':
                widget = QtWidgets.QSpinBox()
                widget.setRange(*spec['range'])
                widget.setValue(spec['value'])
                if 'step' in spec:
                    widget.setSingleStep(spec['step'])
            
            widget.valueChanged.connect(self.update_plots)
            hb.addWidget(widget)
            self.param_widgets[param_name] = widget
        
        # Controls
        hb = QtWidgets.QHBoxLayout(); l0.addLayout(hb)
        
        # Auto-update checkbox
        cb = QtWidgets.QCheckBox("Auto-update"); cb.setChecked(True); hb.addWidget(cb); self.cbAutoUpdate=cb
        
        # Run Button
        btn = QtWidgets.QPushButton("Run Simulation"); btn.clicked.connect(self.run_simulation); hb.addWidget(btn)
        
        # Save/Load buttons
        hb = QtWidgets.QHBoxLayout(); l0.addLayout(hb)
        btnSave = QtWidgets.QPushButton("Save Parameters"); btnSave.clicked.connect(self.save_parameters); hb.addWidget(btnSave)
        btnLoad = QtWidgets.QPushButton("Load Parameters"); btnLoad.clicked.connect(self.load_parameters); hb.addWidget(btnLoad)
        
        # Set the central widget and initialize
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.init_simulation()
    
    def get_param_values(self):
        """Get current values of all parameters"""
        return {name: widget.value() for name, widget in self.param_widgets.items()}
    
    def set_param_values(self, values):
        """Set values for all parameters"""
        for name, value in values.items():
            if name in self.param_widgets:
                self.param_widgets[name].setValue(value)
    
    def save_parameters(self):
        """Save parameters to JSON file"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON files (*.json)")
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            with open(filename, 'w') as f:
                json.dump(self.get_param_values(), f, indent=4)
    
    def load_parameters(self):
        """Load parameters from JSON file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON files (*.json)")
        if filename:
            with open(filename, 'r') as f:
                values = json.load(f)
                self.set_param_values(values)
                self.run_simulation()

    def init_simulation(self):
        params = self.get_param_values()
        
        # Initialize geometry
        nsite = params['nsite']
        R = params['radius']
        
        # Setup sites on circle
        phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
        self.spos = np.zeros((nsite,3))
        self.spos[:,0] = np.cos(phis)*R
        self.spos[:,1] = np.sin(phis)*R
        
        # Setup multipoles and site energies
        self.Esite = [params['Esite']] * nsite
        rots = chr.makeRotMats(phis + params['phiRot'], nsite)
        mpols = np.zeros((nsite,10))
        mpols[:,0] = params['Q0']  # Q0
        mpols[:,4] = params['Qzz'] # Qzz
        
        # Initialize global parameters
        chr.initRingParams(self.spos, self.Esite, rot=rots, MultiPoles=mpols, E_Fermi=0.0, cCouling=params['cCouling'],temperature=params['temperature'], onSiteCoulomb=params['onSiteCoulomb'])
    
    def run_simulation(self):
        self.init_simulation()
        params = self.get_param_values()
        
        # Setup scanning grid
        extent = [-params['L'], params['L'], -params['L'], params['L']]
        ps, _, _ = chr.makePosXY(n=params['npix'], L=params['L'], p0=(0.0, 0.0, params['z_tip']))
        Qtips = np.ones(len(ps)) * params['Q_tip']
        
        # Calculate occupancies and STM maps
        Q_1,_,_  = chr.solveSiteOccupancies(ps, Qtips)
        I_1      = chr.getSTM_map(ps, Qtips, Q_1.reshape(-1,len(self.Esite)), decay=params['decay'])
        
        Q_2,_,_  = chr.solveSiteOccupancies(ps, Qtips+params['dQ'])
        I_2      = chr.getSTM_map(ps, Qtips+params['dQ'], Q_2.reshape(-1,len(self.Esite)), decay=params['decay'])
        
        dIdQ = (I_2-I_1)/params['dQ']
        
        # Reshape for plotting
        Q_1 = Q_1.reshape((params['npix'],params['npix'],len(self.Esite)))
        I_1 = I_1.reshape((params['npix'],params['npix']))
        dIdQ = dIdQ.reshape((params['npix'],params['npix']))
        
        # Clear the entire figure and recreate subplots
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        # Plot results
        im1 = self.ax1.imshow(np.sum(Q_1,axis=2), origin="lower", extent=extent)
        self.ax1.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im1, ax=self.ax1)
        self.ax1.set_title("Total Charge")
        
        im2 = self.ax2.imshow(I_1, origin="lower", extent=extent)
        self.ax2.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im2, ax=self.ax2)
        self.ax2.set_title("STM")
        
        im3 = self.ax3.imshow(dIdQ, origin="lower", extent=extent)
        self.ax3.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im3, ax=self.ax3)
        self.ax3.set_title("dI/dQ")
        
        # Adjust layout to prevent overlapping
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_plots(self):
        if self.cbAutoUpdate.isChecked():
            self.run_simulation()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
