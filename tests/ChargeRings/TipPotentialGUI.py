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
import TipMultipole as tmul

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Tip Potential GUI")
        self.main_widget = QtWidgets.QWidget(self)
        
        # Parameter specifications
        self.param_specs = {
            # Tip Parameters
            'VBias': {'group': 'Tip Parameters', 'widget': 'double', 'range': (0.0, 2.0), 'value': 1.0, 'step': 0.1},
            'Rtip': {'group': 'Tip Parameters', 'widget': 'double', 'range': (0.5, 5.0), 'value': 1.0, 'step': 0.1},
            'z_tip': {'group': 'Tip Parameters', 'widget': 'double', 'range': (1.0, 10.0), 'value': 3.0, 'step': 0.5},
            
            # Mirror Parameters
            'zV0': {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 0.0), 'value': -2.5, 'step': 0.1},
            'zQd': {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0), 'value': 0.0, 'step': 0.1},
            
            # Visualization Parameters
            'npix': {'group': 'Visualization', 'widget': 'int', 'range': (50, 500), 'value': 100, 'step': 50},
            'L': {'group': 'Visualization', 'widget': 'double', 'range': (5.0, 50.0), 'value': 10.0, 'step': 1.0},
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
        btn = QtWidgets.QPushButton("Update Plots"); btn.clicked.connect(self.update_plots); hb.addWidget(btn)
        
        # Set the central widget and initialize
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.update_plots()
    
    def get_param_values(self):
        """Get current values of all parameters"""
        return {name: widget.value() for name, widget in self.param_widgets.items()}
    
    def update_plots(self):
        """Update plots with current parameters"""
        params = self.get_param_values()
        
        # Clear the figure
        self.fig.clear()
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        # Plot 1D potential
        Vtip, ps = self.plot1DpoentialX(**params)
        self.ax1.plot(ps[:,0], Vtip)
        self.ax1.set_title("1D Potential")
        self.ax1.set_xlabel("x [Å]")
        self.ax1.set_ylabel("V [V]")
        self.ax1.grid()
        
        # Plot 2D tip potential
        Vtip, ps = self.plotTipPotXZ(**params)
        self.ax2.set_title("Tip Potential")
        self.ax2.set_xlabel("x [Å]")
        self.ax2.set_ylabel("z [Å]")
        self.ax2.grid()
        
        # Plot 2D site potential
        Esites, ps = self.plot2DpoentialXZ(**params)
        self.ax3.set_title("Site Potential")
        self.ax3.set_xlabel("x [Å]")
        self.ax3.set_ylabel("z [Å]")
        self.ax3.grid()
        
        # Adjust layout and draw
        self.fig.tight_layout()
        self.canvas.draw()
    
    def plot1DpoentialX(self, VBias=1.0, Rtip=1.0, z_tip=6.0, zV0=-2.5, zQd=0.0, npix=100, L=10.0):
        ps = np.zeros((npix,3))
        ps[:,0] = np.linspace(-L,L,npix, endpoint=False)
        ps[:,2] = z_tip+Rtip
        Vtip = tmul.compute_site_energies(ps, np.array([[0.0,0.0,zQd],]), VBias, Rtip, zV0=zV0)
        return Vtip, ps
    
    def plotTipPotXZ(self, VBias=1.0, Rtip=1.0, z_tip=3.0, zV0=-2.5, zQd=0.0, npix=100, L=10.0):
        zT = z_tip+Rtip
        ps,Xs,Ys = chr.makePosXY(n=npix, L=L, axs=(0,2,1))
        Vtip = tmul.compute_V_mirror(np.array([0.0,0.0,zT]), ps, VBias=VBias, Rtip=Rtip, zV0=zV0)
        Vtip = Vtip.reshape(npix,npix)
        extent = [-L,L,-L,L]
        circ1,_ = tmul.makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0,0.0,zT))
        circ2,_ = tmul.makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0,0.0,2*zV0-zT))
        self.ax2.imshow(Vtip, extent=extent, cmap='bwr', origin='lower', vmin=-VBias, vmax=VBias)
        self.ax2.plot(circ1[:,0], circ1[:,2], ':k')
        self.ax2.plot(circ2[:,0], circ2[:,2], ':k')
        return Vtip, ps
    
    def plot2DpoentialXZ(self, VBias=1.0, Rtip=1.0, z_tip=3.0, zV0=-2.5, zQd=0.0, npix=100, L=10.0):
        pSites = np.array([[0.0,0.0,zQd],])
        ps,Xs,Ys = chr.makePosXY(n=npix, L=L, axs=(0,2,1))
        Esites = tmul.compute_site_energies(ps, pSites, VBias, Rtip, zV0=zV0)
        Esites = Esites.reshape(npix,npix, len(pSites))
        self.ax3.imshow(Esites[:,:,0], extent=[-L,L,-L,L], cmap='bwr', origin='lower', vmin=-VBias, vmax=VBias)
        self.ax3.axhline(zV0, ls='--',c='k', label='mirror surface')
        self.ax3.axhline(zQd, ls='--',c='g', label='Qdot height')
        self.ax3.legend()
        return Esites, ps

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())