#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("ChargeRings GUI")
        self.main_widget = QtWidgets.QWidget(self)
        
        # --- Main Layout
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        
        # --- Matplotlib Canvas
        self.fig = Figure(figsize=(15, 5))
        self.canvas = FigureCanvas(self.fig)
        l00.addWidget(self.canvas)
        
        # --- Control Panel
        l0 = QtWidgets.QVBoxLayout(); l00.addLayout(l0)
        
        # System Parameters
        gb = QtWidgets.QGroupBox("System Parameters"); l0.addWidget(gb)
        vb = QtWidgets.QVBoxLayout(gb)
        
        # Q_tip
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Q_tip:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0,2.0); bx.setValue(0.6); bx.setSingleStep(0.1); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxQtip=bx
        
        # cCouling
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("cCouling:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,1.0); bx.setValue(0.02); bx.setSingleStep(0.01); bx.setDecimals(3); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxCouling=bx
        
        # z_tip
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("z_tip:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(1.0,20.0); bx.setValue(6.0); bx.setSingleStep(0.5); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxZtip=bx
        
        # Temperature
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Temperature:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.1,100.0); bx.setValue(10.0); bx.setSingleStep(1.0); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxTemp=bx
        
        # Ring Parameters
        gb = QtWidgets.QGroupBox("Ring Parameters"); l0.addWidget(gb)
        vb = QtWidgets.QVBoxLayout(gb)
        
        # nsite
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Number of sites:")); bx = QtWidgets.QSpinBox(); bx.setRange(1,10); bx.setValue(3); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxNsite=bx
        
        # R
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Ring Radius:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(1.0,20.0); bx.setValue(5.0); bx.setSingleStep(0.5); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxRadius=bx
        
        # Run Button
        btn = QtWidgets.QPushButton("Run Simulation"); btn.clicked.connect(self.run_simulation); l0.addWidget(btn)
        
        # Set the central widget and initialize
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        # Initialize simulation parameters
        self.L = 20.0
        self.npix = 100
        self.decay = 0.7
        self.dQ = 0.02
        self.init_simulation()
        
    def init_simulation(self):
        # Initialize geometry
        nsite = self.bxNsite.value()
        R = self.bxRadius.value()
        
        # Setup sites on circle
        phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
        self.spos = np.zeros((nsite,3))
        self.spos[:,0] = np.cos(phis)*R
        self.spos[:,1] = np.sin(phis)*R
        
        # Setup multipoles and site energies
        self.Esite = [-1.0] * nsite
        rots = chr.makeRotMats(phis - 1.0, nsite)
        mpols = np.zeros((nsite,10))
        mpols[:,0] = 1.0  # Q0
        
        # Initialize global parameters
        chr.initRingParams(self.spos, self.Esite, rot=rots, MultiPoles=mpols,  E_Fermi=0.0, cCouling=self.bxCouling.value(), temperature=self.bxTemp.value(), onSiteCoulomb=3.0)
    
    def run_simulation(self):
        self.init_simulation()
        
        # Setup scanning grid
        extent = [-self.L, self.L, -self.L, self.L]
        ps = chr.makePosXY(n=self.npix, L=self.L, z0=self.bxZtip.value())
        Qtips = np.ones(len(ps)) * self.bxQtip.value()
        
        # Calculate occupancies and STM maps
        Q_1, _ = chr.solveSiteOccupancies(ps, Qtips)
        I_1 = chr.getSTM_map(ps, Qtips, Q_1.reshape(-1,len(self.Esite)), decay=self.decay)
        
        Q_2, _ = chr.solveSiteOccupancies(ps, Qtips+self.dQ)
        I_2 = chr.getSTM_map(ps, Qtips+self.dQ, Q_2.reshape(-1,len(self.Esite)), decay=self.decay)
        
        dIdQ = (I_2-I_1)/self.dQ
        
        # Reshape for plotting
        Q_1 = Q_1.reshape((self.npix,self.npix,len(self.Esite)))
        I_1 = I_1.reshape((self.npix,self.npix))
        dIdQ = dIdQ.reshape((self.npix,self.npix))
        
        # Plot results
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()
        
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
        
        self.canvas.draw()
    
    def update_plots(self):
        self.run_simulation()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
