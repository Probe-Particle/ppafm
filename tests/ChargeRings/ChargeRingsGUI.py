#!/usr/bin/python

import sys
import os
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
        
        # --- Main Layout
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        
        # --- Matplotlib Canvas
        self.fig = Figure(figsize=(15, 5))
        self.canvas = FigureCanvas(self.fig)
        l00.addWidget(self.canvas)
        
        # --- Control Panel
        l0 = QtWidgets.QVBoxLayout(); l00.addLayout(l0)
        
        # Tip Parameters
        gb = QtWidgets.QGroupBox("Tip Parameters"); l0.addWidget(gb)
        vb = QtWidgets.QVBoxLayout(gb)
        
        # Q_tip
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Q_tip:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0,2.0); bx.setValue(0.6); bx.setSingleStep(0.1); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxQtip=bx
        
        # z_tip
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("z_tip:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(1.0,20.0); bx.setValue(6.0); bx.setSingleStep(0.5); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxZtip=bx
        
        # System Parameters
        gb = QtWidgets.QGroupBox("System Parameters"); l0.addWidget(gb)
        vb = QtWidgets.QVBoxLayout(gb)
        
        # cCouling
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("cCouling:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,1.0); bx.setValue(0.02); bx.setSingleStep(0.01); bx.setDecimals(3); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxCouling=bx
        
        # onSiteCoulomb
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("onSiteCoulomb:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,10.0); bx.setValue(3.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxCoulomb=bx

        # Temperature
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Temperature:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.1,100.0); bx.setValue(10.0); bx.setSingleStep(1.0); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxTemp=bx
                
        # Ring Geometry
        gb = QtWidgets.QGroupBox("Ring Geometry"); l0.addWidget(gb)
        vb = QtWidgets.QVBoxLayout(gb)
        
        # nsite
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Number of sites:")); bx = QtWidgets.QSpinBox(); bx.setRange(1,10); bx.setValue(3); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxNsite=bx
        
        # R
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Ring Radius:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(1.0,20.0); bx.setValue(5.0); bx.setSingleStep(0.5); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxRadius=bx
        
        # phiRot
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Rotation (phi):")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-10.0,10.0); bx.setValue(-1.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxPhiRot=bx
        
        # Site Properties
        gb = QtWidgets.QGroupBox("Site Properties"); l0.addWidget(gb)
        vb = QtWidgets.QVBoxLayout(gb)
        
        # Esite
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Site Energy:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-10.0,10.0); bx.setValue(-1.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxEsite=bx
        
        # Q0
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Q0 (monopole):")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-10.0,10.0); bx.setValue(1.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxQ0=bx
        
        # Qzz
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Qzz (quadrupole):")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-20.0,20.0); bx.setValue(0.0); bx.setSingleStep(0.5); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxQzz=bx
        
        # Visualization Parameters
        gb = QtWidgets.QGroupBox("Visualization"); l0.addWidget(gb)
        vb = QtWidgets.QVBoxLayout(gb)
        
        # Canvas Size (L)
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Canvas Size (L):")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(5.0,50.0); bx.setValue(20.0); bx.setSingleStep(1.0); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxL=bx
        
        # Grid Points (npix)
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Grid Points:")); bx = QtWidgets.QSpinBox(); bx.setRange(50,500); bx.setValue(200); bx.setSingleStep(50); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxNpix=bx
        
        # decay
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("Decay:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.1,2.0); bx.setValue(0.7); bx.setSingleStep(0.1); bx.setDecimals(2); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxDecay=bx
        
        # dQ
        hb = QtWidgets.QHBoxLayout(); vb.addLayout(hb)
        hb.addWidget(QtWidgets.QLabel("dQ:")); bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.001,0.1); bx.setValue(0.02); bx.setSingleStep(0.001); bx.setDecimals(3); bx.valueChanged.connect(self.update_plots); hb.addWidget(bx); self.bxDQ=bx
        
        # Run Button
        btn = QtWidgets.QPushButton("Run Simulation"); btn.clicked.connect(self.run_simulation); l0.addWidget(btn)
        
        # Set the central widget and initialize
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
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
        self.Esite = [self.bxEsite.value()] * nsite
        rots = chr.makeRotMats(phis + self.bxPhiRot.value(), nsite)
        mpols = np.zeros((nsite,10))
        mpols[:,0] = self.bxQ0.value()  # Q0
        mpols[:,4] = self.bxQzz.value() # Qzz
        
        # Initialize global parameters
        chr.initRingParams(self.spos, self.Esite, rot=rots, MultiPoles=mpols,
                         E_Fermi=0.0, cCouling=self.bxCouling.value(),
                         temperature=self.bxTemp.value(), 
                         onSiteCoulomb=self.bxCoulomb.value())
    
    def run_simulation(self):
        self.init_simulation()
        
        # Setup scanning grid
        self.L = self.bxL.value()
        self.npix = self.bxNpix.value()
        extent = [-self.L, self.L, -self.L, self.L]
        ps = chr.makePosXY(n=self.npix, L=self.L, z0=self.bxZtip.value())
        Qtips = np.ones(len(ps)) * self.bxQtip.value()
        
        # Calculate occupancies and STM maps
        Q_1, _ = chr.solveSiteOccupancies(ps, Qtips)
        I_1 = chr.getSTM_map(ps, Qtips, Q_1.reshape(-1,len(self.Esite)), decay=self.bxDecay.value())
        
        dQ = self.bxDQ.value()
        Q_2, _ = chr.solveSiteOccupancies(ps, Qtips+dQ)
        I_2 = chr.getSTM_map(ps, Qtips+dQ, Q_2.reshape(-1,len(self.Esite)), decay=self.bxDecay.value())
        
        dIdQ = (I_2-I_1)/dQ
        
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
