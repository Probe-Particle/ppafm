#!/usr/bin/env python3

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases

import sys
import os
import time
import random
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
from enum import Enum

from pyProbeParticle import basUtils
from pyProbeParticle import PPPlot
from pyProbeParticle.AFMulatorOCL_Simple import AFMulator
from pyProbeParticle.fieldOCL            import HartreePotential, hartreeFromFile
import pyProbeParticle.GridUtils  as GU
import pyProbeParticle.common     as PPU
import pyProbeParticle.oclUtils   as oclu
import pyProbeParticle.fieldOCL   as FFcl
import pyProbeParticle.GUIWidgets as guiw

DataViews = Enum('DataViews','df FFin FFout FFel FFpl')
Multipoles = Enum('Multipoles', 's pz dz2')

Presets = {
    'CO (Z8, dz2, Q-0.1, K0.25)': {
        'Z': 8,
        'Multipole': 'dz2',
        'Q': -0.1,
        'Sigma': 0.71,
        'K': [0.25, 0.25, 30.0],
        'EqPos': [0.0, 0.0, 3.0]
    },
    'Xe (Z54, s, Q0.3, K0.25)': {
        'Z': 54,
        'Multipole': 's',
        'Q': 0.3,
        'Sigma': 0.71,
        'K': [0.25, 0.25, 30.0],
        'EqPos': [0.0, 0.0, 3.0]
    },
    'Cl (Z17, s, Q-0.3, K0.50)': {
        'Z': 54,
        'Multipole': 's',
        'Q': -0.3,
        'Sigma': 0.71,
        'K': [0.50, 0.50, 30.0],
        'EqPos': [0.0, 0.0, 3.0]
    }
}

TTips = {
    'Preset': 'Preset: Apply a probe parameter oreset.',
    'Z': 'Z: Probe atomic number. Determines the Lennard-Jones parameters of the force field.',
    'Multipole': 'Multipole: Probe charge multipole type:\ns: monopole\npz: dipole\ndz2: quadrupole.',
    'Q': 'Q: Probe charge/multipole magnitude.',
    'Sigma': 'Sigma: Probe charge distribution width.',
    'K': 'K: Force constants for harmonic force holding the probe to the tip in x, y, and radial directions.',
    'EqPos': 'Eq. Pos: Probe equilibrium position with respect to the tip in x, y, and radial directions. Non-zero values for x and y models asymmetry in tip adsorption.',
    'ScanStep': 'Scan step: Size of pixels in x and y directions and size of oscillation step in z direction.',
    'ScanSize': 'Scan size: Total size of scan region in x and y directions.',
    'ScanCenter': 'Scan center: center position of scan region in x and y directions.',
    'Distance': 'Distance: Average tip distance from the center of the closest atom',
    'Amplitude': 'Amplitude: Peak-to-peak oscillation amplitude for the tip.',
    'k': 'k: Cantilever spring constant. Only appears as a scaling constant.',
    'f0': 'f0: Cantilever eigenfrequency. Only appears as a scaling constant.',
    'df_steps': 'Number of steps in df approach curve when clicking on image.',
    'edit_geom': 'Edit Geometry: Edit the positions, atomic numbers, and charges of atoms.',
    'edit_ff': 'Edit FF: Edit Lennard-Jones parameters of forcefield.'
}

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser('ppm')
    parser.add_argument("-i", "--input", action="store", type=str, help="Input file.")
    parser.add_argument("-d", "--device", action="store", type=int, default=0, help="Choose OpenCL device.")
    parser.add_argument("-l", "--list-devices", action="store_true", help="List available devices and exit.")
    parser.add_argument("-v", '--verbosity', action="store", type=int, default=0, help="Set verbosity level (0-2)")
    args = parser.parse_args()
    return args.input, args.device, args.list_devices, args.verbosity

class ApplicationWindow(QtWidgets.QMainWindow):

    sw_pad = 4.0 # Default padding for scan window on each side of the molecule in xy plane

    def __init__(self, input_file, device, verbose=0):

        self.df = None
        self.xyzs = None
        self.Zs = None
        self.qs = None
        self.df_points = []

        # Initialize OpenCL environment on chosen device and create an afmulator instance to use for simulations
        oclu.init_env(device)
        self.afmulator = AFMulator()

        # Set verbosity level to same value everywhere
        if verbose > 0: print(f'Verbosity level = {verbose}')
        self.verbose = verbose
        self.afmulator.verbose = verbose
        self.afmulator.forcefield.verbose = verbose
        self.afmulator.scanner.verbose = verbose
        FFcl.bRuntime = verbose > 1

        # --- init QtMain
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Probe Particle Model")
        self.main_widget = QtWidgets.QWidget(self)
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        self.figCan = guiw.FigImshow( parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100, verbose=verbose)
        l00.addWidget(self.figCan, 2)
        l0 = QtWidgets.QVBoxLayout(self.main_widget); l00.addLayout(l0, 1)

        # ------- Probe Settings
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Probe settings"); lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font(); font.setPointSize(12); lb.setFont(font); lb.setMaximumHeight(50)
        vb.addWidget(lb)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Preset"); lb.setToolTip(TTips['Preset']); vb.addWidget(lb, 1)
        sl = QtWidgets.QComboBox(); self.slPreset = sl; sl.addItems(Presets.keys())
        sl.currentIndexChanged.connect(self.applyPreset); sl.setToolTip(TTips['Preset']); vb.addWidget(sl, 6)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Z"); lb.setToolTip(TTips['Z']); vb.addWidget(lb, 1)
        bx = QtWidgets.QSpinBox(); bx.setRange(0, 200); bx.setValue(8); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['Z']); vb.addWidget(bx, 2); self.bxZPP=bx

        lb = QtWidgets.QLabel("Multipole"); lb.setToolTip(TTips['Multipole']); vb.addWidget(lb, 2)
        sl = QtWidgets.QComboBox(); self.slMultipole = sl; sl.addItems([m.name for m in Multipoles]); sl.setCurrentIndex(sl.findText(Multipoles.dz2.name))
        sl.currentIndexChanged.connect(self.updateParams); sl.setToolTip(TTips['Multipole']); vb.addWidget(sl, 2)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Q [e]"); lb.setToolTip(TTips['Q']); vb.addWidget(lb, 1)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(-0.1); bx.setSingleStep(0.05); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['Q']); vb.addWidget(bx, 2); self.bxQ=bx
        lb = QtWidgets.QLabel("Sigma [Å]"); lb.setToolTip(TTips['Sigma']); vb.addWidget(lb, 2)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 2.0); bx.setValue(0.71); bx.setSingleStep(0.05); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['Sigma']); vb.addWidget(bx, 2); self.bxS=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        bxl = QtWidgets.QVBoxLayout(); vb.addLayout(bxl, 1)
        bxr = QtWidgets.QVBoxLayout(); vb.addLayout(bxr, 3)

        lb = QtWidgets.QLabel("K (x,y,R) [N/m]"); lb.setToolTip(TTips['K']); bxl.addWidget(lb)
        lb = QtWidgets.QLabel("Eq. pos (x,y,R) [Å]"); lb.setToolTip(TTips['EqPos']); bxl.addWidget(lb)

        vb = QtWidgets.QHBoxLayout(); bxr.addLayout(vb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.25); bx.setSingleStep(0.05); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['K']); vb.addWidget(bx); self.bxKx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.25); bx.setSingleStep(0.05); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['K']); vb.addWidget(bx); self.bxKy=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(30.0); bx.setSingleStep(5.0); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['K']); vb.addWidget(bx); self.bxKr=bx

        vb = QtWidgets.QHBoxLayout(); bxr.addLayout(vb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['EqPos']); vb.addWidget(bx); self.bxP0x=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['EqPos']); vb.addWidget(bx); self.bxP0y=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange( 0.0, 10.0); bx.setValue(3.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['EqPos']); vb.addWidget(bx); self.bxP0r=bx

        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        # ------- Scan settings
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Scan settings"); lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font(); font.setPointSize(12); lb.setFont(font); lb.setMaximumHeight(50)
        vb.addWidget(lb)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        bxl = QtWidgets.QVBoxLayout(); vb.addLayout(bxl, 1)
        bxr = QtWidgets.QVBoxLayout(); vb.addLayout(bxr, 3)

        lb = QtWidgets.QLabel("Scan step (x,y,z)[Å]"); lb.setToolTip(TTips['ScanStep']); bxl.addWidget(lb)
        lb = QtWidgets.QLabel("Scan size (x,y)[Å]"); lb.setToolTip(TTips['ScanSize']); bxl.addWidget(lb)
        lb = QtWidgets.QLabel("Scan center (x,y)[Å]"); lb.setToolTip(TTips['ScanCenter']); bxl.addWidget(lb)

        vb = QtWidgets.QHBoxLayout(); bxr.addLayout(vb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1); bx.setSingleStep(0.02); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['ScanStep']); vb.addWidget(bx); self.bxStepX=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1); bx.setSingleStep(0.02); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['ScanStep']); vb.addWidget(bx); self.bxStepY=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1); bx.setSingleStep(0.02); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['ScanStep']); vb.addWidget(bx); self.bxStepZ=bx 

        vb = QtWidgets.QHBoxLayout(); bxr.addLayout(vb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(16.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['ScanSize']); vb.addWidget(bx); self.bxSSx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(16.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['ScanSize']); vb.addWidget(bx); self.bxSSy=bx

        vb = QtWidgets.QHBoxLayout(); bxr.addLayout(vb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0, 100.0); bx.setValue(0.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['ScanCenter']); vb.addWidget(bx); self.bxSCx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0, 100.0); bx.setValue(0.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['ScanCenter']); vb.addWidget(bx); self.bxSCy=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Distance [Å]"); lb.setToolTip(TTips['Distance']); vb.addWidget(lb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(6.5); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['Distance']); vb.addWidget(bx); self.bxD=bx
        lb = QtWidgets.QLabel("Amplitude [Å]"); lb.setToolTip(TTips['Amplitude']); vb.addWidget(lb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(1.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['Amplitude']); vb.addWidget(bx); self.bxA=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Periodic boundary conditions"); vb.addWidget(lb)
        bx = QtWidgets.QCheckBox(); bx.setChecked(True); bx.toggled.connect(self.updateScanWindow); vb.addWidget(bx); self.bxPBC = bx

        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        # ------- df Conversion
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("df settings"); lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font(); font.setPointSize(12); lb.setFont(font); lb.setMaximumHeight(50)
        vb.addWidget(lb)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("k [kN/m]"); lb.setToolTip(TTips['k']); vb.addWidget(lb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,1000.0); bx.setSingleStep(0.1); bx.setValue(1.8);  bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['k']); vb.addWidget(bx); self.bxCant_K=bx
        lb = QtWidgets.QLabel("f0 [kHz]"); lb.setToolTip(TTips['f0']); vb.addWidget(lb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,2000.0); bx.setSingleStep(1.0); bx.setValue(30.3); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['f0']); vb.addWidget(bx); self.bxCant_f0=bx
        lb = QtWidgets.QLabel("df steps"); lb.setToolTip(TTips['df_steps']); vb.addWidget(lb, 1)
        bx = QtWidgets.QSpinBox(); bx.setRange(0, 50); bx.setValue(10); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['df_steps']); vb.addWidget(bx, 2); self.bxdfst=bx

        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        # ------- Buttons
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        
        # Geometry editor
        self.geomEditor = None
        bt = QtWidgets.QPushButton('Edit Geometry', self)
        bt.setToolTip(TTips['edit_geom'])
        bt.clicked.connect(self.showGeomEditor)
        self.btEditAtoms = bt; vb.addWidget(bt)
        
        # Forcefield parameter editor
        self.FFEditor = guiw.LJParamEditor(self.afmulator.typeParams, self)
        bt = QtWidgets.QPushButton('Edit FF', self)
        bt.setToolTip(TTips['edit_ff'])
        bt.clicked.connect(self.FFEditor.show)
        self.btEditFF = bt; vb.addWidget(bt)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        
        # --- btLoad
        self.btLoad = QtWidgets.QPushButton('Open File...', self)
        self.btLoad.setToolTip('Open new file.')
        self.btLoad.clicked.connect(self.openFile)
        vb.addWidget( self.btLoad )
        
        # --- btSave
        self.btSave = QtWidgets.QPushButton('Save Image...', self)
        self.btSave.setToolTip('Save current image.')
        self.btSave.clicked.connect(self.saveFig)
        vb.addWidget( self.btSave )

        # --- btSaveW (W- wsxm)
        self.btSaveW = QtWidgets.QPushButton('Save df...', self)
        self.btSaveW.setToolTip('Save current frequency shift data.')
        self.btSaveW.clicked.connect(self.saveDataW)
        vb.addWidget( self.btSaveW )

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.figCurv = guiw.PlotWindow( parent=self, width=5, height=4, dpi=100)

        if input_file:
            self.loadInput(input_file)

    def setScanWindow(self, scan_size, scan_center, step, distance, amplitude):
        '''Set scan window in AFMulator and update input fields'''

        if self.xyzs is None: return

        # Make scan size and amplitude multiples of the step size
        scan_dim = np.round([
            scan_size[0] / step[0] + 1,
            scan_size[1] / step[1] + 1,
            amplitude    / step[2]
        ]).astype(np.int32)
        scan_size = (scan_dim[:2] - 1) * step[:2]
        amplitude = scan_dim[2] * step[2]
        z_extra_steps = self.bxdfst.value() - 1
        scan_dim[2] += z_extra_steps
        z = self.xyzs[:, 2].max() + distance
        z_min = z - amplitude / 2
        z_max = z + amplitude / 2 + z_extra_steps * step[2]
        scan_window = (
            (scan_center[0] - scan_size[0] / 2, scan_center[1] - scan_size[1] / 2, z_min),
            (scan_center[0] + scan_size[0] / 2, scan_center[1] + scan_size[1] / 2, z_max)
        )
        self.afmulator.kCantilever = self.bxCant_K.value()
        self.afmulator.f0Cantilever = self.bxCant_f0.value()
        self.afmulator.npbc = (1, 1, 0) if self.bxPBC.isChecked() else (0, 0, 0)
        if self.verbose > 0: print("setScanWindow", step, scan_size, scan_center, scan_dim, scan_window)

        # Set new values to the fields
        guiw.set_box_value(self.bxSSx, scan_size[0])
        guiw.set_box_value(self.bxSSy, scan_size[1])
        guiw.set_box_value(self.bxSCx, scan_center[0])
        guiw.set_box_value(self.bxSCy, scan_center[1])
        guiw.set_box_value(self.bxD, distance)
        guiw.set_box_value(self.bxA, amplitude)

        # Set scan size and amplitude increments to match the set step size
        self.bxSSx.setSingleStep(step[0])
        self.bxSSy.setSingleStep(step[1])
        self.bxA.setSingleStep(step[2])

        # Set new scan window and dimension in AFMulator, and infer FF lvec from the scan window
        self.afmulator.df_steps = scan_dim[2] - z_extra_steps
        self.afmulator.setScanWindow(scan_window, tuple(scan_dim))
        self.afmulator.setLvec()

        if self.verbose > 0: print('lvec:\n', self.afmulator.forcefield.nDim, self.afmulator.lvec)

    def scanWindowFromGeom(self):
        '''Infer and set scan window from current geometry'''
        if self.xyzs is None: return
        scan_size = self.xyzs[:, :2].max(axis=0) - self.xyzs[:, :2].min(axis=0) + 2 * self.sw_pad
        scan_center = (self.xyzs[:, :2].max(axis=0) + self.xyzs[:, :2].min(axis=0)) / 2
        step = np.array([self.bxStepX.value(), self.bxStepY.value(), self.bxStepZ.value()])
        distance = self.bxD.value()
        amplitude = self.bxA.value()
        self.setScanWindow(scan_size, scan_center, step, distance, amplitude)

    def updateScanWindow(self):
        '''Get scan window from input fields and update'''
        if self.xyzs is None: return
        scan_size = np.array([self.bxSSx.value(), self.bxSSy.value()])
        scan_center = np.array([self.bxSCx.value(), self.bxSCy.value()])
        step = np.array([self.bxStepX.value(), self.bxStepY.value(), self.bxStepZ.value()])
        distance = self.bxD.value()
        amplitude = self.bxA.value()
        self.setScanWindow(scan_size, scan_center, step, distance, amplitude)
        self.update()

    def updateParams(self):
        '''Get parameter values from input fields and update'''

        if self.xyzs is None: return

        Q = self.bxQ.value()
        sigma = self.bxS.value()
        multipole = self.slMultipole.currentText()
        tipStiffness = [self.bxKx.value(), self.bxKy.value(), 0.0, self.bxKr.value()]
        tipR0 = [self.bxP0x.value(), self.bxP0y.value(), self.bxP0r.value()]

        if multipole == 's':
            Qs = [Q, 0, 0, 0]
            QZs = [0, 0, 0, 0]
        elif multipole == 'pz':
            Qs = [Q, -Q, 0, 0]
            QZs = [sigma, -sigma, 0, 0]
        elif multipole == 'dz2':
            Qs = [Q, -2*Q, Q, 0]
            QZs = [sigma, 0, -sigma, 0]

        self.afmulator.iZPP = int(self.bxZPP.value())
        self.afmulator.setQs(Qs, QZs)
        if isinstance(self.qs, HartreePotential):
            self.afmulator.setRho({multipole: Q}, sigma)
        self.afmulator.scanner.stiffness = np.array(tipStiffness, dtype=np.float32) / -PPU.eVA_Nm
        self.afmulator.tipR0 = tipR0

        self.update()

    def applyPreset(self):
        '''Get current preset, apply parameters, and update'''
        preset = Presets[self.slPreset.currentText()]
        if 'Z' in preset: guiw.set_box_value(self.bxZPP, preset['Z'])
        if 'Q' in preset: guiw.set_box_value(self.bxQ, preset['Q'])
        if 'Sigma' in preset: guiw.set_box_value(self.bxS, preset['Sigma'])
        if 'K' in preset:
            guiw.set_box_value(self.bxKx, preset['K'][0])
            guiw.set_box_value(self.bxKy, preset['K'][1])
            guiw.set_box_value(self.bxKr, preset['K'][2])
        if 'EqPos' in preset:
            guiw.set_box_value(self.bxP0x, preset['EqPos'][0])
            guiw.set_box_value(self.bxP0y, preset['EqPos'][1])
            guiw.set_box_value(self.bxP0r, preset['EqPos'][2])
        if 'Multipole' in preset:
            sl = self.slMultipole
            sl.blockSignals(True)
            sl.setCurrentIndex(sl.findText(preset['Multipole']))
            sl.blockSignals(False)
        self.updateParams()

    def update(self):
        '''Run simulation, and show the result'''
        if self.xyzs is None: return
        if self.verbose > 1: t0 = time.perf_counter()
        self.df = self.afmulator(self.xyzs, self.Zs, self.qs)
        if self.verbose > 1: print(f'AFMulator total time [s]: {time.perf_counter() - t0}')
        self.updateDataView()

    def loadInput(self, file_path):
        '''Load input file and show result
        
        Arguments:
            file_path: str. File to load. Has to be .xsf, .cube, or .xyz.
        '''

        # Load input file
        if file_path.endswith('.xsf') or file_path.endswith('.cube'):
            qs, xyzs, Zs = hartreeFromFile(file_path)
        elif file_path.endswith('.xyz'):
            with open(file_path, 'r') as f:
                xyzs, Zs, _, qs = basUtils.loadAtomsLines(f.readlines())
        else:
            raise ValueError(f'Unsupported file format for file `{file_path}`.')

        self.xyzs = xyzs
        self.Zs = Zs
        self.qs = qs
        self.df_points = []

        # Create geometry editor widget
        self.createGeomEditor()

        # Infer scan window from loaded geometry and run
        self.scanWindowFromGeom()
        self.updateParams()

    def createGeomEditor(self):
        '''Create a new geometry editor. Replace old one if it exists.'''
        if self.geomEditor:
            self.geomEditor.deleteLater()
            self.geomEditor = None
        enable_qs = not isinstance(self.qs, HartreePotential)
        self.geomEditor = guiw.GeomEditor(len(self.xyzs), enable_qs=enable_qs, parent=self,
            title="Geometry Editor")

    def showGeomEditor(self):
        if self.xyzs is None: return
        self.geomEditor.updateValues()
        self.geomEditor.show()

    def openFile(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '', '*.xyz *.xsf *.cube')
        if file_path:
            self.loadInput(file_path)
        
    def saveFig(self):
        if self.xyzs is None: return
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Save image","","Image files (*.png)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".png" )
            if self.verbose > 0: print("Saving image to :", fileName)
            self.figCan.fig.savefig( fileName,bbox_inches='tight')

    def saveDataW(self):
        if self.df is None: return
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Save df","","WSxM files (*.xyz)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".xyz" )
            if self.verbose > 0: print("Saving data to to :", fileName)
            npdata = self.df
            xs = np.arange(npdata.shape[1] )
            ys = np.arange(npdata.shape[0] )
            Xs, Ys = np.meshgrid(xs,ys)
            GU.saveWSxM_2D(fileName, npdata, Xs, Ys)

    def updateDataView(self):

        t1 = time.perf_counter()

        # Compute current coordinates of df line points
        points = []
        for x, y in self.df_points:
            x_min, y_min = self.afmulator.scan_window[0][:2]
            x_step, y_step = self.bxStepX.value(), self.bxStepY.value()
            ix = (x - x_min) / x_step
            iy = (y - y_min) / y_step
            points.append((ix, iy))

        # Plot df
        try:
            data = self.df.transpose(2, 1, 0)
            z = self.afmulator.scan_window[0][2] + self.afmulator.amplitude / 2
            self.figCan.plotSlice(data, -1, title=f'z = {z:.2f}Å', points=points)
        except Exception as e:
            print("Failed to plot df slice")
            if self.verbose > 1: print(e)

        if self.verbose > 1: print(f"plotSlice time {time.perf_counter() - t1:.5f} [s]")

    def clickImshow(self, ix, iy):
        if self.df is None: return

        # Remember x and y coordinates of point
        x_min, y_min = self.afmulator.scan_window[0][:2]
        x_step, y_step = self.bxStepX.value(), self.bxStepY.value()
        x = x_min + ix * x_step
        y = y_min + iy * y_step
        self.df_points.append((x, y))
        if self.verbose > 0: print('clickImshow', ix, iy, x, y)

        # Update line plot
        z_min = self.afmulator.scan_window[0][2] + self.afmulator.amplitude / 2
        z_max = self.afmulator.scan_window[1][2] - self.afmulator.amplitude / 2
        df_steps = self.bxdfst.value()
        zs = np.linspace(z_max, z_min, df_steps)
        ys = self.df[ix, iy, :]
        self.figCurv.show()
        self.figCurv.figCan.plotDatalines(zs, ys, "%i_%i" %(ix,iy))

    def clearPoints(self):
        self.df_points = []
        self.figCan.point_plots = []
        self.updateDataView()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    input_file, device, list_devices, verbosity = parse_args()
    if list_devices:
        print('\nAvailable OpenCL platforms:')
        oclu.print_platforms()
        sys.exit(0)
    aw = ApplicationWindow(input_file, device, verbosity)
    aw.show()
    sys.exit(qApp.exec_())

