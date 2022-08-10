#!/usr/bin/env python3

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases

import sys
import os
import time
import random
import traceback
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
    'Preset': 'Preset: Apply a probe parameter preset.',
    'Z': 'Z: Probe atomic number. Determines the Lennard-Jones parameters of the force field.',
    'Multipole': 'Multipole: Probe charge multipole type:\ns: monopole\npz: dipole\ndz2: quadrupole.',
    'Q': 'Q: Probe charge/multipole magnitude.',
    'Sigma': 'Sigma: Probe charge distribution width.',
    'point_charge': 'Use point-charge approximation for probe charge distribution. Faster but less accurate.',
    'K': 'K: Force constants for harmonic force holding the probe to the tip in x, y, and radial directions.',
    'EqPos': 'Eq. Pos: Probe equilibrium position with respect to the tip in x, y, and radial directions. Non-zero values for x and y models asymmetry in tip adsorption.',
    'ScanStep': 'Scan step: Size of pixels in x and y directions and size of oscillation step in z direction.',
    'ScanSize': 'Scan size: Total size of scan region in x and y directions.',
    'ScanCenter': 'Scan center: center position of scan region in x and y directions.',
    'Distance': 'Distance: Average tip distance from the center of the closest atom',
    'Amplitude': 'Amplitude: Peak-to-peak oscillation amplitude for the tip.',
    'PBC': 'Periodic Boundaries: Lattice vectors for periodic images of atoms.',
    'k': 'k: Cantilever spring constant. Only appears as a scaling constant.',
    'f0': 'f0: Cantilever eigenfrequency. Only appears as a scaling constant.',
    'z_steps': 'z steps: Number of steps in the df approach curve in z direction when clicking on image.',
    'view_geom': 'View Geometry: Show system geometry in ASE GUI.',
    'edit_geom': 'Edit Geometry: Edit the positions, atomic numbers, and charges of atoms.',
    'view_ff': 'View Forcefield: View forcefield components in a separate window.',
    'edit_ff': 'Edit Forcefield: Edit Lennard-Jones parameters of forcefield.'
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
    zoom_step = 1.0 # How much to increase/reduce scan size on zoom

    def __init__(self, input_file, device, verbose=0):

        self.df = None
        self.xyzs = None
        self.Zs = None
        self.qs = None
        self.pbc_lvec = None
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
        l1 = QtWidgets.QVBoxLayout(self.main_widget); l00.addLayout(l1, 2)
        self.figCan = guiw.FigImshow( parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100, verbose=verbose)
        l1.addWidget(self.figCan)
        l0 = QtWidgets.QVBoxLayout(self.main_widget); l00.addLayout(l0, 1)
        self.resize(1000, 600)

        # -------- Status Bar
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.dim_label = QtWidgets.QLabel('')
        self.status_bar.addPermanentWidget(self.dim_label)
        l1.addWidget(self.status_bar)

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
        lb = QtWidgets.QLabel("Point Charges"); lb.setToolTip(TTips['point_charge']); vb.addWidget(lb)
        bx = QtWidgets.QCheckBox(); bx.setChecked(True); bx.toggled.connect(self.updateParams); bx.setToolTip(TTips['point_charge']); vb.addWidget(bx); self.bxPC = bx

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
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-1000.0, 1000.0); bx.setValue(6.5); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['Distance']); vb.addWidget(bx); self.bxD=bx
        lb = QtWidgets.QLabel("Amplitude [Å]"); lb.setToolTip(TTips['Amplitude']); vb.addWidget(lb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(1.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['Amplitude']); vb.addWidget(bx); self.bxA=bx

        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        # ------- Periodic Boundaries
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Periodic Boundaries"); lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font(); font.setPointSize(12); lb.setFont(font); lb.setMaximumHeight(50)
        vb.addWidget(lb)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Use periodic boundary conditions"); vb.addWidget(lb)
        bx = QtWidgets.QCheckBox(); bx.setChecked(True); bx.toggled.connect(self.updatePBC); vb.addWidget(bx); self.bxPBC = bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        bxl = QtWidgets.QVBoxLayout(); vb.addLayout(bxl, 1)
        bxr = QtWidgets.QVBoxLayout(); vb.addLayout(bxr, 3)

        lb = QtWidgets.QLabel("A (x, y, z) [Å]"); lb.setToolTip(TTips['PBC']); bxl.addWidget(lb)
        lb = QtWidgets.QLabel("B (x, y, z) [Å]"); lb.setToolTip(TTips['PBC']); bxl.addWidget(lb)
        lb = QtWidgets.QLabel("C (x, y, z) [Å]"); lb.setToolTip(TTips['PBC']); bxl.addWidget(lb)
        
        vb = QtWidgets.QHBoxLayout(); bxr.addLayout(vb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(50); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCAx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCAy=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCAz=bx

        vb = QtWidgets.QHBoxLayout(); bxr.addLayout(vb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCBx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(50); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCBy=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCBz=bx

        vb = QtWidgets.QHBoxLayout(); bxr.addLayout(vb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCCx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCCy=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setValue(100); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updatePBC); bx.setToolTip(TTips['PBC']); vb.addWidget(bx); self.bxPBCCz=bx

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
        lb = QtWidgets.QLabel("z steps"); lb.setToolTip(TTips['z_steps']); vb.addWidget(lb, 1)
        bx = QtWidgets.QSpinBox(); bx.setRange(1, 50); bx.setValue(10); bx.valueChanged.connect(self.updateScanWindow); bx.setToolTip(TTips['z_steps']); vb.addWidget(bx, 2); self.bxdfst=bx

        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        # ------- Buttons
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 

        # Geometry viewer
        bt = QtWidgets.QPushButton('View Geometry', self)
        bt.setToolTip(TTips['view_geom'])
        bt.clicked.connect(self.showGeometry)
        self.btViewGeom = bt; vb.addWidget(bt)
        
        # Geometry editor
        self.geomEditor = None
        bt = QtWidgets.QPushButton('Edit Geometry', self)
        bt.setToolTip(TTips['edit_geom'])
        bt.clicked.connect(self.showGeomEditor)
        self.btEditAtoms = bt; vb.addWidget(bt)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 

        # Forcefield viewer
        self.FFViewer = guiw.FFViewer(self, verbose=verbose)
        bt = QtWidgets.QPushButton('View Forcefield', self)
        bt.setToolTip(TTips['view_ff'])
        bt.clicked.connect(self.showFFViewer)
        self.btViewFF = bt; vb.addWidget(bt)
        
        # Forcefield parameter editor
        self.FFEditor = guiw.LJParamEditor(self.afmulator.typeParams, self)
        bt = QtWidgets.QPushButton('Edit Forcefield', self)
        bt.setToolTip(TTips['edit_ff'])
        bt.clicked.connect(self.FFEditor.show)
        self.btEditFF = bt; vb.addWidget(bt)

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

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        if input_file:
            self.loadInput(input_file)

    def status_message(self, msg):
        self.status_bar.showMessage(msg)
        self.status_bar.repaint()

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

        # Update status bar info
        ff_dim = self.afmulator.forcefield.nDim
        self.dim_label.setText(f'Scan dim: {scan_dim[0]}x{scan_dim[1]}x{scan_dim[2]} | '
            f'FF dim: {ff_dim[0]}x{ff_dim[1]}x{ff_dim[2]}')

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
        use_point_charge = self.bxPC.isChecked()

        if multipole == 's':
            Qs = [Q, 0, 0, 0]
            QZs = [0, 0, 0, 0]
        elif multipole == 'pz':
            Qs = [Q, -Q, 0, 0]
            QZs = [sigma, -sigma, 0, 0]
        elif multipole == 'dz2':
            Qs = [Q, -2*Q, Q, 0]
            QZs = [sigma, 0, -sigma, 0]

        if self.verbose > 0: print('updateParams', Q, sigma, multipole, tipStiffness, tipR0,
            use_point_charge, type(self.qs))

        self.afmulator.iZPP = int(self.bxZPP.value())
        self.afmulator.setQs(Qs, QZs)
        if use_point_charge:
            self.afmulator.setRho(None, sigma)
        elif isinstance(self.qs, HartreePotential):
            self.afmulator.setRho({multipole: Q}, sigma)
        self.afmulator.scanner.stiffness = np.array(tipStiffness, dtype=np.float32) / -PPU.eVA_Nm
        self.afmulator.tipR0 = tipR0

        self.update()

    def setPBC(self, lvec, enabled):
        '''Set periodic boundary condition lattice'''

        if self.verbose > 0: print('setPBC', lvec, enabled)

        if enabled:
            self.pbc_lvec = lvec
            self.afmulator.npbc = (1, 1, 1)
        else:
            self.pbc_lvec = None
            self.afmulator.npbc = (0, 0, 0)

        # Set check-box state
        self.bxPBC.blockSignals(True)
        self.bxPBC.setChecked(enabled)
        self.bxPBC.blockSignals(False)

        # Set lattice vector values
        if lvec is not None:
            guiw.set_box_value(self.bxPBCAx, lvec[0][0])
            guiw.set_box_value(self.bxPBCAy, lvec[0][1])
            guiw.set_box_value(self.bxPBCAz, lvec[0][2])
            guiw.set_box_value(self.bxPBCBx, lvec[1][0])
            guiw.set_box_value(self.bxPBCBy, lvec[1][1])
            guiw.set_box_value(self.bxPBCBz, lvec[1][2])
            guiw.set_box_value(self.bxPBCCx, lvec[2][0])
            guiw.set_box_value(self.bxPBCCy, lvec[2][1])
            guiw.set_box_value(self.bxPBCCz, lvec[2][2])

        # Disable lattice vector boxes if PBC not enabled
        for bx in [
            self.bxPBCAx, self.bxPBCAy, self.bxPBCAz,
            self.bxPBCBx, self.bxPBCBy, self.bxPBCBz,
            self.bxPBCCx, self.bxPBCCy, self.bxPBCCz
            ]:
            bx.setDisabled(not enabled)
    
    def updatePBC(self):
        '''Get PBC lattice from from input fields and update'''
        lvec = np.array([
            [self.bxPBCAx.value(), self.bxPBCAy.value(), self.bxPBCAz.value()],
            [self.bxPBCBx.value(), self.bxPBCBy.value(), self.bxPBCBz.value()],
            [self.bxPBCCx.value(), self.bxPBCCy.value(), self.bxPBCCz.value()]
        ])
        toggle = self.bxPBC.isChecked()
        self.setPBC(lvec, toggle)
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
        self.status_message('Running simulation...')
        self.df = self.afmulator(self.xyzs, self.Zs, self.qs, pbc_lvec=self.pbc_lvec)
        if self.verbose > 1: print(f'AFMulator total time [s]: {time.perf_counter() - t0}')
        self.status_message('Updating plot...')
        self.updateDataView()
        if self.FFViewer.isVisible():
            self.status_message('Updating Force field viewer...')
            self.FFViewer.updateFF()
            self.FFViewer.updateView()
        self.status_message('Ready')

    def loadInput(self, file_path):
        '''Load input file and show result
        
        Arguments:
            file_path: str. File to load. Has to be POSCAR, .in, .xsf, .cube, or .xyz.
        '''

        # Load input file
        file_name = os.path.split(file_path)[1].lower()
        ext = os.path.splitext(file_name)[1]
        if self.verbose > 0: print(f'loadInput: {file_path}, {file_name}, {ext}')
        if file_name in ['poscar', 'contcar']:
            xyzs, Zs, lvec = basUtils.loadPOSCAR(file_path)
            qs = np.zeros(len(Zs))
            lvec = lvec[1:]
        elif ext == '.in':
            xyzs, Zs, lvec = basUtils.loadGeometryIN(file_path)
            qs = np.zeros(len(Zs))
            lvec = lvec[1:] if len(lvec) > 0 else None
        elif ext in ['.xsf', '.cube']:
            qs, xyzs, Zs = hartreeFromFile(file_path)
            lvec = qs.lvec[1:]
        elif ext == '.xyz':
            with open(file_path, 'r') as f:
                xyzs, Zs, _, qs = basUtils.loadAtomsLines(f.readlines())
            lvec = None
        else:
            raise ValueError(f'Unsupported file format for file `{file_path}`.')

        self.xyzs = xyzs
        self.Zs = Zs
        self.qs = qs
        self.df_points = []

        self.bxPC.blockSignals(True)
        if isinstance(self.qs, HartreePotential):
            # Default to no point-charge tip for Hartree potentials
            self.bxPC.setChecked(False)
            self.bxPC.setDisabled(False)
        else:
            # Point-charge systems only support point-charge tips for now
            self.bxPC.setChecked(True)
            self.bxPC.setDisabled(True)
        self.bxPC.blockSignals(False)

        # Create geometry editor widget
        self.createGeomEditor()

        # Infer scan window from loaded geometry and run
        self.scanWindowFromGeom()
        self.setPBC(lvec, lvec is not None)
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

    def showFFViewer(self):
        if self.xyzs is None: return
        self.FFViewer.updateFF()
        self.FFViewer.updateView()
        self.FFViewer.show()

    def showGeometry(self):
        try:
            from ase import Atoms
            from ase.visualize import view
        except ModuleNotFoundError as e:
            print('No ase installation detected. Cannot show molecule geometry.')
            if self.verbose > 1: traceback.print_exc()
            return
        atoms = Atoms(positions=self.xyzs, numbers=self.Zs, cell=self.pbc_lvec, pbc=self.afmulator.npbc)
        view(atoms)

    def openFile(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '',
            '*.xyz *.in *.xsf *.cube POSCAR CONTCAR')
        if file_path:
            self.status_message('Opening file...')
            self.loadInput(file_path)
        
    def saveFig(self):
        if self.xyzs is None: return
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Save image","","Image files (*.png)")
        if fileName:
            self.status_message('Saving image...')
            fileName = guiw.correct_ext( fileName, ".png" )
            if self.verbose > 0: print("Saving image to :", fileName)
            self.figCan.fig.savefig( fileName,bbox_inches='tight')
            self.status_message('Ready')

    def saveDataW(self):
        if self.df is None: return
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save df", "df.xsf",
            "XCrySDen files (*.xsf);; WSxM files (*.xyz)")
        if not fileName: return
        ext = os.path.splitext(fileName)[1]
        if ext not in ['.xyz', '.xsf']:
            self.status_message('Unsupported file type in df save file path')
            print(f'Unsupported file type in df save file path `{fileName}`')
            return
        self.status_message('Saving data...')
        if self.verbose > 0: print(f'Saving df data to {fileName}...')
        if ext == '.xyz':
            data = self.df[:, :, -1].T
            xs = np.linspace(0, self.bxSSx.value(), data.shape[1], endpoint=False)
            ys = np.linspace(0, self.bxSSy.value(), data.shape[0], endpoint=False)
            Xs, Ys = np.meshgrid(xs,ys)
            GU.saveWSxM_2D(fileName, data, Xs, Ys)
        elif ext == '.xsf':
            data = self.df.transpose(2, 1, 0)[::-1]
            sw = self.afmulator.scan_window
            size = np.array(sw[1]) - np.array(sw[0])
            size[2] -= self.afmulator.amplitude - self.bxStepZ.value()
            lvecScan = np.array([
                [sw[0][0], sw[0][1], sw[0][2] - self.bxP0r.value()],
                [size[0],       0,       0],
                [      0, size[1],       0],
                [      0,       0, size[2]],
            ])
            if self.pbc_lvec is not None:
                lvec = np.append([[0, 0, 0]], self.pbc_lvec, axis=0)
                atomstring = basUtils.primcoords2Xsf(self.Zs, self.xyzs.T, lvec)
            else:
                atomstring = GU.XSF_HEAD_DEFAULT
            GU.saveXSF(fileName, data, lvecScan, head=atomstring, verbose=0)
        else:
            raise RuntimeError('This should not happen. Missing file format check?')
        if self.verbose > 0: print("Done saving df data.")
        self.status_message('Ready')

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
            title = f'z = {z:.2f}Å'
            if not isinstance(self.qs, HartreePotential) and np.allclose(self.qs, 0):
                title += ' (No electrostatics)'
            self.figCan.plotSlice(data, -1, title=title, points=points)
        except Exception as e:
            print("Failed to plot df slice")
            traceback.print_exc()

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
        self.updateDataView()

    def zoomTowards(self, ix, iy, zoom_direction):

        if self.verbose > 0: print('zoomTowards', ix, iy, zoom_direction)

        scan_size = np.array([self.bxSSx.value(), self.bxSSy.value()])
        scan_center = np.array([self.bxSCx.value(), self.bxSCy.value()])
        frac_coord = np.array([ix, iy]) / (np.array(self.df.shape[:2]) - 1) - 0.5
        offset = self.zoom_step * frac_coord

        if zoom_direction == 'in':
            scan_size -= self.zoom_step
            scan_center += offset
        elif zoom_direction == 'out':
            scan_size += self.zoom_step
            scan_center -= offset

        guiw.set_box_value(self.bxSSx, scan_size[0])
        guiw.set_box_value(self.bxSSy, scan_size[1])
        guiw.set_box_value(self.bxSCx, scan_center[0])
        guiw.set_box_value(self.bxSCy, scan_center[1])

        self.updateScanWindow()

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

