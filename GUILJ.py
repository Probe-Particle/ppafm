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
from pyProbeParticle.fieldOCL            import hartreeFromFile
import pyProbeParticle.GridUtils  as GU
import pyProbeParticle.common     as PPU
import pyProbeParticle.oclUtils   as oclu
import pyProbeParticle.GUIWidgets as guiw

DataViews = Enum('DataViews','df FFin FFout FFel FFpl')
Multipoles = Enum('Multipoles', 's pz dz2')

TTips = {
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
    'f0': 'f0: Cantilever eigenfrequency. Only appears as a scaling constant. '
}

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser('ppm')
    parser.add_argument( "-i", "--input", action="store", type=str, help="Input file.")
    parser.add_argument( "-d", "--device", action="store", type=int, default=0, help="Choose OpenCL device.")
    parser.add_argument( "-l", "--list-devices", action="store_true", help="List available devices and exit.")
    args = parser.parse_args()
    return args.input, args.device, args.list_devices

class ApplicationWindow(QtWidgets.QMainWindow):

    sw_pad = 4.0 # Default padding for scan window on each side of the molecule in xy plane

    def __init__(self, input_file, device):

        # Initialize OpenCL environment on chosen device and create an afmulator instance to use for simulations
        oclu.init_env(device)
        self.afmulator = AFMulator()

        # --- init QtMain
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Probe Particle Model")
        self.main_widget = QtWidgets.QWidget(self)
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        self.figCan = guiw.FigImshow( parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100)
        l00.addWidget(self.figCan, 2)
        l0 = QtWidgets.QVBoxLayout(self.main_widget); l00.addLayout(l0, 1)

        # -------- Data view
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)

        # lb = QtWidgets.QLabel("Data View"); vb.addWidget(lb)
        # sl = QtWidgets.QComboBox(); self.slDataView = sl; vb.addWidget(sl)
        # sl.addItems([d.name for d in DataViews])
        # sl.setCurrentIndex( sl.findText(DataViews.df.name))
        # sl.currentIndexChanged.connect(self.updateDataView)

        # ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        # ------- Probe Settings
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Probe settings"); lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font(); font.setPointSize(12); lb.setFont(font); lb.setMaximumHeight(50)
        vb.addWidget(lb)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("Z"); lb.setToolTip(TTips['Z']); vb.addWidget(lb, 1)
        bx = QtWidgets.QSpinBox(); bx.setRange(0, 200); bx.setValue(8); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['Z']); vb.addWidget(bx, 2); self.bxZPP=bx

        lb = QtWidgets.QLabel("Multipole"); lb.setToolTip(TTips['Multipole']); vb.addWidget(lb, 2)
        sl = QtWidgets.QComboBox(); self.slMultipole = sl; sl.addItems([m.name for m in Multipoles]); sl.setCurrentIndex(sl.findText(Multipoles.dz2.name))
        sl.currentIndexChanged.connect(self.updateParams); bx.setToolTip(TTips['Multipole']); vb.addWidget(sl, 2)

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

        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        # ------- df Conversion
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("df settings"); lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font(); font.setPointSize(12); lb.setFont(font); lb.setMaximumHeight(50)
        vb.addWidget(lb)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb)
        lb = QtWidgets.QLabel("k [kN/m]"); lb.setToolTip(TTips['k']); vb.addWidget(lb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,1000.0); bx.setSingleStep(0.1); bx.setValue(1.8);  bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['k']); vb.addWidget(bx); self.bxCant_K=bx
        lb = QtWidgets.QLabel("f0 [kHz]"); lb.setToolTip(TTips['f0']); vb.addWidget(lb)
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,2000.0); bx.setSingleStep(1.0); bx.setValue(30.3); bx.valueChanged.connect(self.updateParams); bx.setToolTip(TTips['f0']); vb.addWidget(bx); self.bxCant_f0=bx

        # === buttons
        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        
        # --- EditAtoms
        # self.geomEditor = guiw.EditorWindow(self,title="Geometry Editor")
        # bt = QtWidgets.QPushButton('Edit Geom', self)
        # bt.setToolTip('Edit atomic structure')
        # bt.clicked.connect(self.geomEditor.show)
        # self.btEditAtoms = bt; vb.addWidget( bt )
        
        # --- EditFFparams
        # self.speciesEditor = guiw.EditorWindow(self,title="Species Editor")
        # bt = QtWidgets.QPushButton('Edit Params', self)
        # bt.setToolTip('Edit atomic structure')
        # bt.clicked.connect(self.speciesEditor.show)
        # self.btEditParams = bt; vb.addWidget( bt )

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

        # Make scan size and amplitude multiples of the step size
        scan_dim = np.round([
            scan_size[0] / step[0] + 1,
            scan_size[1] / step[1] + 1,
            amplitude    / step[2]
        ]).astype(np.int32)
        scan_size = (scan_dim[:2] - 1) * step[:2]
        amplitude = scan_dim[2] * step[2]
        z = self.xyzs[:, 2].max() + distance
        scan_window = (
            (scan_center[0] - scan_size[0] / 2, scan_center[1] - scan_size[1] / 2, z - amplitude / 2),
            (scan_center[0] + scan_size[0] / 2, scan_center[1] + scan_size[1] / 2, z + amplitude / 2)
        )
        print("setScanWindow", step, scan_size, scan_center, scan_dim, scan_window)

        # Set new values to the fields. Need to temporarily block the signals to do so.
        self.bxSSx.blockSignals(True); self.bxSSx.setValue(scan_size[0]); self.bxSSx.blockSignals(False)
        self.bxSSy.blockSignals(True); self.bxSSy.setValue(scan_size[1]); self.bxSSy.blockSignals(False)
        self.bxSCx.blockSignals(True); self.bxSCx.setValue(scan_center[0]); self.bxSCx.blockSignals(False)
        self.bxSCy.blockSignals(True); self.bxSCy.setValue(scan_center[1]); self.bxSCy.blockSignals(False)
        self.bxD.blockSignals(True); self.bxD.setValue(distance); self.bxD.blockSignals(False)
        self.bxA.blockSignals(True); self.bxA.setValue(amplitude); self.bxA.blockSignals(False)

        # Set scan size and amplitude increments to match the set step size
        self.bxSSx.setSingleStep(step[0])
        self.bxSSy.setSingleStep(step[1])
        self.bxA.setSingleStep(step[2])

        # Set new scan window and dimension in AFMulator, and infer FF lvec from the scan window
        self.afmulator.df_steps = scan_dim[2]
        self.afmulator.setScanWindow(scan_window, tuple(scan_dim))
        self.afmulator.setLvec()

        print(self.afmulator.lvec)

    def scanWindowFromGeom(self):
        '''Infer and set scan window from current geometry'''
        scan_size = self.xyzs[:, :2].max(axis=0) - self.xyzs[:, :2].min(axis=0) + 2 * self.sw_pad
        scan_center = (self.xyzs[:, :2].max(axis=0) + self.xyzs[:, :2].min(axis=0)) / 2
        step = np.array([self.bxStepX.value(), self.bxStepY.value(), self.bxStepZ.value()])
        distance = self.bxD.value()
        amplitude = self.bxA.value()
        self.setScanWindow(scan_size, scan_center, step, distance, amplitude)

    def updateScanWindow(self):
        '''Get scan window from input fields and update'''
        scan_size = np.array([self.bxSSx.value(), self.bxSSy.value()])
        scan_center = np.array([self.bxSCx.value(), self.bxSCy.value()])
        step = np.array([self.bxStepX.value(), self.bxStepY.value(), self.bxStepZ.value()])
        distance = self.bxD.value()
        amplitude = self.bxA.value()
        self.setScanWindow(scan_size, scan_center, step, distance, amplitude)
        self.update()

    def updateParams(self):
        '''Get parameter values from input fields and update'''

        Q = self.bxQ.value()
        sigma = self.bxS.value()
        multipole = self.slMultipole.currentText()
        print(Q, sigma, multipole)
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
        self.afmulator.setRho({multipole: Q}, sigma)
        self.afmulator.scanner.stiffness = np.array(tipStiffness, dtype=np.float32) / -PPU.eVA_Nm
        self.afmulator.tipR0 = tipR0
        self.afmulator.kCantilever = self.bxCant_K.value()
        self.afmulator.f0Cantilever = self.bxCant_f0.value()

        self.update()

    def update(self):
        '''Run simulation, and show the result'''
        self.df = self.afmulator(self.xyzs, self.Zs, self.qs)
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

        # Infer scan window from loaded geometry and run
        self.scanWindowFromGeom()
        self.updateParams()

    def openFile(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '', '*.xyz *.xsf *.cube')
        if file_path:
            self.loadInput(file_path)
        
    def saveFig(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Save image","","Image files (*.png)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".png" )
            print("saving image to :", fileName)
            self.figCan.fig.savefig( fileName,bbox_inches='tight')

    def saveDataW(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Save df","","WSxM files (*.xyz)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".xyz" )
            print("saving data to to :", fileName)
            npdata = self.selectDataView()
            xs = np.arange(npdata.shape[1] )
            ys = np.arange(npdata.shape[0] )
            Xs, Ys = np.meshgrid(xs,ys)
            GU.saveWSxM_2D(fileName, npdata, Xs, Ys)
    
    def selectDataView(self):   # !!! Everything from TOP now !!! #
        # dview = self.slDataView.currentText()
        # data = None;
        # if   dview == DataViews.df.name:
        #     data = self.df
        # elif dview == DataViews.FFout.name:
        #     data = np.transpose( self.FEout[:,:,:,2], (2,1,0) )
        # elif dview == DataViews.FFin.name:
        #     data = self.FEin[::-1,:,:,2]
        # elif dview == DataViews.FFpl.name:
        #     data = self.FF  [::-1,:,:,2]
        # elif dview == DataViews.FFel.name:
        #     data = self.FFel[::-1,:,:,2]
        #     #print "data FFel.shape[1:3]", data.shape[1:3]
        #     #iz = data.shape[0]-iz-1
        return self.df

    def updateDataView(self):
        t1 = time.perf_counter()
        data = self.selectDataView()
        self.viewed_data = data
        try:
            data = data.transpose(2, 1, 0)
            z = (self.afmulator.scan_window[0][2] + self.afmulator.scan_window[1][2]) / 2
            self.figCan.plotSlice(data, 0, title=f'z = {z:.2f}Å')
        except:
            print("cannot plot slice #", 0)
        t2 = time.perf_counter(); print("plotSlice time %f [s]" %(t2-t1))

    def clickImshow(self,ix,iy):
        ys = self.viewed_data[ :, iy, ix ]
        self.figCurv.show()
        self.figCurv.figCan.plotDatalines( ( list(range(len(ys))), ys, "%i_%i" %(ix,iy) )  )

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    input_file, device, list_devices = parse_args()
    if list_devices:
        print('\nAvailable OpenCL platforms:')
        oclu.print_platforms()
        sys.exit(0)
    aw = ApplicationWindow(input_file, device)
    aw.show()
    sys.exit(qApp.exec_())

