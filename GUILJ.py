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

DataViews = Enum( 'DataViews','FFin FFout df FFel FFpl' )

# Pre-autosetting:
# zmin = 1.0; zmax = 20.0; ymax = 20.0; xmax =20.0;
# atoms0, nDim0, lvec0 = basUtils.loadXSFGeom( "FFel_z.xsf" )
# zmin = round(max(atoms0[3]),1); zmax=round(lvec0[3][2],1); ymax=round(lvec0[1][1]+lvec0[2][1],1); xmax=round(lvec0[1][0]+lvec0[2][0],1)
zmin, zmax, ymax, xmax = 8.0, 10.0, 16.0, 16.0

print("zmin, zmax, ymax, xmax")
print(zmin, zmax, ymax, xmax)

class ApplicationWindow(QtWidgets.QMainWindow):

    sw_pad = 4.0 # Default padding for scan window on each side of the molecule in xy plane
    distance = 6.0 # Distance between the closest atom and the closest approach of the tip

    def __init__(self):

        # Initialize OpenCL environment on some device and create an afmulator instance to use for simulations
        oclu.init_env()
        self.afmulator = AFMulator()

        # --- init QtMain
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Probe Particle Model")
        self.main_widget = QtWidgets.QWidget(self)
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        self.figCan = guiw.FigImshow( parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100)
        l00.addWidget(self.figCan)
        l0 = QtWidgets.QVBoxLayout(self.main_widget); l00.addLayout(l0);

        # -------------- Potential
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb);

        sl = QtWidgets.QComboBox(); self.slDataView = sl; vb.addWidget(sl)
        sl.addItem( DataViews.FFpl.name ); sl.addItem( DataViews.FFel.name ); sl.addItem( DataViews.FFin.name ); sl.addItem( DataViews.FFout.name ); sl.addItem(DataViews.df.name );
        sl.setCurrentIndex( sl.findText( DataViews.df.name ) )
        sl.currentIndexChanged.connect(self.updateDataView)
        
        
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Z") )
        bx = QtWidgets.QSpinBox(); bx.setRange(0, 200); bx.setValue(8); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxZPP=bx
        
        # -------------- Relaxation 
        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Q [e]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(-0.1); bx.setSingleStep(0.05); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxQ=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("K {x,y,R} [N/m]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.25); bx.setSingleStep(0.05); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxKx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.25); bx.setSingleStep(0.05); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxKy=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(30.0); bx.setSingleStep(5.0); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxKr=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Eq. pos {x,y,R} [A]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxP0x=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxP0y=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange( 0.0, 10.0); bx.setValue(3.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxP0r=bx
   
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Scan start {x,y,z}[A]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(0.0);   bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxSpanMinX=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(0.0);   bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxSpanMinY=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(zmin);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxSpanMinZ=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Scan end {x,y,z}[A]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(xmax);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxSpanMaxX=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(ymax);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxSpanMaxY=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(zmax);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxSpanMaxZ=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Scan step {x,y,z}[A]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1);  bx.setSingleStep(0.02); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxStepX=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1);  bx.setSingleStep(0.02); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxStepY=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1);  bx.setSingleStep(0.02); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxStepZ=bx 

        # -------------- df Conversion & plotting
        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("{ iz (down from top), nAmp }") )
        bx = QtWidgets.QSpinBox();bx.setRange(-300,500); bx.setSingleStep(1); bx.setValue(0); bx.valueChanged.connect(self.updateDataView); vb.addWidget(bx); self.bxZ=bx
        bx = QtWidgets.QSpinBox();bx.setRange(0,50 ); bx.setSingleStep(1); bx.setValue(10); bx.valueChanged.connect(self.updateScanWindow); vb.addWidget(bx); self.bxA=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("{ k[kN/m], f0 [kHz] }") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,1000.0); bx.setSingleStep(0.1); bx.setValue(1.8);  bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxCant_K=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,2000.0); bx.setSingleStep(1.0); bx.setValue(30.3); bx.valueChanged.connect(self.updateParams); vb.addWidget(bx); self.bxCant_f0=bx

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
        self.btLoad = QtWidgets.QPushButton('Load', self)
        self.btLoad.setToolTip('Load inputs')
        self.btLoad.clicked.connect(self.loadInput)
        vb.addWidget( self.btLoad )
        
        # --- btSave
        self.btSave = QtWidgets.QPushButton('save fig', self)
        self.btSave.setToolTip('save current figure')
        self.btSave.clicked.connect(self.saveFig)
        vb.addWidget( self.btSave )

        # --- btSaveW (W- wsxm)
        self.btSaveW = QtWidgets.QPushButton('save data', self)
        self.btSaveW.setToolTip('save current figure data')
        self.btSaveW.clicked.connect(self.saveDataW)
        vb.addWidget( self.btSaveW )

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.figCurv = guiw.PlotWindow( parent=self, width=5, height=4, dpi=100)

        # Load input file
        self.loadInput('mol.xyz')

    def setScanWindow(self, rmin, rmax, step, df_steps):
        '''Set scan window in AFMulator and update input fields'''

        # Make scan window be a multiple of the step size
        scan_dim = np.round((rmax - rmin) / step).astype(np.int32)
        scan_window = (tuple(rmin), tuple(rmin + step * scan_dim))
        scan_dim[:2] += 1
        print("setScanWindow", step, rmin, rmax, scan_dim, scan_window, self.afmulator.lvec)

        # Set new values to the fields. Need to temporarily block the signals to do so.
        self.bxSpanMinX.blockSignals(True); self.bxSpanMinY.blockSignals(True); self.bxSpanMinZ.blockSignals(True)
        self.bxSpanMaxX.blockSignals(True); self.bxSpanMaxY.blockSignals(True); self.bxSpanMaxZ.blockSignals(True)
        self.bxSpanMinX.setValue(scan_window[0][0]); self.bxSpanMinY.setValue(scan_window[0][1]); self.bxSpanMinZ.setValue(scan_window[0][2])
        self.bxSpanMaxX.setValue(scan_window[1][0]); self.bxSpanMaxY.setValue(scan_window[1][1]); self.bxSpanMaxZ.setValue(scan_window[1][2])
        self.bxSpanMinX.blockSignals(False); self.bxSpanMinY.blockSignals(False); self.bxSpanMinZ.blockSignals(False)
        self.bxSpanMaxX.blockSignals(False); self.bxSpanMaxY.blockSignals(False); self.bxSpanMaxZ.blockSignals(False)

        # Set new scan window and dimension in AFMulator, and infer FF lvec from the scan window
        self.afmulator.df_steps = df_steps
        self.afmulator.setScanWindow(scan_window, tuple(scan_dim))
        self.afmulator.setLvec()

    def scanWindowFromGeom(self):
        '''Infer and set scan window from current geometry'''
        step = np.array( [ float(self.bxStepX.value()), float(self.bxStepY.value()), float(self.bxStepZ.value()) ] )
        df_steps = int(self.bxA.value())
        rmin = self.xyzs.min(axis=0)
        rmax = self.xyzs.max(axis=0)
        rmin[:2] -= self.sw_pad
        rmax[:2] += self.sw_pad
        rmin[2] += self.distance
        rmax[2] = rmin[2] + df_steps * step[2]
        self.setScanWindow(rmin, rmax, step, df_steps)

    def updateScanWindow(self):
        '''Get scan window from input fields and update'''
        step = np.array( [ float(self.bxStepX   .value()), float(self.bxStepY   .value()), float(self.bxStepZ   .value()) ] )
        rmin = np.array( [ float(self.bxSpanMinX.value()), float(self.bxSpanMinY.value()), float(self.bxSpanMinZ.value()) ] )
        rmax = np.array( [ float(self.bxSpanMaxX.value()), float(self.bxSpanMaxY.value()), float(self.bxSpanMaxZ.value()) ] )
        df_steps = int(self.bxA.value())
        self.setScanWindow(rmin, rmax, step, df_steps)
        self.update()

    def updateParams(self):
        '''Get parameter values from input fields and update'''

        Qs, QZs = [self.bxQ.value(), 0, 0, 0], [0, 0, 0, 0] # TODO allow non-monopole
        print(Qs, QZs)
        tipStiffness = [self.bxKx.value(), self.bxKy.value(), 0.0, self.bxKr.value()]
        tipR0 = [self.bxP0x.value(), self.bxP0y.value(), self.bxP0r.value()]

        self.afmulator.iZPP = int(self.bxZPP.value())
        self.afmulator.setQs(Qs, QZs)
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
        
    def saveFig(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Image files (*.png)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".png" )
            print("saving image to :", fileName)
            self.figCan.fig.savefig( fileName,bbox_inches='tight')

    def saveDataW(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","WSxM files (*.xyz)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".xyz" )
            print("saving data to to :", fileName)
            iz,data = self.selectDataView()
            npdata=data[iz]
            xs = np.arange(npdata.shape[1] )
            ys = np.arange(npdata.shape[0] )
            Xs, Ys = np.meshgrid(xs,ys)
            GU.saveWSxM_2D(fileName, npdata, Xs, Ys)
    
    def selectDataView(self):   # !!! Everything from TOP now !!! #
        dview = self.slDataView.currentText()
        #print "DEBUG dview : ", dview
        iz    = int( self.bxZ.value() )
        data = None;
        if   dview == DataViews.df.name:
            data = self.df
        elif dview == DataViews.FFout.name:
            data = np.transpose( self.FEout[:,:,:,2], (2,1,0) )
        elif dview == DataViews.FFin.name:
            data = self.FEin[::-1,:,:,2]
        elif dview == DataViews.FFpl.name:
            data = self.FF  [::-1,:,:,2]
        elif dview == DataViews.FFel.name:
            data = self.FFel[::-1,:,:,2]
            #print "data FFel.shape[1:3]", data.shape[1:3]
            #iz = data.shape[0]-iz-1
        return iz, data

    def updateDataView(self):
        t1 = time.perf_counter()
        iz, data = self.selectDataView()
        self.viewed_data = data 
        #self.figCan.plotSlice_iz(iz)
        try:
            data = data.transpose(2, 0, 1)
            print(data.shape)
            self.figCan.plotSlice( data, iz, title=f'iz = {iz}')
        except:
            print("cannot plot slice #", iz)
        t2 = time.perf_counter(); print("plotSlice time %f [s]" %(t2-t1))

    def clickImshow(self,ix,iy):
        ys = self.viewed_data[ :, iy, ix ]
        self.figCurv.show()
        self.figCurv.figCan.plotDatalines( ( list(range(len(ys))), ys, "%i_%i" %(ix,iy) )  )

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())

