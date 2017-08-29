#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases

from __future__ import unicode_literals
import sys
import os
import time
import random
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets, QtGui
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

#sys.path.append("/home/prokop/git/ProbeParticleModel_OCL") 
#import pyProbeParticle.GridUtils as GU

from   pyProbeParticle import basUtils
from   pyProbeParticle import PPPlot 
import pyProbeParticle.GridUtils as GU
import pyProbeParticle.common    as PPU
import pyProbeParticle.cpp_utils as cpp_utils

import pyopencl as cl
import pyProbeParticle.oclUtils    as oclu 
import pyProbeParticle.fieldOCL    as FFcl 
import pyProbeParticle.RelaxOpenCL as oclr

Modes = Enum('Mode', 'MorseFFel  LJQ')

class MyDynamicMplCanvas(FigureCanvas):
    """A canvas that updates itself every second with a new plot."""

    cbar = None 
    
    def __init__(self, parent=None, width=5, height=4, dpi=100 ):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        #self.compute_initial_figure()
        FigureCanvas.__init__(self, self.fig )
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
            
    def plotSlice(self, F ):
        self.axes.cla()
        self.img = self.axes.imshow( F, origin='image', cmap='gray', interpolation='nearest' )
        if self.cbar is None:
            self.cbar = self.fig.colorbar( self.img )
        self.cbar.set_clim( vmin=F.min(), vmax=F.max() )
        self.cbar.update_normal(self.img)
        self.draw()

# TODO : we use now QDialog instead
class AtomEditor(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(AtomEditor, self).__init__(parent)
        self.parent = parent
        self.textEdit = QtWidgets.QTextEdit()
        if parent.str_Atoms is not None:
            self.textEdit.setText(parent.str_Atoms)
        
        self.setCentralWidget(self.textEdit)
        self.resize(600, 800)
        
        l0 = QtWidgets.QVBoxLayout(self)
        bt = QtWidgets.QPushButton('Update', self)
        bt.setToolTip('recalculate using this atomic structure')
        bt.clicked.connect(parent.updateFF)
        self.btUpdate = bt; 
        
        l0.addWidget( self.textEdit )
        l0.addWidget( bt )
        #self.setCentralWidget(l0)


class ApplicationWindow(QtWidgets.QMainWindow):

    df   = None
    FFel = None
    FF   = None
    FEin = None
    str_Atoms = None
    mode = Modes.LJQ.name
    #Q          = -0.25;
    Q           = 0.0;
    relax_dim   = (120,120,60)

    def __init__(self):
    
        print "oclu.ctx    ", oclu.ctx
        print "oclu.queue  ", oclu.queue
    
        FFcl.init()
        oclr.init()
    
        # --- init QtMain
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")
        self.main_widget = QtWidgets.QWidget(self)
        l0 = QtWidgets.QVBoxLayout(self.main_widget)
        self.mplc1 = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l0.addWidget(self.mplc1)
        
        sl = QtWidgets.QComboBox(); self.slMode = sl; l0.addWidget(sl)
        sl.addItem( Modes.LJQ.name       )
        sl.addItem( Modes.MorseFFel.name )
        sl.currentIndexChanged.connect(self.selectMode)
        
        # --- bxZ
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("{ iz, nAmp }") )
        bx = QtWidgets.QSpinBox();bx.setRange(0,300); bx.setSingleStep(1); bx.setValue(90); bx.valueChanged.connect(self.plotSlice); vb.addWidget(bx); self.bxZ=bx
        bx = QtWidgets.QSpinBox();bx.setRange(0,50 ); bx.setSingleStep(1); bx.setValue(10); bx.valueChanged.connect(self.F2df); vb.addWidget(bx); self.bxA=bx
 
        # === tip params
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("K {x,y,R} [N/m]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKy=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,  2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKz=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(30.0); bx.setSingleStep(5.0); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKr=bx
        
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Q [e]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(self.Q); bx.setSingleStep(0.05); bx.valueChanged.connect(self.upload_and_relax); vb.addWidget(bx); self.bxQ=bx
        
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("eq.pos {x,y,R} [A]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0x=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0y=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0z=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange( 0.0, 10.0); bx.setValue(4.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0r=bx
   
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("relax {dt,damp}") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,10.0); bx.setValue(0.1);  bx.setSingleStep(0.005); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bx_dt=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,1.0); bx.setValue(0.9);   bx.setSingleStep(0.05);  bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bx_damp=bx     
        
        # === buttons
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        # --- btLoad
        self.btLoad = QtWidgets.QPushButton('Load', self)
        self.btLoad.setToolTip('Load inputs')
        self.btLoad.clicked.connect(self.loadInputs)
        vb.addWidget( self.btLoad )
        
        # --- EditAtoms
        bt = QtWidgets.QPushButton('Edit', self)
        bt.setToolTip('Edit atomic structure')
        bt.clicked.connect(self.EditAtoms)
        self.btEdit = bt; vb.addWidget( bt )
        
        # --- btFF
        self.btFF = QtWidgets.QPushButton('getFF', self)
        self.btFF.setToolTip('Get ForceField')
        self.btFF.clicked.connect(self.getFF)
        vb.addWidget( self.btFF )
        
        # --- btRelax
        self.btRelax = QtWidgets.QPushButton('relax', self)
        self.btRelax.setToolTip('relaxed scan')
        self.btRelax.clicked.connect(self.relax)
        vb.addWidget( self.btRelax )
        
        # --- btSave
        self.btSave = QtWidgets.QPushButton('save fig', self)
        self.btSave.setToolTip('save current figure')
        self.btSave.clicked.connect(self.saveFig)
        vb.addWidget( self.btSave )

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        self.editor = AtomEditor(self) # TODO : we use now QDialog instead
        '''
        self.loadInputs()
        self.getFF()
        
        self.Q    = -0.25;
        self.FEin = self.FF[:,:,:,:4] + self.Q*self.FF[:,:,:,4:] 
        
        self.invCell     = oclr.getInvCell(self.lvec)
        self.relax_dim   = (120,120,60)
        self.relax_poss  = oclr.preparePoss( self.relax_dim, z0=16.0, start=(0.0,0.0), end=(12.0,12.0) )
        self.relax_args  = oclr.prepareBuffers( self.FEin, self.relax_dim )
        '''
    
    def initRelax(self, relax_dim=None, z0=16.0, start=(0.0,0.0), end=(12.0,12.0) ):
        if relax_dim is not None:
            self.relax_dim = relax_dim
        self.FEin = np.zeros( self.ff_nDim+(4,), dtype=np.float32)  # TODO: what is the point of making empty buffer ?
        if self.FF is not None:
            self.composeTotalFF()
        self.relax_poss  = oclr.preparePoss   ( self.relax_dim, z0=z0, start=start, end=end )
        self.relax_args  = oclr.prepareBuffers( self.FEin, self.relax_dim )

    def EditAtoms(self):        
        self.editor.show() # TODO : we use now QDialog instead
    
    def updateFF(self):
        #print self.str_Atoms;
        self.str_Atoms = self.editor.textEdit.toPlainText() 
        #print self.str_Atoms;
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( self.str_Atoms.split('\n') )
        Zs, xyzs, qs      = PPU.PBCAtoms( Zs, xyzs, qs, avec=self.lvec[1], bvec=self.lvec[2] )
        self.atoms        = FFcl.xyzq2float4(xyzs,qs);
        
        #self.ff_args      = FFcl.updateArgsLJC( self.ff_args, atoms=self.atoms )
        self.ff_args      = self.func_updateFFArgs( self.ff_args, atoms=self.atoms )
        self.getFF()
        #self.FF           = FFcl.runLJC( self.ff_args, self.ff_nDim )
        self.upload_and_relax()

    def loadInputMorseFFel(self):

        try:
            self.TypeParams   = PPU.loadSpecies( 'atomtypes.ini' )
        except:
            self.TypeParams   = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )

        print self.TypeParams; # exit()

        atoms, nDim, lvec = basUtils.loadXSFGeom( "FFel_z.xsf" )
        Zs   = atoms[0] 
        xyzs = np.transpose( np.array( [atoms[1],atoms[2],atoms[3]] ) ).copy()

        print "Zs", Zs
        print "xyzs", xyzs
        print "lvec", lvec

        self.lvec    = np.array( lvec )
        self.invCell = oclr.getInvCell(self.lvec)
        Zs, xyzs, qs = PPU.PBCAtoms( Zs, xyzs, atoms[4], avec=self.lvec[1], bvec=self.lvec[2] )

        self.atoms   = FFcl.xyzq2float4(xyzs,qs)
        REAs         = PPU.getAtomsREA( 8, Zs, self.TypeParams )
        self.REAs    = REAs.astype(np.float32)
        #self.clREAs  = FFcl.REA2float4(REAs);

        # load grid
        #FFel, lvec, nDim, head = GU.loadVecFieldXsf( "FFel" )
        #self.FFel = FFcl.XYZ2float4(FFel[:,:,:,0],FFel[:,:,:,1],FFel[:,:,:,2])

        poss         = FFcl.getposs( self.lvec )
        self.ff_nDim = poss.shape[:3]
        print "ff_dim", self.ff_nDim
        self.ff_args = FFcl.initArgsMorse( self.atoms, self.REAs, poss )

    def loadInputs(self):
        
        self.str_Atoms=open('input.xyz').read()
        
        self.TypeParams   = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
        xyzs,Zs,enames,qs = basUtils.loadAtomsNP( 'input.xyz' )
        self.lvec         = np.genfromtxt('cel.lvs')
        print "self.lvec: ", self.lvec
        
        Zs, xyzs, qs = PPU.PBCAtoms( Zs, xyzs, qs, avec=self.lvec[1], bvec=self.lvec[2] )
        cLJs_        = PPU.getAtomsLJ     ( 8, Zs, self.TypeParams );
        self.atoms   = FFcl.xyzq2float4(xyzs,qs);
        self.cLJs    = cLJs_.astype(np.float32)
        
        poss         = FFcl.getposs( self.lvec )
        self.ff_nDim = poss.shape[:3]
        print "ff_dim", self.ff_nDim
        self.ff_args = FFcl.initArgsLJC( self.atoms, self.cLJs, poss )

    def getFF(self):
        t1 = time.clock() 
        #self.FF    = FFcl.runLJC( self.ff_args, self.ff_nDim )
        self.FF    = self.func_runFF( self.ff_args, self.ff_nDim )
        print "getFF : self.FF.shape", self.FF.shape;
        self.plot_FF = True
        t2 = time.clock(); print "FFcl.runLJC time %f [s]" %(t2-t1)
        self.plotSlice()
        
    def relax(self):
        t1 = time.clock() 
        stiffness    = np.array([self.bxKx.value(),self.bxKy.value(),0.0,self.bxKr.value()], dtype=np.float32 ); stiffness/=-16.0217662; #print "stiffness", stiffness
        dpos0        = np.array([self.bxP0x.value(),self.bxP0y.value(),0.0,self.bxP0r.value()], dtype=np.float32 ); dpos0[2] = -np.sqrt( dpos0[3]**2 - dpos0[0]**2 + dpos0[1]**2 ); #print "dpos0", dpos0
        relax_params = np.array([self.bx_dt.value(),self.bx_damp.value(),self.bx_dt.value()*0.2,self.bx_dt.value()*5.0], dtype=np.float32 ); #print "relax_params", relax_params
        self.FEout = oclr.relax( self.relax_args, self.relax_dim, self.invCell, poss=self.relax_poss, dpos0=dpos0, stiffness=stiffness, relax_params=relax_params  )
        t2 = time.clock(); print "oclr.relax time %f [s]" %(t2-t1)
        self.F2df()
    
    def composeTotalFF(self):
        self.Q  = self.bxQ.value()
        if np.abs(self.Q)<1e-4 : 
            self.FEin[:,:,:,:] = self.FF[:,:,:,:4]
        else:
            if self.FFel is None:
                self.FEin[:,:,:,:] = self.FF[:,:,:,:4] + self.Q*self.FF[:,:,:,4:] 
            else:
                self.FEin[:,:,:,:] = self.FF[:,:,:,:4] + self.Q*self.FFel[:,:,:,4:]

    def upload_and_relax(self):
        self.composeTotalFF()
        region = self.FEin.shape[:3]; region = region[::-1]; print "upload FEin : ", region
        cl.enqueue_copy( oclr.oclu.queue, self.relax_args[0], self.FEin, origin=(0,0,0), region=region  )
        self.relax()
        
    def F2df(self):
        nAmp = self.bxA.value()
        if nAmp > 0:
            t1 = time.clock() 
            self.df = -PPU.Fz2df( np.transpose( self.FEout[:,:,:,2], (2,0,1) ), dz=0.1, k0=1800.0, f0=30300.0, n=nAmp )
            t2 = time.clock(); print "F2df time %f [s]" %(t2-t1)
        else:
            self.df = None
        self.plot_FF = False
        self.plotSlice()
        
    def plotSlice(self):
        t1 = time.clock() 
        val = int( self.bxZ.value() )
        if self.plot_FF:
            Fslice = self.FF[val,:,:,2]
        elif self.df is not None:
            Fslice = self.df[self.df.shape[0]-val-1,:,:]
        else:
            Fslice = self.FEout[:,:,self.FEout.shape[2]-val-1,2]
        self.mplc1.plotSlice( Fslice )
        t2 = time.clock(); print "plotSlice time %f [s]" %(t2-t1)

    def saveFig(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Image files (*.png)")
        if fileName:
            print "saving image to :", fileName
            self.mplc1.fig.savefig( fileName,bbox_inches='tight')

    def selectMode(self):
        self.mode = self.slMode.currentText()
        if   self.mode == Modes.LJQ.name:
            self.func_updateFFArgs = FFcl.updateArgsLJC
            self.func_runFF        = FFcl.runLJC
            self.loadInputs()
        elif self.mode == Modes.MorseFFel.name:
            #self.loadInputMorseFFel()
            self.func_updateFFArgs = FFcl.updateArgsMorse
            self.func_runFF        = FFcl.runMorse
            self.loadInputMorseFFel()
        else:
            print "No such mode : " , self.mode 
        self.getFF()
        self.initRelax()
        print self.mode

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())

