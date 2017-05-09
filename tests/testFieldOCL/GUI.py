#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases

from __future__ import unicode_literals
import sys
import os
import random
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/home/prokop/git/ProbeParticleModel_OCL") 
#import pyProbeParticle.GridUtils as GU

from   pyProbeParticle import basUtils
from   pyProbeParticle import PPPlot 
import pyProbeParticle.GridUtils as GU
import pyProbeParticle.common    as PPU
import pyProbeParticle.cpp_utils as cpp_utils

import pyProbeParticle.oclUtils    as oclu 
import pyProbeParticle.fieldOCL    as FFcl 
import pyProbeParticle.RelaxOpenCL as oclr

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
        self.img = self.axes.imshow( F, origin='image', cmap='gray' )
        if self.cbar is None:
            self.cbar = self.fig.colorbar( self.img )
        self.cbar.set_clim( vmin=F.min(), vmax=F.max() )
        self.cbar.update_normal(self.img)
        self.draw()

class ApplicationWindow(QtWidgets.QMainWindow):
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
        
        # --- bxZ
        self.bxZ = QtWidgets.QSpinBox()
        self.bxZ.setRange(0, 300)
        self.bxZ.setSingleStep(1)
        self.bxZ.setValue(90)
        self.bxZ.valueChanged.connect(self.plotSlice)
        
        self.bxY = QtWidgets.QSpinBox()
        vb = QtWidgets.QHBoxLayout()
        vb.addWidget( QtWidgets.QLabel("iz,iy") )
        vb.addWidget( self.bxZ )
        vb.addWidget( self.bxY )
        l0.addLayout(vb)
        #l = QtWidgets.QFormLayout(); l0.addLayout(l); l.addRow( QtWidgets.QLabel("iz"), vb )
        

        '''
        DEFAULT_dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
        DEFAULT_stiffness    = np.array( [-0.03,-0.03, -0.03,-1.0 ], dtype=np.float32 );
        DEFAULT_dpos0        = np.array( [ 0.0 , 0.0 ,  4.0 , 4.0 ], dtype=np.float32 );
        DEFAULT_relax_params = np.array( [ 0.01 , 0.9 , 0.01, 0.3 ], dtype=np.float32 );
        '''
        
        # === tip params
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("K {x,y,R} [N/m]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKx=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKy=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,  2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKz=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(30.0); bx.setSingleStep(5.0); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKr=bx
        
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("eq.pos {x,y,R} [A]") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0x=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0y=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0z=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange( 0.0, 10.0); bx.setValue(4.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0r=bx
   
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("relax {dt,damp}") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,10.0); bx.setValue(0.01);  bx.setSingleStep(0.005); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bx_dt=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,1.0); bx.setValue(0.9);   bx.setSingleStep(0.05);  bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bx_damp=bx     
        
        # === buttons
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        # --- btLoad
        self.btLoad = QtWidgets.QPushButton('Load', self)
        self.btLoad.setToolTip('Load inputs')
        self.btLoad.clicked.connect(self.loadInputs)
        vb.addWidget( self.btLoad )
        
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

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        self.loadInputs()
        self.getFF()
        
        self.Q    = -0.25;
        self.FEin = self.FF[:,:,:,:4] + self.Q*self.FF[:,:,:,4:] 
        
        self.invCell     = oclr.getInvCell(self.lvec)
        self.relax_dim   = (100,100,60)
        self.relax_poss  = oclr.preparePoss( self.relax_dim, z0=16.0, start=(0.0,0.0), end=(10.0,10.0) )
        self.relax_args  = oclr.prepareBuffers( self.FEin, self.relax_dim )
        
        
    def loadInputs(self):
        self.TypeParams   = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
        xyzs,Zs,enames,qs = basUtils.loadAtomsNP( 'input_wrap.xyz' )
        self.lvec         = np.genfromtxt('cel.lvs')
        
        Zs, xyzs, qs = PPU.PBCAtoms( Zs, xyzs, qs, avec=self.lvec[1], bvec=self.lvec[2] )
        cLJs_        = PPU.getAtomsLJ     ( 8, Zs, self.TypeParams );
        self.atoms   = FFcl.xyzq2float4(xyzs,qs);
        self.cLJs    = cLJs_.astype(np.float32)
        
        poss         = FFcl.getposs( self.lvec )
        self.ff_nDim = poss.shape[:3]
        print "ff_dim", self.ff_nDim
        self.ff_args = FFcl.initArgsLJC( self.atoms, self.cLJs, poss )

    def getFF(self):
        self.FF    = FFcl.runLJC( self.ff_args, self.ff_nDim )
        self.plot_FF = True
        self.plotSlice()
        
    def relax(self):
        stiffness    = np.array([self.bxKx.value(),self.bxKy.value(),0.0,self.bxKr.value()], dtype=np.float32 ); stiffness/=-16.0217662; print "stiffness", stiffness
        dpos0        = np.array([self.bxP0x.value(),self.bxP0y.value(),0.0,self.bxP0r.value()], dtype=np.float32 ); dpos0[2] = -np.sqrt( dpos0[3]**2 - dpos0[0]**2 + dpos0[1]**2 );  print "dpos0", dpos0
        relax_params = np.array([self.bx_dt.value(),self.bx_damp.value(),self.bx_dt.value()*0.2,self.bx_dt.value()*5.0], dtype=np.float32 ); print "relax_params", relax_params
        self.FEout = oclr.relax( self.relax_args, self.relax_dim, self.invCell, poss=self.relax_poss, dpos0=dpos0, stiffness=stiffness, relax_params=relax_params  )
        #self.FEout = oclr.relax( self.relax_args, self.relax_dim, self.invCell, poss=self.relax_poss )
        #oclr.saveResults()
        self.plot_FF = False
        self.plotSlice()
        
    def plotSlice(self):
        val = int( self.bxZ.value() )
        if self.plot_FF:
            Fslice = self.FF[val,:,:,2]
        else:
            Fslice = self.FEout[:,:,self.FEout.shape[2]-val-1,2]
        self.mplc1.plotSlice( Fslice )
        
    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About", " Fucking about !!!")

if __name__ == "__main__":

    #E,lvec, nDim, head = GU.loadXSF('ELJ_cl.xsf' );
    #import matplotlib.pyplot as plt
    #plt.imshow( E[50,:,:] )
    #plt.show()
    
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
    
    
    

