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
        l = QtWidgets.QVBoxLayout(self.main_widget)
        self.mplc1 = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(self.mplc1)
        
        # --- bxZ
        scaleLabel = QtWidgets.QLabel("Frequency <%d .. %d> Hz" %(0.0, 20.0))
        l.addWidget( scaleLabel )
        self.bxZ = QtWidgets.QSpinBox()
        self.bxZ.setRange(0, 300)
        self.bxZ.setSingleStep(1)
        self.bxZ.setValue(90)
        self.bxZ.valueChanged.connect(self.plotSlice)
        l.addWidget( self.bxZ )
        
        # --- btLoad
        self.btLoad = QtWidgets.QPushButton('Load', self)
        self.btLoad.setToolTip('Load inputs')
        self.btLoad.clicked.connect(self.loadInputs)
        l.addWidget( self.btLoad )
        
        # --- btFF
        self.btFF = QtWidgets.QPushButton('getFF', self)
        self.btFF.setToolTip('Get ForceField')
        self.btFF.clicked.connect(self.getFF)
        l.addWidget( self.btFF )
        
        # --- btRelax
        self.btRelax = QtWidgets.QPushButton('relax', self)
        self.btRelax.setToolTip('relaxed scan')
        self.btRelax.clicked.connect(self.relax)
        l.addWidget( self.btRelax )

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        self.loadInputs()
        self.getFF()
        
        self.Q    = -0.25;
        self.FEin = self.FF[:,:,:,:4] + self.Q*self.FF[:,:,:,4:] 
        
        self.invCell     = oclr.getInvCell(self.lvec)
        self.relax_dim   = (100,100,60)
        self.relax_poss  = oclr.preparePoss( self.relax_dim, start=(0.0,0.0), end=(10.0,10.0), z0=10.0 )
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
        self.FEout = oclr.relax( self.relax_args, self.relax_dim, self.invCell, poss=self.relax_poss )
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
    
    
    

