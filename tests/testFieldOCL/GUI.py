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
import pyProbeParticle.common as PPU
import pyProbeParticle.cpp_utils as cpp_utils

import RelaxOpenCL as oclr

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
        
        #E,lvec, nDim, head = GU.loadXSF('mini.xsf' );
        #self.axes.imshow( E[5,:,:] )
            
    def plotSlice(self, F ):
        self.axes.cla()
        self.cax = self.axes.imshow( F, origin='image', cmap='gray' )
        if self.cbar is None:
            self.cbar = self.fig.colorbar( self.cax )
        self.cax.set_clim([F.min(), F.max()])
        #self.fig.colorbar(im, self.axes, orientation='horizontal')
        #self.axes.colorbar( F )
        self.draw()

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
    
        # --- init RelaxOpenCL
        oclr.prepareProgram()
    
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
        self.bxZ.setRange(0, 20)
        self.bxZ.setSingleStep(1)
        self.bxZ.setValue(0)
        self.bxZ.valueChanged.connect(self.plotSlice)
        l.addWidget( self.bxZ )
        
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

    def getFF(self):
        
        #self.E,lvec, nDim, head = GU.loadXSF('ELJ_cl.xsf' );
        #val = self.bxZ.value()
        #self.mplc1.plotSlice( E[val,:,:] )
        #self.plotSlice( val)
        
        self.kargs, self.relaxShape = oclr.prepareBuffers()
        val = self.bxZ.value()
        self.plotSlice( val)
        
    def relax(self):
        oclr.relax( self.kargs, self.relaxShape )
        oclr.saveResults()
        val = self.bxZ.value()
        self.plotSlice( val)
        
    def plotSlice(self, val):
        val = int(val)
        print val
        #Fslice = oclr.FE  [val,:,:,3]
        Fslice = oclr.FEout[:,:,oclr.FEout.shape[2]-val-1,2]
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
    
    
    

