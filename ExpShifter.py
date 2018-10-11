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
import numpy as np
from enum import Enum
import glob

import pyProbeParticle.GuiWigets   as guiw
import pyProbeParticle.file_dat    as file_dat

class ApplicationWindow(QtWidgets.QMainWindow):
    PATH='./'

    def __init__(self):

        # --- init QtMain
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")
        self.main_widget = QtWidgets.QWidget(self)
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        self.figCan = guiw.FigImshow( parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100)
        l00.addWidget(self.figCan)
        l0 = QtWidgets.QVBoxLayout(self.main_widget); l00.addLayout(l0);

        # -------------- Potential
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb);

        #sl = QtWidgets.QComboBox(); self.slMode = sl; vb.addWidget(sl)
        #sl.currentIndexChanged.connect(self.selectMode)

        #sl = QtWidgets.QComboBox(); self.slDataView = sl; vb.addWidget(sl)
        #sl.addItem( DataViews.FFpl.name ); sl.addItem( DataViews.FFel.name ); sl.addItem( DataViews.FFin.name ); sl.addItem( DataViews.FFout.name ); sl.addItem(DataViews.df.name );
        #sl.setCurrentIndex( sl.findText( DataViews.FFpl.name ) )
        #sl.currentIndexChanged.connect(self.updateDataView)
        
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("{iZPP, fMorse[1]}") )
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.valueChanged.connect(self.updateDataView); vb.addWidget(bx); self.bxZ=bx
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.valueChanged.connect(self.shiftData); vb.addWidget(bx); self.bxX=bx
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.valueChanged.connect(self.shiftData); vb.addWidget(bx); self.bxY=bx

        # === buttons
        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 

        # --- btFF
        self.btFF = QtWidgets.QPushButton('getFF', self)
        self.btFF.setToolTip('Get ForceField')
        #self.btFF.clicked.connect(self.getFF)
        vb.addWidget( self.btFF )
        
        # --- btRelax
        self.btRelax = QtWidgets.QPushButton('relax', self)
        self.btRelax.setToolTip('relaxed scan')
        #self.btRelax.clicked.connect(self.relax)
        vb.addWidget( self.btRelax )
        
        # --- btSave
        self.btSave = QtWidgets.QPushButton('save fig', self)
        self.btSave.setToolTip('save current figure')
        #self.btSave.clicked.connect(self.saveFig)
        vb.addWidget( self.btSave )

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.fnames   = glob.glob(self.PATH+'*.dat')
        self.fnames.sort()
        self.data = self.loadData();
        self.shifts = [[0,0]]*len(self.data)
        self.bxZ.setRange( 0, len(self.data));
        self.bxX.setRange( -1000, +1000);
        self.bxY.setRange( -1000, +1000);

        
        #self.geomEditor    = guiw.EditorWindow(self,title="Geometry Editor")
        #self.speciesEditor = guiw.EditorWindow(self,title="Species Editor")
        #self.figCurv       = guiw.PlotWindow( parent=self, width=5, height=4, dpi=100)

    def loadData(self ):
        #print file_list
        #fnames
        data = []
        for fname in self.fnames:
            print fname
            imgs = file_dat.readDat(fname)
            data.append( imgs[1] )
            fnames = fname
        return data

    def shiftData(self):
        iz = int(self.bxZ.value())
        ix = int(self.bxX.value()); dix = ix - self.shifts[iz][0]; self.shifts[iz][0] = ix
        iy = int(self.bxY.value()); diy = iy - self.shifts[iz][1]; self.shifts[iz][1] = iy
        print ix,iy
        self.data[iz] = np.roll( self.data[iz], dix, axis=0 )
        self.data[iz] = np.roll( self.data[iz], diy, axis=1 )
        self.updateDataView()

    def selectDataView(self):
        iz    = int( self.bxZ.value() )
        self.bxX.setValue( self.shifts[iz][0] )
        self.bxY.setValue( self.shifts[iz][1] )
        #print "iz : ", iz
        return iz

    def updateDataView(self):
        t1 = time.clock() 
        iz = self.selectDataView()
        try:
            self.figCan.plotSlice( self.data[iz] )
        except:
            print "cannot plot slice #", iz
        t2 = time.clock(); print "plotSlice time %f [s]" %(t2-t1)

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())

