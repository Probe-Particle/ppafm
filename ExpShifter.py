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
    #path='./'
    path="/u/25/prokoph1/unix/Desktop/CATAM/Exp_Data/Camphor/Orientation_4/"

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
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("path") )
        el = QtWidgets.QLineEdit(); el.setText(self.path); vb.addWidget(el);  self.txPath=el
        bt = QtWidgets.QPushButton('Load', self); bt.setToolTip('load file from dir'); bt.clicked.connect(self.loadData); vb.addWidget( bt ); self.btLoad = bt

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("slice ") )
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.valueChanged.connect(self.updateDataView); vb.addWidget(bx); self.bxZ=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("shift ix,iy") )
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.valueChanged.connect(self.shiftData); vb.addWidget(bx); self.bxX=bx
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.valueChanged.connect(self.shiftData); vb.addWidget(bx); self.bxY=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Ninter ") )
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(1); vb.addWidget(bx); self.bxNi=bx
        bt = QtWidgets.QPushButton('interpolate', self); bt.setToolTip('interpolate'); bt.clicked.connect(self.interpolate); vb.addWidget( bt ); self.btLoad = bt

        #sl = QtWidgets.QComboBox(); self.slMode = sl; vb.addWidget(sl)
        #sl.currentIndexChanged.connect(self.selectMode)

        #sl = QtWidgets.QComboBox(); self.slDataView = sl; vb.addWidget(sl)
        #sl.addItem( DataViews.FFpl.name ); sl.addItem( DataViews.FFel.name ); sl.addItem( DataViews.FFin.name ); sl.addItem( DataViews.FFout.name ); sl.addItem(DataViews.df.name );
        #sl.setCurrentIndex( sl.findText( DataViews.FFpl.name ) )
        #sl.currentIndexChanged.connect(self.updateDataView)

        # === buttons
        ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 

        # --- btSave
        self.btSave = QtWidgets.QPushButton('Save', self)
        self.btSave.setToolTip('save data stack to .npy')
        self.btSave.clicked.connect(self.saveData)
        vb.addWidget( self.btSave )

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        #self.geomEditor    = guiw.EditorWindow(self,title="Geometry Editor")
        #self.speciesEditor = guiw.EditorWindow(self,title="Species Editor")
        #self.figCurv       = guiw.PlotWindow( parent=self, width=5, height=4, dpi=100)

    def loadData(self):
        #print file_list
        #fnames

        self.path = self.txPath.text()
        print self.path

        '''
        https://www.tutorialspoint.com/pyqt/pyqt_qfiledialog_widget.htm
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFilter("Text files (*.txt)")
        filenames = QStringList()
        if dlg.exec_():
            filenames = dlg.selectedFiles()

        self.path = 
        '''

        self.fnames   = glob.glob(self.path+'*.dat')
        self.fnames.sort()
        #self.data = self.loadData();
        print self.fnames
        data = []
        fnames  = []
        for fname in self.fnames:
            #print fname
            fname_ = os.path.basename(fname); fnames.append( fname_ )
            #print os.path.basename(fname)
            imgs = file_dat.readDat(fname)
            data.append( imgs[1] )
        self.fnames = fnames
        #return data
        self.data = data
        self.shifts = [[0,0]]*len(self.data)
        self.bxZ.setRange( 0, len(self.data)-1 );
        self.bxX.setRange( -1000, +1000);
        self.bxY.setRange( -1000, +1000);

        self.updateDataView()

    def interpolate(self):
        iz    = int( self.bxZ.value() )
        ni    = int( self.bxNi.value() )
        dat1  = self.data[iz  ]
        dat2  = self.data[iz+1]
        for i in range(ni):
            c = (i+1)/float(ni+1)
            print c
            dat = c*dat1 + (1.0-c)*dat2
            #dat[:100,:] = dat1[:100,:]
            #dat[100:,:] = dat2[100:,:]
            self.data  .insert( iz+1, dat )
            self.shifts.insert( iz+1, [0,0] )
            self.fnames.insert( iz+1, "c%1.3f" %c )
        self.bxZ.setRange( 0, len(self.data)-1 )

    def saveData(self):
        arr = np.array(self.data)
        print arr.shape 
        np.save( self.path+"data.npy", arr)

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
            self.figCan.plotSlice( self.data[iz], self.fnames[iz] )
        except:
            print "cannot plot slice #", iz
        t2 = time.clock(); print "plotSlice time %f [s]" %(t2-t1)

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())

