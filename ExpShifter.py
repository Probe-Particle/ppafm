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
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
from enum import Enum
import glob
import pickle
import scipy.ndimage as nimg
import pyProbeParticle.GuiWigets   as guiw
import pyProbeParticle.file_dat    as file_dat
import copy

def crosscorel_2d_fft(im0,im1):
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)
    renorm = 1/( np.std(f0)*np.std(f1) )
    #return abs(np.fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    #return np.abs( np.fft.ifft2( (    f0 * f1.conjugate() / ( np.abs(f0) * np.abs(f1) )  ) ) )
    return abs(np.fft.ifft2( f0 * f1.conjugate() ) ) * renorm

def trans_match_fft(im0, im1):
    """Return translation vector to register images."""
    print 'we are in trans_match_fft'
    shape = im0.shape
    '''
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    '''
    ir = crosscorel_2d_fft(im0,im1)
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    #if t0 > shape[0] // 2:
    #    t0 -= shape[0]
    #if t1 > shape[1] // 2:
    #    t1 -= shape[1]
    return [t0, t1]


def roll2d( a , shift=(10,10) ):
    a_ =np.roll( a, shift[0], axis=0 )
    return np.roll( a_, shift[1], axis=1 )

class ApplicationWindow(QtWidgets.QMainWindow):
    #path='./'
    path="/u/25/prokoph1/unix/Desktop/CATAM/Exp_Data/Camphor/Orientation_4/"

    divNX = 8
    divNY = 8
    bSaveDivisible = True

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
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("load") )
        bt = QtWidgets.QPushButton('Load dat', self); bt.setToolTip('load .dat files from dir'); bt.clicked.connect(self.loadData); vb.addWidget( bt ); self.btLoad = bt
        bt = QtWidgets.QPushButton('Load npy', self); bt.setToolTip('load .npy file  from dir'); bt.clicked.connect(self.loadNPY ); vb.addWidget( bt ); self.btLoad = bt

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("slice ") )
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.valueChanged.connect(self.selectDataView); vb.addWidget(bx); self.bxZ=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("shift ix,iy") )
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.setRange(-1000,1000); bx.valueChanged.connect(self.shiftData); vb.addWidget(bx); self.bxX=bx
        bx = QtWidgets.QSpinBox(); bx.setSingleStep(1); bx.setValue(0); bx.setRange(-1000,1000); bx.valueChanged.connect(self.shiftData); vb.addWidget(bx); self.bxY=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) ; bt = QtWidgets.QPushButton('Magic fit', self); bt.setToolTip('Fit to colser slice'); bt.clicked.connect(self.magicFit); vb.addWidget( bt ); self.btLoad = bt

        l0.addLayout(vb) ; bt = QtWidgets.QPushButton('MagicAll', self); bt.setToolTip('Fit all slices'); bt.clicked.connect(self.magicFitAll);      vb.addWidget( bt ); self.btLoad = bt
        l0.addLayout(vb) ; bt = QtWidgets.QPushButton('SaveImgs', self); bt.setToolTip('save images'); bt.clicked.connect(self.saveImg);      vb.addWidget( bt ); self.btLoad = bt

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

    def magicFit(self):
        print 'magic fit'
        iz = int(self.bxZ.value())
        print 'iz=',iz  
        if (iz<len(self.data)-1 ):
            print 'we are in if'            
            '''    
            image=np.float32(self.data2[iz])
            image-=image.mean()
            vmax=image.max()
            if vmax>0:
                image /= vmax
            image_target=np.float32(self.data2[iz+1])
            image_target-=image_target.mean()
            vmax=image_target.max()
            if vmax>0:
                image_target /= vmax
            '''
            [ix,iy] = trans_match_fft(self.data2[iz],self.data[iz+1]) 
            print 'ix,iy=',-ix,-iy
            if abs(int(ix))>self.data[iz].shape[0]:
                ix=ix/abs(int(ix))*(abs(int(ix))-self.data[iz].shape[0])
            if abs(int(iy))>self.data[iz].shape[1]:
                iy=iy/abs(int(iy))*(abs(int(iy))-self.data[iz].shape[1])
            if abs(int(ix))>self.data[iz].shape[0]//2:
                ix=ix/abs(int(ix))*(abs(int(ix))-self.data[iz].shape[0])
            if abs(int(iy))>self.data[iz].shape[1]//2:
                iy=iy/abs(int(iy))*(abs(int(iy))-self.data[iz].shape[1])

            self.data[iz]=nimg.shift (self.data[iz], (-ix-self.shifts[iz][0],-iy-self.shifts[iz][1]), order=3,mode='mirror' )   
            self.shifts[iz][0] =  -ix
            self.shifts[iz][1] =  -iy
            print self.shifts
            self.updateDataView()

    def magicFitAll(self):
        izs = range( len(self.data)-1 )
        for iz in izs[::-1]:
            self.bxZ.setValue(iz);
            self.magicFit()

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
        data2 = []
        fnames  = []
        for fname in self.fnames:
            #print fname
            fname_ = os.path.basename(fname); fnames.append( fname_ )
            #print os.path.basename(fname)
            Header = {}
            imgs = file_dat.readDat(fname, Header=Header )
            print fname, "Size ", Header['ScanRange_Y'],Header['ScanRange_X']," N ", imgs[0].shape," dxy ",  Header['ScanRange_Y']/imgs[0].shape[0], Header['ScanRange_X']/imgs[0].shape[1]
            data.append( imgs[1] )
            #data2.append( imgs[1] )
        self.fnames = fnames
        #return data
        self.data = data
        #z=np.arange(25)
        data2=copy.copy(data)        
        self.data2= data2 #np.reshape(z, (5,5)) #data
        print 'data dat loaded'
        self.shifts = [ [0,0] for i in range(len(self.data)) ]
        self.bxZ.setRange( 0, len(self.data)-1 );
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
            self.data2  .insert( iz+1, dat )
            self.shifts.insert( iz+1, [0,0] )
            self.fnames.insert( iz+1, "c%1.3f" %c )
        self.bxZ.setRange( 0, len(self.data)-1 )

    def saveData(self):
        arr = np.array(self.data)
        print "saveData: arr.shape ", arr.shape 
        if ( self.bSaveDivisible ):
            print "dat.shape ", arr.shape
            nx=arr.shape[1]/self.divNX * self.divNX
            ny=arr.shape[2]/self.divNY * self.divNY
            arr_ = arr[:,:nx,:ny]
            arr_ = arr_.transpose((1,2,0))
            print "saveData: arr_.shape ", arr_.shape
            np.save( self.path+"data.npy", arr_)
        else:
            np.save( self.path+"data.npy", arr  )

        with open(self.path+'data.pickle', 'wb') as fp:
            pickle.dump( self.fnames, fp)
            pickle.dump( self.shifts, fp)

    def loadNPY(self):
        with open ( self.path+'data.pickle', 'rb') as fp:
            self.fnames = pickle.load(fp)
            self.shifts = pickle.load(fp)
        for item in zip(self.fnames,self.shifts):
            print item[0], " : ", item[1]
        data = np.load(self.path+'data.npy')
        self.data = [ s for s in data ]
        self.bxZ.setRange( 0, len(self.data)-1 );
        self.updateDataView()

    def saveImg(self):
        n = len(self.data)
        plt.figure( figsize=(n*5,5) )
        for i in range(n):
            #print i
            plt.subplot(1,n,i+1)
            plt.imshow(self.data[i], origin='image')
        plt.savefig(self.path+"data.png", bbox_inches='tight')

    def shiftData(self):
        print "shiftData"
        iz = int(self.bxZ.value())
        ix = int(self.bxX.value()); dix = ix - self.shifts[iz][0]; self.shifts[iz][0] = ix
        iy = int(self.bxY.value()); diy = iy - self.shifts[iz][1]; self.shifts[iz][1] = iy
        print 'self.original[iz]=',self.data2[iz][:3,:3]
        print 'dix,diy=', dix,diy

        print self.shifts
        image=self.data2[iz]
        self.data[iz]=nimg.shift (image, (iy,ix), order=3,mode='mirror' )   
        #self.data[iz] = np.roll( self.data[iz], dix, axis=0 )
        

        #self.data[iz] = np.roll( self.data[iz], diy, axis=1 )

        self.updateDataView()

    def selectDataView(self):
        iz    = int( self.bxZ.value() )
        print " selectDataView iz,ix,iy ", iz, self.shifts[iz][0], self.shifts[iz][1]
        self.bxX.blockSignals(True); self.bxX.setValue( self.shifts[iz][0] ); self.bxX.blockSignals(False);
        self.bxY.blockSignals(True); self.bxY.setValue( self.shifts[iz][1] ); self.bxY.blockSignals(False);
        print "selectDataView bxXY      ", self.bxX.value(), self.bxY.value()
        self.updateDataView()

    def updateDataView(self):
        iz    = int( self.bxZ.value() )
        #t1 = time.clock() 
        #iz = self.selectDataView()
        try:
            self.figCan.plotSlice( self.data[iz], self.fnames[iz] )
        except:
            print "cannot plot slice #", iz
        #t2 = time.clock(); print "plotSlice time %f [s]" %(t2-t1)

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())

