#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases


import sys
import os
import time
import random
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets, QtGui
#from numpy import arange, sin, pi
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
#import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT

import pyProbeParticle.GuiWigets   as guiw

import photonMap as phmap

def makeBox( pos, rot, a=10.0,b=20.0 ):
    ca=np.cos(rot)
    sa=np.sin(rot)
    xs=[pos,pos-ca*a,pos-ca*a+sa*b,pos+sa*b,pos]
    ys=[pos,pos+sa*a,pos+sa*a+ca*b,pos+ca*b,pos]
    return xs,ys

class ApplicationWindow(QtWidgets.QMainWindow):

    def __init__(self, opts ):

        self.opts=opts

        self.HOMOname     = opts.homo
        self.LUMOname     = opts.lumo
        self.rhoTransName = opts.transdens

        self.rhoTrans = None
        self.tryLoadInputGrids()

        # --- init QtMain
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")
        self.main_widget = QtWidgets.QWidget(self)
        #l0 = QtWidgets.QVBoxLayout(self.main_widget)
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        self.fig1 = guiw.FigImshow( parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100); l00.addWidget(self.fig1)
        self.fig2 = guiw.FigImshow( parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100); l00.addWidget(self.fig2)
        self.fig3 = guiw.FigImshow( parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100); l00.addWidget(self.fig3)
        l0 = QtWidgets.QVBoxLayout(self.main_widget); l00.addLayout(l0);

        # -------------- Potential
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb);

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel(" z_tip[A], sigma[A] ") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,10.0); bx.setSingleStep(0.1); bx.setValue(5.0); bx.valueChanged.connect(self.eval); vb.addWidget(bx); self.bx_z=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,10.0); bx.setSingleStep(0.1); bx.setValue(3.0); bx.valueChanged.connect(self.eval); vb.addWidget(bx); self.bx_w=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("canvas {nx,ny} pix") )
        bx = QtWidgets.QSpinBox();bx.setRange(0,2000); bx.setSingleStep(50); bx.setValue(500); bx.valueChanged.connect(self.eval); vb.addWidget(bx); self.bx_nx=bx
        bx = QtWidgets.QSpinBox();bx.setRange(0,2000); bx.setSingleStep(50); bx.setValue(500); bx.valueChanged.connect(self.eval); vb.addWidget(bx); self.bx_ny=bx
        #pho, Vtip, rho =  phmap.photonMap2D_stamp( self.rhoTrans, tipDict, self.lvec, z=0.5, sigma=1.0, multipole_dict={'s':1.0}, rots=rots, poss=poss, coefs=coefs, ncanv=(500,500) )


        # === buttons
        #ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        
        # --- EditAtoms
        #bt = QtWidgets.QPushButton('Edit Geom', self); bt.setToolTip('Edit atomic structure'); bt.clicked.connect(self.editAtoms); self.btEditAtoms = bt; vb.addWidget( bt )
        #bt = QtWidgets.QPushButton('Edit Params', self); bt.setToolTip('Edit atomic structure'); bt.clicked.connect(self.editSpecies); self.btEditParams = bt; vb.addWidget( bt )
        #bt = QtWidgets.QPushButton('getFF', self); bt.setToolTip('Get ForceField'); bt.clicked.connect(self.getFF); self.btFF=bt; vb.addWidget( self.btFF )
        
        # --- btRun
        bt = QtWidgets.QPushButton('Run', self)
        bt.setToolTip('evaluate convolution')
        bt.clicked.connect(self.eval)
        vb.addWidget( bt )
        self.btRun = bt

        #self.main_widget.setFocus()
        #self.setCentralWidget(self.main_widget)

        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        #self.btLoad = QtWidgets.QPushButton('Load', self); self.btLoad.setToolTip('Load inputs'); self.btLoad.clicked.connect(self.loadInputs_New); vb.addWidget( self.btLoad )
        #self.btSave = QtWidgets.QPushButton('save fig', self); self.btSave.setToolTip('save current figure'); self.btSave.clicked.connect(self.saveFig); vb.addWidget( self.btSave )
        #self.btSaveW = QtWidgets.QPushButton('save data', self); self.btSaveW.setToolTip('save current figure data'); self.btSaveW.clicked.connect(self.saveDataW); vb.addWidget( self.btSaveW )


        vb = QtWidgets.QVBoxLayout(); l0.addLayout(vb) 
        vb.addWidget( QtWidgets.QLabel("x y [A]  rot[deg]  coef {Re,Im}") )
        tx = QtWidgets.QTextEdit()
        vb.addWidget( tx )
        tx.setText('''-5.0 -10.0      0.0     1.0 0.0\n10.0 5.0     90.0     -1.0 0.0''')
        self.txPoses = tx
        #self.centralLayout.addWidget( bt )

        # ==========  Child Windows
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        #self.geomEditor    = guiw.EditorWindow(self,title="Geometry Editor")
        #self.speciesEditor = guiw.EditorWindow(self,title="Species Editor")
        #self.figCurv       = guiw.PlotWindow( parent=self, width=5, height=4, dpi=100)

        #self.selectMode()

        if self.rhoTrans is not None : 
            self.eval()

    #def editAtoms(self):        
    #    self.geomEditor.show()    # TODO : we use now QDialog instead
    
    #def editSpecies(self):        
    #    self.speciesEditor.show() # TODO : we use now QDialog instead

    def parsePoses(self):
        text = self.txPoses.toPlainText().split('\n')
        n = len(text)
        rots  = [ ]
        poss  = [ ]
        coefs = [ ]
        for i in range(n):
            ws = text[i].split()
            print("ws ", ws)
            poss .append( [float(ws[0]),float(ws[1])] )
            rots .append(  float(ws[2])*np.pi/180.0 )
            coefs.append(  [float(ws[3]),float(ws[4])] )
        self.rots  = rots
        self.poss  = poss
        self.coefs = coefs

    def tryLoadInputGrids(self):
        print(self.rhoTransName, self.HOMOname, self.LUMOname) 
        if   len(self.rhoTransName)>0 :
            print("tryLoadInputGrids  : rhoTransName ")
            self.rhoTrans, self.lvec, self.ndim = GU.load_scal_field(self.rhoTransName)
            return True
        elif len(self.HOMOname)>0 and len(self.LUMOname)>0 :
            print("tryLoadInputGrids  : HOMO / LUMO ")
            homo, self.lvec, self.ndim = GU.load_scal_field(self.HOMOname)
            lumo, self.lvec, self.ndim = GU.load_scal_field(self.LUMOname)
            self.rhoTrans = homo*lumo
            return True
        return False


    def drawPoses(self, fig, d=10.0):
        '''
        n = len( self.poss )
        for i in range(n):
            #fig.axes.plot( self.poss[i][0], self.poss[i][0], 'o' )
            #x  = self.poss[i][0]
            #y  = self.poss[i][1]
            #dx = np.cos(self.rots[i])*d
            #dy = np.sin(self.rots[i])*d
            xs,ys = makeBox( self.poss[i], self.rots[i], a=10.0,b=20.0 )
            fig.axes.plot( xs, ys, '-' )
        '''
        phmap.plotBoxes( self.poss, self.rots, self.lvec, ax=fig.axes )
        fig.draw()

    def eval(self):
        self.parsePoses()

        if self.rhoTrans is None :
            print("tryLoadInputGrids ") 
            self.tryLoadInputGrids()
        
        # --- ToDo : rots, poss, coefs should be read from input box, and plotted on the picture
        #rots =[0.0]
        #poss =[ [200.0,200.0] ]
        #coefs=[1.0]
        #tip =  { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        tipDict =  { 's': 1.0  }
        pho, Vtip, rho, dd =  phmap.photonMap2D_stamp( self.rhoTrans, self.lvec, z=self.bx_z.value(), sigma=self.bx_w.value(), multipole_dict=tipDict, rots=self.rots, poss=self.poss, coefs=self.coefs, ncanv=(self.bx_nx.value(),self.bx_ny.value()) )
        #self.figCan.plotSlice( pho, title="photon map" )
        #self.fig1.plotSlice( pho.real**2+pho.imag**2, title="photon map" )
        #self.fig2.plotSlice( rho.real**2+rho.imag**2, title="rhoTrans"   )

        sh    =pho.shape
        extent=( -sh[0]*dd[0]*0.5,sh[0]*dd[0]*0.5,   -sh[1]*dd[1]*0.5, sh[1]*dd[1]*0.5   )

        self.fig1.plotSlice( pho.real**2+pho.imag**2,extent=extent, title="photon map" )
        self.fig2.plotSlice( rho.real,               extent=extent, title="rhoTrans"   )
        self.fig3.plotSlice( Vtip.real,              extent=extent, title="Vtip" )

        self.drawPoses( self.fig1 )
        self.drawPoses( self.fig2 )
        self.drawPoses( self.fig3 )

    def loadHOMO(self):
        pass

    def loadLUMO(self):
        pass
        
    def saveFig(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Image files (*.png)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".png" )
            print(( "saving image to :", fileName ))
            self.figCan.fig.savefig( fileName,bbox_inches='tight')
            

    '''
    def saveDataW(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","WSxM files (*.xyz)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".xyz" )
            print "saving data to to :", fileName
            iz,data = self.selectDataView()
            npdata=data[iz]
            xs = np.arange(npdata.shape[0] )
            ys = np.arange(npdata.shape[1] )
            Xs, Ys = np.meshgrid(xs,ys)
            GU.saveWSxM_2D(fileName, npdata, Xs, Ys)
    '''

    '''
    def updateDataView(self):
        t1 = time.clock() 
        iz,data = self.selectDataView()
        self.viewed_data = data 
        #self.figCan.plotSlice_iz(iz)
        try:
            print data.shape
            self.figCan.plotSlice( data[iz], title=self.slDataView.currentText()+':iz= '+str(int( self.bxZ.value() )) )
        except:
            print "cannot plot slice #", iz
        t2 = time.clock(); print "plotSlice time %f [s]" %(t2-t1)
    '''

    #def clickImshow(self,ix,iy):
    #    ys = self.viewed_data[ :, iy, ix ]
    #    self.figCurv.show()
    #    self.figCurv.figCan.plotDatalines( ( range(len(ys)), ys, "%i_%i" %(ix,iy) )  )

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from optparse import OptionParser

    parser = OptionParser()
    #parser.add_option( "-H", "--homo",   action="store", type="string", default="homo.xsf", help="orbital of electron hole;    3D data-file (.xsf,.cube)")
    #parser.add_option( "-L", "--lumo",   action="store", type="string", default="lumo.xsf", help="orbital of excited electron; 3D data-file (.xsf,.cube)")
    #parser.add_option( "-T", "--transdens",   action="store", type="string", default="transRho.xsf", help="transition density; 3D data-file (.xsf,.cube)")
        
    parser.add_option( "-H", "--homo",       action="store", type="string", default="", help="orbital of electron hole;    3D data-file (.xsf,.cube)")
    parser.add_option( "-L", "--lumo",       action="store", type="string", default="", help="orbital of excited electron; 3D data-file (.xsf,.cube)")
    parser.add_option( "-T", "--transdens",  action="store", type="string", default="", help="transition density; 3D data-file (.xsf,.cube)")

    parser.add_option( "-R", "--radius", action="store", type="float",  default="1.0", help="tip radius")
    parser.add_option( "-z", "--ztip",   action="store", type="float",  default="5.0", help="tip above substrate")
    parser.add_option( "-t", "--tip",    action="store", type="string", default="s",   help="tip compositon s,px,py,pz,d...")
    (opts, args) = parser.parse_args()


    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow( opts )
    aw.show()
    sys.exit(qApp.exec_())
