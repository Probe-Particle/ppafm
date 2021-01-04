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

        # --- Mode select
        #sl = QtWidgets.QComboBox(); self.slMode = sl; vb.addWidget(sl)
        #sl.addItem( Modes.LJQ.name       )
        #sl.addItem( Modes.MorseFFel.name )
        #sl.setCurrentIndex( sl.findText( Modes.MorseFFel.name ) )
        #sl.currentIndexChanged.connect(self.selectMode)

        # --- Data View selection
        #sl = QtWidgets.QComboBox(); self.slDataView = sl; vb.addWidget(sl)
        #sl.addItem( DataViews.FFpl.name ); sl.addItem( DataViews.FFel.name ); sl.addItem( DataViews.FFin.name ); sl.addItem( DataViews.FFout.name ); sl.addItem(DataViews.df.name );
        #sl.setCurrentIndex( sl.findText( DataViews.FFpl.name ) )
        #sl.currentIndexChanged.connect(self.updateDataView)
        
        # --- Parameters  iZPP, fMorse
        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("{iZPP, fMorse[1]}") )
        #bx = QtWidgets.QSpinBox();       bx.setRange(0, 200);    bx.setValue(8);                             bx.valueChanged.connect(self.updateFromFF); vb.addWidget(bx); self.bxZPP=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.25, 4.0); bx.setValue(1.00);  bx.setSingleStep(0.05); bx.valueChanged.connect(self.updateFromFF); vb.addWidget(bx); self.bxMorse=bx
        
        # -------------- Relaxation 
        #ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("Q [e]") )
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(self.Q); bx.setSingleStep(0.05); bx.valueChanged.connect(self.upload_and_relax); vb.addWidget(bx); self.bxQ=bx

        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("K {x,y,R} [N/m]") )
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.24);  bx.setSingleStep(0.05); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKx=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,   2.0); bx.setValue(0.24);  bx.setSingleStep(0.05); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKy=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 100.0); bx.setValue(30.0); bx.setSingleStep(5.0); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxKr=bx

        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("eq.pos {x,y,R} [A]") )
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0x=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-2.0, 2.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0y=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0, 2.0); bx.setValue(0.5);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0z=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange( 0.0, 10.0); bx.setValue(4.0); bx.setSingleStep(0.1); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bxP0r=bx
   
        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("relax {dt,damp}") )
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,10.0); bx.setValue(0.1);  bx.setSingleStep(0.005); bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bx_dt=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,1.0); bx.setValue(0.9);   bx.setSingleStep(0.05);  bx.valueChanged.connect(self.relax); vb.addWidget(bx); self.bx_damp=bx  

        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("relax_min {x,y,z}[A]") )
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxSpanMinX=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(0.0);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxSpanMinY=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100.0,100.0); bx.setValue(zmin);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxSpanMinZ=bx

        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("relax_max {x,y,z}[A]") )
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,100.0); bx.setValue(xmax);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxSpanMaxX=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,100.0); bx.setValue(ymax);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxSpanMaxY=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.0,100.0); bx.setValue(zmax);  bx.setSingleStep(0.1); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxSpanMaxZ=bx

        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("relax_step {x,y,z}[A]") )
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1);  bx.setSingleStep(0.02); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxStepX=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1);  bx.setSingleStep(0.02); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxStepY=bx
        #bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0.02,0.5); bx.setValue(0.1);  bx.setSingleStep(0.02); bx.valueChanged.connect(self.shapeRelax); vb.addWidget(bx); self.bxStepZ=bx 

        #vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("In the beginning & after chaging anything rather push relax") )

        # -------------- df Conversion & plotting
        #ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)



        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel(" z_tip[A], sigma[A] ") )
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,10.0); bx.setSingleStep(0.1); bx.setValue(0.5); bx.valueChanged.connect(self.eval); vb.addWidget(bx); self.bx_z=bx
        bx = QtWidgets.QDoubleSpinBox(); bx.setRange(0,10.0); bx.setSingleStep(0.1); bx.setValue(1.0); bx.valueChanged.connect(self.eval); vb.addWidget(bx); self.bx_w=bx

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb); vb.addWidget( QtWidgets.QLabel("canvas {nx,ny} pix") )
        bx = QtWidgets.QSpinBox();bx.setRange(0,2000); bx.setSingleStep(50); bx.setValue(500); bx.valueChanged.connect(self.eval); vb.addWidget(bx); self.bx_nx=bx
        bx = QtWidgets.QSpinBox();bx.setRange(0,2000); bx.setSingleStep(50); bx.setValue(500); bx.valueChanged.connect(self.eval); vb.addWidget(bx); self.bx_ny=bx
        #pho, Vtip, rho =  phmap.photonMap2D_stamp( self.rhoTrans, tipDict, self.lvec, z=0.5, sigma=1.0, multipole_dict={'s':1.0}, rots=rots, poss=poss, coefs=coefs, ncanv=(500,500) )


        # === buttons
        #ln = QtWidgets.QFrame(); l0.addWidget(ln); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setFrameShadow(QtWidgets.QFrame.Sunken)
        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        
        # --- EditAtoms
        #bt = QtWidgets.QPushButton('Edit Geom', self)
        #bt.setToolTip('Edit atomic structure')
        #bt.clicked.connect(self.editAtoms)
        #self.btEditAtoms = bt; vb.addWidget( bt )
        
        # --- EditFFparams
        #bt = QtWidgets.QPushButton('Edit Params', self)
        #bt.setToolTip('Edit atomic structure')
        #bt.clicked.connect(self.editSpecies)
        #self.btEditParams = bt; vb.addWidget( bt )
        
        # --- btFF
        #self.btFF = QtWidgets.QPushButton('getFF', self)
        #self.btFF.setToolTip('Get ForceField')
        #self.btFF.clicked.connect(self.getFF)
        #vb.addWidget( self.btFF )
        
        # --- btRelax
        bt = QtWidgets.QPushButton('Run', self)
        bt.setToolTip('evaluate convolution')
        bt.clicked.connect(self.eval)
        vb.addWidget( bt )
        self.btRun = bt

        #self.main_widget.setFocus()
        #self.setCentralWidget(self.main_widget)

        vb = QtWidgets.QHBoxLayout(); l0.addLayout(vb) 
        
        # --- btLoad
        #self.btLoad = QtWidgets.QPushButton('Load', self)
        #self.btLoad.setToolTip('Load inputs')
        #self.btLoad.clicked.connect(self.loadInputs_New)
        #vb.addWidget( self.btLoad )
        
        # --- btSave
        #self.btSave = QtWidgets.QPushButton('save fig', self)
        #self.btSave.setToolTip('save current figure')
        #self.btSave.clicked.connect(self.saveFig)
        #vb.addWidget( self.btSave )

        # --- btSaveW (W- wsxm)
        #self.btSaveW = QtWidgets.QPushButton('save data', self)
        #self.btSaveW.setToolTip('save current figure data')
        #self.btSaveW.clicked.connect(self.saveDataW)
        #vb.addWidget( self.btSaveW )


        vb = QtWidgets.QVBoxLayout(); l0.addLayout(vb) 
        tx = QtWidgets.QTextEdit()
        vb.addWidget( tx )
        tx.setText('''200.0 200.0      0.0     1.0\n280.0 300.0     90.0     -1.0''')
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
            poss .append( [float(ws[0]),float(ws[1])] )
            rots .append(  float(ws[2])*np.pi/180.0 )
            coefs.append(  float(ws[3]) )
        self.rots  = rots
        self.poss  = poss
        self.coefs = coefs

    def tryLoadInputGrids(self):
        print self.rhoTransName, self.HOMOname, self.LUMOname 
        if   len(self.rhoTransName)>0 :
            print "tryLoadInputGrids  : rhoTransName "
            self.rhoTrans, self.lvec, self.ndim = GU.load_scal_field(self.rhoTransName)
            return True
        elif len(self.HOMOname)>0 and len(self.LUMOname)>0 :
            print "tryLoadInputGrids  : HOMO / LUMO "
            homo, self.lvec, self.ndim = GU.load_scal_field(self.HOMOname)
            lumo, self.lvec, self.ndim = GU.load_scal_field(self.LUMOname)
            self.rhoTrans = homo*lumo
            return True
        return False

    def eval(self):
        self.parsePoses()

        if self.rhoTrans is None :
            print "tryLoadInputGrids " 
            self.tryLoadInputGrids()
        
        # --- ToDo : rots, poss, coefs should be read from input box, and plotted on the picture
        #rots =[0.0]
        #poss =[ [200.0,200.0] ]
        #coefs=[1.0]
        #tip =  { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        tipDict =  { 's': 1.0  }
        pho, Vtip, rho =  phmap.photonMap2D_stamp( self.rhoTrans, self.lvec, z=self.bx_z.value(), sigma=self.bx_w.value(), multipole_dict=tipDict, rots=self.rots, poss=self.poss, coefs=self.coefs, ncanv=(self.bx_nx.value(),self.bx_ny.value()) )
        #self.figCan.plotSlice( pho, title="photon map" )
        self.fig1.plotSlice( pho,   title="photon map" )
        self.fig2.plotSlice( rho,   title="rhoTrans" )
        self.fig3.plotSlice( Vtip,  title="Vtip" )

    def loadHOMO(self):
        pass

    def loadLUMO(self):
        pass
        
    def saveFig(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Image files (*.png)")
        if fileName:
            fileName = guiw.correct_ext( fileName, ".png" )
            print( "saving image to :", fileName )
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
        
    parser.add_option( "-H", "--homo",   action="store", type="string", default="", help="orbital of electron hole;    3D data-file (.xsf,.cube)")
    parser.add_option( "-L", "--lumo",   action="store", type="string", default="", help="orbital of excited electron; 3D data-file (.xsf,.cube)")
    parser.add_option( "-T", "--transdens",   action="store", type="string", default="", help="transition density; 3D data-file (.xsf,.cube)")

    parser.add_option( "-R", "--radius", action="store", type="float",  default="1.0", help="tip radius")
    parser.add_option( "-z", "--ztip",   action="store", type="float",  default="5.0", help="tip above substrate")
    parser.add_option( "-t", "--tip",    action="store", type="string", default="s",   help="tip compositon s,px,py,pz,d...")
    (opts, args) = parser.parse_args()


    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow( opts )
    aw.show()
    sys.exit(qApp.exec_())
