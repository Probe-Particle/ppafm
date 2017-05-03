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
import numpy as np

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

class MyDynamicMplCanvas(FigureCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, parent=None, width=5, height=4, dpi=100 ):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        #self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
         
    def plot(self, val ):
        #print ">>", val
        xs = np.linspace(-1.0,1.0, 100);
        ys = np.sin(xs*val);
        self.axes.cla()
        self.axes.plot( xs, ys, '.-' )
        self.draw()
   
    def plot2d(self, val ):
        #print ">>", val
        xs = np.linspace(-1.0,1.0, 100);
        ys = np.linspace(-1.0,1.0, 100);
        Xs,Ys = np.meshgrid(xs,ys)
        self.axes.cla()
        self.axes.imshow(np.sin(val*np.sqrt(Xs**2+Ys**2)))
        self.draw()

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.main_widget = QtWidgets.QWidget(self)
        l = QtWidgets.QVBoxLayout(self.main_widget)
        self.mplc1 = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(self.mplc1)
        
        # --- QDoubleSpinBox
        scaleLabel = QtWidgets.QLabel("Frequency <%d .. %d> Hz" %(0.0, 20.0))
        self.scaleSpinBox = QtWidgets.QDoubleSpinBox()
        self.scaleSpinBox.setRange(0.0, 20.0)
        self.scaleSpinBox.setSingleStep(1.0)
        self.scaleSpinBox.setValue(5.0)
        self.scaleSpinBox.setSuffix(' Hz')
        self.scaleSpinBox.valueChanged.connect(self.valuechange)
        l.addWidget( scaleLabel )
        l.addWidget( self.scaleSpinBox )
        
        # --- QPushButton
        self.button1 = QtWidgets.QPushButton('run!', self)
        self.button1.setToolTip('This will unleash tremendous computational effort')
        self.button1.move(100,70) 
        self.button1.clicked.connect(self.on_click)
        l.addWidget( self.button1 )

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def on_click(self):
        val = self.scaleSpinBox.value()
        print 'running (val=%f) ... ' %val 
        self.mplc1.plot2d(val)
        
    def valuechange(self, val):
        print val
        self.mplc1.plot(val)
        

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About", " Fucking about !!!")

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle("%s" % progname)
    aw.show()
    sys.exit(qApp.exec_())

