import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtWidgets, QtGui
import os

def correct_ext(fname, ext ):
    _, fext = os.path.splitext( fname )
    if( fext.capitalize() != ext.capitalize() ):
        fname += ext
    return fname


# =======================
#     FigCanvas
# =======================

class FigCanvas(FigureCanvasQTAgg):
    """A canvas that updates itself every second with a new plot."""
    
    def __init__(self, parentWiget=None, parentApp=None,  width=5, height=4, dpi=100 ):
        self.fig  = Figure( figsize=(width, height), dpi=dpi )
        self.axes = self.fig.add_subplot(111)
                #super(self.__class__, self).__init__( self.fig )
        FigureCanvasQTAgg.__init__(self, self.fig )
        self.parent = parentApp
        self.setParent(parentWiget)
        FigureCanvasQTAgg.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

# =======================
#      FigLinePlot
# =======================

class FigPlot(FigCanvas):
    """A canvas that updates itself every second with a new plot."""
    
    def __init__(self, parentWiget=None, parentApp=None,  width=5, height=4, dpi=100 ):
        super(self.__class__, self).__init__(parentWiget=parentWiget, parentApp=parentApp,  width=width, height=height, dpi=dpi )
        self.defaultPlotAxis()
        #cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def defaultPlotAxis(self):
        self.axes.grid()
        #self.axes.axhline(0.0, ls="--", c="k")

    def plotDatalines( self, dline ):
        self.axes.plot( dline[0], dline[1], label=dline[2] )
        self.draw()  
        #self.updatePlotAxis()

# =======================
#      FigLinePlot
# =======================

class FigImshow(FigCanvas):
    """A canvas that updates itself every second with a new plot."""
    #data = None
    cbar = None 
    
    def __init__(self, parentWiget=None, parentApp=None,  width=5, height=4, dpi=100 ):
        super(self.__class__, self).__init__(parentWiget=parentWiget, parentApp=parentApp,  width=width, height=height, dpi=dpi )
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            
    def plotSlice(self, F , title=None):
        self.axes.cla()
        #self.img = self.axes.imshow( F, origin='image', cmap='gray', interpolation='nearest' )
        self.img = self.axes.imshow( F, origin='image', cmap='gray', interpolation='bicubic' )

        if self.cbar is None:
            self.cbar = self.fig.colorbar( self.img )
        self.cbar.set_clim( vmin=F.min(), vmax=F.max() )
        self.cbar.update_normal(self.img)
        self.axes.set_xlim(0,F.shape[1])
        self.axes.set_ylim(0,F.shape[0])
        self.axes.set_title(title)
        self.fig.tight_layout()
        self.draw()

        #self.parent.figCurv.figCan.axes.cla()
        #self.parent.figCurv.figCan.defaultPlotAxis()
        #self.parent.figCurv.figCan.draw()
    
    def onclick(self, event):
        #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
        ix = int(event.xdata)
        iy = int(event.ydata) 
        self.axes.plot( [ix] , [iy], 'o' )
        self.draw()
        #ys = self.data[ :, iy, ix ]
        #self.parent.figCurv.show()
        #self.parent.figCurv.figCan.plotDatalines( ( range(len(ys)), ys, "%i_%i" %(ix,iy) )  )
        #self.axes.plot( range(len(ys)), ys )
        self.parent.clickImshow(ix,iy)
        return iy, ix

# =======================
#     SlaveWindow
# =======================

class SlaveWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, title="SlaveWindow" ):
        #super(self.__class__, self).__init__(parent)
        #QtWidgets.QMainWindow.__init__(parent)
        super(SlaveWindow, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle( title )
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.centralLayout = QtWidgets.QVBoxLayout()
        self.centralWidget().setLayout(self.centralLayout)

        #self.figCan = MyFigCanvas( parent, width=width, height=height, dpi=dpi )
        #l0.addWidget(self.figCan)

# =======================
#      PlotWindow
# =======================

class PlotWindow(SlaveWindow):
    def __init__(self, parent=None, title="PlotWindow", width=5, height=4, dpi=100 ):
        super(self.__class__, self).__init__( parent=parent, title=title );
        #SlaveWindow.__init__( parent=parent, title=title );
        #super(PlotWindow, self).__init__(parent=parent, title=title)
        self.figCan = FigPlot( parent, width=width, height=height, dpi=dpi )
        self.centralLayout.addWidget(self.figCan)

        vb = QtWidgets.QHBoxLayout(); self.centralLayout.addLayout(vb); #vb.addWidget( QtWidgets.QLabel("{iZPP, fMorse[1]}") )
        self.btSaveDat =bt= QtWidgets.QPushButton('Save.dat', self); bt.setToolTip('Save Curves to .dat file'); bt.clicked.connect(self.save_dat); vb.addWidget( bt )
        self.btSavePng =bt= QtWidgets.QPushButton('Save.png', self); bt.setToolTip('Save Figure to .png file'); bt.clicked.connect(self.save_png); vb.addWidget( bt )
        self.btClear   =bt= QtWidgets.QPushButton('Clear', self);    bt.setToolTip('Clear figure');             bt.clicked.connect(self.clearFig); vb.addWidget( bt )

        
        vb.addWidget( QtWidgets.QLabel("Xmin (top):") ); self.leXmin=wg=QtWidgets.QLineEdit(); wg.returnPressed.connect(self.setRange); vb.addWidget(wg)
        vb.addWidget( QtWidgets.QLabel("Xmax (bottom):") ); self.leXmax=wg=QtWidgets.QLineEdit(); wg.returnPressed.connect(self.setRange); vb.addWidget(wg)
        vb.addWidget( QtWidgets.QLabel("Ymin:") ); self.leYmin=wg=QtWidgets.QLineEdit(); wg.returnPressed.connect(self.setRange); vb.addWidget(wg)
        vb.addWidget( QtWidgets.QLabel("Ymax:") ); self.leYmax=wg=QtWidgets.QLineEdit(); wg.returnPressed.connect(self.setRange); vb.addWidget(wg)

    def save_dat(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","data files (*.dat)")
        if fileName:
            fileName = correct_ext( fileName, ".dat" )
            print "saving data to :", fileName
            data = []
            data.append( np.array(self.figCan.axes.lines[0].get_xdata()) )
            for line in self.figCan.axes.lines:
                #print "for line ", line
                data.append( line.get_ydata() )
            #print "data = ", data
            data = np.transpose( np.array(data) )
            #print "data = ", data
            #np.savetxt( fileName, data, fmt='%.6e', delimiter='\t', newline='\n', header='', footer='', comments='# ')
            np.savetxt( fileName, data )

    def save_png(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Image files (*.png)")
        if fileName:
            fileName = correct_ext( fileName, ".png" )
            print "saving image to :", fileName
            self.figCan.fig.savefig( fileName,bbox_inches='tight')

    def clearFig(self):
        self.figCan.axes.cla()
        self.figCan.defaultPlotAxis()
        self.figCan.draw()
    
    def setRange(self):
        xmin=None;xmax=None;ymin=None;ymax=None
        try:
            xmin = float( self.leXmin.text() )
            xmax = float( self.leXmax.text() )
            self.figCan.axes.set_xlim( xmin, xmax )
        except:
            pass
        try:
            ymin = float( self.leYmin.text() )
            ymax = float( self.leYmax.text() )
            self.figCan.axes.set_ylim( ymin, ymax )
        except:
            pass
        self.figCan.draw()
        print "range: ", xmin, xmax, ymin, ymax

# =======================
#         Editor
# =======================

class EditorWindow(SlaveWindow):
    def __init__(self, parent=None, title="Editor" ):
        super(self.__class__, self).__init__( parent=parent, title=title );
        #super(EditorWindow, self).__init__(parent=parent, title=title)
        #SlaveWindow.__init__( parent=parent, title=title );
        bt = QtWidgets.QPushButton('Update', self); 
        bt.setToolTip('recalculate using this atomic structure'); 
        bt.clicked.connect(parent.updateFromFF)
        self.btUpdate = bt; 

        self.textEdit = QtWidgets.QTextEdit()

        self.centralLayout.addWidget( self.textEdit )
        self.centralLayout.addWidget( bt )
