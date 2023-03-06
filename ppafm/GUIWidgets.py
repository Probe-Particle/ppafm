import os
import time

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from . import GridUtils as GU
from . import io

import matplotlib; matplotlib.use('Qt5Agg')

def correct_ext(fname, ext ):
    _, fext = os.path.splitext( fname )
    if( fext.capitalize() != ext.capitalize() ):
        fname += ext
    return fname

def set_box_value(box, value):
    box.blockSignals(True)
    box.setValue(value)
    box.blockSignals(False)

# =======================
#     FigCanvas
# =======================

class FigCanvas(FigureCanvasQTAgg):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, parentWiget=None, parentApp=None,  width=5, height=4, dpi=100 ):
        self.fig  = Figure( figsize=(width, height), dpi=dpi )
        self.axes = self.fig.add_subplot(111)
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

    def defaultPlotAxis(self):
        self.axes.grid()

    def plotDatalines(self, x, y, label):
        self.axes.plot(x, y, 'x-', label=label)
        self.draw()

# =======================
#      FigLinePlot
# =======================

class FigImshow(FigCanvas):
    """A canvas that updates itself every second with a new plot."""
    cbar = None

    def __init__(self, parentWiget=None, parentApp=None,  width=5, height=4, dpi=100, verbose=0):
        super(self.__class__, self).__init__(parentWiget=parentWiget, parentApp=parentApp,  width=width, height=height, dpi=dpi )
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.verbose = verbose
        self.img = None
        self.cbar = None

    def plotSlice(self, F_stack , z_slice, title=None, margins=None, grid_selector = 0,
            slice_length = None, points=[], cbar_range=None, extent=None):

        F = F_stack[z_slice]

        if self.verbose > 0: print("plotSlice F.shape, F.min(), F.max(), margins", F.shape, F.min(), F.max(), margins)

        if self.img is None or self.img.get_array().shape != F.shape:
            self.axes.cla()
            self.img = self.axes.imshow(F, origin='lower', cmap='gray', interpolation='bicubic')
            for ix, iy in points:
                self.axes.plot([ix] , [iy], 'o')
        else:
            self.img.set_data(F)
            if len(points) == 0:
                for line in self.axes.lines:
                    line.remove()
                self.axes.set_prop_cycle(None) # Reset color cycle
                if self.verbose > 0: print('plotSlice: reset points')
            else:
                for p, (ix, iy) in zip(self.axes.lines, points):
                    p.set_data((ix, iy))

        if cbar_range:
            if self.cbar == None:
                self.cbar = self.fig.colorbar(self.img, ax=self.axes)
                self.cbar.set_label('df (Hz)')
                if self.verbose > 0: print('plotSlice: added colorbar')
            self.img.set_clim(vmin=cbar_range[0], vmax=cbar_range[1])
            self.cbar.mappable.set_clim(vmin=cbar_range[0], vmax=cbar_range[1])
        else:
            self.img.autoscale()
            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None
                if self.verbose > 0: print('plotSlice: removed colorbar')

        if extent is not None:
            self.img.set_extent(extent)
            self.axes.set_xlabel('x (Å)')
            self.axes.set_ylabel('y (Å)')

        if margins:
            self.axes.add_patch(matplotlib.patches.Rectangle((margins[0], margins[1]),F.shape[1]-margins[2]-margins[0], F.shape[0]-margins[3]-margins[1], linewidth=2,edgecolor='r',facecolor='none'))
            textRes = 'output size: '+str(F.shape[1]-margins[2]-margins[0])+ 'x'+ str(F.shape[0]-margins[3]-margins[1])
            if slice_length:
                textRes += '     length [A] ='+f'{slice_length[0]:03.4f}, {slice_length[1]:03.4f}'
            self.axes.set_xlabel(textRes)

        self.axes.set_title(title)

        if (grid_selector > 0):
            self.axes.grid(True,  linestyle='dotted', color='blue')
        else:
            self.axes.grid(False)

        self.fig.tight_layout()
        self.draw()

    def plotSlice2(self, F_stack, title=None , margins=None, grid_selector = 0, slice_length = None, big_len_image = None, alpha = 0.0):

        self.axes.cla()

        F = F_stack
        print("plotSlice F.shape, F.min(), F.max() ", F.shape, F.min(), F.max())


        print('self.margins', margins)
        if alpha>0 and big_len_image is not None:
            F = F*(1-alpha) + big_len_image*alpha
        self.img = self.axes.imshow( F, origin='image', cmap='viridis', interpolation='bicubic' )

        j_min,i_min = np.unravel_index(F.argmin(), F.shape)
        j_max,i_max = np.unravel_index(F.argmax(), F.shape)

        if margins:
            self.axes.add_patch(matplotlib.patches.Rectangle((margins[0], margins[1]),margins[2], margins[3], linewidth=2,edgecolor='r',facecolor='none'))
            textRes = 'output size: '+str(margins[2])+ 'x'+ str(margins[3])
            if slice_length:
                textRes += '     length [A] ='+f'{slice_length[0]:03.4f}, {slice_length[1]:03.4f}'
            self.axes.set_xlabel(textRes)

        self.axes.set_xlim(0,F.shape[1])
        self.axes.set_ylim(0,F.shape[0])
        self.axes.set_title(title)
        if (grid_selector > 0):
            self.axes.grid(True,  linestyle='dotted', color='blue')
        else:
            self.axes.grid(False)

        self.fig.tight_layout()
        self.draw()


    def onclick(self, event):
        if not self.parent: return
        try:
            x = float(event.xdata)
            y = float(event.ydata)
        except TypeError:
            if self.verbose > 0: print('Invalid click event.')
            return
        self.axes.plot([x], [y], 'o', scalex=False, scaley=False)
        self.draw()
        self.parent.clickImshow(x, y)
        return y, x

    def onscroll(self, event):
        if not self.parent: return
        try:
            x = float(event.xdata)
            y = float(event.ydata)
        except TypeError:
            if self.verbose > 0: print('Invalid scroll event.')
            return
        if event.button == 'up':
            direction = 'in'
        elif event.button == 'down':
            direction = 'out'
        else:
            print(f'Invalid scroll direction {event.button}')
            return
        self.parent.zoomTowards(x, y, direction)

# =======================
#     SlaveWindow
# =======================

class SlaveWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, title="SlaveWindow" ):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle( title )
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.centralLayout = QtWidgets.QVBoxLayout()
        self.centralWidget().setLayout(self.centralLayout)

# =======================
#      PlotWindow
# =======================

class PlotWindow(SlaveWindow):
    def __init__(self, parent=None, title="PlotWindow", width=5, height=4, dpi=100 ):
        super(self.__class__, self).__init__( parent=parent, title=title );
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
        default_path = os.path.join(os.path.split(self.parent.file_path)[0], 'df_curve.dat')
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save df curve raw data", default_path,
            "Data files (*.dat)")
        if fileName:
            fileName = correct_ext( fileName, ".dat" )
            print("saving data to :", fileName)
            data = []
            data.append( np.array(self.figCan.axes.lines[0].get_xdata()) )
            for line in self.figCan.axes.lines:
                data.append( line.get_ydata() )
            data = np.transpose( np.array(data) )
            np.savetxt( fileName, data )

    def save_png(self):
        default_path = os.path.join(os.path.split(self.parent.file_path)[0], 'df_curve.png')
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save df curve image", default_path,
            "Image files (*.png)")
        if fileName:
            fileName = correct_ext( fileName, ".png" )
            print("saving image to :", fileName)
            self.figCan.fig.savefig( fileName,bbox_inches='tight')

    def clearFig(self):
        self.figCan.axes.cla()
        self.figCan.defaultPlotAxis()
        self.figCan.draw()
        self.parent.clearPoints()

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
        print("range: ", xmin, xmax, ymin, ymax)

# =======================
#         Editor
# =======================

class GeomEditor(SlaveWindow):

    def __init__(self, n_atoms, enable_qs=True, parent=None, title="Geometry editor" ):

        super().__init__(parent=parent, title=title)

        self.n_atoms = n_atoms
        self.enable_qs = enable_qs

        # Layout everything on a (n_atoms + 1) x 5 grid
        grid = QtWidgets.QGridLayout()
        self.centralLayout.addLayout(grid)

        # Labels
        for j, text in enumerate('Zxyzq'):
            lb = QtWidgets.QLabel(text)
            lb.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(lb, 0, j)

        # Atoms
        self.input_boxes = []
        for i in range(n_atoms):

            bZ = QtWidgets.QSpinBox(); bZ.setRange(0, 200)
            bx = QtWidgets.QDoubleSpinBox(); bx.setRange(-100, 100); bx.setSingleStep(0.1); bx.setDecimals(3)
            by = QtWidgets.QDoubleSpinBox(); by.setRange(-100, 100); by.setSingleStep(0.1); by.setDecimals(3)
            bz = QtWidgets.QDoubleSpinBox(); bz.setRange(-100, 100); bz.setSingleStep(0.05); bz.setDecimals(3)
            bQ = QtWidgets.QDoubleSpinBox(); bQ.setRange(-2.0, 2.0); bQ.setSingleStep(0.02); bQ.setDecimals(3)

            for j, b in enumerate([bZ, bx, by, bz, bQ]):
                b.valueChanged.connect(self.updateParent)
                grid.addWidget(b, i+1, j)

            self.input_boxes.append([bZ, bx, by, bz, bQ])

        # Since the box may be very tall, make it scrollable
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.main_widget)
        self.scroll.setMinimumWidth(420)
        self.setCentralWidget(self.scroll)

    def updateValues(self):

        xyzs = self.parent.xyzs
        Zs = self.parent.Zs

        if self.enable_qs:
            qs = self.parent.qs
        else:
            qs = np.zeros(len(xyzs))
            for ab in self.input_boxes:
                ab[4].setDisabled(True)

        for xyz, Z, q, boxes in zip(xyzs, Zs, qs, self.input_boxes):
            set_box_value(boxes[0], Z)
            set_box_value(boxes[1], xyz[0])
            set_box_value(boxes[2], xyz[1])
            set_box_value(boxes[3], xyz[2])
            set_box_value(boxes[4], q)

    def updateParent(self):
        xyzs, Zs, qs = [], [], []
        for boxes in self.input_boxes:
            xyzs.append([boxes[1].value(), boxes[2].value(), boxes[3].value()])
            Zs.append(boxes[0].value())
            qs.append(boxes[4].value())
        self.parent.xyzs = np.array(xyzs)
        self.parent.Zs = np.array(Zs, dtype=np.int32)
        if self.enable_qs:
            self.parent.qs = np.array(qs)
        self.parent.update()

class LJParamEditor(QtWidgets.QMainWindow):

    def __init__(self, type_params, parent=None, title="Edit Forcefield"):

        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(title)
        self.grid = QtWidgets.QGridLayout()
        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setLayout(self.grid)

        self.type_params = np.array(type_params)

        # Labels
        for j, text in enumerate(['Z', 'r_0', 'eps_0']):
            lb = QtWidgets.QLabel(text)
            lb.setAlignment(QtCore.Qt.AlignCenter)
            self.grid.addWidget(lb, 0, j)

        # Lennard-Jones parameters
        self.input_boxes = []
        for i, (r0, e0, _, _, el) in enumerate(self.type_params):

            lb = QtWidgets.QLabel(f'{i+1}: ' + el.decode('UTF-8')); self.grid.addWidget(lb, i+1, 0)
            br = QtWidgets.QDoubleSpinBox(); br.setRange(0, 5); br.setSingleStep(0.01); br.setDecimals(4); br.setValue(r0)
            be = QtWidgets.QDoubleSpinBox(); be.setRange(0, 0.1); be.setSingleStep(0.0002); be.setDecimals(6); be.setValue(e0)

            for j, b in enumerate([br, be]):
                b.valueChanged.connect(self.updateParent)
                self.grid.addWidget(b, i+1, j+1)

            self.input_boxes.append([br, be])

        # Since the box is very tall, make it scrollable
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.main_widget)
        self.scroll.setMinimumWidth(250)
        self.setCentralWidget(self.scroll)

    def updateParent(self):
        for i, (br, be) in enumerate(self.input_boxes):
            p = self.type_params[i]
            r0 = br.value()
            e0 = be.value()
            self.type_params[i] = (r0, e0, p[2], p[3], p[4])
        self.parent.afmulator.typeParams = [tuple(p) for p in self.type_params]
        self.parent.update()

class FFViewer(SlaveWindow):

    def __init__(self, parent=None, title="View Forcefield", width=5, height=4, dpi=100,
            verbose=0):
        super().__init__(parent=parent, title=title)
        self.verbose=verbose

        self.figCan = FigImshow(parent, width=width, height=height, dpi=dpi)
        self.centralLayout.addWidget(self.figCan)

        tooltips = [
            'Component: Choose which component of forcefield to view:\nFx: Force x-component'
                + '\nFy: Force y-component\nFz: Force z-component\nE: Potential Energy',
            'z index: index of z coordinate to view.'
        ]
        components = ['Fx', 'Fy', 'Fz', 'E']

        hb = QtWidgets.QHBoxLayout(); self.centralLayout.addLayout(hb)

        hbl = QtWidgets.QHBoxLayout(); hb.addLayout(hbl)
        lb = QtWidgets.QLabel('Component:'); lb.setToolTip(tooltips[0]); hbl.addWidget(lb)
        sl = QtWidgets.QComboBox(); self.slComponent = sl; sl.addItems(components)
        sl.setCurrentIndex(sl.findText(components[2])); sl.currentIndexChanged.connect(self.updateView)
        sl.setToolTip(tooltips[0]); hbl.addWidget(sl)

        hbr = QtWidgets.QHBoxLayout(); hb.addLayout(hbr)
        lb = QtWidgets.QLabel('z index:'); lb.setToolTip(tooltips[1]); hbr.addWidget(lb)
        bx = QtWidgets.QSpinBox(); bx.setRange(0, 200); bx.setValue(0)
        bx.valueChanged.connect(self.updateView); bx.setToolTip(tooltips[1])
        hbr.addWidget(bx); self.bxInd = bx
        hbr.setContentsMargins(10, 0, 10, 0)

        hb.addStretch(1)

        bt = QtWidgets.QPushButton('Save to file...', self)
        bt.setToolTip('Save current force field component to a .xsf file.')
        bt.clicked.connect(self.saveFF)
        self.btSaveFF = bt; hb.addWidget(bt)


    def updateFF(self):

        afmulator = self.parent.afmulator

        self.FE = afmulator.forcefield.downloadFF()
        self.bxInd.setRange(0, self.FE.shape[2] - 1)

        self.z_min = afmulator.lvec[0, 2]
        self.z_step = 1 / afmulator.pixPerAngstrome

        z = afmulator.scan_window[0][2] + afmulator.amplitude / 2 - afmulator.tipR0[2]
        iz = round((z - self.z_min) / self.z_step)
        if not self.isVisible():
            set_box_value(self.bxInd, iz)

        if self.verbose > 0: print('FFViewer.updateFF', self.FE.shape, iz, self.z_step, self.z_min)

    def updateView(self):

        t0 = time.perf_counter()

        ic = self.slComponent.currentIndex()
        iz = self.bxInd.value()
        z = self.z_min + iz * self.z_step
        data = self.FE[..., ic].transpose(2, 1, 0)
        self.figCan.plotSlice(data, iz, title=f'z = {z:.2f}Å')

        if self.verbose > 0: print('FFViewer.updateView', ic, iz, data.shape)
        if self.verbose > 1: print('updateView time [s]', time.perf_counter() - t0)

    def saveFF(self):

        comp = self.slComponent.currentText()
        default_path = os.path.join(os.path.split(self.parent.file_path)[0], f'{comp}.xsf')
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save force field data", default_path,
            "XCrySDen files (*.xsf)")
        if not fileName: return
        ext = os.path.splitext(fileName)[1]
        if ext != '.xsf':
            self.parent.status_message('Unsupported file type in force field save file path')
            print(f'Unsupported file type in force field save file path `{fileName}`')
            return
        self.parent.status_message('Saving data...')

        if self.verbose > 0: print(f'Saving force field data to {fileName}...')
        ic = self.slComponent.currentIndex()
        data = self.FE.copy()
        # Clamp large values for easier visualization
        if ic < 3:
            GU.limit_vec_field(data, Fmax=1000)
            data = data[..., ic].transpose(2, 1, 0)
        else:
            data = data[..., ic].transpose(2, 1, 0)
            data[data > 1000] = 1000
        lvec = self.parent.afmulator.lvec
        xyzs = self.parent.xyzs - lvec[0]
        atomstring = io.primcoords2Xsf(self.parent.Zs, xyzs.T, lvec)
        GU.saveXSF(fileName, data, lvec, head=atomstring, verbose=0)

        if self.verbose > 0: print("Done saving force field data.")
        self.parent.status_message('Ready')
