#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases


import copy
import glob
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nimg
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

import ppafm.file_dat as file_dat
import ppafm.GUIWidgets as guiw

matplotlib.use("Qt5Agg")


def crosscorel_2d_fft(im0, im1):
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)
    renorm = 1 / (np.std(f0) * np.std(f1))
    # return abs(np.fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    # return np.abs( np.fft.ifft2( (    f0 * f1.conjugate() / ( np.abs(f0) * np.abs(f1) )  ) ) )
    return abs(np.fft.ifft2(f0 * f1.conjugate())) * renorm


def trans_match_fft(im0, im1):
    """Return translation vector to register images."""
    print("we are in trans_match_fft")
    shape = im0.shape
    """
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    """
    ir = crosscorel_2d_fft(im0, im1)
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    # if t0 > shape[0] // 2:
    #    t0 -= shape[0]
    # if t1 > shape[1] // 2:
    #    t1 -= shape[1]
    return [t0, t1]


def roll2d(a, shift=(10, 10)):
    a_ = np.roll(a, shift[0], axis=0)
    return np.roll(a_, shift[1], axis=1)


class ApplicationWindow(QtWidgets.QMainWindow):
    # path='./'
    # path="/u/25/prokoph1/unix/Desktop/CATAM/Exp_Data/Camphor/Orientation_4/"
    # path_CO='/Users/urtevf1/Documents/20december/Xylotetraose/CO/CO 1'
    path_Xe = "/Users/urtevf1/Documents/21april/Xylo/Xe/Xe 1"
    path_CO = "/Users/urtevf1/Documents/21april/Xylo/CO/CO 1"
    # path_Xe=''
    ind_biggest_AFM = 0
    cur_rotation = 0

    def __init__(self):
        # --- init QtMain
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")
        self.main_widget = QtWidgets.QWidget(self)
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        self.figCan = guiw.FigImshow(parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100)
        l00.addWidget(self.figCan)
        l0 = QtWidgets.QVBoxLayout(self.main_widget)
        l00.addLayout(l0)

        # -------------- Potential
        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        vb.addWidget(QtWidgets.QLabel("CO"))
        el = QtWidgets.QLineEdit()
        el.setText(self.path_CO)
        vb.addWidget(el)
        el.setToolTip("path to folder with CO tip AFM .dat files")
        self.txPath_CO = el
        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        vb.addWidget(QtWidgets.QLabel("Xe"))
        el = QtWidgets.QLineEdit()
        el.setText(self.path_Xe)
        vb.addWidget(el)
        el.setToolTip("path to folder with Xe tip AFM .dat files")
        self.txPath_Xe = el

        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        vb.addWidget(QtWidgets.QLabel("load"))
        bt = QtWidgets.QPushButton("Load all *.dat", self)
        bt.setToolTip("load .dat files from dir")
        bt.clicked.connect(self.loadData)
        vb.addWidget(bt)
        self.btLoad = bt

        bx = QtWidgets.QComboBox()
        bx.addItem("CO")
        bx.addItem("Xe")
        bx.activated[str].connect(self.selectDataTip)
        vb.addWidget(bx)
        bx.setToolTip("edit AFM data of specific tip")
        bx.setEnabled(False)
        self.bxTip = bx

        vb.addWidget(QtWidgets.QLabel("channel"))
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setValue(1)
        bx.valueChanged.connect(self.selectDataChannel)
        vb.addWidget(bx)
        bx.setToolTip("select available channels with data")
        bx.setEnabled(False)
        self.bxChannel = bx
        vb.addWidget(QtWidgets.QLabel("alpha"))
        bx = QtWidgets.QDoubleSpinBox()
        bx.setSingleStep(0.1)
        bx.setValue(0.0)
        bx.setRange(0.0, 1.0)
        bx.valueChanged.connect(self.selectAlpha)
        vb.addWidget(bx)
        bx.setToolTip("select level of overlay for smaller size tips AFM image")
        bx.setEnabled(False)
        self.bxAlpha = bx
        ln = QtWidgets.QFrame()
        l0.addWidget(ln)
        ln.setFrameShape(QtWidgets.QFrame.HLine)
        ln.setFrameShadow(QtWidgets.QFrame.Sunken)
        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        vb.addWidget(QtWidgets.QLabel("slice "))
        bx = QtWidgets.QSpinBox()
        bx.setToolTip("select available slices from stack")
        bx.setSingleStep(1)
        bx.setValue(0)
        bx.valueChanged.connect(self.selectDataView)
        vb.addWidget(bx)
        bx.setEnabled(False)
        self.bxZ = bx

        checkbox = QtWidgets.QCheckBox("show grid")
        vb.addWidget(checkbox)
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self.ChkBxGrid)
        self.checkbox = checkbox
        checkbox.setEnabled(False)
        vb.addWidget(QtWidgets.QLabel(" rotate"))
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setValue(0)
        bx.setRange(-180, 180)
        bx.valueChanged.connect(self.selectRotate)
        vb.addWidget(bx)
        bx.setToolTip("select rotatation angle in degrees for current AFM tip")
        bx.setEnabled(False)
        self.bxAngle = bx

        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        vb.addWidget(QtWidgets.QLabel("shift ix,iy"))
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setToolTip("adjust vertical shift")
        bx.setValue(0)
        bx.setRange(-1000, 1000)
        bx.valueChanged.connect(self.shiftData)
        vb.addWidget(bx)
        bx.setEnabled(False)
        self.bxX = bx
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setValue(0)
        bx.setToolTip("adjust horizontal shift")
        bx.setRange(-1000, 1000)
        bx.valueChanged.connect(self.shiftData)
        vb.addWidget(bx)
        bx.setEnabled(False)
        self.bxY = bx

        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        bt = QtWidgets.QPushButton("Magic fit", self)
        bt.setToolTip("Fit to colser slice")
        bt.clicked.connect(self.magicFit)
        vb.addWidget(bt)
        bt.setEnabled(False)
        self.btMagic = bt

        l0.addLayout(vb)
        bt = QtWidgets.QPushButton("MagicAll", self)
        bt.setToolTip("Fit all slices")
        bt.setEnabled(False)
        bt.clicked.connect(self.magicFitAll)
        vb.addWidget(bt)
        self.btMagicAll = bt
        l0.addLayout(vb)
        bt = QtWidgets.QPushButton("SaveImgs", self)
        bt.setToolTip("save images")
        bt.setEnabled(False)
        bt.clicked.connect(self.saveImg)
        vb.addWidget(bt)
        self.btSaveImg = bt
        ln = QtWidgets.QFrame()
        l0.addWidget(ln)
        ln.setFrameShape(QtWidgets.QFrame.HLine)
        ln.setFrameShadow(QtWidgets.QFrame.Sunken)
        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        vb.addWidget(QtWidgets.QLabel("Ninter "))
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setValue(1)
        vb.addWidget(bx)
        bx.setEnabled(False)
        self.bxNi = bx
        bt = QtWidgets.QPushButton("interpolate", self)
        bt.setToolTip("select amount of slices to add before slice")
        bt.clicked.connect(self.interpolate)
        bt.setEnabled(False)
        vb.addWidget(bt)
        self.btInterp = bt

        # sl = QtWidgets.QComboBox(); self.slMode = sl; vb.addWidget(sl)
        # sl.currentIndexChanged.connect(self.selectMode)

        # sl = QtWidgets.QComboBox(); self.slDataView = sl; vb.addWidget(sl)
        # sl.addItem( DataViews.FFpl.name ); sl.addItem( DataViews.FFel.name ); sl.addItem( DataViews.FFin.name ); sl.addItem( DataViews.FFout.name ); sl.addItem(DataViews.df.name );
        # sl.setCurrentIndex( sl.findText( DataViews.FFpl.name ) )
        # sl.currentIndexChanged.connect(self.updateDataView)

        # === buttons

        ln = QtWidgets.QFrame()
        l0.addWidget(ln)
        ln.setFrameShape(QtWidgets.QFrame.HLine)
        ln.setFrameShadow(QtWidgets.QFrame.Sunken)

        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)

        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        vb.addWidget(QtWidgets.QLabel("margins:"))
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setValue(0)
        bx.setRange(0, 1000)
        bx.valueChanged.connect(self.marginData)
        bx.setToolTip("trim left border")
        vb.addWidget(QtWidgets.QLabel("L"))
        vb.addWidget(bx)
        bx.setEnabled(False)
        self.marginX0 = bx
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setValue(0)
        bx.setRange(0, 1000)
        bx.valueChanged.connect(self.marginData)
        bx.setToolTip("trim bottom border")
        vb.addWidget(QtWidgets.QLabel("B"))
        vb.addWidget(bx)
        bx.setEnabled(False)
        self.marginY0 = bx
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setValue(0)
        bx.setRange(0, 1000)
        bx.valueChanged.connect(self.marginData)
        bx.setToolTip("adjust crop width")
        vb.addWidget(QtWidgets.QLabel("W"))
        vb.addWidget(bx)
        bx.setEnabled(False)
        self.marginW = bx
        bx = QtWidgets.QSpinBox()
        bx.setSingleStep(1)
        bx.setValue(0)
        bx.setRange(0, 1000)
        bx.valueChanged.connect(self.marginData)
        bx.setToolTip("adjust crop height")
        vb.addWidget(QtWidgets.QLabel("H"))
        vb.addWidget(bx)
        bx.setEnabled(False)
        self.marginH = bx

        ln = QtWidgets.QFrame()
        l0.addWidget(ln)
        ln.setFrameShape(QtWidgets.QFrame.HLine)
        ln.setFrameShadow(QtWidgets.QFrame.Sunken)
        # --- btSave
        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        vb.addWidget(QtWidgets.QLabel("slices"))
        selSliceSave = QtWidgets.QLineEdit()
        vb.addWidget(selSliceSave)
        selSliceSave.setToolTip("select slices to save, i.e.: 1,3-11")
        selSliceSave.setEnabled(False)
        self.txSliceSave = selSliceSave
        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        self.btSave = QtWidgets.QPushButton("Save .npz", self)
        self.btSave.setToolTip("save data stack to .npz")
        self.btSave.clicked.connect(self.saveData)
        self.btSave.setEnabled(False)
        vb.addWidget(self.btSave)
        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        bt = QtWidgets.QPushButton("Load .npz", self)
        bt.setToolTip("load .npz file  from dir")
        bt.clicked.connect(self.loadNPZ)
        vb.addWidget(bt)
        self.btLoad2 = bt
        self.btLoad2.setEnabled(False)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        # self.geomEditor    = guiw.EditorWindow(self,title="Geometry Editor")
        # self.speciesEditor = guiw.EditorWindow(self,title="Species Editor")
        # self.figCurv       = guiw.PlotWindow( parent=self, width=5, height=4, dpi=100)

    def magicFit(self):
        print("magic fit")
        iz = int(self.bxZ.value())
        print("iz=", iz)
        if iz < len(self.data) - 1:
            # print ('we are in if')
            """
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
            """
            [ix, iy] = trans_match_fft(self.data2[iz], self.data[iz + 1])
            print("ix,iy=", -ix, -iy)
            if abs(int(ix)) > self.data[iz].shape[0]:
                ix = ix / abs(int(ix)) * (abs(int(ix)) - self.data[iz].shape[0])
            if abs(int(iy)) > self.data[iz].shape[1]:
                iy = iy / abs(int(iy)) * (abs(int(iy)) - self.data[iz].shape[1])
            if abs(int(ix)) > self.data[iz].shape[0] // 2:
                ix = ix / abs(int(ix)) * (abs(int(ix)) - self.data[iz].shape[0])
            if abs(int(iy)) > self.data[iz].shape[1] // 2:
                iy = iy / abs(int(iy)) * (abs(int(iy)) - self.data[iz].shape[1])

            self.data[iz] = nimg.shift(self.data[iz], (-ix - self.shifts[iz][0], -iy - self.shifts[iz][1]), order=3, mode="mirror")
            self.shifts[iz][0] = -ix
            self.shifts[iz][1] = -iy
            self.bxX.setValue(self.shifts[iz][0])
            self.bxY.setValue(self.shifts[iz][1])

            print(self.shifts)
            self.updateDataView()

    def magicFitAll(self):
        izs = range(len(self.data) - 1)
        print("izs = ", izs)
        for iz in izs[::-1]:
            self.bxZ.setValue(iz)
            self.magicFit()

    def loadData(self):
        # print file_list
        # fnames
        self.pathes = [self.txPath_CO.text(), self.txPath_Xe.text()]
        assert self.pathes[0]
        self.channel = int(self.bxChannel.value())
        print(self.path_CO)
        print(self.path_Xe)
        """
        https://www.tutorialspoint.com/pyqt/pyqt_qfiledialog_widget.htm
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFilter("Text files (*.txt)")
        filenames = QStringList()
        if dlg.exec_():
            filenames = dlg.selectedFiles()

        self.path =
        """
        self.fnames = []
        self.data = []
        self.data_orig = []
        self.headers = []
        self.slice_lengths = []
        self.tip = 0

        self.shifts = []
        self.margins = []
        self.max_length = []
        self.marged_size = []
        for i, path in enumerate(self.pathes):
            if path:
                if path[-1] != "/":
                    path += "/"

                file_pathes = glob.glob(path + "*.dat")
                file_pathes.sort()

                data = []
                headers = []
                fnames = []
                for fname in file_pathes:
                    # print fname
                    fname_ = os.path.basename(fname)
                    fnames.append(fname_)
                    # print os.path.basename(fname)
                    Header = {}
                    imgs = file_dat.readDat(fname, Header=Header)
                    amountCh = len(imgs)
                    # print fname, "Size ", Header['ScanRange_Y'],Header['ScanRange_X']," N ", imgs[0].shape," dxy ",  Header['ScanRange_Y']/imgs[0].shape[0], Header['ScanRange_X']/imgs[0].shape[1]
                    # print "Lentgh [x] ", Header['LengthX']
                    # print "Lentgh [y] ", Header['LengthY']
                    data.append(imgs[self.channel])
                    headers.append(Header)
                    # data2.append( imgs[1] )

                # return data
                data = [d[::-1, :] for d in data]

                self.data.append(data)
                self.headers.append(headers)
                self.fnames.append(fnames)
                # z=np.arange(25)
                self.data_orig.append(copy.copy(data))
                image_shape = self.data[i][0].shape
                print(f"Data from {path} loaded")
                print(f"data len = {len(self.data[i])}")
                print(f"image.shape = {image_shape}")

                self.shifts.append([[0, 0] for i in range(len(self.data[i]))])
                self.margins.append([0, 0, image_shape[0], image_shape[1]])
                self.bxZ.setRange(0, len(self.data[i]) - 1)
                self.marged_size.append([image_shape[0], image_shape[1]])
                # set proper scale for all slices depends from parameters:  Header['LengthX']; Header['LengthY']

                slice_lengths = [[x["LengthX"], x["LengthY"]] for x in headers]
                # print ('slice_lengths = ', slice_lengths)
                max_length = [np.max(np.array(slice_lengths)[:, 0]), np.max(np.array(slice_lengths)[:, 1])]
                print("max slice_length = ", max_length)

                for z_slice in range(len(self.data[i])):
                    if slice_lengths[z_slice][0] != max_length[0]:
                        # here we fit slice of AFM data, but according to only x axis scale. So ir works only for square images.
                        # print ('slice_lengths/max_lengths = ',slice_lengths[z_slice][0]/max_length[0])
                        scaled_size = int(image_shape[0] * slice_lengths[z_slice][0] / max_length[0])

                        start_xy = int((image_shape[0] - scaled_size) / 2)
                        scaled_image = np.array(Image.fromarray(self.data[i][z_slice]).resize((scaled_size, scaled_size), Image.BILINEAR))
                        # scaled_afm_slice[start_xy:start_xy+scaled_size, start_xy:start_xy+scaled_size] =  scaled_image
                        # scaled_afm_slice[:start_xy,:] = np.flipud(scaled_image[:start_xy,:])
                        pad_sh = (start_xy, image_shape[0] - scaled_size - start_xy)
                        self.data[i][z_slice] = np.pad(scaled_image, (pad_sh, pad_sh), "symmetric")
                        self.data_orig[i][z_slice] = np.pad(scaled_image, (pad_sh, pad_sh), "symmetric")
                        self.margins[i] = [pad_sh[0], pad_sh[0], scaled_size, scaled_size]

                print(f"self.margins[{i}] = {self.margins[i]}")

                for z_slice in range(len(self.data[i])):
                    marged_size = [self.margins[i][2], self.margins[i][3]]
                    self.marged_size[i] = marged_size
                    # print 'self.max_length  =', max_length
                    # slice_lengths[z_slice] = [marged_size[1]*  max_length[1]/image_shape[0]   , marged_size[0]*  max_length[0]/image_shape[1] ]
                max_length = [np.max(np.array(slice_lengths)[:, 0]), np.max(np.array(slice_lengths)[:, 1])]

                # print ('slice_lengths = ',  slice_lengths )

                self.slice_lengths.append(slice_lengths)
                self.max_length.append(max_length)

                # print 'amountCh = ', amountCh
                self.bxChannel.setRange(0, amountCh - 1)

            else:
                pass
        self.tip = i

        # now we should adjust lenghts of two tips AFM images
        if self.tip > 0:
            print("self.marged_size = ", self.marged_size)
            print("self.max_length = ", self.max_length)
            print("self.image shape = ", [self.data[0][0].shape, self.data[1][0].shape])
            diff_lenghts_2tip = np.abs(self.max_length[0][0] - self.max_length[1][0])
            diff_pixels_2tip = np.abs(self.data[0][0].shape[0] - self.data[1][0].shape[0])
            print("diff_lenghts_2tip  = ", diff_lenghts_2tip)
            # print ('slice_lengths = ', self.slice_lengths)
            if diff_lenghts_2tip > 1e-1 or diff_pixels_2tip > 2:  # if difference is more then 2 pixels or more than 0.1 Ang
                ind_biggest_AFM = np.argmax([self.max_length[0][0], self.max_length[1][0]])
                self.ind_biggest_AFM = ind_biggest_AFM
                print("biggest AFM image = ", ind_biggest_AFM)
                pix2ang_big = self.marged_size[ind_biggest_AFM][0] / self.max_length[ind_biggest_AFM][0]
                pixels_size_small = int(pix2ang_big * self.max_length[1 - ind_biggest_AFM][0])
                print("self.margins[1-ind_biggest_AFM] = ", self.margins[1 - ind_biggest_AFM])

                self.margins[1 - ind_biggest_AFM] = [int(x * pixels_size_small / self.data[1 - ind_biggest_AFM][0].shape[0]) for x in self.margins[1 - ind_biggest_AFM]]

                print("pix2ang_big = ", pix2ang_big)
                print("pixels_size_small = ", pixels_size_small)
                for z_slice in range(len(self.data[1 - ind_biggest_AFM])):
                    scaled_image = np.array(Image.fromarray(self.data[1 - ind_biggest_AFM][z_slice]).resize((pixels_size_small, pixels_size_small), Image.BILINEAR))
                    self.data[1 - ind_biggest_AFM][z_slice] = scaled_image
                    self.data_orig[1 - ind_biggest_AFM][z_slice] = scaled_image

                self.margins[1 - ind_biggest_AFM][0]
                self.margins[1 - ind_biggest_AFM][1]
                imarginW = self.margins[1 - ind_biggest_AFM][2]
                imarginH = self.margins[1 - ind_biggest_AFM][3]

                self.marged_size[1 - ind_biggest_AFM] = [imarginW, imarginH]
                print("self.margins[1-ind_biggest_AFM] = ", self.margins[1 - ind_biggest_AFM])
                print("self.marged_size = ", self.marged_size)
                print("slice_lengths[1-ind_biggest_AFM] = ", self.slice_lengths[1 - ind_biggest_AFM])

        # now we select the smaller marged shape of two tips AFM images
        self.min_margin_size_ind = np.argmin([self.marged_size[0][0], self.marged_size[1][0]])
        self.combined_margin_size = self.marged_size[self.min_margin_size_ind]
        self.tip = 1 - self.min_margin_size_ind
        start_xy = int((self.data[self.tip][0].shape[0] - self.combined_margin_size[0]) / 2)
        pad_sh = (start_xy, self.data[self.tip][0].shape[0] - self.combined_margin_size[0] - start_xy)
        self.margins[self.tip] = [start_xy, start_xy, self.combined_margin_size[0], self.combined_margin_size[0]]
        print("self.margins = ", self.margins)
        self.bxChannel.setRange(0, amountCh - 1)
        self.bxZ.setRange(0, len(self.data[self.tip]) - 1)
        self.marginX0.blockSignals(True)
        self.marginX0.setValue(self.margins[self.tip][0])
        self.marginX0.blockSignals(False)
        self.marginY0.blockSignals(True)
        self.marginY0.setValue(self.margins[self.tip][1])
        self.marginY0.blockSignals(False)
        self.marginW.blockSignals(True)
        self.marginW.setValue(self.margins[self.tip][2])
        self.marginW.blockSignals(False)

        self.marginH.blockSignals(True)
        self.marginH.setValue(self.margins[self.tip][3])
        self.marginH.blockSignals(False)
        iz = int(self.bxZ.value())
        self.bxX.blockSignals(True)
        self.bxX.setValue(self.shifts[self.tip][iz][0])
        self.bxX.blockSignals(False)
        self.bxY.blockSignals(True)
        self.bxY.setValue(self.shifts[self.tip][iz][1])
        self.bxY.blockSignals(False)
        self.bxTip.setCurrentIndex(self.tip)

        self.updateDataView()
        self.bxChannel.setEnabled(True)
        self.bxZ.setEnabled(True)
        self.bxX.setEnabled(True)
        self.bxY.setEnabled(True)
        self.bxAlpha.setEnabled(True)
        self.bxAngle.setEnabled(True)
        # self.btMagicAll.setEnabled(True)
        # self.btMagic.setEnabled(True)
        # self.btSaveImg.setEnabled(True)
        # self.bxNi.setEnabled(True)
        # self.btInterp.setEnabled(True)
        self.marginX0.setEnabled(True)
        self.marginW.setEnabled(True)
        self.marginY0.setEnabled(True)
        self.marginH.setEnabled(True)
        self.btSave.setEnabled(True)
        self.txSliceSave.setEnabled(True)
        self.checkbox.setEnabled(True)

        self.bxTip.setEnabled(True)
        self.btLoad.setEnabled(False)

    def interpolate(self):
        iz = int(self.bxZ.value())
        ni = int(self.bxNi.value())
        dat1 = self.data[0][iz]
        dat2 = self.data[0][iz + 1]
        for i in range(ni):
            c = (i + 1) / float(ni + 1)
            print(c)
            dat = c * dat1 + (1.0 - c) * dat2
            # dat[:100,:] = dat1[:100,:]
            # dat[100:,:] = dat2[100:,:]
            self.data[0].insert(iz + 1, dat)
            self.data_orig[0].insert(iz + 1, dat)
            self.slice_lengths.insert(iz + 1, self.slice_lengths[iz])
            self.shifts[i].insert(iz + 1, [0, 0])
            self.fnames.insert(iz + 1, "c%1.3f" % c)
        self.bxZ.setRange(0, len(self.data[0]) - 1)
        print("slice_lengths = ", self.slice_lengths)

    def saveData(self):
        if self.tip == 0:
            save_name = self.txPath_CO.text()
        else:
            save_name = self.txPath_Xe.text()
        if save_name[-1] != "/":
            save_name += "/"

        save_name += "data_" + self.bxTip.currentText() + ".npz"

        self.slices_to_save = str(self.txSliceSave.text())
        if self.slices_to_save:
            print("slices_to_save = ", self.slices_to_save)
            slices_nums = [s.strip() for s in re.split(r"[,;]+| ,", self.slices_to_save) if s]
            # print 'slices_nums = ', slices_nums
            linearrframes = [int(i) for i in slices_nums if "-" not in i]
            linearrdiapasones = sum([list(range(int(i.split("-")[0]), int(i.split("-")[1]) + 1)) for i in slices_nums if "-" in i], [])
            # print 'linearrframes = ', linearrframes
            # print 'linearrdiapasones = ', linearrdiapasones

            linearrframes.extend(linearrdiapasones)
            linearrframes = list(set(linearrframes))
            slices_indexes = [int(i) for i in linearrframes]

            lengthX = self.marged_size[self.min_margin_size_ind][0] * self.max_length[self.tip][0] / self.data[self.tip][0].shape[0]
            lengthY = self.marged_size[self.min_margin_size_ind][1] * self.max_length[self.tip][1] / self.data[self.tip][0].shape[1]
            print("lnegth yx:  ", lengthX, lengthY)

            try:
                arr = np.array(self.data[self.tip])
                startx = self.margins[self.tip][0]
                starty = self.margins[self.tip][1]
                endx = startx + self.margins[self.tip][2]
                endy = starty + self.margins[self.tip][3]

                arr = arr[slices_indexes, starty:endy, startx:endx]

                arr = arr.transpose((2, 1, 0))  # save array with AFM data in (x,y,z) format
                print("dat.shape ", arr.shape)
                np.savez_compressed(save_name, data=arr, lengthX=lengthX, lengthY=lengthY)
                print(f"Data {self.bxTip.currentText()} saved to {save_name}.")
            except:
                QtWidgets.QMessageBox.critical(self, "Error", "Slices selected to save have incorrect range.\nPlease correct indexes for output.")

        else:
            arr = np.array(self.data[self.tip])

            startx = self.margins[self.tip][0]
            starty = self.margins[self.tip][1]
            endx = startx + self.margins[self.tip][2]
            endy = starty + self.margins[self.tip][3]
            arr = arr[:, starty:endy, startx:endx]

            lengthX = self.marged_size[self.min_margin_size_ind][0] * self.max_length[self.tip][0] / self.data[self.tip][0].shape[0]
            lengthY = self.marged_size[self.min_margin_size_ind][1] * self.max_length[self.tip][1] / self.data[self.tip][0].shape[1]
            print("lnegth yx:  ", lengthX, lengthY)

            arr = arr.transpose((2, 1, 0))  # save array with AFM data in (x,y,z) format
            print("dat.shape ", arr.shape)
            np.savez_compressed(save_name, data=arr, lengthX=lengthX, lengthY=lengthY)
            print(f"Data {self.bxTip.currentText()} saved to {save_name}.")

    def loadNPZ(self):
        self.path = self.txPath.text()
        if self.path[-1] != "/":
            self.path += "/"

        # load image data from data.npz
        data = []
        data2 = []
        npzfile = np.load(self.path + "data.npz")
        data = npzfile["data"]
        lenX = npzfile["lengthX"].astype(float).item()
        lenY = npzfile["lengthY"].astype(float).item()
        self.slice_lengths = [[lenX, lenY]] * data.shape[2]
        self.fnames = [""] * data.shape[2]
        print("self.slice_lengths loaded from npz")

        data = data.transpose((2, 1, 0))

        self.data = [s for s in data]

        data2 = copy.copy(self.data)
        self.data2 = data2  # np.reshape(z, (5,5)) #data
        print("data npz loaded")

        # load meta data from data.pickle about file names, shifts, lengths

        self.shifts = [[0, 0] for i in range(len(self.data))]

        self.margins = [0, 0, 0, 0]
        self.bxZ.setRange(0, len(self.data) - 1)
        self.bxChannel.setEnabled(False)
        iz = int(self.bxZ.value())
        self.bxX.setValue(self.shifts[iz][0])
        self.bxY.setValue(self.shifts[iz][1])

        self.marginX0.blockSignals(True)
        self.marginX0.setValue(self.margins[0])
        self.marginX0.blockSignals(False)
        self.marginW.blockSignals(True)
        self.marginW.setValue(self.margins[2])
        self.marginW.blockSignals(False)
        self.marginY0.blockSignals(True)
        self.marginY0.setValue(self.margins[1])
        self.marginY0.blockSignals(False)
        self.marginH.blockSignals(True)
        self.marginH.setValue(self.margins[3])
        self.marginH.blockSignals(False)
        self.updateDataView()
        self.bxChannel.setEnabled(True)
        self.bxZ.setEnabled(True)
        self.bxX.setEnabled(True)
        self.bxY.setEnabled(True)
        self.btMagicAll.setEnabled(True)
        self.btMagic.setEnabled(True)
        self.btSaveImg.setEnabled(True)
        self.bxNi.setEnabled(True)
        self.btInterp.setEnabled(True)
        self.marginX0.setEnabled(True)
        self.marginW.setEnabled(True)
        self.marginY0.setEnabled(True)
        self.marginH.setEnabled(True)
        self.btSave.setEnabled(True)
        self.txSliceSave.setEnabled(True)
        self.checkbox.setEnabled(True)

    def saveImg(self):
        n = len(self.data)
        plt.figure(figsize=(n * 5, 5))
        for i in range(n):
            print(i)
            plt.subplot(1, n, i + 1)
            plt.imshow(self.data[i], origin="lower")  # ,cmap='gray')
            print("image path = ", self.path + "data.png")
        plt.savefig(self.path + "data.png", bbox_inches="tight")

    def shiftData(self):
        print("shiftData")
        iz = int(self.bxZ.value())
        ix = int(self.bxX.value())
        dix = ix - self.shifts[self.tip][iz][0]
        self.shifts[self.tip][iz][0] = ix
        iy = int(self.bxY.value())
        diy = iy - self.shifts[self.tip][iz][1]
        self.shifts[self.tip][iz][1] = iy
        print("self.original[iz]=", self.data_orig[self.tip][iz][:3, :3])
        print("dix,diy=", dix, diy)

        print(self.shifts[self.tip])
        image = self.data_orig[self.tip][iz]
        self.data[self.tip][iz] = nimg.shift(image, (iy, ix), order=3, mode="mirror")
        # self.data[iz] = np.roll( self.data[iz], dix, axis=0 )

        # self.data[iz] = np.roll( self.data[iz], diy, axis=1 )

        self.updateDataView()

    def marginData(self):
        int(self.bxZ.value())

        imarginx0 = int(self.marginX0.value())
        imarginW = int(self.marginW.value())
        print("self.data[self.tip][0].shape = ", self.data[self.tip][0].shape)
        print("imarginW+imarginx0 = ", imarginW + imarginx0)
        print("self.margins[self.tip] = ", self.margins[self.tip])
        if imarginW + imarginx0 < self.data[self.tip][0].shape[0]:
            print(f"good value. changed")
            self.margins[self.tip][0] = imarginx0
            self.margins[self.tip][2] = imarginW
            self.margins[1 - self.tip][2] = imarginW
        else:
            print(f"we overloaded value x, so set {self.margins[self.tip][2]}")
            self.marginX0.setValue(self.margins[self.tip][0])
            self.marginW.setValue(self.margins[self.tip][2])

            imarginx0 = int(self.marginX0.value())
            imarginW = int(self.marginW.value())

        imarginy0 = int(self.marginY0.value())
        imarginH = int(self.marginH.value())
        if imarginH + imarginy0 < self.data[self.tip][0].shape[1]:
            self.margins[self.tip][1] = imarginy0
            self.margins[self.tip][3] = imarginH
            self.margins[1 - self.tip][3] = imarginH
        else:
            self.marginY0.setValue(self.margins[self.tip][1])
            self.marginH.setValue(self.margins[self.tip][3])

            imarginy0 = int(self.marginY0.value())
            imarginH = int(self.marginH.value())
        image_shape = self.data[self.tip][0].shape
        self.marged_size[0] = [imarginW, imarginH]
        self.marged_size[1] = [imarginW, imarginH]
        print("image_shape =", image_shape)

        print("margins = ", self.margins[self.tip])
        print("slice_lengths = ", self.slice_lengths[self.tip])

        self.updateDataView()

    def ChkBxGrid(self):
        # print "grid selector      ", self.checkbox.checkState()
        self.updateDataView()

    def selectDataView(self):
        iz = int(self.bxZ.value())
        print(" selectDataView iz,ix,iy ", iz, self.shifts[self.tip][iz][0], self.shifts[self.tip][iz][1])
        self.bxX.blockSignals(True)
        self.bxX.setValue(self.shifts[self.tip][iz][0])
        self.bxX.blockSignals(False)
        self.bxY.blockSignals(True)
        self.bxY.setValue(self.shifts[self.tip][iz][1])
        self.bxY.blockSignals(False)

        print("selectDataView bxXY      ", self.bxX.value(), self.bxY.value())
        self.updateDataView()

    def selectAlpha(self):
        self.updateDataView()

    def selectRotate(self):
        if self.tip == self.ind_biggest_AFM:
            irot = int(self.bxAngle.value())
            delta_rot = irot - self.cur_rotation
            if delta_rot != 0:
                for j in range(len(self.data[self.tip])):
                    self.data[self.tip][j] = nimg.rotate(self.data[self.tip][j], delta_rot, reshape=False, mode="mirror")
                    self.data_orig[self.tip][j] = nimg.rotate(self.data_orig[self.tip][j], delta_rot, reshape=False, mode="mirror")
                self.cur_rotation = irot
                self.updateDataView()

    def selectDataChannel(self):
        ichannel = int(self.bxChannel.value())
        self.loadData()
        print(" selectDataChannel ", ichannel)

        self.updateDataView()

    def selectDataTip(self, tip_selector):
        if tip_selector == "CO":
            self.tip = 0
        else:
            self.tip = 1
        iz = min(int(self.bxZ.value()), len(self.data[self.tip]) - 1)
        self.bxZ.setRange(0, len(self.data[self.tip]) - 1)
        self.marginX0.blockSignals(True)
        self.marginX0.setValue(self.margins[self.tip][0])
        self.marginX0.blockSignals(False)
        self.marginW.blockSignals(True)
        self.marginW.setValue(self.margins[self.tip][2])
        self.marginW.blockSignals(False)
        self.marginY0.blockSignals(True)
        self.marginY0.setValue(self.margins[self.tip][1])
        self.marginY0.blockSignals(False)
        self.marginH.blockSignals(True)
        self.marginH.setValue(self.margins[self.tip][3])
        self.marginH.blockSignals(False)

        self.bxX.blockSignals(True)
        self.bxX.setValue(self.shifts[self.tip][iz][0])
        self.bxX.blockSignals(False)
        self.bxY.blockSignals(True)
        self.bxY.setValue(self.shifts[self.tip][iz][1])
        self.bxY.blockSignals(False)
        if self.tip == self.ind_biggest_AFM:
            self.bxAngle.setValue(self.cur_rotation)
            self.bxAngle.setEnabled(True)
            self.bxAlpha.setEnabled(True)
        else:
            self.bxAngle.setValue(0)
            self.bxAngle.setEnabled(False)
            self.bxAlpha.setEnabled(False)

        self.updateDataView()

    def updateDataView(self):
        iz = int(self.bxZ.value())
        grid_selector = int(self.checkbox.checkState())
        alpha = float(self.bxAlpha.value())
        print("self.cur_rotation = ", self.cur_rotation)
        print("iz = ", iz)
        print("len slice lengths = ", len(self.slice_lengths[self.tip]))
        # t1 = time.clock()
        # iz = self.selectDataView()
        # print 'self.margins', self.margins
        # f =np.fft.fft2(self.data[iz])
        # f_shift = np.fft.fftshift(f)
        # print 'f_shift.shape = ',f_shift.shape
        # f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
        # f_abs = np.abs(f_shift) + 1 # lie between 1 and 1e6
        # f_bounded = 20 * np.log(f_abs)
        # f_img = 255 * f_bounded / np.max(f_bounded)
        # f_img = f_img.astype(np.uint8)
        if self.tip == self.ind_biggest_AFM:
            big_len_image = self.data[self.tip][iz].copy()

            imarginx0 = self.margins[1 - self.tip][0]
            imarginW = self.margins[1 - self.tip][2]
            imarginy0 = self.margins[1 - self.tip][1]
            imarginH = self.margins[1 - self.tip][3]

            small_len_image = self.data[1 - self.tip][-2][imarginy0 : imarginy0 + imarginH, imarginx0 : imarginx0 + imarginW]

            big_len_image = (big_len_image - np.mean(big_len_image)) / (np.std(big_len_image) + 1e-5)
            small_len_image = (small_len_image - np.mean(small_len_image)) / (np.std(small_len_image) + 1e-5)

            big_len_image[
                self.margins[self.tip][1] : self.margins[self.tip][1] + self.margins[self.tip][3], self.margins[self.tip][0] : self.margins[self.tip][0] + self.margins[self.tip][2]
            ] = small_len_image

            main_afm_image = self.data[self.tip][iz]
            main_afm_image = (main_afm_image - np.mean(main_afm_image)) / (np.std(main_afm_image) + 1e-5)
            self.figCan.plotSlice2(main_afm_image, self.fnames[self.tip][iz], self.margins[self.tip], grid_selector, self.slice_lengths[self.tip][iz], big_len_image, alpha)

        else:
            self.figCan.plotSlice2(self.data[self.tip][iz], self.fnames[self.tip][iz], self.margins[self.tip], grid_selector, self.slice_lengths[self.tip][iz])


def main():
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())


if __name__ == "__main__":
    main()
