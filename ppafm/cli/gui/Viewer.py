#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases


import os
import sys

import matplotlib
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import ppafm.GUIWidgets as guiw
from ppafm import PPPlot, io

matplotlib.use("Qt5Agg")


class ApplicationWindow(QtWidgets.QMainWindow):
    data = None
    label = ""

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")
        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        # l0 = QtWidgets.QVBoxLayout(self.main_widget)
        l0 = QtWidgets.QVBoxLayout(self.main_widget)
        self.figCan = guiw.FigImshow(parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100)
        l0.addWidget(self.figCan)
        # l0 = QtWidgets.QVBoxLayout(self.main_widget); l00.addLayout(l0);

        # -------------- Potential
        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        self.leDir = wg = QtWidgets.QLineEdit()
        wg.returnPressed.connect(self.reloadAll)
        vb.addWidget(wg)

        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        self.leFile1 = wg = QtWidgets.QLineEdit()
        wg.returnPressed.connect(lambda: self.load(0))
        vb.addWidget(wg)
        self.bxCoef1 = wg = QtWidgets.QDoubleSpinBox()
        wg.setRange(-10.0, 10.0)
        wg.setValue(1.0)
        wg.setSingleStep(0.05)
        wg.valueChanged.connect(self.updateLincomb)
        vb.addWidget(wg)

        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        self.leFile2 = wg = QtWidgets.QLineEdit()
        wg.returnPressed.connect(lambda: self.load(1))
        vb.addWidget(wg)
        self.bxCoef2 = wg = QtWidgets.QDoubleSpinBox()
        wg.setRange(-10.0, 10.0)
        wg.setValue(0.0)
        wg.setSingleStep(0.05)
        wg.valueChanged.connect(self.updateLincomb)
        vb.addWidget(wg)

        vb = QtWidgets.QHBoxLayout()
        l0.addLayout(vb)
        self.bxZ = wg = QtWidgets.QSpinBox()
        wg.setRange(0, 300)
        wg.setSingleStep(1)
        wg.setValue(10)
        wg.valueChanged.connect(self.updateDataView)
        vb.addWidget(wg)

        self.items = [
            # [ None, None, self.leFile1, self.btLoad1, self.bxCoef1 ],
            # [ None, None, self.leFile2, self.btLoad2, self.bxCoef2 ],
            [None, None, self.leFile1, self.bxCoef1],
            [None, None, self.leFile2, self.bxCoef2],
        ]

        # print self.bxCoef1.value()
        self.leDir.setText("/home/prokop/Desktop/WORK/Phtalocyanine_distortion/Sim/dz2-Morse/")
        self.leFile1.setText("singlet/FFLJ_z.xsf")
        self.leFile2.setText("singlet/FFel_z.xsf")

        self.figCurv = guiw.PlotWindow(parent=self, width=5, height=4, dpi=100)

    def reloadAll(self):
        for item in self.items:
            item[0] = None
        self.tryLoadEmpty()

    def tryLoadEmpty(self):
        print(" tryLoadEmpty ")
        for i, item in enumerate(self.items):
            if item[0] is None:
                self.load(i)

    def load(self, idata):
        item = self.items[idata]
        fname = self.leDir.text() + item[2].text()
        _, fext = os.path.splitext(fname)
        try:
            if fext == ".xsf":
                F, lvec, nDim, head = io.loadXSF(fname)
            elif fext == ".cube":
                F, lvec, nDim, head = io.loadCUBE(fname)
            item[0] = F
            item[1] = lvec
            self.updateLincomb()
        except Exception as e:
            print("cannot load file: ", fname)
            print(e)

    def updateLincomb(self):
        self.label = ""
        if self.items[0][0] is not None:
            self.data = np.zeros(self.items[0][0].shape)
            for item in self.items:
                if item[0] is not None:
                    coef = item[3].value()
                    print(coef)
                    self.data += item[0] * coef
                    self.label += f" + {coef:g} * ({item[2].text()}) "
            self.updateDataView()

    def clickImshow(self, ix, iy):
        print("ix, iy", ix, iy)
        ys = self.data[:, iy, ix]
        self.figCurv.show()
        label = self.label + ("_%i_%i" % (ix, iy))
        lvec = self.items[0][1]
        z0 = lvec[3][0]
        xs = np.linspace(z0, z0 + lvec[3][2], len(ys), endpoint=False)
        self.figCurv.figCan.plotDatalines((xs, ys, label))

    def updateDataView(self):
        if self.data is None:
            self.tryLoadEmpty()
            self.updateLincomb()
        iz = self.bxZ.value()
        try:
            self.figCan.plotSlice(self.data[iz])
        except:
            print("cannot plot slice #", iz)


def main():
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())


if __name__ == "__main__":
    main()
