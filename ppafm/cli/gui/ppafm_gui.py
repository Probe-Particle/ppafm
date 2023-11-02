#!/usr/bin/env python3

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases

import os
import sys
import time
import traceback
from argparse import ArgumentParser
from enum import Enum

import matplotlib
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import ppafm.common as PPU
import ppafm.GUIWidgets as guiw
import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu
from ppafm import PPPlot, io
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.field import ElectronDensity, HartreePotential, TipDensity

matplotlib.use("Qt5Agg")


Multipoles = Enum("Multipoles", "s pz dz2")

# fmt: off
Presets = {
    'CO (Z8, dz2, Q-0.1, K0.25)': {
        'Z': 8,
        'Multipole': 'dz2',
        'Q': -0.1,
        'Sigma': 0.71,
        'K': [0.25, 0.25, 30.0],
        'EqPos': [0.0, 0.0, 3.0]
    },
    'Xe (Z54, s, Q0.3, K0.25)': {
        'Z': 54,
        'Multipole': 's',
        'Q': 0.3,
        'Sigma': 0.71,
        'K': [0.25, 0.25, 30.0],
        'EqPos': [0.0, 0.0, 3.0]
    },
    'Cl (Z17, s, Q-0.3, K0.50)': {
        'Z': 17,
        'Multipole': 's',
        'Q': -0.3,
        'Sigma': 0.71,
        'K': [0.50, 0.50, 30.0],
        'EqPos': [0.0, 0.0, 3.0]
    }
}
# fmt: on

TTips = {
    "Preset": "Preset: Apply a probe parameter preset.",
    "Z": "Z: Probe atomic number. Determines the Lennard-Jones parameters of the force field.",
    "Multipole": "Multipole: Probe charge multipole type:\ns: monopole\npz: dipole\ndz2: quadrupole.",
    "Q": "Q: Probe charge/multipole magnitude.",
    "Sigma": "Sigma: Probe charge distribution width.",
    "point_charge": "Use point-charge approximation for probe charge distribution. Faster but less accurate.",
    "K": "K: Force constants for harmonic force holding the probe to the tip in x, y, and radial directions.",
    "EqPos": "Eq. Pos: Probe equilibrium position with respect to the tip in x, y, and radial directions. Non-zero values for x and y models asymmetry in tip adsorption.",
    "ScanStep": "Scan step: Size of pixels in x and y directions and size of oscillation step in z direction.",
    "ScanSize": "Scan size: Total size of scan region in x and y directions.",
    "ScanStart": "Scan start: bottom left position of scan region in x and y directions.",
    "Distance": "Distance: Average tip distance from the nucleus of the closest atom.",
    "Amplitude": "Amplitude: Peak-to-peak oscillation amplitude for the tip.",
    "Rotation": "Rotation: Set sample counter-clockwise rotation angle around center of atom coordinates.",
    "fdbm_V0": "FDBM V0: Prefactor in Pauli interaction integral in the full-density based model.",
    "fdbm_alpha": "FDBM alpha: Exponent in Pauli interaction integral in the full-density based model.",
    "PBCz": "z periodicity: When checked, the lattice is also periodic in z direction. This is usually not required, since the scan is aligned with the xy direction.",
    "PBC": "Periodic Boundaries: Lattice vectors for periodic images of atoms. Does not affect electrostatics calculated from a Hartree potential file, which is always assumed to be periodic.",
    "k": "k: Cantilever spring constant. Only appears as a scaling constant.",
    "f0": "f0: Cantilever eigenfrequency. Only appears as a scaling constant.",
    "z_steps": "z steps: Number of steps in the df approach curve in z direction when clicking on image.",
    "df_colorbar": "Colorbar: Add a colorbar of df values to plot.",
    "df_range": "df range: Minimum and maximum df value in colorbar.",
    "df_reset": "Reset Range: Reset df colorbar range.",
    "view_geom": "View Geometry: Show system geometry in ASE GUI.",
    "edit_geom": "Edit Geometry: Edit the positions, atomic numbers, and charges of atoms.",
    "view_ff": "View Forcefield: View forcefield components in a separate window.",
    "edit_ff": "Edit Forcefield: Edit Lennard-Jones parameters of forcefield.",
}


def parse_args():
    parser = ArgumentParser(description="Probe Particle Model graphical user interface")
    # fmt: off
    parser.add_argument("input", nargs='*', help="Optional input file(s). The first file is the main input file containing the xyz geometry or Hartree potential. The subsequent optional files, in order, are the sample electron density, tip electron density, and tip electron delta density.")
    parser.add_argument("-d", "--device",       action="store", type=int, default=0,  help="Choose OpenCL device.")
    parser.add_argument("-l", "--list-devices", action="store_true",                  help="List available OpenCL devices and exit.")
    parser.add_argument("-v", '--verbosity',    action="store", type=int, default=0,  help="Set verbosity level (0-2).")
    # fmt: on
    args = parser.parse_args()
    if args.input:
        inputs = {"main_input": args.input[0]}
        if len(args.input) > 1:
            inputs["rho_sample"] = args.input[1]
        if len(args.input) > 2:
            inputs["rho_tip"] = args.input[2]
        if len(args.input) > 3:
            inputs["rho_tip_delta"] = args.input[3]
        args.input = inputs

    return args


class ApplicationWindow(QtWidgets.QMainWindow):
    sw_pad = 4.0  # Default padding for scan window on each side of the molecule in xy plane
    zoom_step = 1.0  # How much to increase/reduce scan size on zoom
    df_range = (-1, 1)  # min and max df value in colorbar
    fixed_df_range = False  # Keep track if df range was fixed by user or should be set automatically

    def __init__(self, input_files=None, device=0, verbose=0):
        self.df = None
        self.xyzs = None
        self.Zs = None
        self.qs = None
        self.rho_sample = None
        self.rho_tip = None
        self.rho_tip_delta = None
        self.sample_lvec = None
        self.rot = np.eye(3)
        self.df_points = []

        # Initialize OpenCL environment on chosen device and create an afmulator instance to use for simulations
        oclu.init_env(device)
        self.afmulator = AFMulator()

        # Set verbosity level to same value everywhere
        if verbose > 0:
            print(f"Verbosity level = {verbose}")
        self.verbose = verbose
        self.afmulator.verbose = verbose
        self.afmulator.forcefield.verbose = verbose
        self.afmulator.scanner.verbose = verbose
        FFcl.bRuntime = verbose > 1

        # --- init QtMain
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Probe Particle Model")
        self.main_widget = QtWidgets.QWidget(self)
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        l1 = QtWidgets.QVBoxLayout(self.main_widget)
        l00.addLayout(l1, 2)
        self.figCan = guiw.FigImshow(parentWiget=self.main_widget, parentApp=self, width=5, height=4, dpi=100, verbose=verbose)
        l1.addWidget(self.figCan)
        self.resize(1200, 600)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        # -------- Status Bar
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.dim_label = QtWidgets.QLabel("")
        self.status_bar.addPermanentWidget(self.dim_label)
        l1.addWidget(self.status_bar)

        # -------- Settings
        self.l0 = QtWidgets.QVBoxLayout(self.main_widget)
        l00.addLayout(self.l0, 1)
        self._create_probe_settings_ui()
        _separator_line(self.l0)
        self._create_scan_settings_ui()
        _separator_line(self.l0)
        self._create_pbc_settings_ui()
        _separator_line(self.l0)
        self._create_df_settings_ui()
        _separator_line(self.l0)
        self._create_buttons_ui()

        # Create curve plotting window
        self.figCurv = guiw.PlotWindow(parent=self, width=5, height=4, dpi=100)

        if input_files:
            self.loadInput(**input_files)

    def status_message(self, msg):
        self.status_bar.showMessage(msg)
        self.status_bar.repaint()

    def setScanWindow(self, scan_size, scan_start, step, distance, amplitude):
        """Set scan window in AFMulator and update input fields"""

        if self.xyzs is None:
            return

        # Make scan size and amplitude multiples of the step size
        scan_dim = np.round([scan_size[0] / step[0] + 1, scan_size[1] / step[1] + 1, amplitude / step[2]]).astype(np.int32)
        scan_size = (scan_dim[:2] - 1) * step[:2]
        amplitude = scan_dim[2] * step[2]
        z_extra_steps = self.bxdfst.value() - 1
        scan_dim[2] += z_extra_steps
        z = self.xyzs[:, 2].max() + distance
        z_min = z - amplitude / 2
        z_max = z + amplitude / 2 + z_extra_steps * step[2]
        # fmt: off
        scan_window = (
            (scan_start[0]               , scan_start[1]               , z_min),
            (scan_start[0] + scan_size[0], scan_start[1] + scan_size[1], z_max)
        )  # fmt: on
        self.afmulator.kCantilever = self.bxCant_K.value() * 1000
        self.afmulator.f0Cantilever = self.bxCant_f0.value() * 1000
        if self.verbose > 0:
            print("setScanWindow", step, scan_size, scan_start, scan_dim, scan_window)

        # Set new values to the fields
        guiw.set_widget_value(self.bxSSx, scan_size[0])
        guiw.set_widget_value(self.bxSSy, scan_size[1])
        guiw.set_widget_value(self.bxSCx, scan_start[0])
        guiw.set_widget_value(self.bxSCy, scan_start[1])
        guiw.set_widget_value(self.bxD, distance)
        guiw.set_widget_value(self.bxA, amplitude)

        # Set scan size and amplitude increments to match the set step size
        self.bxSSx.setSingleStep(step[0])
        self.bxSSy.setSingleStep(step[1])
        self.bxA.setSingleStep(step[2])

        # Set new scan window and dimension in AFMulator, and infer FF lvec from the scan window
        self.afmulator.setScanWindow(scan_window, tuple(scan_dim), scan_dim[2] - z_extra_steps)
        self.afmulator.setLvec()

        # Update status bar info
        ff_dim = self.afmulator.forcefield.nDim
        self.dim_label.setText(f"Scan dim: {scan_dim[0]}x{scan_dim[1]}x{scan_dim[2]} | " f"FF dim: {ff_dim[0]}x{ff_dim[1]}x{ff_dim[2]}")
        if self.verbose > 0:
            print("lvec:\n", self.afmulator.forcefield.nDim, self.afmulator.lvec)

    def scanWindowFromGeom(self):
        """Infer and set scan window from current geometry"""
        if self.xyzs is None:
            return
        scan_size = self.xyzs[:, :2].max(axis=0) - self.xyzs[:, :2].min(axis=0) + 2 * self.sw_pad
        scan_size = np.minimum(scan_size, [25, 25])  # Let's not make it automatically too large, so that we don't run out of memory
        scan_start = (self.xyzs[:, :2].max(axis=0) + self.xyzs[:, :2].min(axis=0)) / 2 - scan_size / 2
        step = np.array([self.bxStepX.value(), self.bxStepY.value(), self.bxStepZ.value()])
        distance = self.bxD.value()
        amplitude = self.bxA.value()
        self.setScanWindow(scan_size, scan_start, step, distance, amplitude)

    def updateScanWindow(self):
        """Get scan window from input fields and update"""
        if self.xyzs is None:
            return
        scan_size = np.array([self.bxSSx.value(), self.bxSSy.value()])
        scan_start = np.array([self.bxSCx.value(), self.bxSCy.value()])
        step = np.array([self.bxStepX.value(), self.bxStepY.value(), self.bxStepZ.value()])
        distance = self.bxD.value()
        amplitude = self.bxA.value()
        self.setScanWindow(scan_size, scan_start, step, distance, amplitude)
        self.update()

    def updateRotation(self):
        """Get rotation from input field and update"""
        a = self.bxRot.value() / 180 * np.pi
        # fmt: off
        self.rot = np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a),  np.cos(a), 0],
            [        0,          0, 1]
        ])  # fmt: on
        if self.verbose > 0:
            print("updateRotation", a, self.rot)
        self.update()

    def updateParams(self, preset_none=True):
        """Get parameter values from input fields and update"""

        if self.xyzs is None:
            return

        if preset_none:
            guiw.set_widget_value(self.slPreset, -1)

        Q = self.bxQ.value()
        sigma = self.bxS.value()
        multipole = self.slMultipole.currentText()
        tipStiffness = [self.bxKx.value(), self.bxKy.value(), 0.0, self.bxKr.value()]
        tipR0 = [self.bxP0x.value(), self.bxP0y.value(), self.bxP0r.value()]
        use_point_charge = self.bxPC.isChecked()
        A_pauli = self.bxV0.value()
        B_pauli = self.bxAlpha.value() if (self.rho_sample is not None) else 1.0

        if multipole == "s":
            Qs = [Q, 0, 0, 0]
            QZs = [0, 0, 0, 0]
        elif multipole == "pz":
            Qs = [Q, -Q, 0, 0]
            QZs = [sigma, -sigma, 0, 0]
        elif multipole == "dz2":
            Qs = [Q, -2 * Q, Q, 0]
            QZs = [sigma, 0, -sigma, 0]

        if self.verbose > 0:
            print("updateParams", Q, sigma, multipole, tipStiffness, tipR0, use_point_charge, A_pauli, B_pauli, type(self.qs))

        self.afmulator.iZPP = int(self.bxZPP.value())
        if self.rho_sample is not None:  # Use FDBM
            if self.afmulator.B_pauli != B_pauli:
                self.afmulator.setRho(self.rho_tip, B_pauli=B_pauli)
        else:
            if self.rho_tip is None:  # else we just keep the tip density loaded from file
                if use_point_charge:
                    self.afmulator.setQs(Qs, QZs)
                    self.afmulator.setRho(None, sigma)
                elif isinstance(self.qs, HartreePotential):
                    self.afmulator.setRho({multipole: Q}, sigma)
        self.afmulator.scanner.stiffness = np.array(tipStiffness, dtype=np.float32) / -PPU.eVA_Nm
        self.afmulator.tipR0 = tipR0
        self.afmulator.A_pauli = A_pauli

        self.update()

    def setDfRange(self, df_range):
        """Set df range in input boxes"""
        guiw.set_widget_value(self.bxDfMin, df_range[0])
        guiw.set_widget_value(self.bxDfMax, df_range[1])

    def dfRangeFromData(self):
        """Set colorbar df range from current df data"""
        if self.df is None:
            return
        self.df_range = (self.df[:, :, -1].min(), self.df[:, :, -1].max())
        self.setDfRange(self.df_range)

    def updateDfRange(self):
        """Get df range from input field and update plot"""
        self.fixed_df_range = True
        self.df_range = (self.bxDfMin.value(), self.bxDfMax.value())
        self.updateDataView()

    def resetDfRange(self):
        """Reset df range to min-max values in current image and update plot"""
        self.fixed_df_range = False
        self.dfRangeFromData()
        self.updateDataView()

    def updateDfColorbar(self):
        """Get colorbar state and update plot"""
        if self.bxDfCbar.isChecked():
            self.bxDfMin.setDisabled(False)
            self.bxDfMax.setDisabled(False)
            self.btDfReset.setDisabled(False)
        else:
            self.fixed_df_range = False
            self.dfRangeFromData()
            self.bxDfMin.setDisabled(True)
            self.bxDfMax.setDisabled(True)
            self.btDfReset.setDisabled(True)
        self.updateDataView()

    def setPBC(self, lvec, enabled):
        """Set periodic boundary condition lattice"""

        if self.verbose > 0:
            print("setPBC", lvec, enabled)

        if enabled:
            self.sample_lvec = lvec
            if self.bxPBCz.isChecked():
                self.afmulator.npbc = (1, 1, 1)
            else:
                self.afmulator.npbc = (1, 1, 0)
        else:
            self.sample_lvec = None
            self.afmulator.npbc = (0, 0, 0)

        # Set check-box state
        guiw.set_widget_value(self.bxPBC, enabled)

        # Set lattice vector values
        if lvec is not None:
            guiw.set_widget_value(self.bxPBCAx, lvec[0][0])
            guiw.set_widget_value(self.bxPBCAy, lvec[0][1])
            guiw.set_widget_value(self.bxPBCAz, lvec[0][2])
            guiw.set_widget_value(self.bxPBCBx, lvec[1][0])
            guiw.set_widget_value(self.bxPBCBy, lvec[1][1])
            guiw.set_widget_value(self.bxPBCBz, lvec[1][2])
            guiw.set_widget_value(self.bxPBCCx, lvec[2][0])
            guiw.set_widget_value(self.bxPBCCy, lvec[2][1])
            guiw.set_widget_value(self.bxPBCCz, lvec[2][2])

        # Disable lattice vector boxes if PBC not enabled
        for bx in [self.bxPBCAx, self.bxPBCAy, self.bxPBCAz, self.bxPBCBx, self.bxPBCBy, self.bxPBCBz, self.bxPBCCx, self.bxPBCCy, self.bxPBCCz]:
            bx.setDisabled(not enabled)

    def updatePBC(self):
        """Get PBC lattice from from input fields and update"""
        lvec = np.array(
            [
                [self.bxPBCAx.value(), self.bxPBCAy.value(), self.bxPBCAz.value()],
                [self.bxPBCBx.value(), self.bxPBCBy.value(), self.bxPBCBz.value()],
                [self.bxPBCCx.value(), self.bxPBCCy.value(), self.bxPBCCz.value()],
            ]
        )
        toggle = self.bxPBC.isChecked()
        self.setPBC(lvec, toggle)
        self.update()

    def applyPreset(self, update=True):
        """Get current preset, apply parameters, and update"""
        preset = Presets[self.slPreset.currentText()]
        if "Z" in preset:
            guiw.set_widget_value(self.bxZPP, preset["Z"])
        if "Q" in preset:
            guiw.set_widget_value(self.bxQ, preset["Q"])
        if "Sigma" in preset:
            guiw.set_widget_value(self.bxS, preset["Sigma"])
        if "K" in preset:
            guiw.set_widget_value(self.bxKx, preset["K"][0])
            guiw.set_widget_value(self.bxKy, preset["K"][1])
            guiw.set_widget_value(self.bxKr, preset["K"][2])
        if "EqPos" in preset:
            guiw.set_widget_value(self.bxP0x, preset["EqPos"][0])
            guiw.set_widget_value(self.bxP0y, preset["EqPos"][1])
            guiw.set_widget_value(self.bxP0r, preset["EqPos"][2])
        if "Multipole" in preset:
            guiw.set_widget_value(self.slMultipole, self.slMultipole.findText(preset["Multipole"]))
        if update:
            self.updateParams(preset_none=False)

    def update(self):
        """Run simulation, and show the result"""
        if self.xyzs is None:
            return
        if self.verbose > 1:
            t0 = time.perf_counter()
        self.status_message("Running simulation...")
        try:
            self.df = self.afmulator(
                self.xyzs,
                self.Zs,
                self.qs,
                rho_sample=self.rho_sample,
                sample_lvec=self.sample_lvec,
                rot=self.rot,
            )
        except Exception:
            traceback.print_exc()
            guiw.show_warning(self, f"Error during simulation! Error message:\n{traceback.format_exc()}", "Simulation error!")
        if self.verbose > 1:
            print(f"AFMulator total time [s]: {time.perf_counter() - t0}")
        self.status_message("Updating plot...")
        self.updateDataView()
        if self.FFViewer.isVisible():
            self.status_message("Updating Force field viewer...")
            self.FFViewer.updateFF()
            self.FFViewer.updateView()
        self.status_message("Ready")

    def loadInput(self, main_input, rho_sample=None, rho_tip=None, rho_tip_delta=None):
        """Load input file(s) and show the result

        Arguments:
            main_input: str. Main input file containing xyz geometry or Hartree potential. xyz, POSCAR, in, xsf, or cube format.
            rho_sample: str. Optional sample electron density. xsf format.
            rho_tip: str. Optional tip electron density. xsf format.
            rho_tip_delta: str. Optional tip delta electron density. xsf format.
        """

        # If there are existing arrays in memory, release them
        if isinstance(self.qs, HartreePotential):
            self.qs.release()
        if self.rho_sample is not None:
            self.rho_sample.release()
        if self.rho_tip is not None:
            self.rho_tip.release()
        if self.rho_tip_delta is not None:
            self.rho_tip_delta.release()

        # Load input file
        file_name = os.path.split(main_input)[1].lower()
        ext = os.path.splitext(file_name)[1]
        if self.verbose > 0:
            print(f"loadInput: {main_input}, {file_name}, {ext}")
        if file_name in ["poscar", "contcar"]:
            xyzs, Zs, lvec = io.loadPOSCAR(main_input)
            qs = np.zeros(len(Zs))
            lvec = lvec[1:]
        elif ext == ".in":
            xyzs, Zs, lvec = io.loadGeometryIN(main_input)
            qs = np.zeros(len(Zs))
            lvec = lvec[1:] if len(lvec) > 0 else None
        elif ext in [".xsf", ".cube"]:
            # Scale=-1.0 for correct units of potential (V) instead of energy (eV)
            qs, xyzs, Zs = HartreePotential.from_file(main_input, scale=-1.0)
            lvec = qs.lvec[1:]
        elif ext == ".xyz":
            xyzs, Zs, qs, comment = io.loadXYZ(main_input)
            lvec = io.parseLvecASE(comment)
        else:
            raise ValueError(f"Unsupported file format for file `{main_input}`.")

        # Load auxiliary files (if any)
        if rho_sample:
            self.rho_sample, _, _ = ElectronDensity.from_file(rho_sample)
        else:
            self.rho_sample = None
        if rho_tip:
            self.rho_tip, self.xyzs_tip, self.Zs_tip = TipDensity.from_file(rho_tip)
        else:
            self.rho_tip = None
        if rho_tip_delta:
            self.rho_tip_delta, self.xyzs_tip, self.Zs_tip = TipDensity.from_file(rho_tip_delta)
        else:
            self.rho_tip_delta = None
            if self.rho_tip is not None:
                self.rho_tip_delta = self.rho_tip.subCores(self.xyzs_tip, self.Zs_tip, Rcore=0.7)
                if self.rho_sample is None:
                    # Not FDBM, no need for full tip electron density
                    self.rho_tip.release()
                    self.rho_tip = None

        if (self.rho_tip is None) and (self.rho_tip_delta is not None):
            # Not doing FDBM, just use the delta density for electrostatics
            self.rho_tip = self.rho_tip_delta
            self.rho_tip_delta = None

        if self.rho_tip is not None:
            B_pauli = 1.0 if (self.rho_sample is None) else self.bxAlpha.value()
            self.afmulator.setRho(self.rho_tip, B_pauli=B_pauli)
        if self.rho_tip_delta is not None:
            self.afmulator.setRhoDelta(self.rho_tip_delta)

        self.xyzs = xyzs
        self.Zs = Zs
        self.qs = qs
        self.df_points = []

        if self.rho_tip is not None:
            # Pick the probe particle type from the lowest atom in the tip density geometry
            lowest_ind = np.argmin(self.xyzs_tip[:, 2])
            Zpp = int(self.Zs_tip[lowest_ind])

            if Zpp != 0:  # Zs_tip can be just [0] if the file does not contain a geometry
                self.afmulator.iZPP = Zpp

                for name, preset in Presets.items():  # Try to find a matching preset
                    if preset["Z"] == Zpp:
                        if self.verbose > 0:
                            print(f"Setting preset `{name}` based on tip density file geometry.")
                        self.slPreset.setCurrentText(name)
                        self.applyPreset(update=False)
                        break
                else:
                    if self.verbose > 0:
                        print(f"Setting probe particle type `{Zpp}` based on tip density file geometry.")
                    guiw.set_widget_value(self.bxZPP, Zpp)

        self.bxPC.blockSignals(True)
        if isinstance(self.qs, HartreePotential):
            # Default to no point-charge tip for Hartree potentials
            self.bxPC.setChecked(False)
            self.bxPC.setDisabled(False)
        else:
            # Point-charge systems only support point-charge tips for now
            self.bxPC.setChecked(True)
            self.bxPC.setDisabled(True)
        self.bxPC.blockSignals(False)

        # Some tip parameters cannot be controlled when the tip density is loaded from file
        if self.rho_tip is not None:
            guiw.set_widget_value(self.bxPC, False)
            self._toggleTipControls(enabled=False)
        else:
            self._toggleTipControls(enabled=True)

        # Disable FDBM boxes when it's not in use
        if self.rho_sample is None:
            self.bxV0.setDisabled(True)
            self.bxAlpha.setDisabled(True)
        else:
            self.bxV0.setDisabled(False)
            self.bxAlpha.setDisabled(False)

        # Create geometry editor widget
        self.createGeomEditor()

        # Infer scan window from loaded geometry and run
        self.scanWindowFromGeom()
        self.setPBC(lvec, lvec is not None)
        self.updateParams(preset_none=False)

        # Set current file path to window title
        self.file_path = main_input
        if len(main_input) > 80:
            main_input = f"...{main_input[-80:]}"
        self.setWindowTitle(f"{main_input} - Probe Particle Model")

    def _toggleTipControls(self, enabled):
        for widget in [self.slMultipole, self.bxQ, self.bxS, self.bxPC]:
            widget.setDisabled(not enabled)

    def createGeomEditor(self):
        """Create a new geometry editor. Replace old one if it exists."""
        if self.geomEditor:
            self.geomEditor.deleteLater()
            self.geomEditor = None
        enable_qs = not isinstance(self.qs, HartreePotential)
        self.geomEditor = guiw.GeomEditor(len(self.xyzs), enable_qs=enable_qs, parent=self, title="Geometry Editor")

    def showGeomEditor(self):
        if self.xyzs is None:
            return
        self.geomEditor.updateValues()
        self.geomEditor.show()

    def showFFViewer(self):
        if self.xyzs is None:
            return
        self.FFViewer.updateFF()
        self.FFViewer.updateView()
        self.FFViewer.show()

    def showGeometry(self):
        try:
            from ase import Atoms
            from ase.visualize import view
        except ModuleNotFoundError:
            print("No ase installation detected. Cannot show molecule geometry.")
            if self.verbose > 1:
                traceback.print_exc()
            return
        atoms = Atoms(
            positions=self.xyzs,
            numbers=self.Zs,
            cell=self.sample_lvec,
            pbc=self.afmulator.npbc,
        )
        view(atoms)

    def openFile(self):
        self.openFileDialog.exec()
        file_paths = self.openFileDialog.paths
        if self.verbose > 0:
            print("openFile", file_paths)
        if file_paths is None:
            return
        self.status_message("Opening file(s)...")
        try:
            self.loadInput(**file_paths)
        except:
            traceback.print_exc()
            guiw.show_warning(self, f"Ran into an error while opening a file! Error message:\n{traceback.format_exc()}", "File open error!")
            self.status_message("Ready")

    def saveFig(self):
        if self.xyzs is None:
            return
        default_path = os.path.join(os.path.split(self.file_path)[0], "df.png")
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", default_path, "Image files (*.png)")
        if fileName:
            self.status_message("Saving image...")
            fileName = guiw.correct_ext(fileName, ".png")
            if self.verbose > 0:
                print("Saving image to :", fileName)
            self.figCan.fig.savefig(fileName, bbox_inches="tight")
            self.status_message("Ready")

    def saveDataW(self):
        if self.df is None:
            return
        default_path = os.path.join(os.path.split(self.file_path)[0], "df.xsf")
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save df", default_path, "XCrySDen files (*.xsf);; WSxM files (*.xyz)")
        if not fileName:
            return
        ext = os.path.splitext(fileName)[1]
        if ext not in [".xyz", ".xsf"]:
            self.status_message("Unsupported file type in df save file path")
            print(f"Unsupported file type in df save file path `{fileName}`")
            return
        self.status_message("Saving data...")
        if self.verbose > 0:
            print(f"Saving df data to {fileName}...")
        if ext == ".xyz":
            data = self.df[:, :, -1].T
            xs = np.linspace(0, self.bxSSx.value(), data.shape[1], endpoint=False)
            ys = np.linspace(0, self.bxSSy.value(), data.shape[0], endpoint=False)
            Xs, Ys = np.meshgrid(xs, ys)
            io.saveWSxM_2D(fileName, data, Xs, Ys)
        elif ext == ".xsf":
            data = self.df.transpose(2, 1, 0)[::-1]
            sw = self.afmulator.scan_window
            size = np.array(sw[1]) - np.array(sw[0])
            size[2] -= self.afmulator.amplitude - self.bxStepZ.value()
            # fmt: off
            lvecScan = np.array([
                [sw[0][0], sw[0][1], sw[0][2] - self.bxP0r.value()],
                [size[0],       0,       0],
                [      0, size[1],       0],
                [      0,       0, size[2]],
            ])  # fmt: on
            if self.sample_lvec is not None:
                lvec = np.append([[0, 0, 0]], self.sample_lvec, axis=0)
                atomstring = io.primcoords2Xsf(self.Zs, self.xyzs.T, lvec)
            else:
                atomstring = io.XSF_HEAD_DEFAULT
            io.saveXSF(fileName, data, lvecScan, head=atomstring, verbose=0)
        else:
            raise RuntimeError("This should not happen. Missing file format check?")
        if self.verbose > 0:
            print("Done saving df data.")
        self.status_message("Ready")

    def updateDataView(self):
        if self.df is None:
            return

        t1 = time.perf_counter()

        # Plot df
        try:
            data = self.df.transpose(2, 1, 0)

            # Colorbar
            if not self.fixed_df_range:
                self.dfRangeFromData()
            cbar_range = self.df_range if self.bxDfCbar.isChecked() else None

            # Title
            z = self.afmulator.scan_window[0][2] + self.afmulator.amplitude / 2
            title = f"z = {z:.2f}Å"
            if not isinstance(self.qs, HartreePotential) and np.allclose(self.qs, 0):
                title += " (No electrostatics)"

            # xy limits
            sw = self.afmulator.scan_window
            extent = (sw[0][0], sw[1][0], sw[0][1], sw[1][1])

            # Plot
            self.figCan.plotSlice(data, -1, title=title, points=self.df_points, cbar_range=cbar_range, extent=extent)

        except Exception:
            print(f"Failed to plot df slice.\n{traceback.format_exc()}")
            guiw.show_warning(self, f"Ran into an error while plotting! Error message:\n{traceback.format_exc()}", "Plot error!")

        if self.verbose > 1:
            print(f"plotSlice time {time.perf_counter() - t1:.5f} [s]")

    def clickImshow(self, x, y):
        if self.df is None:
            return

        # Find closest index corresponding to x and y coordinates
        x_min, y_min = self.afmulator.scan_window[0][:2]
        x_step, y_step = self.bxStepX.value(), self.bxStepY.value()
        ix = int(round((x - x_min) / x_step))
        iy = int(round((y - y_min) / y_step))
        if self.verbose > 0:
            print("clickImshow", ix, iy, x, y)

        # Remember coordinates in case scan_start changes
        self.df_points.append((x, y))

        # Update line plot
        z_min = self.afmulator.scan_window[0][2] + self.afmulator.amplitude / 2
        z_max = self.afmulator.scan_window[1][2] - self.afmulator.amplitude / 2
        df_steps = self.bxdfst.value()
        zs = np.linspace(z_max, z_min, df_steps)
        ys = self.df[ix, iy, :]
        self.figCurv.show()
        self.figCurv.figCan.plotDatalines(zs, ys, f"{x:.02f}, {y:.02f}")

    def clearPoints(self):
        self.df_points = []
        self.updateDataView()

    def zoomTowards(self, x, y, zoom_direction):
        if self.verbose > 0:
            print("zoomTowards", x, y, zoom_direction)

        scan_size = np.array([self.bxSSx.value(), self.bxSSy.value()])
        scan_start = np.array([self.bxSCx.value(), self.bxSCy.value()])
        frac_coord = (np.array([x, y]) - scan_start) / scan_size
        offset = self.zoom_step * frac_coord

        if zoom_direction == "in":
            if scan_size[0] > 1.0 and scan_size[1] > 1.0:
                scan_size -= self.zoom_step
                scan_start += offset
        elif zoom_direction == "out":
            scan_size += self.zoom_step
            scan_start -= offset

        guiw.set_widget_value(self.bxSSx, scan_size[0])
        guiw.set_widget_value(self.bxSSy, scan_size[1])
        guiw.set_widget_value(self.bxSCx, scan_start[0])
        guiw.set_widget_value(self.bxSCy, scan_start[1])

        self.updateScanWindow()

    def saveParams(self):
        """
        Save all current parameters to a params.ini file. Bring up a file dialog to decide
        the file location.
        """
        if hasattr(self, "file_path"):
            default_path = os.path.join(os.path.split(self.file_path)[0], "params.ini")
        else:
            default_path = "params.ini"
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save parameters", default_path, "(*.ini)")
        if not fileName:
            return
        if self.verbose > 0:
            print(f"Saving current parameters to `{fileName}`")
        self.afmulator.save_params(fileName)
        self.status_message("Saved parameters")

    def loadParams(self):
        """
        Load all current parameters from a params.ini file. Bring up a file dialog to decide
        the file location.
        """

        if hasattr(self, "file_path"):
            default_path = os.path.join(os.path.split(self.file_path)[0], "params.ini")
        else:
            default_path = "params.ini"
        default_path = default_path if os.path.exists(default_path) else None
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open parameters file", default_path, "(*.ini)")
        if not file_path:
            return
        self.afmulator.load_params(file_path)

        # Set preset to nothing
        guiw.set_widget_value(self.slPreset, -1)

        # Set probe settings
        guiw.set_widget_value(self.bxZPP, self.afmulator.iZPP)
        if isinstance(self.afmulator._rho, dict):
            tip = list(self.afmulator._rho.keys())[0]
            guiw.set_widget_value(self.slMultipole, tip)
            guiw.set_widget_value(self.bxQ, self.afmulator._rho[tip])
            guiw.set_widget_value(self.bxS, self.afmulator.sigma)
        guiw.set_widget_value(self.bxKx, self.afmulator.scanner.stiffness[0] * -PPU.eVA_Nm)
        guiw.set_widget_value(self.bxKy, self.afmulator.scanner.stiffness[1] * -PPU.eVA_Nm)
        guiw.set_widget_value(self.bxKr, self.afmulator.scanner.stiffness[3] * -PPU.eVA_Nm)
        guiw.set_widget_value(self.bxP0x, self.afmulator.tipR0[0])
        guiw.set_widget_value(self.bxP0y, self.afmulator.tipR0[1])
        guiw.set_widget_value(self.bxP0r, self.afmulator.tipR0[2])
        if self.rho_sample is not None:  # Using FDBM
            guiw.set_widget_value(self.bxV0, self.afmulator.A_pauli)
            guiw.set_widget_value(self.bxAlpha, self.afmulator.B_pauli)
        else:
            self.afmulator.setBPauli(B_pauli=1.0)  # Not using the FDBM so make sure the tip density is not being raised to the power

        # Set scan settings
        scan_size = (self.afmulator.scan_window[1][0] - self.afmulator.scan_window[0][0], self.afmulator.scan_window[1][1] - self.afmulator.scan_window[0][1])
        scan_step = ((scan_size[0]) / (self.afmulator.scan_dim[0] - 1), (scan_size[1]) / (self.afmulator.scan_dim[1] - 1), self.afmulator.dz)
        guiw.set_widget_value(self.bxStepX, scan_step[0])
        guiw.set_widget_value(self.bxStepY, scan_step[1])
        guiw.set_widget_value(self.bxStepZ, scan_step[2])
        guiw.set_widget_value(self.bxSSx, scan_size[0])
        guiw.set_widget_value(self.bxSSy, scan_size[1])
        guiw.set_widget_value(self.bxSCx, self.afmulator.scan_window[0][0])
        guiw.set_widget_value(self.bxSCy, self.afmulator.scan_window[0][1])
        if self.xyzs is not None:
            d = self.afmulator.scan_window[0][2] + self.afmulator.amplitude / 2 - self.xyzs[:, 2].max()
            guiw.set_widget_value(self.bxD, d)
        guiw.set_widget_value(self.bxA, self.afmulator.amplitude)

        # Set PBC settings
        guiw.set_widget_value(self.bxPBCz, False)  # The CPU code actually never uses periodic copies in the z direction
        if isinstance(self.qs, HartreePotential):
            # To be consistent with the CPU scripts, we should always prioritize the sample lattice vectors
            # from the .xsf/.cube files
            self.afmulator.sample_lvec = self.qs.lvec[1:]
        self.setPBC(
            self.afmulator.sample_lvec,
            enabled=not (np.array(self.afmulator.npbc) == 0).all(),
        )

        # Set df settings
        guiw.set_widget_value(self.bxCant_K, self.afmulator.kCantilever / 1000)
        guiw.set_widget_value(self.bxCant_f0, self.afmulator.f0Cantilever / 1000)
        guiw.set_widget_value(self.bxdfst, self.afmulator.scan_dim[2] - self.afmulator.df_steps + 1)

        self.update()

    def _create_probe_settings_ui(self):
        # Title
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("Probe settings")
        lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font()
        font.setPointSize(12)
        lb.setFont(font)
        lb.setMaximumHeight(50)
        vb.addWidget(lb)

        # Presets
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("Preset")
        lb.setToolTip(TTips["Preset"])
        vb.addWidget(lb, 1)
        self.slPreset = QtWidgets.QComboBox()
        self.slPreset.addItems(Presets.keys())
        self.slPreset.currentIndexChanged.connect(self.applyPreset)
        self.slPreset.setToolTip(TTips["Preset"])
        vb.addWidget(self.slPreset, 6)

        # Tip type
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("Z")
        lb.setToolTip(TTips["Z"])
        vb.addWidget(lb, 1)
        self.bxZPP = QtWidgets.QSpinBox()
        self.bxZPP.setRange(0, 200)
        self.bxZPP.setValue(8)
        self.bxZPP.valueChanged.connect(self.updateParams)
        self.bxZPP.setToolTip(TTips["Z"])
        self.bxZPP.setKeyboardTracking(False)
        vb.addWidget(self.bxZPP, 2)

        # Charge multipole
        lb = QtWidgets.QLabel("Multipole")
        lb.setToolTip(TTips["Multipole"])
        vb.addWidget(lb, 2)
        self.slMultipole = QtWidgets.QComboBox()
        self.slMultipole.addItems([m.name for m in Multipoles])
        self.slMultipole.setCurrentIndex(self.slMultipole.findText(Multipoles.dz2.name))
        self.slMultipole.currentIndexChanged.connect(self.updateParams)
        self.slMultipole.setToolTip(TTips["Multipole"])
        vb.addWidget(self.slMultipole, 2)

        # Charge magnitude and sigma
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("Q [e]")
        lb.setToolTip(TTips["Q"])
        vb.addWidget(lb, 1)
        self.bxQ = _spin_box((-2.0, 2.0), -0.1, 0.05, self.updateParams, TTips["Q"], vb, 2)
        lb = QtWidgets.QLabel("Sigma [Å]")
        lb.setToolTip(TTips["Sigma"])
        vb.addWidget(lb, 2)
        self.bxS = _spin_box((0.0, 2.0), 0.71, 0.05, self.updateParams, TTips["Sigma"], vb, 2)

        # Point charge toggle
        lb = QtWidgets.QLabel("Point Charges")
        lb.setToolTip(TTips["point_charge"])
        vb.addWidget(lb)
        self.bxPC = QtWidgets.QCheckBox()
        self.bxPC.setChecked(True)
        self.bxPC.toggled.connect(self.updateParams)
        self.bxPC.setToolTip(TTips["point_charge"])
        vb.addWidget(self.bxPC)

        # Left-right divide for labels and input boxes
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        bxl = QtWidgets.QVBoxLayout()
        vb.addLayout(bxl, 1)
        bxr = QtWidgets.QVBoxLayout()
        vb.addLayout(bxr, 3)

        # Spring constants
        lb = QtWidgets.QLabel("K (x,y,R) [N/m]")
        lb.setToolTip(TTips["K"])
        bxl.addWidget(lb)
        vb = QtWidgets.QHBoxLayout()
        bxr.addLayout(vb)
        self.bxKx = _spin_box((0.0, 2.0), 0.25, 0.05, self.updateParams, TTips["K"], vb)
        self.bxKy = _spin_box((0.0, 2.0), 0.25, 0.05, self.updateParams, TTips["K"], vb)
        self.bxKr = _spin_box((0.0, 100.0), 30.00, 5.00, self.updateParams, TTips["K"], vb)

        # Probe equilibrium position
        lb = QtWidgets.QLabel("Eq. pos (x,y,R) [Å]")
        lb.setToolTip(TTips["EqPos"])
        bxl.addWidget(lb)
        vb = QtWidgets.QHBoxLayout()
        bxr.addLayout(vb)
        self.bxP0x = _spin_box((-2.0, 2.0), 0.0, 0.1, self.updateParams, TTips["EqPos"], vb)
        self.bxP0y = _spin_box((-2.0, 2.0), 0.0, 0.1, self.updateParams, TTips["EqPos"], vb)
        self.bxP0r = _spin_box((0.0, 10.0), 3.0, 0.1, self.updateParams, TTips["EqPos"], vb)

        # FDBM settings
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("FDBM settings:")
        vb.addWidget(lb)
        l = QtWidgets.QHBoxLayout()
        vb.addLayout(l)
        lb = QtWidgets.QLabel("V0")
        lb.setToolTip(TTips["fdbm_V0"])
        l.addWidget(lb)
        self.bxV0 = _spin_box((0.1, 999), 18.0, 1.0, self.updateParams, TTips["fdbm_V0"], l)
        lb = QtWidgets.QLabel("alpha")
        lb.setToolTip(TTips["fdbm_alpha"])
        l.addWidget(lb)
        self.bxAlpha = _spin_box((0.1, 9.9), 1.0, 0.02, self.updateParams, TTips["fdbm_alpha"], l)

    def _create_scan_settings_ui(self):
        # Title
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("Scan settings")
        lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font()
        font.setPointSize(12)
        lb.setFont(font)
        lb.setMaximumHeight(50)
        vb.addWidget(lb)

        # Left-right divide for labels and input boxes
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        bxl = QtWidgets.QVBoxLayout()
        vb.addLayout(bxl, 1)
        bxr = QtWidgets.QVBoxLayout()
        vb.addLayout(bxr, 3)

        # Scan step
        lb = QtWidgets.QLabel("Scan step (x,y,z)[Å]")
        lb.setToolTip(TTips["ScanStep"])
        bxl.addWidget(lb)
        vb = QtWidgets.QHBoxLayout()
        bxr.addLayout(vb)
        self.bxStepX = _spin_box((0.02, 0.5), 0.1, 0.02, self.updateScanWindow, TTips["ScanStep"], vb)
        self.bxStepY = _spin_box((0.02, 0.5), 0.1, 0.02, self.updateScanWindow, TTips["ScanStep"], vb)
        self.bxStepZ = _spin_box((0.02, 0.5), 0.1, 0.02, self.updateScanWindow, TTips["ScanStep"], vb)

        # Scan size
        lb = QtWidgets.QLabel("Scan size (x,y)[Å]")
        lb.setToolTip(TTips["ScanSize"])
        bxl.addWidget(lb)
        vb = QtWidgets.QHBoxLayout()
        bxr.addLayout(vb)
        self.bxSSx = _spin_box((1, 100), 16, 0.1, self.updateScanWindow, TTips["ScanSize"], vb)
        self.bxSSy = _spin_box((1, 100), 16, 0.1, self.updateScanWindow, TTips["ScanSize"], vb)

        # Scan start
        lb = QtWidgets.QLabel("Scan start (x,y)[Å]")
        lb.setToolTip(TTips["ScanStart"])
        bxl.addWidget(lb)
        vb = QtWidgets.QHBoxLayout()
        bxr.addLayout(vb)
        self.bxSCx = _spin_box((-100, 100), 0, 0.1, self.updateScanWindow, TTips["ScanStart"], vb)
        self.bxSCy = _spin_box((-100, 100), 0, 0.1, self.updateScanWindow, TTips["ScanStart"], vb)

        # Distance, amplitude, rotation
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("Distance [Å]")
        lb.setToolTip(TTips["Distance"])
        vb.addWidget(lb)
        self.bxD = _spin_box((-1000, 1000), 6.5, 0.1, self.updateScanWindow, TTips["Distance"], vb)
        lb = QtWidgets.QLabel("Amplitude [Å]")
        lb.setToolTip(TTips["Amplitude"])
        vb.addWidget(lb)
        self.bxA = _spin_box((0.0, 100), 1.0, 0.1, self.updateScanWindow, TTips["Amplitude"], vb)
        lb = QtWidgets.QLabel("Rotation")
        lb.setToolTip(TTips["Rotation"])
        vb.addWidget(lb)
        self.bxRot = _spin_box((-360.0, 360.0), 0.0, 5.0, self.updateRotation, TTips["Rotation"], vb)

    def _create_pbc_settings_ui(self):
        # Title
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("Periodic Boundaries")
        lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font()
        font.setPointSize(12)
        lb.setFont(font)
        lb.setMaximumHeight(50)
        vb.addWidget(lb)

        # Toggle PBC
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("Use periodic boundary conditions")
        lb.setToolTip(TTips["PBC"])
        vb.addWidget(lb)
        self.bxPBC = QtWidgets.QCheckBox()
        self.bxPBC.setChecked(True)
        self.bxPBC.toggled.connect(self.updatePBC)
        vb.addWidget(self.bxPBC)

        # Toggle z PBC
        lb = QtWidgets.QLabel("z periodicity")
        lb.setToolTip(TTips["PBCz"])
        vb.addWidget(lb)
        self.bxPBCz = QtWidgets.QCheckBox()
        self.bxPBCz.setChecked(False)
        self.bxPBCz.toggled.connect(self.updatePBC)
        self.bxPBCz.setToolTip(TTips["PBCz"])
        vb.addWidget(self.bxPBCz)

        # Left-right divide for labels and input boxes
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        bxl = QtWidgets.QVBoxLayout()
        vb.addLayout(bxl, 1)
        bxr = QtWidgets.QVBoxLayout()
        vb.addLayout(bxr, 3)

        # Lattice vector A
        lb = QtWidgets.QLabel("A (x, y, z) [Å]")
        lb.setToolTip(TTips["PBC"])
        bxl.addWidget(lb)
        vb = QtWidgets.QHBoxLayout()
        bxr.addLayout(vb)
        self.bxPBCAx = _spin_box((-1000, 1000), 50, 0.1, self.updatePBC, TTips["PBC"], vb)
        self.bxPBCAy = _spin_box((-1000, 1000), 0, 0.1, self.updatePBC, TTips["PBC"], vb)
        self.bxPBCAz = _spin_box((-1000, 1000), 0, 0.1, self.updatePBC, TTips["PBC"], vb)

        # Lattice vector B
        lb = QtWidgets.QLabel("B (x, y, z) [Å]")
        lb.setToolTip(TTips["PBC"])
        bxl.addWidget(lb)
        vb = QtWidgets.QHBoxLayout()
        bxr.addLayout(vb)
        self.bxPBCBx = _spin_box((-1000, 1000), 0, 0.1, self.updatePBC, TTips["PBC"], vb)
        self.bxPBCBy = _spin_box((-1000, 1000), 50, 0.1, self.updatePBC, TTips["PBC"], vb)
        self.bxPBCBz = _spin_box((-1000, 1000), 0, 0.1, self.updatePBC, TTips["PBC"], vb)

        # Lattice vector C
        lb = QtWidgets.QLabel("C (x, y, z) [Å]")
        lb.setToolTip(TTips["PBC"])
        bxl.addWidget(lb)
        vb = QtWidgets.QHBoxLayout()
        bxr.addLayout(vb)
        self.bxPBCCx = _spin_box((-1000, 1000), 0, 0.1, self.updatePBC, TTips["PBC"], vb)
        self.bxPBCCy = _spin_box((-1000, 1000), 0, 0.1, self.updatePBC, TTips["PBC"], vb)
        self.bxPBCCz = _spin_box((-1000, 1000), 50, 0.1, self.updatePBC, TTips["PBC"], vb)

    def _create_df_settings_ui(self):
        # Title
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("df settings")
        lb.setAlignment(QtCore.Qt.AlignCenter)
        font = lb.font()
        font.setPointSize(12)
        lb.setFont(font)
        lb.setMaximumHeight(50)
        vb.addWidget(lb)

        # Cantilevel settings
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)
        lb = QtWidgets.QLabel("k [kN/m]")
        lb.setToolTip(TTips["k"])
        vb.addWidget(lb)
        self.bxCant_K = _spin_box((0, 1000), 1.8, 0.1, self.updateScanWindow, TTips["k"], vb)
        lb = QtWidgets.QLabel("f0 [kHz]")
        lb.setToolTip(TTips["f0"])
        vb.addWidget(lb)
        self.bxCant_f0 = _spin_box((0, 1000), 30.3, 1.0, self.updateScanWindow, TTips["k"], vb)

        # Number of z-steps in df curve
        lb = QtWidgets.QLabel("z steps")
        lb.setToolTip(TTips["z_steps"])
        vb.addWidget(lb, 1)
        self.bxdfst = QtWidgets.QSpinBox()
        self.bxdfst.setRange(1, 100)
        self.bxdfst.setValue(10)
        self.bxdfst.valueChanged.connect(self.updateScanWindow)
        self.bxdfst.setToolTip(TTips["z_steps"])
        self.bxdfst.setKeyboardTracking(False)
        vb.addWidget(self.bxdfst, 2)

        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)

        # Colorbar toggle
        lb = QtWidgets.QLabel("Colorbar")
        lb.setToolTip(TTips["df_colorbar"])
        vb.addWidget(lb)
        self.bxDfCbar = QtWidgets.QCheckBox()
        self.bxDfCbar.setChecked(False)
        self.bxDfCbar.toggled.connect(self.updateDfColorbar)
        self.bxDfCbar.setToolTip(TTips["df_colorbar"])
        vb.addWidget(self.bxDfCbar)

        # Colorbar range
        lb = QtWidgets.QLabel("df range")
        lb.setToolTip(TTips["df_range"])
        vb.addWidget(lb)
        self.bxDfMin = _spin_box((-1000, 1000), -1.0, 0.1, self.updateDfRange, TTips["df_range"], vb)
        self.bxDfMax = _spin_box((-1000, 1000), 1.0, 0.1, self.updateDfRange, TTips["df_range"], vb)
        self.bxDfMin.setDisabled(True)
        self.bxDfMax.setDisabled(True)

        # Colorbar range reset button
        self.btDfReset = QtWidgets.QPushButton("Reset", self)
        self.btDfReset.setToolTip(TTips["df_reset"])
        self.btDfReset.clicked.connect(self.resetDfRange)
        self.btDfReset.setDisabled(True)
        vb.addWidget(self.btDfReset)

    def _create_buttons_ui(self):
        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)

        # Geometry viewer
        bt = QtWidgets.QPushButton("View Geometry", self)
        bt.setToolTip(TTips["view_geom"])
        bt.clicked.connect(self.showGeometry)
        self.btViewGeom = bt
        vb.addWidget(bt)

        # Geometry editor
        self.geomEditor = None
        bt = QtWidgets.QPushButton("Edit Geometry", self)
        bt.setToolTip(TTips["edit_geom"])
        bt.clicked.connect(self.showGeomEditor)
        self.btEditAtoms = bt
        vb.addWidget(bt)

        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)

        # Forcefield viewer
        self.FFViewer = guiw.FFViewer(self, verbose=self.verbose)
        bt = QtWidgets.QPushButton("View Forcefield", self)
        bt.setToolTip(TTips["view_ff"])
        bt.clicked.connect(self.showFFViewer)
        self.btViewFF = bt
        vb.addWidget(bt)

        # Forcefield parameter editor
        self.FFEditor = guiw.LJParamEditor(self.afmulator.typeParams, self)
        bt = QtWidgets.QPushButton("Edit Forcefield", self)
        bt.setToolTip(TTips["edit_ff"])
        bt.clicked.connect(self.FFEditor.show)
        self.btEditFF = bt
        vb.addWidget(bt)

        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)

        # Buttons for saving/loading parameters
        self.btSaveParams = QtWidgets.QPushButton("Save parameters...", self)
        self.btSaveParams.setToolTip("Save all current parameters into a params.ini file.")
        self.btSaveParams.clicked.connect(self.saveParams)
        vb.addWidget(self.btSaveParams)
        self.btLoadParams = QtWidgets.QPushButton("Load parameters...", self)
        self.btLoadParams.setToolTip("Load parameters from a params.ini file.")
        self.btLoadParams.clicked.connect(self.loadParams)
        vb.addWidget(self.btLoadParams)

        vb = QtWidgets.QHBoxLayout()
        self.l0.addLayout(vb)

        # --- btLoad
        self.openFileDialog = guiw.FileOpen(self)
        self.btLoad = QtWidgets.QPushButton("Open File...", self)
        self.btLoad.setToolTip("Open new file.")
        self.btLoad.clicked.connect(self.openFile)
        vb.addWidget(self.btLoad)

        # --- btSave
        self.btSave = QtWidgets.QPushButton("Save Image...", self)
        self.btSave.setToolTip("Save current image.")
        self.btSave.clicked.connect(self.saveFig)
        vb.addWidget(self.btSave)

        # --- btSaveW (W- wsxm)
        self.btSaveW = QtWidgets.QPushButton("Save df...", self)
        self.btSaveW.setToolTip("Save current frequency shift data.")
        self.btSaveW.clicked.connect(self.saveDataW)
        vb.addWidget(self.btSaveW)


def _separator_line(parent):
    ln = QtWidgets.QFrame()
    ln.setFrameShape(QtWidgets.QFrame.HLine)
    ln.setFrameShadow(QtWidgets.QFrame.Sunken)
    parent.addWidget(ln)
    return ln


def _spin_box(value_range, value, step, connect_func, tool_tip, parent, stretch=2):
    bx = QtWidgets.QDoubleSpinBox()
    bx.setRange(value_range[0], value_range[1])
    bx.setValue(value)
    bx.setSingleStep(step)
    bx.valueChanged.connect(connect_func)
    bx.setToolTip(tool_tip)
    bx.setKeyboardTracking(False)
    parent.addWidget(bx, stretch)
    return bx


def main():
    qApp = QtWidgets.QApplication(sys.argv)
    args = parse_args()
    if args.list_devices:
        print("\nAvailable OpenCL platforms:")
        oclu.print_platforms()
        sys.exit(0)
    try:
        aw = ApplicationWindow(args.input, args.device, args.verbosity)
        aw.show()
        sys.exit(qApp.exec_())
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        guiw.show_warning(None, f"Ran into an error!\n\n{traceback.format_exc()}", "Error!")


if __name__ == "__main__":
    main()
