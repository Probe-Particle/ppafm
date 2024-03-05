#!/usr/bin/python3

import os
import warnings

import numpy as np
import pyopencl as cl

from .. import common as PPU
from .. import elements, io
from ..PPPlot import plotImages
from . import field as FFcl
from . import oclUtils as oclu
from . import relax as oclr
from .field import ElectronDensity, HartreePotential, MultipoleTipDensity, TipDensity

VALID_SIZES = np.array([16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048])


class AFMulator:
    """
    Simulate Atomic force microscope images of molecules.

    Arguments:
        pixPerAngstrome: int. Number of pixels (voxels) per angstrom in force field grid.
        lvec: np.ndarray of shape (4, 3) or None. Unit cell boundaries for force field. First (row) vector
            specifies the origin, and the remaining three vectors specify the edge vectors of the unit cell.
            If None, will be calculated automatically from scan_window and tipR0, leaving some additional space
            on each side.
        scan_dim: tuple of three ints. Number of scan steps in the (x, y, z) dimensions. The size of the resulting
            images have a size of scan_dim[0] x scan_dim[1] and the number images at different heights is
            scan_dim[2] - df_steps + 1.
        scan_window: tuple ((x_min, y_min, z_min), (x_max, y_max, z_max)). The minimum and maximum coordinates of
            scan region in angstroms. The scan region is a rectangular box with the opposing corners defined by the
            coordinates in scan_window. Note that the step size in z dimension is the scan window size in z divided by
            scan_dim[2], and the scan in z-direction proceeds for scan_dim[2] steps, so the final step is one step short
            of z_min.
        iZPPs: int. Element of probe particle.
        QZs: Array of length 4. Positions of tip charges along z-axis relative to probe-particle center in angstroms.
        Qs: Array of length 4. Values of tip charges in units of e. Some can be zero.
        rho: Dict or :class:`.TipDensity`. Tip charge density. Used with Hartree potentials. Overrides QZs and Qs.
            The dict should contain float entries for at least of one the following 's', 'px', 'py', 'pz', 'dz2',
            'dy2', 'dx2', 'dxy', 'dxz', 'dyz'. The tip charge density will be a linear combination of the specified
            multipole types with the specified weights.
        sigma: float. Width of tip density distribution if rho is a dict of multipole coefficients.
        rho_delta: :class:`.TipDensity`. Tip delta charge density. Used in FDBM approximation for calculating the
            electrostatic interaction force.
        A_pauli: float. Prefactor for Pauli repulsion when using the FDBM.
        B_pauli: float. Exponent for Pauli repulsion when using the FDBM.
        fdbm_vdw_type: 'D3' or 'LJ'. Type of vdW interaction to use with the FDBM. 'D3' is for Grimme-D3 and 'LJ' uses
            standard Lennard-Jones vdW.
        d3_params: str or dict. Functional-specific scaling parameters for Grimme-D3. Can be a str with the functional name
            or a dict with manually specified parameters. Used in FDBM. See :meth:`.add_dftd3` for further explanation.
        lj_vdw_damp: int. Type of damping to use in vdw calculation for FDBM when fdbm_vdw_type=='LJ'.
            -1: no damping, 0: constant, 1: R2, 2: R4, 3: invR4, 4: invR8.
        df_steps: int. Number of steps in z convolution. The total amplitude is df_steps times scan z-step size.
        tipR0: array of length 3. Probe particle equilibrium position (x, y, z) in angstroms.
        tipStiffness: array of length 4. Harmonic spring constants (x, y, z, r) in N/m for holding the probe particle
            to the tip.
        npbc: tuple of three ints. How many periodic images of atoms to use in (x, y, z) dimensions. Used for calculating
            Lennard-Jones force field and electrostatic field from point charges. Electrostatic field from a Hartree
            potential defined on a grid is always considered to be periodic.
        f0Cantilever: float. Resonance frequency of the cantilever in Hz.
        kCantilever: float. Harmonic spring constant of the cantilever in N/m.
    """

    bMergeConv = False  # Should we use merged kernel relaxStrokesTilted_convZ or two separated kernells  ( relaxStrokesTilted, convolveZ  )

    # --- Relaxation
    relaxParams = [0.5, 0.1, 0.1 * 0.2, 0.1 * 5.0]  # (dt,damp, .., .. ) parameters of relaxation, in the moment just dt and damp are used

    # --- Output
    bSaveFF = False  # Save debug ForceField as .xsf
    verbose = 0  # Print information during excecution

    # ==================== Methods =====================

    def __init__(
        self,
        pixPerAngstrome=10,
        lvec=None,
        scan_dim=(128, 128, 30),
        scan_window=((2.0, 2.0, 7.0), (18.0, 18.0, 10.0)),
        iZPP=8,
        QZs=[0.1, 0, -0.1, 0],
        Qs=[-10, 20, -10, 0],
        rho=None,
        sigma=0.71,
        rho_delta=None,
        A_pauli=18.0,
        B_pauli=1.0,
        fdbm_vdw_type="D3",
        d3_params="PBE",
        lj_vdw_damp=2,
        df_steps=10,
        tipR0=[0.0, 0.0, 3.0],
        tipStiffness=[0.25, 0.25, 0.0, 30.0],
        npbc=(1, 1, 0),
        f0Cantilever=30300,
        kCantilever=1800,
    ):
        if not FFcl.oclu or not oclr.oclu:
            oclu.init_env()

        self.forcefield = FFcl.ForceField_LJC()

        self.scanner = oclr.RelaxedScanner()
        self.scanner.relax_params = np.array(self.relaxParams, dtype=np.float32)
        self.scanner.stiffness = np.array(tipStiffness, dtype=np.float32) / -PPU.eVA_Nm

        self.iZPP = iZPP
        self.tipR0 = tipR0
        self.f0Cantilever = f0Cantilever
        self.kCantilever = kCantilever
        self.npbc = npbc
        self.sigma = sigma
        self.A_pauli = A_pauli
        self.B_pauli = B_pauli
        self.fdbm_vdw_type = fdbm_vdw_type
        self.d3_params = d3_params
        self.lj_vdw_damp = lj_vdw_damp
        self.sample_lvec = None

        self.setScanWindow(scan_window, scan_dim, df_steps)
        self.setLvec(lvec, pixPerAngstrome)
        self.setRho(rho, sigma, B_pauli)
        self.setRhoDelta(rho_delta)
        self.setQs(Qs, QZs)

        self.typeParams = PPU.loadSpecies("atomtypes.ini")
        self.saveFFpre = ""

    def eval(self, xyzs, Zs, qs, rho_sample=None, sample_lvec=None, rot=np.eye(3), rot_center=None, REAs=None, X=None, plot_to_dir=None):
        """
        Prepare and evaluate AFM image.

        Arguments:
            xyzs: np.ndarray of shape (num_atoms, 3). Positions of atoms in x, y, and z.
            Zs: np.ndarray of shape (num_atoms,). Elements of atoms.
            qs: np.ndarray of shape (num_atoms,) or :class:`.HartreePotential` or None. Charges of atoms or hartree potential.
                If None, then no electrostatics are used.
            rho_sample: :class:`.ElectronDensity` or None. Sample electron density. If not None, then FDBM is used
                for calculating the Pauli repulsion force.
            sample_lvec: np.ndarray of shape (3, 3) or None. Unit cell lattice vectors for periodic images of atoms.
                If None, periodic boundaries are disabled, unless qs is :class:`.HartreePotential` and the lvec from the
                Hartree potential is used instead. If npbc = (0, 0, 0), then has no function.
            rot: np.ndarray of shape (3, 3). Rotation matrix to apply to atom positions.
            rot_center: np.ndarray of shape (3,). Center for rotation. Defaults to center of atom coordinates.
            REAs: np.ndarray of shape (num_atoms, 4). Lennard Jones interaction parameters. Calculated automatically if None.
            X: np.ndarray of shape (self.scan_dim[0], self.scan_dim[1], self.scan_dim[2]-self.df_steps+1)).
               Array where AFM image will be saved. If None, will be created automatically.
            plot_to_dir: str or None. If not None, plot the generated AFM images to this directory.

        Returns:
            X: np.ndarray. Output AFM images. If input X is not None, this is the same array object as X with values overwritten.
        """
        self.prepareFF(xyzs, Zs, qs, rho_sample, sample_lvec, rot, rot_center, REAs)
        self.prepareScanner()
        X = self.evalAFM(X)
        if plot_to_dir:
            self.plot_images(X, outdir=plot_to_dir)
        return X

    def __call__(self, *args, **kwargs):
        """
        Makes object callable. See :meth:`eval` for input arguments.
        """
        return self.eval(*args, **kwargs)

    def eval_(self, mol):
        return self.eval(mol.xyzs, mol.Zs, mol.qs)

    # ========= Setup =========

    def setLvec(self, lvec=None, pixPerAngstrome=None):
        """Set forcefield lattice vectors. If lvec is not given it is inferred from the scan window."""

        if pixPerAngstrome is not None:
            self.pixPerAngstrome = pixPerAngstrome
        if lvec is not None:
            self.lvec = lvec
        else:
            self.lvec = get_lvec(self.scan_window, tipR0=self.tipR0, pixPerAngstrome=self.pixPerAngstrome)

        # Remember old grid size
        if hasattr(self.forcefield, "nDim"):
            self._old_nDim = self.forcefield.nDim.copy()
        else:
            self._old_nDim = np.zeros(4)

        # Set lvec in force field and scanner
        self.forcefield.initSampling(self.lvec, pixPerAngstrome=self.pixPerAngstrome)
        FEin_shape = self.forcefield.nDim if (self._old_nDim != self.forcefield.nDim).any() else None
        self.scanner.prepareBuffers(lvec=self.lvec, FEin_shape=FEin_shape)

    def setScanWindow(self, scan_window=None, scan_dim=None, df_steps=None):
        """Set scanner scan window."""

        if scan_window is not None:
            self.scan_window = scan_window
        if scan_dim is not None:
            self.scan_dim = scan_dim
        if df_steps is not None:
            if df_steps <= 0 or df_steps > self.scan_dim[2]:
                raise ValueError(f"df_steps should be between 1 and scan_dim[2]({scan_dim[2]}), but got {df_steps}.")
            self.df_steps = df_steps

        # Set df convolution weights
        self.dz = (self.scan_window[1][2] - self.scan_window[0][2]) / self.scan_dim[2]
        self.dfWeight = PPU.get_simple_df_weight(self.df_steps, dz=self.dz).astype(np.float32)
        self.dfWeight *= PPU.eVA_Nm * self.f0Cantilever / self.kCantilever
        self.amplitude = self.dz * len(self.dfWeight)
        self.scanner.zstep = self.dz

        # Prepare buffers
        self.scanner.prepareBuffers(
            scan_dim=self.scan_dim,
            nDimConv=len(self.dfWeight),
            nDimConvOut=(self.scan_dim[2] - len(self.dfWeight) + 1),
        )
        self.scanner.updateBuffers(WZconv=self.dfWeight)
        self.scanner.preparePosBasis(start=self.scan_window[0][:2], end=self.scan_window[1][:2])

    def setRho(self, rho=None, sigma=0.71, B_pauli=1.0):
        """Set tip charge distribution.

        Arguments:
            rho: Dict, :class:`.TipDensity`, or None. Tip charge density. If None, the existing density is deleted.
            sigma: float. Tip charge density distribution when rho is a dict.
            B_pauli: float. Pauli repulsion exponent for tip density when using FDBM.
        """
        if rho is not None:
            self.sigma = sigma
            self.B_pauli = B_pauli
            self._rho = rho  # Remember argument value so that we can recompute the power from grid or the grid from a dict later if needed
            if isinstance(rho, dict):
                self.rho = MultipoleTipDensity(
                    self.forcefield.lvec[:, :3],
                    self.forcefield.nDim[:3],
                    sigma=self.sigma,
                    multipole=rho,
                    ctx=self.forcefield.ctx,
                )
            else:
                if not isinstance(rho, TipDensity):
                    raise ValueError(f"rho should of type `TipDensity`, but got `{type(rho)}`")
                self.rho = rho
            if self.verbose > 0:
                print("AFMulator.setRho: Preparing buffers")
            if not np.allclose(B_pauli, 1.0):
                rho_power = self.rho.power_positive(p=self.B_pauli, in_place=False)
                self.rho.release()  # Let's not keep the original array in device memory to minimize memory foot print
                self.rho = rho_power
            self.forcefield.prepareBuffers(rho=self.rho, bDirect=True)
        else:
            self._rho = None
            self.rho = None
            if self.forcefield.rho is not None:
                if self.verbose > 0:
                    print("AFMulator.setRho: Releasing buffers")
                self.forcefield.rho.release()
                self.forcefield.rho = None

    def setBPauli(self, B_pauli=1.0):
        """Set Pauli repulsion exponent used in FDBM."""
        self.setRho(self._rho, sigma=self.sigma, B_pauli=B_pauli)

    def setRhoDelta(self, rho_delta=None):
        """Set tip electron delta-density that is used for electrostatic interaction in FDBM.

        Arguments:
            rho_delta: :class:`.TipDensity` or None. Tip electron delta-density. If None, the existing density is deleted.
        """
        self.rho_delta = rho_delta
        if self.rho_delta is not None:
            if not isinstance(rho_delta, TipDensity):
                raise ValueError(f"rho_delta should of type `TipDensity`, but got `{type(rho_delta)}`")
            if self.verbose > 0:
                print("AFMulator.setRhoDelta: Preparing buffers")
            self.forcefield.prepareBuffers(rho_delta=self.rho_delta, bDirect=True)
        elif self.forcefield.rho_delta is not None:
            if self.verbose > 0:
                print("AFMulator.setRhoDelta: Releasing buffers")
            self.forcefield.rho_delta.release()
            self.forcefield.rho_delta = None

    def setQs(self, Qs, QZs):
        """Set tip point charges."""
        self.Qs = Qs
        self.QZs = QZs
        self.forcefield.setQs(Qs=Qs, QZs=QZs)

    # ========= Imaging =========

    def prepareFF(self, xyzs, Zs, qs, rho_sample=None, sample_lvec=None, rot=np.eye(3), rot_center=None, REAs=None):
        """
        Prepare molecule parameters and calculate force field.

        Arguments:
            xyzs: np.ndarray of shape (num_atoms, 3). Positions of atoms in x, y, and z.
            Zs: np.ndarray of shape (num_atoms,). Elements of atoms.
            qs: np.ndarray of shape (num_atoms,) or :class:`.HartreePotential` or None. Charges of atoms or hartree potential.
                If None, then no electrostatics are used.
            rho_sample: :class:`.ElectronDensity` or None. Sample electron density. If not None, then FDBM is used
                for calculating the Pauli repulsion force. Requires rho_delta to be set and qs has to
                be :class:`.HartreePotential`.
            sample_lvec: np.ndarray of shape (3, 3) or None. Unit cell lattice vectors for periodic images of atoms.
                If None, periodic boundaries are disabled, unless qs is HartreePotential and the lvec from the
                Hartree potential is used instead. If npbc = (0, 0, 0), then has no function.
            rot: np.ndarray of shape (3, 3). Rotation matrix to apply to atom positions.
            rot_center: np.ndarray of shape (3,). Center for rotation. Defaults to center of atom coordinates.
            REAs: np.ndarray of shape (num_atoms, 4). Lennard Jones interaction parameters. Calculated automatically if None.
        """

        # Check if the scan window extends over any non-periodic boundaries and issue a warning if it does
        self.check_scan_window()

        # (Re)initialize force field if the size of the grid changed since last run.
        if (self._old_nDim != self.forcefield.nDim).any():
            if self.verbose > 0:
                print("(Re)initializing force field buffers.")
            if self.verbose > 1:
                print(f"old nDim: {self._old_nDim}, new nDim: {self.forcefield.nDim}")
            self.forcefield.tryReleaseBuffers()
            if self._rho is not None:
                # The grid size changed so we need to recompute/reinterpolate the tip density grid
                self.setRho(self._rho, self.sigma, self.B_pauli)
                self.setRhoDelta(self.rho_delta)  # self.rho_delta could be None, but then this does nothing
            self.forcefield.prepareBuffers()
            self._old_nDim = self.forcefield.nDim

        # If rho_sample is specified, then we use FDBM. Check that other requirements are satisfied.
        if rho_sample is not None:
            if not isinstance(rho_sample, ElectronDensity):
                raise ValueError(f"rho_sample should of type `ElectronDensity`, but got `{type(rho_sample)}`")
            if not isinstance(qs, HartreePotential):
                raise ValueError(f"qs should be HartreePotential when rho_sample is not None, but got type `{type(qs)}`.")
            if self.rho_delta is None:
                raise ValueError(f"rho_delta should be set when rho_sample is not None.")

        # Check if using point charges or precomputed Hartee potential
        if qs is None:
            pot = None
            qs = np.zeros(len(Zs))
        elif isinstance(qs, HartreePotential):
            pot = qs
            qs = np.zeros(len(Zs))
            self.sample_lvec = sample_lvec if sample_lvec is not None else pot.lvec[1:]
            assert self.sample_lvec.shape == (
                3,
                3,
            ), f"sample_lvec has shape {self.sample_lvec.shape}, but should have shape (3, 3)"
        else:
            pot = None

        # Determine method
        if rho_sample is not None:
            method = "fdbm"
            self.forcefield.setPP(self.iZPP)
        elif pot is not None:
            method = "hartree"
        else:
            method = "point-charge"

        if (method != "fdbm") and (not np.allclose(self.B_pauli, 1.0)):
            warnings.warn(f"Not using FDBM, but tip density exponent is {self.B_pauli}! This is probably not what you want to do!")

        npbc = self.npbc if self.sample_lvec is not None else (0, 0, 0)

        if rot_center is None:
            rot_center = xyzs.mean(axis=0)

        # Get Lennard-Jones parameters and apply periodic boundary conditions to atoms
        if REAs is None:
            REAs = PPU.getAtomsREA(self.iZPP, Zs, self.typeParams, alphaFac=-1.0)
        cLJs = PPU.REA2LJ(REAs)
        if sum(npbc) > 0:
            Zs, xyzs, qs, cLJs, REAs = PPU.PBCAtoms3D_np(Zs, xyzs, qs, cLJs, REAs, self.sample_lvec, npbc=npbc)

        # Compute force field
        self.forcefield.makeFF(
            xyzs,
            cLJs,
            REAs=REAs,
            Zs=Zs,
            method=method,
            qs=qs,
            pot=pot,
            rho_sample=rho_sample,
            rho_delta=None,
            A=self.A_pauli,
            B=self.B_pauli,
            rot=rot,
            rot_center=rot_center,
            fdbm_vdw_type=self.fdbm_vdw_type,
            d3_params=self.d3_params,
            lj_vdw_damp=self.lj_vdw_damp,
            bRelease=False,
            bCopy=False,
            bFinish=False,
        )
        if self.bSaveFF:
            self.saveFF()

    def prepareScanner(self):
        """Prepare scanner. Run after preparing force field."""

        # Copy forcefield array to scanner buffer
        self.scanner.updateFEin(self.forcefield.cl_FE)

        # Subtract origin, because OpenCL kernel for tip relaxation does not take the origin of the FF box into account
        self.pos0 = np.array([0, 0, self.scan_window[1][2]]) - self.lvec[0]

        # Prepare tip position array
        self.scanner.setScanRot(self.pos0, rot=np.eye(3), tipR0=self.tipR0)

    def evalAFM(self, X=None):
        """
        Evaluate AFM image. Run after preparing force field and scanner.

        Arguments:
            X: np.ndarray of shape (self.scan_dim[0], self.scan_dim[1], self.scan_dim[2]-self.df_steps+1)).
               Array where AFM image will be saved. If None, will be created automatically.

        Returns:
            X: np.ndarray. Output AFM images. If input X is not None, this is the same array object as X with values overwritten.
        """

        if self.bMergeConv:
            FEout = self.scanner.run_relaxStrokesTilted_convZ()
        else:
            self.scanner.run_relaxStrokesTilted(bCopy=False, bFinish=False)
            FEout = self.scanner.run_convolveZ()

        if X is None:
            X = FEout[:, :, :, 2].copy()
        else:
            if X.shape != self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight) + 1,):
                raise ValueError(
                    f"Expected an array of shape {self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight) + 1,)} "
                    f"for storing AFM image, but got an array of shape {X.shape} instead."
                )
            X[:, :, :] = FEout[:, :, :, 2]

        return X

    # ========= Save/Load state =========

    @classmethod
    def from_params(cls, file_path="./params.ini"):
        """
        Construct an AFMulator instance from a params.ini file.

        Arguments:
            file_path: str. Path to the params.ini file to load.
        """
        params, sample_lvec = _get_params(file_path)
        afmulator = cls(**params)
        afmulator.sample_lvec = sample_lvec
        return afmulator

    def load_params(self, file_path="./params.ini"):
        """
        Update the parameters of this AFMulator instance from a params.ini file.

        Arguments:
            file_path: str. Path to the params.ini file to load.
        """
        params, sample_lvec = _get_params(file_path)
        if params["tipStiffness"] is not None:
            self.scanner.stiffness = np.array(params["tipStiffness"], dtype=np.float32) / -PPU.eVA_Nm
        self.iZPP = params["iZPP"]
        self.tipR0 = params["tipR0"]
        self.f0Cantilever = params["f0Cantilever"]
        self.kCantilever = params["kCantilever"]
        self.npbc = params["npbc"]
        self.A_pauli = params["A_pauli"]
        self.setScanWindow(params["scan_window"], params["scan_dim"], params["df_steps"])
        self.setLvec(params["lvec"], params["pixPerAngstrome"])
        if (self._rho is None) or isinstance(self._rho, dict):
            self.setRho(params["rho"], params["sigma"])
        else:
            warnings.warn(f'Using existing tip density with exponent {params["B_pauli"]}, instead of parameters from params.ini file.')
            self.setRho(self._rho, sigma=params["sigma"], B_pauli=params["B_pauli"])
        self.sample_lvec = sample_lvec

    def save_params(self, file_path="./params.ini"):
        """
        Save the parameters of this AFMulator instance to a params.ini file.

        Arguments:
            file_path: str. Path to the file where parameters are saved.
        """
        k = self.scanner.stiffness * -PPU.eVA_Nm
        nDim = self.forcefield.nDim
        scan_step = (
            (self.scan_window[1][0] - self.scan_window[0][0]) / (self.scan_dim[0] - 1),
            (self.scan_window[1][1] - self.scan_window[0][1]) / (self.scan_dim[1] - 1),
            (self.scan_window[1][2] - self.scan_window[0][2]) / self.scan_dim[2],
        )
        with open(file_path, "w") as f:
            f.write(f"probeType {self.iZPP}\n")
            if isinstance(self._rho, dict):
                if len(self._rho) > 1:
                    warnings.warn("More than one tip multipole type specified. Only writing the first one into params.ini.")
                tip = list(self._rho)[0]
                f.write(f"tip {tip}\n")
                f.write(f"charge {self._rho[tip]}\n")
                f.write(f"sigma {self.sigma}\n")
            f.write(f"stiffness {k[0]} {k[1]} {k[3]}\n")
            f.write(f"r0Probe {self.tipR0[0]} {self.tipR0[1]} {self.tipR0[2]}\n")
            f.write(f"PBC {(np.array(self.npbc) > 0).any()}\n")
            f.write(f"nPBC {self.npbc[0]} {self.npbc[1]} {self.npbc[2]}\n")
            if self.sample_lvec is not None:
                f.write(f"gridA {self.sample_lvec[0, 0]} {self.sample_lvec[0, 1]} {self.sample_lvec[0, 2]}\n")
                f.write(f"gridB {self.sample_lvec[1, 0]} {self.sample_lvec[1, 1]} {self.sample_lvec[1, 2]}\n")
                f.write(f"gridC {self.sample_lvec[2, 0]} {self.sample_lvec[2, 1]} {self.sample_lvec[2, 2]}\n")
            f.write(f"FFgrid0 {self.lvec[0, 0]} {self.lvec[0, 1]} {self.lvec[0, 2]}\n")
            f.write(f"FFgridA {self.lvec[1, 0]} {self.lvec[1, 1]} {self.lvec[1, 2]}\n")
            f.write(f"FFgridB {self.lvec[2, 0]} {self.lvec[2, 1]} {self.lvec[2, 2]}\n")
            f.write(f"FFgridC {self.lvec[3, 0]} {self.lvec[3, 1]} {self.lvec[3, 2]}\n")
            f.write(f"gridN {nDim[0]} {nDim[1]} {nDim[2]}\n")
            f.write(f"scanMin {self.scan_window[0][0]} {self.scan_window[0][1]} {self.scan_window[0][2]}\n")
            f.write(f"scanMax {self.scan_window[1][0]} {self.scan_window[1][1]} {self.scan_window[1][2]}\n")
            f.write(f"scanStep {scan_step[0]} {scan_step[1]} {scan_step[2]}\n")
            f.write(f"kCantilever {self.kCantilever}\n")
            f.write(f"f0Cantilever {self.f0Cantilever}\n")
            f.write(f"Amplitude {self.amplitude}\n")
            f.write(f"Apauli {self.A_pauli}\n")
            f.write(f"Bpauli {self.B_pauli}\n")

    # ========= Debug/Plot Misc. =========

    def saveFF(self):
        FF = self.forcefield.downloadFF()
        FFx = FF[:, :, :, 0]
        FFy = FF[:, :, :, 1]
        FFz = FF[:, :, :, 2]
        Fr = np.sqrt(FFx**2 + FFy**2 + FFz**2)
        Fbound = 10.0
        mask = Fr.flat > Fbound
        FFx.flat[mask] *= (Fbound / Fr).flat[mask]
        FFy.flat[mask] *= (Fbound / Fr).flat[mask]
        FFz.flat[mask] *= (Fbound / Fr).flat[mask]
        if self.verbose > 0:
            print("FF.shape ", FF.shape)
        self.saveDebugXSF_FF(self.saveFFpre + "FF_x.xsf", FFx)
        self.saveDebugXSF_FF(self.saveFFpre + "FF_y.xsf", FFy)
        self.saveDebugXSF_FF(self.saveFFpre + "FF_z.xsf", FFz)

    def saveDebugXSF_FF(self, fname, F):
        if self.verbose > 0:
            print("saveDebugXSF : ", fname)
        io.saveXSF(fname, F, self.lvec)

    def check_scan_window(self):
        """Check that scan window does not extend beyond any non-periodic boundaries."""
        for i, dim in enumerate("xyz"):
            if self.npbc[i]:
                continue
            lvec_start = self.lvec[0, i]
            lvec_end = lvec_start + self.lvec[i + 1, i]
            scan_start = self.scan_window[0][i] - 0.3  # Small offsets because being close to the boundary
            scan_end = self.scan_window[1][i] + 0.3  # is problematic as well because of PP bending.
            if i == 2:
                scan_start -= self.tipR0[2]
                scan_end -= self.tipR0[2]
            if (scan_start < lvec_start) or (scan_end > lvec_end):
                print(
                    f"Warning: The edge of the scan window in {dim} dimension is very close or extends over "
                    f"the boundary of the force-field grid which is not periodic in {dim} dimension. "
                    "If you get artifacts in the images, please check the boundary conditions and "
                    "the size of the scan window and the force field grid."
                )

    def plot_images(self, X, outdir="afm_images", prefix="df"):
        """
        Plot simulated AFM images and save them to a directory.

        Arguments:
            X: np.ndarray. AFM images to plot.
            outdir: str. Path to directory where files are saved.
            prefix: str. Prefix string for saved files.
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        X = X.transpose(2, 1, 0)[::-1]
        zTips = np.linspace(
            self.scan_window[0][2],
            self.scan_window[1][2] - self.df_steps * self.dz,
            self.scan_dim[2] - self.df_steps + 1,
        )
        zTips += self.amplitude / 2
        extent = [
            self.scan_window[0][0],
            self.scan_window[1][0],
            self.scan_window[0][1],
            self.scan_window[1][1],
        ]
        plotImages(
            os.path.join(outdir, prefix),
            X,
            slices=list(range(0, len(X))),
            zs=zTips,
            extent=extent,
            cmap=PPU.params["colorscale"],
        )


def _get_params(file_path):
    """Get AFMulator arguments from a params.ini file."""
    PPU.loadParams(file_path)
    lvec = np.array(
        [
            PPU.params["FFgrid0"],
            PPU.params["FFgridA"],
            PPU.params["FFgridB"],
            PPU.params["FFgridC"],
        ]
    )
    if (lvec < 0).any():
        lvec = None
    sample_lvec = np.array([PPU.params["gridA"], PPU.params["gridB"], PPU.params["gridC"]])
    if (PPU.params["gridN"] == 0).any() or lvec is None:
        pixPerAngstrome = 10
    else:
        rx, ry, rz = (round(PPU.params["gridN"][i] / np.linalg.norm(lvec[i + 1])) for i in range(3))
        if np.allclose([rx, ry], rz):
            pixPerAngstrome = rx
        else:
            pixPerAngstrome = np.max([rx, ry, rz])
            warnings.warn(
                "Unequal grid densities in x, y, z directions is not supported in the OpenCL version of ppafm. "
                f"Using the maximum of x, y, z directions, {pixPerAngstrome}, for grid point density."
            )
    scan_window = (PPU.params["scanMin"], PPU.params["scanMax"])
    scan_dim = (
        round((scan_window[1][0] - scan_window[0][0]) / PPU.params["scanStep"][0]) + 1,
        round((scan_window[1][1] - scan_window[0][1]) / PPU.params["scanStep"][1]) + 1,
        round((scan_window[1][2] - scan_window[0][2]) / PPU.params["scanStep"][2]),
    )
    iZPP = PPU.params["probeType"]
    iZPP = elements.ELEMENT_DICT[iZPP][0] if iZPP in elements.ELEMENT_DICT else int(iZPP)
    tipStiffness = PPU.params["stiffness"]
    if (tipStiffness < 0).any():
        tipStiffness = [0.25, 0.25, 0.0, 30.0]
    else:
        tipStiffness = np.insert(tipStiffness, 2, 0.0)  # AFMulator additionally has a z-component in the third place
    afmulator_params = {
        "lvec": lvec,
        "pixPerAngstrome": pixPerAngstrome,
        "scan_dim": scan_dim,
        "scan_window": scan_window,
        "iZPP": iZPP,
        "rho": {PPU.params["tip"]: PPU.params["charge"]},
        "sigma": PPU.params["sigma"],
        "A_pauli": PPU.params["Apauli"],
        "B_pauli": PPU.params["Bpauli"],
        "df_steps": round(PPU.params["Amplitude"] / PPU.params["scanStep"][2]),
        "tipR0": PPU.params["r0Probe"],
        "tipStiffness": tipStiffness,
        "npbc": PPU.params["nPBC"] * PPU.params["PBC"],
        "f0Cantilever": PPU.params["f0Cantilever"],
        "kCantilever": PPU.params["kCantilever"],
    }
    return afmulator_params, sample_lvec


def get_lvec(scan_window, pad=(2.0, 2.0, 3.0), tipR0=(0.0, 0.0, 3.0), pixPerAngstrome=10):
    pad = np.array(pad)
    tipR0 = np.array(tipR0)
    center = (np.array(scan_window[0]) + np.array(scan_window[1])) / 2
    box_size = (np.array(scan_window[1]) - np.array(scan_window[0])) + 2 * pad
    nDim = (pixPerAngstrome * box_size).round().astype(np.int32)
    nDim = np.array([VALID_SIZES[VALID_SIZES >= d][0] for d in nDim])
    box_size = nDim / pixPerAngstrome
    origin = center - box_size / 2 - tipR0
    # fmt: off
    lvec = np.array([
        origin,
        [box_size[0],           0,           0],
        [          0, box_size[1],           0],
        [          0,           0, box_size[2]]
    ])
    # fmt: on
    return lvec


def quick_afm(
    file_path,
    scan_size=(16, 16),
    offset=(0, 0),
    distance=8.0,
    scan_step=(0.1, 0.1, 0.1),
    probe_type=8,
    charge=-0.1,
    tip="dz2",
    sigma=0.71,
    num_heights=10,
    amplitude=1.0,
    out_dir=None,
):
    r"""
    Make an AFM simulation from a .cube, .xsf, or .xyz file, and print images to a folder.

    Arguments:
        file_path: str. Path to input file.
        scan_size: tuple of length 2. Size of scan region in angstroms.
        offset: tuple of length 2. Offset to center of scan region in angstroms.
        distance: float. Furthest distance of probe tip from sample.
        scan_step: tuple of length 3. Scan steps in x, y, and z dimensions.
        probe_type: int. Probe particle type.
        charge: float. Tip charge.
        tip: str. Tip multipole type.
        sigma: float. Width of tip charge distribution.
        num_heights: int. Number of different heights to scan.
        amplitude: float. Oscillation amplitude in angstroms.
        out_dir: str or None. Output folder path. If None, defaults to "./afm\_" + input_file_name.
    """

    if not FFcl.oclu or not oclr.oclu:
        oclu.init_env()

    if not out_dir:
        file_name = os.path.split(file_path)[1]
        file_name = os.path.splitext(file_name)[0]
        out_dir = f"afm_{file_name}".replace(" ", "_")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load input file
    if file_path.endswith(".xsf") or file_path.endswith(".cube"):
        qs, xyzs, Zs = FFcl.HartreePotential.from_file(file_path, scale=-1.0)
        multipole = {tip: charge}
        Qs = [0, 0, 0, 0]
        QZs = [0, 0, 0, 0]
    elif file_path.endswith(".xyz"):
        xyzs, Zs, qs, _ = io.loadXYZ(file_path)
        multipole = {}
        if tip == "s":
            Qs = [charge, 0, 0, 0]
            QZs = [0, 0, 0, 0]
        elif tip == "pz":
            Qs = [10 * charge, -10 * charge, 0, 0]
            QZs = [0.1, -0.1, 0, 0]
        elif tip == "dz2":
            Qs = [100 * charge, -200 * charge, 100 * charge, 0]
            QZs = [0.1, 0, -0.1, 0]
        else:
            raise ValueError(f"Unsupported tip type `{tip}` for point charges.")
    else:
        raise ValueError(f"Unsupported file format in file `{file_path}`.")

    # Figure out scan parameters
    z_max = xyzs[:, 2].max() + distance
    xy_center = (xyzs[:, :2].min(axis=0) + xyzs[:, :2].max(axis=0)) / 2 + np.array(offset)
    df_steps = int(amplitude / scan_step[2])
    z_size = (num_heights + df_steps - 1) * scan_step[2]
    scan_window = (
        (xy_center[0] - scan_size[0] / 2, xy_center[1] - scan_size[1] / 2, z_max - z_size),
        (xy_center[0] + scan_size[0] / 2, xy_center[1] + scan_size[1] / 2, z_max),
    )
    scan_dim = (
        int(scan_size[0] / scan_step[0]),
        int(scan_size[1] / scan_step[1]),
        int(z_size / scan_step[2]),
    )

    # Do scan
    afmulator = AFMulator(
        scan_window=scan_window,
        scan_dim=scan_dim,
        rho=multipole,
        iZPP=probe_type,
        Qs=Qs,
        QZs=QZs,
        df_steps=df_steps,
    )
    X = afmulator(xyzs, Zs, qs)

    # Plot
    X = X[:, :, ::-1].T
    extent = [
        scan_window[0][0],
        scan_window[1][0],
        scan_window[0][1],
        scan_window[1][1],
    ]
    zs = np.linspace(scan_window[0][2], scan_window[1][2], scan_dim[2] + 1)[:num_heights]
    plotImages(os.path.join(out_dir, "df"), X, range(len(X)), extent=extent, zs=zs)
