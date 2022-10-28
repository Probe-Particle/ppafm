#!/usr/bin/python3

import os
import numpy as np
import pyopencl as cl

from .. import common   as PPU
from . import field     as FFcl
from . import relax     as oclr
from . import oclUtils  as oclu

from .field import HartreePotential, MultipoleTipDensity, hartreeFromFile
from ..basUtils import loadAtomsLines
from ..PPPlot import plotImages

VALID_SIZES = np.array([16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048])

class AFMulator():
    '''
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
        QZs: Array of length 4. Position tip charges along z-axis relative to probe-particle center in angstroms.
        Qs: Array of length 4. Values of tip charges in units of e. Some can be zero.
        rho: Dict or MultipoleTipDensity. Tip charge density. Used with Hartree potentials. Overrides QZs and Qs.
            The dict should contain float entries for at least of one the following 's', 'px', 'py', 'pz', 'dz2',
            'dy2', 'dx2', 'dxy', 'dxz', 'dyz'. The tip charge density will be a linear combination of the specified
            multipole types with the specified weights.
        sigma: float. Width of tip density distribution if rho is a dict of multipole coefficients.
        df_steps: int. Number of steps in z convolution. The total amplitude is df_steps times scan z-step size.
        tipR0: array of length 3. Probe particle equilibrium position (x, y, z) in angstroms.
        tipStiffness: array of length 4. Harmonic spring constants (x, y, z, r) for holding the probe particle
            to the tip in N/m.
        npbc: tuple of three ints. How many periodic images of atoms to use in (x, y, z) dimensions. Used for calculating
            Lennard-Jones force field and electrostatic field from point charges. Electrostatic field from a Hartree
            potential defined on a grid is always considered to be periodic.
        f0Cantilever: float. Resonance frequency of cantilever in Hz.
        kCantilever: float. Harmonic spring constant of cantilever in N/m.
        initFF: Bool. Whether to initialize buffers. Set to False to modify force field and scanner parameters
            before initialization.
    '''

    bNoPoss    = True    # Use algorithm which does not need to store array of FF_grid-sampling positions in memory (neither GPU-memory nor main-memory)
    bNoFFCopy  = True    # Should we copy Force-Field grid from GPU  to main_mem ?  ( Or we just copy it just internally withing GPU directly to image/texture used by RelaxedScanner )
    bMergeConv = False   # Should we use merged kernel relaxStrokesTilted_convZ or two separated kernells  ( relaxStrokesTilted, convolveZ  )

    # --- Relaxation
    relaxParams  = [ 0.5, 0.1, 0.1*0.2,0.1*5.0 ]    # (dt,damp, .., .. ) parameters of relaxation, in the moment just dt and damp are used

    # --- Output
    bSaveFF  = False    # Save debug ForceField as .xsf
    verbose  = 0        # Print information during excecution

    # ==================== Methods =====================

    def __init__( self, 
        pixPerAngstrome = 10,
        lvec            = None,
        scan_dim        = (128, 128, 30),
        scan_window     = ((2.0, 2.0, 7.0), ( 18.0, 18.0, 10.0)),
        iZPP            = 8,
        QZs             = [ 0.1,  0, -0.1, 0 ],
        Qs              = [ -10, 20,  -10, 0 ],
        rho             = None,
        sigma           = 0.71,
        df_steps        = 10,
        tipR0           = [0.0, 0.0, 3.0],
        tipStiffness    = [0.25, 0.25, 0.0, 30.0],
        npbc            = (1, 1, 0),
        f0Cantilever    = 30300,
        kCantilever     = 1800
    ):

        if not FFcl.oclu or not oclr.oclu:
            oclu.init_env()

        self.forcefield = FFcl.ForceField_LJC()

        self.scanner = oclr.RelaxedScanner()
        self.scanner.relax_params = np.array( self.relaxParams, dtype=np.float32 )
        self.scanner.stiffness = np.array(tipStiffness, dtype=np.float32) / -PPU.eVA_Nm

        self.iZPP = iZPP
        self.tipR0 = tipR0
        self.f0Cantilever = f0Cantilever
        self.kCantilever = kCantilever
        self.npbc = npbc
        self.pot = None

        self.setScanWindow(scan_window, scan_dim, df_steps)
        self.setLvec(lvec, pixPerAngstrome)
        self.setRho(rho, sigma)
        self.setQs(Qs, QZs)

        self.typeParams = PPU.loadSpecies('atomtypes.ini')
        self.saveFFpre = ""
        self.counter = 0
    
    def eval(self, xyzs, Zs, qs, pbc_lvec=None, rot=np.eye(3), rot_center=None, REAs=None, X=None ):
        '''
        Prepare and evaluate AFM image.
        Arguments:
            xyzs: np.ndarray of shape (num_atoms, 3). Positions of atoms in x, y, and z.
            Zs: np.ndarray of shape (num_atoms,). Elements of atoms.
            qs: np.ndarray of shape (num_atoms,) or HartreePotential or None. Charges of atoms or hartree potential.
                If None, then no electrostatics are used.
            pbc_lvec: np.ndarray of shape (3, 3) or None. Unit cell lattice vectors for periodic images of atoms.
                If None, periodic boundaries are disabled, unless qs is HartreePotential and the lvec from the
                Hartree potential is used instead. If npbc = (0, 0, 0), then has no function.
            REAs: np.ndarray of shape (num_atoms, 4). Lennard Jones interaction parameters. Calculated automatically if None.
            X: np.ndarray of shape (self.scan_dim[0], self.scan_dim[1], self.scan_dim[2]-self.df_steps+1)).
               Array where AFM image will be saved. If None, will be created automatically.
        Returns: np.ndarray. AFM image. If X is not None, this is the same array object as X with values overwritten.
        '''
        self.prepareFF(xyzs, Zs, qs, pbc_lvec, rot, rot_center, REAs)
        self.prepareScanner()
        X = self.evalAFM(X)
        return X
    
    def __call__(self, xyzs, Zs, qs, pbc_lvec=None, rot=np.eye(3), rot_center=None, REAs=None, X=None):
        '''
        Makes object callable. See eval for input arguments.
        '''
        return self.eval(xyzs, Zs, qs, pbc_lvec, rot, rot_center, REAs=REAs, X=X)

    def eval_( self, mol ):
        return self.eval( mol.xyzs, mol.Zs, mol.qs )

    # ========= Setup =========

    def setLvec(self, lvec=None, pixPerAngstrome=None):
        ''' Set forcefield lattice vectors. If lvec is not given it is inferred from the scan window.'''

        if pixPerAngstrome is not None:
            self.pixPerAngstrome = pixPerAngstrome
        if lvec is not None:
            self.lvec = lvec
        else:
            self.lvec = get_lvec(self.scan_window, tipR0=self.tipR0,
                pixPerAngstrome=self.pixPerAngstrome)
        
        # Remember old grid size
        if hasattr(self.forcefield, 'nDim'):
            self._old_nDim = self.forcefield.nDim.copy()
        else:
            self._old_nDim = np.zeros(4)

        # Set lvec in force field and scanner
        if self.bNoPoss:
            self.forcefield.initSampling(self.lvec, pixPerAngstrome=self.pixPerAngstrome)
        else:
            self.forcefield.initPoss(lvec=self.lvec, pixPerAngstrome=self.pixPerAngstrome)
        FEin_shape = self.forcefield.nDim if (self._old_nDim != self.forcefield.nDim).any() else None
        self.scanner.prepareBuffers(lvec=self.lvec, FEin_shape=FEin_shape)

    def setScanWindow(self, scan_window=None, scan_dim=None, df_steps=None):
        '''Set scanner scan window.'''

        if scan_window is not None:
            self.scan_window = scan_window
        if scan_dim is not None:
            self.scan_dim = scan_dim
        if df_steps is not None:
            if df_steps <= 0 or df_steps > self.scan_dim[2]:
                raise ValueError(f'df_steps should be between 1 and scan_dim[2]({scan_dim[2]}), but got {df_steps}.')
            self.df_steps = df_steps

        # Set df convolution weights
        self.dz = (self.scan_window[1][2] - self.scan_window[0][2]) / self.scan_dim[2]
        self.dfWeight = PPU.getDfWeight(self.df_steps, dz=self.dz)[0].astype(np.float32)
        self.dfWeight *= PPU.eVA_Nm * self.f0Cantilever / self.kCantilever
        self.amplitude = self.dz * len(self.dfWeight)
        self.scanner.zstep = self.dz

        # Prepare buffers
        self.scanner.prepareBuffers(scan_dim=self.scan_dim, nDimConv=len(self.dfWeight),
            nDimConvOut=(self.scan_dim[2] - len(self.dfWeight) + 1))
        self.scanner.updateBuffers(WZconv=self.dfWeight)
        self.scanner.preparePosBasis(start=self.scan_window[0][:2], end=self.scan_window[1][:2] )

    def setRho(self, rho=None, sigma=0.71):
        '''Set tip charge distribution.'''
        self.sigma = sigma
        if rho is not None:
            self._rho = rho # Remember argument value so that if it's a dict the tip density can be recalculated if necessary
            if isinstance(rho, dict):
                self.rho = MultipoleTipDensity(self.forcefield.lvec[:, :3], self.forcefield.nDim[:3], sigma=self.sigma,
                    multipole=rho, ctx=self.forcefield.ctx)
            else:
                self.rho = rho
            if self.verbose > 0: print('AFMulator.setRho: Preparing buffers')
            self.forcefield.prepareBuffers(rho=self.rho, bDirect=True)
        else:
            self._rho = None
            self.rho = None
            if self.forcefield.rho is not None:
                self.forcefield.rho.release()
                self.forcefield.rho = None

    def setQs(self, Qs, QZs):
        '''Set tip point charges.'''
        self.Qs = Qs
        self.QZs = QZs
        self.forcefield.setQs(Qs=Qs, QZs=QZs)

    # ========= Imaging =========

    def prepareFF(self, xyzs, Zs, qs, pbc_lvec=None, rot=np.eye(3), rot_center=None, REAs=None):
        '''
        Prepare molecule parameters and calculate force field.
        Arguments:
            xyzs: np.ndarray of shape (num_atoms, 3). Positions of atoms in x, y, and z.
            Zs: np.ndarray of shape (num_atoms,). Elements of atoms.
            qs: np.ndarray of shape (num_atoms,) or HartreePotential or None. Charges of atoms or hartree potential.
                If None, then no electrostatics are used.
            pbc_lvec: np.ndarray of shape (3, 3) or None. Unit cell lattice vectors for periodic images of atoms.
                If None, periodic boundaries are disabled, unless qs is HartreePotential and the lvec from the
                Hartree potential is used instead. If npbc = (0, 0, 0), then has no function.
            rot: np.ndarray of shape (3, 3). Rotation matrix to apply to atom positions.
            rot_center: np.ndarray of shape (3,). Center for rotation. Defaults to center of atom coordinates.
            REAs: np.ndarray of shape (num_atoms, 4). Lennard Jones interaction parameters. Calculated automatically if None.
        '''

        # Check if the scan window extends over any non-periodic boundaries and issue a warning if it does
        self.check_scan_window()

        # (Re)initialize force field if the size of the grid changed since last run.
        if (self._old_nDim != self.forcefield.nDim).any():
            if self.verbose > 0: print('(Re)initializing force field buffers.')
            if self.verbose > 1: print(f'old nDim: {self._old_nDim}, new nDim: {self.forcefield.nDim}')
            self.forcefield.tryReleaseBuffers()
            self.setRho(self._rho, self.sigma)
            self.forcefield.prepareBuffers()
        
        # Check if using point charges or precomputed Hartee potential
        if qs is None:
            pot = None
            qs = np.zeros(len(Zs))
        elif isinstance(qs, HartreePotential):
            pot = qs
            qs = np.zeros(len(Zs))
            pbc_lvec = pbc_lvec if pbc_lvec is not None else pot.lvec[1:]
            assert pbc_lvec.shape == (3, 3), f'pbc_lvec has shape {pbc_lvec.shape}, but should have shape (3, 3)'
        else:
            pot = None

        npbc = self.npbc if pbc_lvec is not None else (0, 0, 0)

        if rot_center is None:
            rot_center = xyzs.mean(axis=0)

        # Get Lennard-Jones parameters and apply periodic boundary conditions to atoms
        self.natoms0 = len(Zs)
        if REAs is None:
            self.REAs = PPU.getAtomsREA(self.iZPP, Zs, self.typeParams, alphaFac=-1.0)
        else:
            self.REAs = REAs
        cLJs = PPU.REA2LJ(self.REAs)
        if sum(npbc) > 0:
            Zs, xyzqs, cLJs = PPU.PBCAtoms3D_np(Zs, xyzs, qs, cLJs, pbc_lvec, npbc=npbc)
        else:
            xyzqs = np.concatenate([xyzs, qs[:, None]], axis=1)
        self.Zs = Zs

        # Rotate atom positions
        if pot is None:
            xyzqs[:, :3] -= rot_center
            xyzqs[:, :3] = np.dot(xyzqs[:, :3], rot.T)
            xyzqs[:, :3] += rot_center
            
        # Compute force field
        if self.bNoFFCopy:
            if pot:
                self.forcefield.makeFFHartree(xyzqs[:, :3], cLJs, pot=pot, rho=None, rot=rot,
                    rot_center=rot_center, bRelease=False, bCopy=False, bFinish=False)
            else:
                self.forcefield.makeFF(atoms=xyzqs, cLJs=cLJs, FE=False, Qmix=None,
                    bRelease=False, bCopy=False, bFinish=True, bQZ=True)
            self.atoms = self.forcefield.atoms
            if self.bSaveFF:
                self.saveFF()
        else:
            self.FEin = np.empty(self.forcefield.nDim, np.float32)
            FF, self.atoms = self.forcefield.makeFF(atoms=xyzqs, cLJs=cLJs, FE=self.FEin, Qmix=None,
                bRelease=True,bCopy=True, bFinish=True )
        
        self.atomsNonPBC = self.atoms[:self.natoms0].copy()

    def prepareScanner(self):
        '''
        Prepare scanner. Run after preparing force field.
        '''

        # Copy forcefield array to scanner buffer
        if self.bNoFFCopy:
            self.scanner.updateFEin(self.forcefield.cl_FE)
        else:
            self.scanner.prepareBuffers(self.FEin)
        
        # Subtract origin, because OpenCL kernel for tip relaxation does not take the origin of the FF box into account
        self.pos0 = np.array([0, 0, self.scan_window[1][2]]) - self.lvec[0]

        # Prepare tip position array
        self.scanner.setScanRot(self.pos0, rot=np.eye(3), tipR0=self.tipR0)

    def evalAFM(self, X=None):
        '''
        Evaluate AFM image. Run after preparing force field and scanner.
        Arguments:
            X: np.ndarray of shape (self.scan_dim[0], self.scan_dim[1], self.scan_dim[2]-self.df_steps+1)).
               Array where AFM image will be saved. If None, will be created automatically.
        Returns: np.ndarray. AFM image. If X is not None, this is the same array object as X with values overwritten.
        '''

        if self.bMergeConv:
            FEout = self.scanner.run_relaxStrokesTilted_convZ()
        else:
            self.scanner.run_relaxStrokesTilted(bCopy=False, bFinish=False)
            FEout = self.scanner.run_convolveZ()

        if X is None:
            X = FEout[:,:,:,2].copy()
        else:
            if X.shape != self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight) + 1,):
                raise ValueError(
                    f"Expected an array of shape {self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight) + 1,)} "
                    + f"for storing AFM image, but got an array of shape {X.shape} instead."
                )
            X[:,:,:] = FEout[:,:,:,2]

        return X

    # ========= Debug/Plot Misc. =========

    def saveFF(self):
        FF = self.forcefield.downloadFF()
        FFx=FF[:,:,:,0]
        FFy=FF[:,:,:,1]
        FFz=FF[:,:,:,2]
        Fr = np.sqrt( FFx**2 + FFy**2 + FFz**2 )
        Fbound = 10.0
        mask = Fr.flat > Fbound
        FFx.flat[mask] *= (Fbound/Fr).flat[mask]
        FFy.flat[mask] *= (Fbound/Fr).flat[mask]
        FFz.flat[mask] *= (Fbound/Fr).flat[mask]
        if self.verbose > 0: print("FF.shape ", FF.shape)
        self.saveDebugXSF_FF( self.saveFFpre+"FF_x.xsf", FFx )
        self.saveDebugXSF_FF( self.saveFFpre+"FF_y.xsf", FFy )
        self.saveDebugXSF_FF( self.saveFFpre+"FF_z.xsf", FFz )

    def saveDebugXSF_FF( self, fname, F ):
        if hasattr(self, 'GridUtils'):
            GU = self.GridUtils
        else:
            from . import GridUtils as GU
            self.GridUtils = GU
        if(self.verbose>0): print("saveDebugXSF : ", fname)
        GU.saveXSF( fname, F, self.lvec )

    def check_scan_window(self):
        '''Check that scan window does not extend beyond any non-periodic boundaries.'''
        for i, dim in enumerate('xyz'):
            if self.npbc[i]: continue
            lvec_start = self.lvec[0, i]
            lvec_end = lvec_start + self.lvec[i+1, i]
            scan_start = self.scan_window[0][i] - 0.3   # Small offsets because being close to the boundary
            scan_end = self.scan_window[1][i] + 0.3     # is problematic as well because of PP bending.
            if i == 2:
                scan_start -= self.tipR0[2]
                scan_end -= self.tipR0[2]
            if (scan_start < lvec_start) or (scan_end > lvec_end):
                print(f'Warning: The edge of the scan window in {dim} dimension is very close or extends over '
                    f'the boundary of the force-field grid which is not periodic in {dim} dimension. '
                    'If you get artifacts in the images, please check the boundary conditions and '
                    'the size of the scan window and the force field grid.')

def get_lvec(scan_window, pad=(2.0, 2.0, 3.0), tipR0=(0.0, 0.0, 3.0), pixPerAngstrome=10):
    pad = np.array(pad)
    tipR0 = np.array(tipR0)
    center = (np.array(scan_window[0]) + np.array(scan_window[1])) / 2
    box_size = (np.array(scan_window[1]) - np.array(scan_window[0])) + 2 * pad
    nDim = (pixPerAngstrome * box_size).round().astype(np.int32)
    nDim = np.array([VALID_SIZES[VALID_SIZES >= d][0] for d in nDim])
    box_size = nDim / pixPerAngstrome
    origin = center - box_size / 2 - tipR0
    lvec = np.array([
        origin,
        [box_size[0], 0, 0],
        [0, box_size[1], 0],
        [0, 0, box_size[2]]
    ])
    return lvec

def quick_afm(file_path, scan_size=(16, 16), offset=(0, 0), distance=8.0, scan_step=(0.1, 0.1, 0.1),
        probe_type=8, charge=-0.1, tip='dz2', sigma=0.71, num_heights=10, amplitude=1.0, out_dir=None):
    '''
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
        out_dir: str or None. Output folder path. If None, defaults to afm_ + input_file_name.
    '''

    if not FFcl.oclu or not oclr.oclu:
        oclu.init_env()

    if not out_dir:
        file_name = os.path.split(file_path)[1]
        file_name = os.path.splitext(file_name)[0]
        out_dir = f'afm_{file_name}'.replace(' ', '_')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load input file
    if file_path.endswith('.xsf') or file_path.endswith('.cube'):
        qs, xyzs, Zs = hartreeFromFile(file_path)
        multipole = {tip: charge}
        Qs = [0, 0, 0, 0]
        QZs = [0, 0, 0, 0]
    elif file_path.endswith('.xyz'):
        with open(file_path, 'r') as f:
            xyzs, Zs, _, qs = loadAtomsLines(f.readlines())
        multipole = {}
        if tip == 's':
            Qs = [charge, 0, 0, 0]
            QZs = [0, 0, 0, 0]
        elif tip == 'pz':
            Qs = [10*charge, -10*charge, 0, 0]
            Qzs = [0.1, -0.1, 0, 0]
        elif tip == 'dz2':
            Qs = [100*charge, -200*charge, 100*charge, 0]
            QZs = [0.1, 0, -0.1, 0]
        else:
            raise ValueError(f'Unsupported tip type `{tip}` for point charges.')
    else:
        raise ValueError(f'Unsupported file format in file `{file_path}`.')

    # Figure out scan parameters
    z_max = xyzs[:, 2].max() + distance
    xy_center = (xyzs[:, :2].min(axis=0) + xyzs[:, :2].max(axis=0)) / 2 + np.array(offset)
    df_steps = int(amplitude / scan_step[2])
    z_size = (num_heights + df_steps - 1) * scan_step[2]
    scan_window = (
        (xy_center[0] - scan_size[0] / 2, xy_center[1] - scan_size[1] / 2, z_max - z_size),
        (xy_center[0] + scan_size[0] / 2, xy_center[1] + scan_size[1] / 2, z_max)
    )
    scan_dim = (int(scan_size[0] / scan_step[0]), int(scan_size[1] / scan_step[1]), int(z_size / scan_step[2]))

    # Do scan
    afmulator = AFMulator(
        scan_window=scan_window,
        scan_dim=scan_dim,
        rho=multipole,
        iZPP=probe_type,
        Qs=Qs,
        QZs=QZs,
        df_steps=df_steps
    )
    X = afmulator(xyzs, Zs, qs)

    # Plot
    X = X[:, :, ::-1].T
    extent = [scan_window[0][0], scan_window[1][0], scan_window[0][1], scan_window[1][1]]
    zs = np.linspace(scan_window[0][2], scan_window[1][2], scan_dim[2]+1)[:num_heights]
    plotImages(os.path.join(out_dir, 'df'), X, range(len(X)), extent=extent, zs=zs)
