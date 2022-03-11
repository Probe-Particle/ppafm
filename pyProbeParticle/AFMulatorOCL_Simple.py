#!/usr/bin/python3

import numpy as np

from . import common       as PPU
from . import fieldOCL     as FFcl
from . import RelaxOpenCL  as oclr
from . import HighLevelOCL as hl

from .fieldOCL import HartreePotential

class AFMulator():
    '''
    Simulate Atomic force microscope images of molecules.
    Arguments:
        pixPerAngstrome: int. Number of pixels (voxels) per angstrom in force field grid.
        lvec: np.ndarray of shape (4,3). Unit cell boundaries for force field. First (row) vector specifies the origin,
            and the remaining three vectors specify the edge vectors of the unit cell.
        scan_dim: tuple of three ints. Number of scan steps in the (x, y, z) dimensions. The size of the resulting
            images have a size of scan_dim[0] x scan_dim[1] and the number images at different heights is
            scan_dim[2] - df_steps + 1.
        scan_window: tuple ((x_min, y_min, z_min), (x_max, y_max, z_max)). The minimum and maximum coordinates of
            scan region in angstroms. The scan region is a rectangular box with the opposing corners defined by the
            coordinates in scan_window. Note that the step size in z dimension is the scan window size in z divided by
            scan_dim[2], and the scan in z-direction proceeds for scan_dim[2] steps, so the final step is one step short
            of z_min.
        iZPPs: int. Element of probe particle.
        QZs: Array of lenght 4. Position tip charges along z-axis relative to probe-particle center in angstroms.
        Qs: Array of lenght 4. Values of tip charges in units of e. Some can be zero.
        df_steps: int. Number of steps in z convolution. The total amplitude is df_steps times scan z-step size.
        tipR0: array of length 3. Probe particle equilibrium position (x, y, z) in angstroms.
        tipStiffness: array of length 4. Harmonic spring constants (x, y, z, r) for holding the probe particle
            to the tip in N/m.
        npbc: tuple of three ints. How many periodic images of atoms to use in (x, y, z) dimensions. Used for calculating
            Lennard-Jones force field and electrostatic field from point charges. Electrostatic field from a Hartree
            potential defined on a grid is always considered to be periodic.
        initFF: Bool. Whether to initialize buffers. Set to False to modify force field and scanner parameters
            before initialization.
    '''

    bNoPoss    = True    # Use algorithm which does not need to store array of FF_grid-sampling positions in memory (neither GPU-memory nor main-memory)
    bNoFFCopy  = True    # Should we copy Force-Field grid from GPU  to main_mem ?  ( Or we just copy it just internally withing GPU directly to image/texture used by RelaxedScanner )
    bFEoutCopy = False   # Should we copy un-convolved FEout from GPU to main_mem ? ( Or we should copy oly final convolved FEconv? ) 
    bMergeConv = False   # Should we use merged kernel relaxStrokesTilted_convZ or two separated kernells  ( relaxStrokesTilted, convolveZ  )

    # --- Relaxation
    relaxParams  = [ 0.5, 0.1, 0.1*0.2,0.1*5.0 ]    # (dt,damp, .., .. ) parameters of relaxation, in the moment just dt and damp are used

    # --- Output
    bSaveFF  = False    # Save debug ForceField as .xsf
    verbose  = 0        # Print information during excecution

    # ==================== Methods =====================

    def __init__( self, 
        pixPerAngstrome = 10,
        lvec            = np.array([
                            [ 0.0,  0.0, 0.0],
                            [20.0,  0.0, 0.0],
                            [ 0.0, 20.0, 0.0],
                            [ 0.0,  0.0, 8.0]
                          ]),
        scan_dim        = (128, 128, 30),
        scan_window     = ((2.0, 2.0, 7.0), ( 18.0, 18.0, 10.0)),
        iZPP            = 8,
        QZs             = [ 0.1,  0, -0.1, 0 ],
        Qs              = [ -10, 20,  -10, 0 ],
        df_steps        = 10,
        tipR0           = [0.0, 0.0, 3.0],
        tipStiffness    = [0.25, 0.25, 0.0, 30.0],
        npbc            = (1, 1, 0),
        initFF          = True 
    ):

        if not FFcl.oclu:
            raise RuntimeError('Force field OpenCL context not initialized. Initialize with FFcl.init before creating an AFMulator object.')
        elif not oclr.oclu:
            raise RuntimeError('Scanner OpenCL context not initialized. Initialize with oclr.init before creating an AFMulator object.')
        
        self.pixPerAngstrome = pixPerAngstrome
        self.lvec = lvec
        self.scan_dim = scan_dim
        self.scan_window = scan_window
        self.iZPP = iZPP
        self.df_steps = df_steps
        self.tipR0 = tipR0
        self.npbc = npbc

        self.typeParams = hl.loadSpecies('atomtypes.ini')

        self.forcefield = FFcl.ForceField_LJC()
        self.setQs(Qs, QZs)

        self.scanner = oclr.RelaxedScanner()
        self.scanner.relax_params = np.array( self.relaxParams, dtype=np.float32 )
        self.scanner.stiffness = np.array(tipStiffness, dtype=np.float32) / -16.0217662

        if initFF:
            self.initFF()

        self.saveFFpre = ""

        self.counter = 0
    
    def eval(self, xyzs, Zs, qs=None, REAs=None, X=None ):
        '''
        Prepare and evaluate AFM image.
        Arguments:
            xyzs: np.ndarray of shape (num_atoms, 3). Positions of atoms in x, y, and z.
            Zs: np.ndarray of shape (num_atoms,). Elements of atoms.
            qs: np.ndarray of shape (num_atoms,) or HarteePotential. Charges of atoms or hartree potential.
            REAs: np.ndarray of shape (num_atoms, 4). Lennard Jones interaction parameters. Calculated automatically if None.
            X: np.ndarray of shape (self.scan_dim[0], self.scan_dim[1], self.scan_dim[2]-self.df_steps)).
               Array where AFM image will be saved. If None, will be created automatically.
        Returns: np.ndarray. AFM image. If X is not None, this is the same array object as X with values overwritten.
        '''
        self.prepareFF(xyzs, Zs, qs, REAs)
        self.prepareScanner()
        X = self.evalAFM(X)
        return X
    
    def __call__( self, xyzs, Zs, qs, REAs=None, X=None ):
        '''
        Makes object callable. See eval for input arguments.
        '''
        return self.eval( xyzs, Zs, qs, REAs=REAs, X=X )

    def eval_( self, mol ):
        return self.eval( mol.xyzs, mol.Zs, mol.qs )

    # ========= Setup =========

    def initFF(self):
        '''
        Initialize force field and scanner buffers. Call this method after changing parameters in the scanner or forcefield
        or after modifying any of the following attributes: lvec, pixPerAngstrome, scan_dim, scan_window, dfWeight.
        '''

        # Set df convolution weights
        self.dz = (self.scan_window[1][2] - self.scan_window[0][2]) / self.scan_dim[2]
        self.dfWeight = PPU.getDfWeight(self.df_steps, dz=self.dz)[0].astype(np.float32)
        self.amplitude = self.dz * len(self.dfWeight)

        # Initialize force field
        if self.bNoPoss:
            self.forcefield.initSampling( self.lvec, pixPerAngstrome=self.pixPerAngstrome )
        else:
            self.forcefield.initPoss( lvec=self.lvec, pixPerAngstrome=self.pixPerAngstrome )

        # Initialize scanner
        self.scanner.zstep = self.dz
        if self.bNoFFCopy:
            self.scanner.prepareBuffers(lvec=self.lvec, FEin_cl=self.forcefield.cl_FE,
                FEin_shape=self.forcefield.nDim, scan_dim=self.scan_dim, nDimConv=len(self.dfWeight),
                nDimConvOut=(self.scan_dim[2] - len(self.dfWeight) + 1)
            )
            self.scanner.updateBuffers(WZconv=self.dfWeight)
            self.scanner.preparePosBasis( start=self.scan_window[0][:2], end=self.scan_window[1][:2] )
        else:
            self.FEin = np.empty( self.forcefield.nDim, np.float32 )

    def setQs(self, Qs, QZs):
        '''
        Set tip charges.
        '''
        self.Qs = Qs
        self.QZs = QZs
        self.forcefield.setQs( Qs=Qs, QZs=QZs )

    # ========= Imaging =========

    def prepareFF(self, xyzs, Zs, qs, REAs=None):
        '''
        Prepare molecule parameters and calculate force field.
        Arguments:
            xyzs: np.ndarray of shape (num_atoms, 3). Positions of atoms in x, y, and z.
            Zs: np.ndarray of shape (num_atoms,). Elements of atoms.
            qs: np.ndarray of shape (num_atoms,) or HarteePotential. Charges of atoms or hartree potential.
            REAs: np.ndarray of shape (num_atoms, 4). Lennard Jones interaction parameters. Calculated automatically if None.
        '''

        # Check if using point charges or precomputed Hartee potential
        if isinstance(qs, HartreePotential):
            pot = qs
            qs = np.zeros(len(Zs))
        else:
            pot = None

        # Get Lennard-Jones parameters
        self.natoms0 = len(Zs)
        if REAs is None:
            self.REAs = PPU.getAtomsREA(self.iZPP, Zs, self.typeParams, alphaFac=-1.0)
        else:
            self.REAs = REAs
        cLJs = PPU.REA2LJ(self.REAs)
        if( self.npbc is not None ):
            Zs, xyzqs, cLJs = PPU.PBCAtoms3D_np(Zs, xyzs, qs, cLJs, self.lvec[1:], npbc=self.npbc)
        self.Zs = Zs
            
        # Compute force field
        if self.bNoFFCopy:
            self.forcefield.makeFF( atoms=xyzqs, cLJs=cLJs, pot=pot, FE=False,
                Qmix=None, bRelease=False, bCopy=False, bFinish=True, bQZ=True)
            self.atoms = self.forcefield.atoms
            if self.bSaveFF:
                FF = self.forcefield.downloadFF( )
                FFx=FF[:,:,:,0]
                FFy=FF[:,:,:,1]
                FFz=FF[:,:,:,2]
                Fr = np.sqrt( FFx**2 + FFy**2 + FFz**2 )
                Fbound = 10.0
                mask = Fr.flat > Fbound
                FFx.flat[mask] *= (Fbound/Fr).flat[mask]
                FFy.flat[mask] *= (Fbound/Fr).flat[mask]
                FFz.flat[mask] *= (Fbound/Fr).flat[mask]
                print("FF.shape ", FF.shape)
                self.saveDebugXSF_FF( self.saveFFpre+"FF_x.xsf", FFx )
                self.saveDebugXSF_FF( self.saveFFpre+"FF_y.xsf", FFy )
                self.saveDebugXSF_FF( self.saveFFpre+"FF_z.xsf", FFz )
                #self.saveDebugXSF_FF( "FF_E.xsf", FF[:,:,:,3] )
        else:
            FF,self.atoms  = self.forcefield.makeFF( atoms=xyzqs, cLJs=cLJs, FE=self.FEin, Qmix=None, bRelease=True, bCopy=True, bFinish=True )
        self.atomsNonPBC = self.atoms[:self.natoms0].copy()

    def prepareScanner(self):
        '''
        Prepare scanner. Run after preparing force field.
        '''
        if self.bNoFFCopy:
            self.scanner.updateFEin( self.forcefield.cl_FE )
        else:
            if(self.counter>0): # not first step
                if(self.verbose > 1): print("scanner.releaseBuffers()")
                self.scanner.releaseBuffers()
            self.scanner.prepareBuffers( self.FEin, self.lvec, scan_dim=self.scan_dim, nDimConv=len(self.dfWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight)+1 )
            self.scanner.preparePosBasis( start=self.scan_window[0][:2], end=self.scan_window[1][:2] )
        
        # Subtract origin, because OpenCL kernel for tip relaxation does not take the origin of the FF box into account
        self.pos0 = np.array([0, 0, self.scan_window[1][2]]) - self.lvec[0]

        self.scanner.setScanRot(self.pos0, rot=np.eye(3), tipR0=self.tipR0)

    def evalAFM( self, X=None ):
        '''
        Evaluate AFM image. Run after preparing force field and scanner.
        Arguments:
            X: np.ndarray of shape (self.scan_dim[0], self.scan_dim[1], self.scan_dim[2]-self.df_steps)).
               Array where AFM image will be saved. If None, will be created automatically.
        Returns: np.ndarray. AFM image. If X is not None, this is the same array object as X with values overwritten.
        '''
        if self.bMergeConv:
            FEout = self.scanner.run_relaxStrokesTilted_convZ()
        else:
            if self.bFEoutCopy:
                FEout = self.scanner.run_relaxStrokesTilted( bCopy=True, bFinish=True )
            else:
                #print("DEBUG  HERE !!!! ")
                self.scanner.run_relaxStrokesTilted( bCopy=False, bFinish=True )
            if( len(self.dfWeight) != self.scanner.scan_dim[2] - self.scanner.nDimConvOut + 1):
                raise ValueError(
                    "len(dfWeight) must be self.scanner.scan_dim[2] - self.scanner.nDimConvOut + 1 but got "
                    + f"len(self.dfWeight) = {len(self.dfWeight)}, "
                    + f"self.scanner.scan_dim[2] = {self.scanner.scan_dim[2]}, "
                    + f"self.scanner.nDimConvOut = {self.scanner.nDimConvOut}"
                )
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

    def saveDebugXSF_FF( self, fname, F ):
        if hasattr(self, 'GridUtils'):
            GU = self.GridUtils
        else:
            from . import GridUtils as GU
            self.GridUtils = GU
        if(self.verbose>0): print("saveDebugXSF : ", fname)
        GU.saveXSF( fname, F, self.lvec )

# ==========================================================
# ==========================================================
# ====================== TEST RUN ==========================
# ==========================================================
# ==========================================================

if __name__ == "__main__":

    import matplotlib as mpl #; mpl.use('Agg')
    import matplotlib.pyplot as plt
    import time
    import os

    from . import basUtils
    from . import oclUtils as oclu

    molecules = ["out2", "out3","benzeneBrCl2"]
    args = {
        'pixPerAngstrome'   : 8,
        'lvec'              : np.array([
                                [ 0.0,  0.0, 0.0],
                                [20.0,  0.0, 0.0],
                                [ 0.0, 20.0, 0.0],
                                [ 0.0,  0.0, 5.0]
                              ]),
        'scan_dim'          : (128, 128, 30),
        'scan_window'       : ((2.0, 2.0, 5.0), (18.0, 18.0, 8.0)),
        'iZPP'              : 8,
        'QZs'               : [ 0.1,  0, -0.1, 0 ],
        'Qs'                : [ -10, 20,  -10, 0 ],
        'amplitude'         : 1.0,
        'df_steps'          : 10,
        'initFF'            : True
    }

    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    afmulator = AFMulator(**args)
    afmulator.npbc = (1,1,0)

    afmulator            .verbose  = 1
    afmulator.forcefield .verbose  = 1
    afmulator.scanner    .verbose  = 1

    FFcl.bRuntime = True

    for mol in molecules:

        filename = f'{mol}/pos.xyz'
        atom_lines = open( filename ).readlines()
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( atom_lines )
        xyzs[:,:2] = xyzs[:,:2] - xyzs[:,:2].mean(axis=0) + np.array([10, 10]) 
        xyzs[:,2] = xyzs[:,2] - xyzs[:,2].max() - 1.5

        t0 = time.time()
        X = afmulator(xyzs, Zs, qs)
        print(f'Simulation time for {mol}: {time.time() - t0}')
        print(X.shape)

        rows, cols = 4, 5
        fig = plt.figure(figsize=(3.2*cols,2.5*rows))
        for k in range(X.shape[-1]):
            fig.add_subplot(rows, cols, k+1)
            plt.imshow(X[:,:,k].T, cmap='afmhot', origin="lower")
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{mol}/afm.png')
        plt.show()
        plt.close()