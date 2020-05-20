
import numpy as np

from . import basUtils
from . import elements
from . import oclUtils     as oclu
from . import fieldOCL     as FFcl
from . import RelaxOpenCL  as oclr

class AuxMapBase:
    '''
    Base class for AuxMap subclasses.
    Arguments:
        scan_dim: tuple of two ints. Indicates the pixel size of the scan in x and y.
        scan_window: tuple ((start_x, start_y), (end_x, end_y)). The start and end coordinates of scan region in angstroms.
        zmin: float. Deepest coordinate that is still included. Top is defined to be at 0.
    '''
    def __init__(self, scan_dim, scan_window, zmin=None):
        if not FFcl.oclu:
            raise RuntimeError('OpenCL context not initialized. Initialize with fieldOCL.init before creating an AuxMap object.')
        self.scan_dim = scan_dim
        self.scan_window = scan_window
        self.projector = FFcl.AtomProcjetion()
        if zmin:
            self.projector.zmin = zmin
        self.nChan = 1
        
    def prepare_projector(self, xyzqs, Zs, pos0, bonds2atoms=None):
        rot = np.eye(3)
        coefs = self.projector.makeCoefsZR( Zs, elements.ELEMENTS )
        self.projector.tryReleaseBuffers()
        self.projector.prepareBuffers( xyzqs.astype(np.float32), self.scan_dim[:2]+(self.nChan,), coefs=coefs, bonds2atoms=bonds2atoms )
        return oclr.preparePossRot( self.scan_dim, pos0, rot[0], rot[1], self.scan_window[0], self.scan_window[1] )
          
class vdwSpheres(AuxMapBase):
    '''
    Generate vdW Spheres descriptors for molecules. Each atom is represented by a projection of a sphere 
    with radius equal to the vdW radius of the element.
    Arguments:
        Rpp: float. A constant that is added to the vdW radius of each atom.
    '''

    def __init__(self, scan_dim=(128, 128), scan_window=((-8, -8), (8, 8)), zmin=-1.5, Rpp=-0.5):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rpp = Rpp
                
    def eval(self, xyzqs, Zs):
        assert xyzqs.shape[1] == 4
        coefs = self.projector.makeCoefsZR( Zs, elements.ELEMENTS )
        pos0 = [0, 0, (xyzqs[:,2]+coefs[:,3]).max()+self.projector.Rpp]
        poss = self.prepare_projector(xyzqs, Zs, pos0)
        return self.projector.run_evalSpheres( poss = poss, tipRot=oclr.mat3x3to4f(np.eye(3)) )[:,:,0]

class AtomicDisks(AuxMapBase):
    '''
    Generate Atomic Disks descriptors for molecules. Each atom is represented by a conically decaying disk.
    Arguments:
        zmax_s: float. The maximum depth of vdW sphere shell when diskMode='sphere'.
        Rpp: float. A constant that is added to the vdW radius of each atom.
        diskMode: 'sphere' or 'center'. With 'center' only the center coordinates are considered, when deciding
                  whether an atom is too deep. With 'sphere' also the effective size of the atom is take into
                  account.
    '''
    def __init__(self, scan_dim=(128, 128), scan_window=((-8, -8), (8, 8)), zmin=-1.5, zmax_s=2.0, Rpp=-0.5, diskMode='sphere'):
        super().__init__(scan_dim, scan_window)
        self.projector.dzmax = -zmin
        self.projector.dzmax_s = zmax_s
        self.projector.Rpp = Rpp
        if diskMode == 'sphere':
            self.offset = 0.0
        elif diskMode == 'center':
            self.projector.dzmax_s = np.Inf
            self.offset = -1.0
        else:
            raise ValueError(f'Unknown diskMode {diskMode}. Should be either sphere or center')
                
    def eval(self, xyzqs, Zs):
        assert xyzqs.shape[1] == 4
        pos0 = [0, 0, xyzqs[:,2].max()]
        poss = self.prepare_projector(xyzqs, Zs, pos0)
        return self.projector.run_evaldisks( poss = poss, tipRot=oclr.mat3x3to4f(np.eye(3)), offset=self.offset )[:,:,0]
        
class MultiMapSpheres(AuxMapBase):
    '''
    Generate Multimap vdW Spheres descriptors for molecules. Each atom is represented by a projection of a sphere
    with radius equal to the vdW radius of the element. Different sizes of atoms are separated into different
    channels based on their vdW radii.
    Arguments:
        Rpp: float. A constant that is added to the vdW radius of each atom.
        nChan: int. Number of channels.
        Rmin: float. Minimum radius.
        Rstep: float. Size range per bin.
        bOccl: 0 or 1. Switch occlusion of atoms 0=off 1=on.
    '''

    def __init__(self, scan_dim=(128, 128), scan_window=((-8, -8), (8, 8)), zmin=-1.5, Rpp=-0.5, nChan=3, Rmin=1.4, Rstep=0.3, bOccl=0):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rpp = Rpp
        self.nChan = nChan
        self.Rmin = Rmin
        self.Rstep = Rstep
        self.bOccl = bOccl
                
    def eval(self, xyzqs, Zs):
        assert xyzqs.shape[1] == 4
        coefs = self.projector.makeCoefsZR( Zs, elements.ELEMENTS )
        pos0 = [0, 0, (xyzqs[:,2]+coefs[:,3]).max()+self.projector.Rpp]
        poss = self.prepare_projector(xyzqs, Zs, pos0)
        return self.projector.run_evalMultiMapSpheres( poss = poss, tipRot=oclr.mat3x3to4f(np.eye(3)), bOccl=self.bOccl, Rmin=self.Rmin, Rstep=self.Rstep )
        
class Bonds(AuxMapBase):
    '''
    Generate Bonds descriptors for molecules. Bonds between atoms are represented by ellipses.
    Arguments:
        Rfunc: numpy array of numpy.float32. Radial function of bonds&atoms potential.
        Rmax: float. Cutoff for radial function.
        drStep: float. Step dx (dr) for sampling of radial function.
        ellipticity: float. Ratio between major and minor semiaxis.
    '''
    def __init__(self, scan_dim=(128, 128), scan_window=((-8, -8), (8, 8)), zmin=-1.5, Rfunc=None, Rmax=10.0, drStep=0.1, ellipticity=0.5):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rmax = Rmax
        self.projector.drStep = drStep
        self.projector.elipticity = ellipticity
        if not Rfunc:
            xs = np.linspace(0.0,10.0,100)
            dx = xs[1]-xs[0];
            xs -= dx
            ys = np.exp( -5*xs )
            self.projector.Rfunc = ys.astype(np.float32)
        else:
            assert Rfunc.dtype == np.float32
            self.projector.Rfunc = Rfunc
                
    def eval(self, xyzqs, Zs):
        assert xyzqs.shape[1] == 4
        pos0 = [0, 0, xyzqs[:,2].max()]
        bonds2atoms = np.array( basUtils.findBonds_( xyzqs, Zs, 1.2, ELEMENTS=elements.ELEMENTS ), dtype=np.int32 )
        poss = self.prepare_projector(xyzqs, Zs, pos0, bonds2atoms=bonds2atoms)
        return self.projector.run_evalBondEllipses( poss = poss, tipRot=oclr.mat3x3to4f(np.eye(3)) )[:,:,0]
        
class AtomRfunc(AuxMapBase):
    '''
    Generate AtomRfunc descriptors for molecules. Atoms are represented by disks with decay determined by Rfunc.
    Arguments:
        Rfunc: numpy.ndarray of numpy.float32. Radial function of bonds&atoms potential.
        Rmax: float. Cutoff for radial function.
        drStep: float. Step dx (dr) for sampling of radial function.
    '''
    def __init__(self, scan_dim=(128, 128), scan_window=((-8, -8), (8, 8)), zmin=-1.5, Rfunc=None, Rmax=10.0, drStep=0.1):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rmax = Rmax
        self.projector.drStep = drStep
        if not Rfunc:
            xs = np.linspace(0.0,10.0,100)
            dx = xs[1]-xs[0];
            xs -= dx
            ys = np.exp( -5*xs )
            self.projector.Rfunc = ys.astype(np.float32)
        else:
            assert Rfunc.dtype == np.float32
            self.projector.Rfunc = Rfunc
                
    def eval(self, xyzqs, Zs):
        assert xyzqs.shape[1] == 4
        pos0 = [0, 0, xyzqs[:,2].max()]
        bonds2atoms = np.array( basUtils.findBonds_( xyzqs, Zs, 1.2, ELEMENTS=elements.ELEMENTS ), dtype=np.int32 )
        poss = self.prepare_projector(xyzqs, Zs, pos0, bonds2atoms=bonds2atoms)
        return self.projector.run_evalAtomRfunc( poss = poss, tipRot=oclr.mat3x3to4f(np.eye(3)) )[:,:,0]

aux_map_dict = {
    'vdwSpheres': vdwSpheres,
    'AtomicDisks': AtomicDisks,
    'MultiMapSpheres': MultiMapSpheres,
    'Bonds': Bonds,
    'AtomRfunc': AtomRfunc
}

class AuxMapFactory:

    def __call__(self, map_type, args={}):
        try:
            aux_map = aux_map_dict[map_type](**args)
        except KeyError:
            recognized_types = ", ".join([f'{key}' for key in aux_map_dict.keys()])
            raise ValueError(f'Unrecognized AuxMap type {map_type}. Should be one of {recognized_types}.')
        return aux_map

AuxMaps = AuxMapFactory()

