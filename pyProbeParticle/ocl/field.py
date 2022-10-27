#!/usr/bin/python

import os
import time
import numpy as np 

import pyopencl as cl
from pyopencl import array

from ..GridUtils import loadCUBE, loadXSF
from ..basUtils import loadAtomsCUBE, loadXSFGeom
from ..fieldFFT import getProbeDensity

try:
    from reikna.cluda import ocl_api, dtypes
    from reikna.fft import FFT
    from reikna.core import Annotation, Type, Transformation, Parameter
    fft_available = True
except ModuleNotFoundError:
    fft_available = False

DEFAULT_FD_STEP = 0.05

cl_program = None
oclu       = None

def init(env):
    global cl_program
    global oclu
    cl_program = env.loadProgram(env.CL_PATH+"/FF.cl")
    oclu = env

verbose    = 0
bRuntime   = False

# ========= init Args 

def getCtxQueue():
    return oclu.ctx, oclu.queue

def initArgsCoulomb( atoms, poss ):
    ctx,queue = getCtxQueue()
    nbytes     =  0;
    nAtoms     = np.int32( len(atoms) ) 
    mf         = cl.mem_flags
    cl_atoms   = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
    cl_poss    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes
    cl_FE      = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   ); nbytes+=poss.nbytes
    kargs = ( nAtoms, cl_atoms, cl_poss, cl_FE )
    if(verbose>0): print("initArgsCoulomb.nbytes ", nbytes)
    return kargs 

def initArgsLJC( atoms, cLJs, poss ):
    ctx,queue = getCtxQueue()
    nbytes     =  0;
    nAtoms   = np.int32( len(atoms) ) 
    mf       = cl.mem_flags
    cl_atoms = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
    cl_cLJs  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=cLJs  ); nbytes+=cLJs.nbytes
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes   # float4
    cl_FE    = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes ); nbytes+=poss.nbytes # float4     # we are using Qmix now
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    if(verbose>0):print("initArgsLJC.nbytes ", nbytes)
    return kargs

def initArgsLJ(atoms,cLJs, poss ):
    ctx,queue = getCtxQueue()
    nbytes     =  0;
    nAtoms   = np.int32( len(atoms) ) 
    mf       = cl.mem_flags
    cl_atoms = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
    cl_cLJs  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=cLJs  ); nbytes+=cLJs.nbytes
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes   # float4
    cl_FE    = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   ); nbytes+=poss.nbytes   # float4
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    if(verbose>0):print("initArgsLJ.nbytes ", nbytes)
    return kargs

def initArgsMorse(atoms,REAs, poss ):
    ctx,queue = getCtxQueue()
    nbytes     =  0;
    nAtoms   = np.int32( len(atoms) ) 
    mf       = cl.mem_flags
    cl_atoms = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
    cl_REAs  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=REAs  ); nbytes+=REAs.nbytes
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes # float4
    cl_FE    = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   ); nbytes+=poss.nbytes # float4
    kargs = ( nAtoms, cl_atoms, cl_REAs, cl_poss, cl_FE )
    if(verbose>0):print("initArgsMorse.nbytes ", nbytes)
    return kargs

def releaseArgs( kargs ):
    for karg in kargs[1:]:
        karg.release()

# ========= Update Args 

def updateArgsLJC( kargs_old, atoms=None, cLJs=None, poss=None ):
    ctx,queue = getCtxQueue()
    mf       = cl.mem_flags
    if kargs_old is None:
        return initArgsLJC( atoms, cLJs, poss )
    else:
        if atoms is not None:
            nAtoms   = np.int32( len(atoms) )
            if (kargs_old[0] != nAtoms):
                if(verbose>0): print(" kargs_old[0] != nAtoms; TRY only")
                return initArgsLJC( atoms, cLJs, poss )
            else:
                cl_atoms=kargs_old[1]
                cl.enqueue_copy( queue, cl_atoms, atoms )
        else:
            cl_atoms=kargs_old[1]
        if cLJs is not None:
            cl_cLJs=kargs_old[2]
            cl.enqueue_copy( queue, cl_cLJs, cLJs )
        else:
            cl_cLJs=kargs_old[2]
        if poss is not None:
            cl_poss=kargs_old[3]
            cl.enqueue_copy( queue, cl_poss, poss )

        else:
            cl_poss=kargs_old[3]

    cl_FE=kargs_old[4]
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    return kargs

def updateArgsMorse( kargs_old=None, atoms=None, REAs=None, poss=None ):
    ctx,queue = getCtxQueue()
    mf       = cl.mem_flags
    if kargs_old is None:
        return initArgsMorse( atoms, REAs, poss )
    else:
        if atoms is not None:
            nAtoms   = np.int32( len(atoms) )
            if (kargs_old[0] != nAtoms):
                if(verbose>0): print(" kargs_old[0] != nAtoms; TRY only")#; exit()
                return initArgsMorse( atoms, REAs, poss )
            else:
                cl_atoms=kargs_old[1]
                cl.enqueue_copy( queue, cl_atoms, atoms )
        else:
            cl_atoms=kargs_old[1]

        if REAs is not None:
            cl_cREAs=kargs_old[2]
            cl.enqueue_copy( queue, cl_cREAs, REAs )
        else:
            cl_cREAs=kargs_old[2]

        if poss is not None:
            cl_poss=kargs_old[3]
            cl.enqueue_copy( queue, cl_poss, poss )
        else:
            cl_poss=kargs_old[3]

        cl_FE=kargs_old[4]
        kargs = ( nAtoms, cl_atoms, cl_cREAs, cl_poss, cl_FE )
        return kargs

def updateArgsLJ( kargs_old, atoms=None, cLJs=None, poss=None ):
    ctx,queue = getCtxQueue()
    mf       = cl.mem_flags
    if kargs_old is None:
        return initArgsLJ( atoms, cLJs, poss )
    else:
        if atoms is not None:
            nAtoms   = np.int32( len(atoms) )
            if (kargs_old[0] != nAtoms):
                if(verbose>0): print(" kargs_old[0] != nAtoms; TRY only")
                return initArgsLJ( atoms, cLJs, poss )
            else:
                cl_atoms=kargs_old[1]
                cl.enqueue_copy( queue, cl_atoms, atoms )
        else:
            cl_atoms=kargs_old[1]
        if cLJs is not None:
            cl_cLJs=kargs_old[2]
            cl.enqueue_copy( queue, cl_cLJs, cLJs )
        else:
            cl_cLJs=kargs_old[2]
        if poss is not None:
            cl_poss=kargs_old[3]
            cl.enqueue_copy( queue, cl_poss, poss )
        else:
            cl_poss=kargs_old[3]

    cl_FE=kargs_old[4]
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    return kargs

def makeDivisibleUp( num, divisor ):
    rest = num % divisor;
    if rest > 0: num += (divisor-rest)
    return num

# ========= Run Job

def runCoulomb( kargs, nDim, local_size=(32,) ):
    ctx,queue = getCtxQueue()
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,) 
    FE          = np.zeros( nDim+(4,) , dtype=np.float32 )
    cl_program.evalLJC ( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy    ( queue, FE, kargs[3] );
    queue.finish()
    return FE

def runLJC( kargs, nDim, local_size=(32,) ):
    ctx,queue = getCtxQueue()
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,) # TODO make sure divisible by local_size
    FE = np.zeros( nDim+(8,), dtype=np.float32 ) # float8
    if(verbose>0): print("FE.shape ", FE.shape)
    cl_program.evalLJC( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy( queue, FE, kargs[4] )
    queue.finish()
    return FE

def runLJ( kargs, nDim, local_size=(32,) ):  # slowed down, because of problems with the field far away
    ctx,queue = getCtxQueue()
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,)
    FE          = np.zeros( nDim+(4,) , dtype=np.float32 ) # float4
    cl_program.evalLJ( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy( queue, FE, kargs[4] )
    queue.finish()
    return FE

def runMorse( kargs, nDim, local_size=(32,) ):
    ctx,queue = getCtxQueue()
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,)
    FE          = np.zeros( nDim+(4,) , dtype=np.float32 ) # float4
    cl_program.evalMorse( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy( queue, FE, kargs[4] )
    queue.finish()
    return FE

# ========= getPos

def genFFSampling( lvec, pixPerAngstrome=10 ):
    nDim = np.array([
        int(round(pixPerAngstrome * np.sqrt(np.dot(lvec[1],lvec[1])) )),
        int(round(pixPerAngstrome * np.sqrt(np.dot(lvec[2],lvec[2])) )),
        int(round(pixPerAngstrome * np.sqrt(np.dot(lvec[3],lvec[3])) )),
        4,
    ], np.int32 )
    return nDim

def getPos(lvec, nDim=None, step=(0.1,0.1,0.1) ):
    if nDim is None:
        nDim = (    int(np.linalg.norm(lvec[3,:])/step[2]),
                    int(np.linalg.norm(lvec[2,:])/step[1]),
                    int(np.linalg.norm(lvec[1,:])/step[0]))
    dCell = np.array( ( lvec[1,:]/nDim[2], lvec[2,:]/nDim[1], lvec[3,:]/nDim[0] ) ) 
    ABC   = np.mgrid[0:nDim[0],0:nDim[1],0:nDim[2]]
    X = lvec[0,0] + ABC[2]*dCell[0,0] + ABC[1]*dCell[1,0] + ABC[0]*dCell[2,0]
    Y = lvec[0,1] + ABC[2]*dCell[0,1] + ABC[1]*dCell[1,1] + ABC[0]*dCell[2,1] 
    Z = lvec[0,2] + ABC[2]*dCell[0,2] + ABC[1]*dCell[1,2] + ABC[0]*dCell[2,2] 
    return X, Y, Z
	
def XYZ2float4(X,Y,Z):
    nDim = X.shape
    XYZW = np.zeros( (nDim[0],nDim[1],nDim[2],4), dtype=np.float32)
    XYZW[:,:,:,0] = X
    XYZW[:,:,:,1] = Y
    XYZW[:,:,:,2] = Z
    return XYZW

def getposs( lvec, nDim=None, step=(0.1,0.1,0.1) ):
    X,Y,Z   = getPos( lvec, nDim=nDim, step=step ); 
    poss    = XYZ2float4(X,Y,Z)
    return poss
    
def atoms2float4(atoms):
    atoms_   = np.zeros( (len(atoms[0]),4), dtype=np.float32)
    atoms_[:,0] = np.array( atoms[1] )
    atoms_[:,1] = np.array( atoms[2] )
    atoms_[:,2] = np.array( atoms[3] )
    atoms_[:,3] = np.array( atoms[4] )
    return atoms_
    
def xyzq2float4(xyzs,qs):
    atoms_       = np.zeros( (len(qs),4), dtype=np.float32)
    atoms_[:,:3] = xyzs[:,:]
    atoms_[:, 3] = qs[:]      
    return atoms_

def CLJ2float2(C6s,C12s):
    cLJs      = np.zeros( (len(C6s),2), dtype=np.float32)
    cLJs[:,0] = C6s
    cLJs[:,1] = C12s
    return cLJs

def hartreeFromFile(file_path):
    '''
    Load hartree potential and atoms from a .cube or .xsf file.

    Arguments:
        file_path: str. Path to file to load.

    Returns: tuple (pot, xyzs, Zs)
        | pot: HartreePotential.
        | xyzs: np.ndarray of shape (num_atoms, 3). Atom coordinates.
        | Zs: np.ndarray of shape (num_atoms,). Atomic numbers.
    '''

    if file_path.endswith('.cube'):
        FF, lvec, _, _ = loadCUBE(file_path, xyz_order=True, verbose=False)
        Zs, x, y, z, _ = loadAtomsCUBE(file_path)
    elif file_path.endswith('.xsf'):
        FF, lvec, _, _ = loadXSF(file_path, xyz_order=True, verbose=False)
        (Zs, x, y, z, _), _, _ = loadXSFGeom(file_path)
    else:
        raise ValueError(f'Unsupported file format in file `{file_path}`')

    FF *= -1
    pot = HartreePotential(FF, lvec)
    xyzs = np.stack([x, y, z], axis=1)

    return pot, xyzs, Zs

# ========= classes

class HartreePotential:
    '''
    Class for holding data of a Hartree potential on a grid.

    Arguments:
        array: np.ndarray. Potential values on a 3D grid.
        lvec: array-like of shape (4, 3). Unit cell boundaries. First (row) vector specifies the origin,
            and the remaining three vectors specify the edge vectors of the unit cell.
        ctx: pyopencl.Context. OpenCL context for device buffer. Defaults to oclu.ctx.
    '''
    def __init__(self, array, lvec, ctx=None):
        assert isinstance(array, np.ndarray), 'array should be a numpy.ndarray'
        if array.dtype != np.float32 or not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array, dtype=np.float32)
        self.array = array
        self.lvec = np.array(lvec)
        self.origin = self.lvec[0]
        assert self.lvec.shape == (4, 3), 'lvec should have shape (4, 3)'
        self.ctx = ctx or oclu.ctx
        self._cl_array = None
        self.nbytes = 0

    @property
    def shape(self):
        return self.array.shape

    @property
    def step(self):
        return np.stack([self.lvec[i+1] / self.array.shape[i] for i in range(3)])

    @property
    def cl_array(self):
        if self._cl_array is None:
            mf = cl.mem_flags
            self._cl_array = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.array)
            self.nbytes += 4 * np.prod(self.shape)
            if (verbose > 0): print(f'HartreePotential.nbytes {self.nbytes}')
        return self._cl_array

    def release(self):
        '''Release device buffers.'''
        if self._cl_array is not None:
            self._cl_array.release()
            self._cl_array = None
            self.nbytes -= 4 * np.prod(self.shape)

class MultipoleTipDensity:
    '''
    Multipole probe tip charge density on a periodic grid.

    Arguments:
        lvec: np.ndarray of shape (3, 3). Grid lattice vectors.
        nDim: array-like of length 3. Grid shape.
        center: array-like of length 3. Center position of charge density in the grid.
        sigma: float. Width of charge distribution.
        multipole: Dict. Charge multipole types. The dict should contain float entries for at least
            of one the following 's', 'px', 'py', 'pz', 'dz2', 'dy2', 'dx2', 'dxy' 'dxz', 'dyz'.
            The tip charge density will be a linear combination of the specified multipole types
            with the specified weights.
        tilt: float. Tip charge tilt angle in radians.
        ctx: pyopencl.Context. OpenCL context for device buffer. Defaults to oclu.ctx.
    '''

    def __init__(self, lvec, nDim, center=[0, 0, 0], sigma=0.71, multipole={'dz2': -0.1}, tilt=0.0, ctx=None):

        self.lvec = lvec
        self.lvec_len = np.linalg.norm(self.lvec, axis=1)
        self.nDim = np.array(nDim)
        self.center = np.array(center)
        self.step = self.lvec_len / self.nDim
        self.sigma = sigma
        self.multipole = multipole
        self.tilt = tilt
        self.ctx = ctx or oclu.ctx
        self._cl_array = None
        self.nbytes = 0

        if (self.center < 0).any() or (self.center > lvec.sum(axis=0)).any():
            raise ValueError('Center position is outside the grid.')

        # Make tip density grid as a numpy array
        self.array = self._make_tip_density()

    def _make_tip_density(self):
        if(bRuntime): t0 = time.perf_counter()
        xyz = []
        for i in range(3):
            c = np.linspace(0, self.lvec_len[i] * (1 - 1/self.nDim[i]), self.nDim[i]) - self.center[i]
            c[c >= self.lvec_len[i] / 2] -= self.lvec_len[i]
            c[c <= -self.lvec_len[i] / 2] += self.lvec_len[i]
            xyz.append(c)
        X, Y, Z = np.meshgrid(*xyz, indexing='ij')
        rho = getProbeDensity(self.lvec, X, Y, Z, self.step, sigma=self.sigma,
            multipole_dict=self.multipole, tilt=self.tilt)
        if(bRuntime): print("runtime(FFTConvolution._make_tip_density) [s]: ", time.perf_counter() - t0)
        return rho.astype(np.float32)

    @property
    def cl_array(self):
        if self._cl_array is None:
            mf = cl.mem_flags
            self._cl_array = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.array)
            self.nbytes += 4 * np.prod(self.nDim)
            if (verbose > 0): print(f'MultipoleTipDensity.nbytes {self.nbytes}')
        return self._cl_array

    def release(self):
        '''Release device buffers.'''
        if self._cl_array is not None:
            self._cl_array.release()
            self._cl_array = None
            self.nbytes -= 4 * np.prod(self.nDim)

class FFTConvolution:
    '''
    Do circular convolution of Hartree potential with tip charge density via FFT.

    Arguments:
        rho: MultipoleTipDensity. Tip charge density.
        queue: pyopencl.CommandQueue. OpenCL queue on which operations are performed. Defaults to oclu.queue.
    '''

    def __init__(self, rho, queue=None):
        if not fft_available:
            raise RuntimeError('Cannot do FFT because reikna is not installed.')
        self.shape = rho.array.shape
        self.queue = queue or oclu.queue
        self.ctx = self.queue.context
        self.nbytes = 0
        self._make_transforms()
        self._make_fft()
        self._set_rho(rho)
        if (verbose > 0): print(f'FFTConvolution.nbytes {self.nbytes}')

    # https://github.com/fjarri/reikna/issues/57
    def _make_transforms(self):
        self.r2c = Transformation(
            [Parameter('output', Annotation(Type(np.complex64, self.shape), 'o')),
            Parameter('input', Annotation(Type(np.float32, self.shape), 'i'))],
            """
            ${output.store_same}(
                COMPLEX_CTR(${output.ctype})(
                    ${input.load_same},
                    0));
            """
        )
        self.c2r = Transformation(
            [Parameter("output", Annotation(Type(np.float32, self.shape), "o")),
            Parameter("input", Annotation(Type(np.complex64, self.shape), "i")),
            Parameter("scale", Annotation(np.float32))],
            """
            ${output.store_same}(${input.load_same}.x * ${scale});
            """
        )

    def _make_fft(self):

        if bRuntime: t0 = time.perf_counter()

        thr = ocl_api().Thread(self.queue)
        self.pot_hat_cl = array.empty(self.queue, self.shape, dtype=np.complex64)
        self.rho_hat_cl = array.empty(self.queue, self.shape, dtype=np.complex64)
        self.nbytes += 2 * np.prod(self.shape) * 8

        fft_f = FFT(self.r2c.output)
        fft_f.parameter.input.connect(self.r2c, self.r2c.output, new_input=self.r2c.input)
        self.fft_f = fft_f.compile(thr)

        fft_i = FFT(self.c2r.input)
        fft_i.parameter.output.connect(self.c2r, self.c2r.input, new_output=self.c2r.output, scale=self.c2r.scale)
        self.fft_i = fft_i.compile(thr)

        if(bRuntime): print("runtime(FFTConvolution._make_fft) [s]: ", time.perf_counter() - t0)

    def _set_rho(self, rho):
        self.rho = rho
        self.fft_f(self.rho_hat_cl, rho.cl_array, inverse=0)

    def convolve(self, pot, E=None, bCopy=True, bFinish=True):
        '''
        Convolve Hartree potential with tip charge density.

        Arguments:
            pot: HartreePotential or pyopencl.Buffer. Hartree potential to convolve. Has to be same shape as rho.
            E: np.ndarray, pyopencl.Buffer or None. Output energy. Created automatically,
                if None. For bCopy==True it is a np.ndarray and for bCopy==False it is a
                pyopencl.Buffer.
            bCopy: Bool. Whether to return the output energy to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns: np.ndarray if bCopy == True or pyopencl.Buffer otherwise.
        '''

        if bRuntime: t0 = time.perf_counter()

        if isinstance(pot, HartreePotential):
            assert pot.shape == self.shape, 'pot array shape does not match rho array shape'
            pot = pot.cl_array
        
        mf = cl.mem_flags
        if bCopy:
            E = E or np.empty(self.shape, dtype=np.float32)
            assert E.shape == self.shape, 'E array shape does not match'
            E_cl = cl.Buffer(self.ctx, mf.READ_WRITE, size=4*np.prod(self.shape))
        else:
            E = E or cl.Buffer(self.ctx, mf.READ_WRITE, size=4*np.prod(self.shape))
            E_cl = E

        if(bRuntime):
            self.queue.finish()
            print("runtime(FFTConvolution.convolve.pre) [s]: ", time.perf_counter() - t0)

        # Do convolution
        self.fft_f(output=self.pot_hat_cl, new_input=pot, inverse=0)
        self.fft_i(new_output=E_cl, input=self.pot_hat_cl * self.rho_hat_cl, scale=self.rho.step.prod(), inverse=1)

        if bCopy: cl.enqueue_copy(self.queue, E, E_cl)
        if bFinish or bRuntime: self.queue.finish()
        if(bRuntime): print("runtime(FFTConvolution.convolve) [s]: ", time.perf_counter() - t0)

        return E

class ForceField_LJC:
    '''
        to evaluate ForceField on GPU
    '''

    verbose = 0

    def __init__( self ):
        self.ctx   = oclu.ctx; 
        self.queue = oclu.queue
        self.cl_poss   = None
        self.cl_FE     = None
        self.cl_Efield = None
        self.pot       = None
        self.rho       = None

    def initSampling(self, lvec, pixPerAngstrome=10, nDim=None ):
        if nDim is None:
            nDim = genFFSampling( lvec, pixPerAngstrome=pixPerAngstrome )
        self.nDim = nDim
        self.setLvec(lvec, nDim=nDim )

    def initPoss(self, poss=None, nDim=None, lvec=None, pixPerAngstrome=10 ):
        if poss is None:
            self.initSampling( lvec, pixPerAngstrome=10, nDim=None )
        self.prepareBuffers(poss=poss)

    def setLvec(self, lvec, nDim = None ):
        if nDim is not None:
            self.nDim = np.array([nDim[0],nDim[1],nDim[2],4], dtype=np.int32)
        elif self.nDim is not None:
            nDim = self.nDim
        else:
            print("ERROR : nDim must be set somewhere"); exit()
        self.lvec0       = np.zeros( 4, dtype=np.float32 ) 
        self.lvec        = np.zeros( (3,4), dtype=np.float32 ) 
        self.dlvec       = np.zeros( (3,4), dtype=np.float32 )
        self.lvec0[:3]    = lvec[  0,:3]
        self.lvec[:,:3]  = lvec[1:4,:3]
        self.dlvec[0,:]  =  self.lvec[0,:] / nDim[0]
        self.dlvec[1,:]  =  self.lvec[1,:] / nDim[1]
        self.dlvec[2,:]  =  self.lvec[2,:] / nDim[2]

    def setQs(self, Qs=[100,-200,100,0],QZs=[0.1,0,-0.1,0]):
        if ( len(Qs) != 4 ) or ( len(QZs) != 4 ):
            print("Qs and Qzs must have length 4 ") 
            exit()
        self.Qs  = np.array(Qs ,dtype=np.float32)
        self.QZs = np.array(QZs,dtype=np.float32)

    def prepareBuffers(self, atoms=None, cLJs=None, poss=None, bDirect=False, nz=20, pot=None, E_field=False, rho=None):
        '''
        allocate all necessary buffers in GPU memory
        '''
        nbytes   =  0;
        mf       = cl.mem_flags
        nb_float = np.dtype(np.float32).itemsize
        if atoms is not None:
            self.nAtoms   = np.int32( len(atoms) ) 
            atoms = atoms.astype(np.float32)
            self.cl_atoms = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
        if cLJs is not None:
            cLJs = cLJs.astype(np.float32)
            self.cl_cLJs  = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=cLJs  ); nbytes+=cLJs.nbytes
        if poss is not None:
            self.nDim = np.array( poss.shape, dtype=np.int32 )
            self.cl_poss  = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes   # float4
        if (self.cl_FE is None) and not bDirect:
            nb = self.nDim[0]*self.nDim[1]*self.nDim[2] * 4 * nb_float
            self.cl_FE    = cl.Buffer(self.ctx, mf.WRITE_ONLY , nb ); nbytes+=nb
            if(self.verbose>0): print(" forcefield.prepareBuffers() :  self.cl_FE  ", self.cl_FE)
        if pot is not None:
            assert isinstance(pot, HartreePotential), 'pot should be a HartreePotential object'
            self.pot = pot
            self.pot.cl_array # Accessing the cl_array attribute copies the pot to the device
        if E_field:
            self.cl_Efield = cl.Buffer(self.ctx, mf.READ_WRITE, size=4*np.prod(self.nDim)); nbytes+=4*np.prod(self.nDim)
        if rho is not None:
            assert isinstance(rho, MultipoleTipDensity), 'rho should be a MultipoleTipDensity object'
            self.rho = rho
            self.fft_conv = FFTConvolution(rho)
        if(self.verbose>0): print("initArgsLJC.nbytes ", nbytes)

    def updateBuffers(self, atoms=None, cLJs=None, poss=None):
        '''
        update content of all buffers
        '''
        if(self.verbose>0): print(" ForceField_LJC.updateBuffers ")
        oclu.updateBuffer(atoms, self.cl_atoms )
        oclu.updateBuffer(cLJs,  self.cl_cLJs  )
        oclu.updateBuffer(poss,  self.cl_poss  )

    def tryReleaseBuffers(self):
        '''
        release all buffers
        '''
        if(self.verbose>0): print(" ForceField_LJC.tryReleaseBuffers ")
        try: 
            self.cl_atoms.release() 
            self.cl_atoms = None
        except: 
            pass
        try: 
            self.cl_cLJs.release() 
            self.cl_cLJs = None
        except: 
            pass
        try: 
            self.cl_poss.release() 
            self.cl_poss = None
        except: 
            pass
        try: 
            self.cl_FE.release() 
            self.cl_FE = None
        except: 
            pass
        try: 
            self.pot.release()
        except: 
            pass
        try: 
            self.cl_Efield.release() 
            self.cl_Efield = None
        except: 
            pass
        try:
            self.rho.release()
        except:
            pass

    def run(self, FE=None, local_size=(32,), bCopy=True, bFinish=True ):
        '''
        generate force-field
        '''
        if(bRuntime): t0 = time.time()
        if bCopy and (FE is None):
            FE = np.zeros( self.nDim[:3]+(8,), dtype=np.float32 )
            if(self.verbose>0): print("FE.shape", FE.shape, self.nDim)
        ntot = self.nDim[0]*self.nDim[1]*self.nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_poss,
            self.cl_FE,
        )
        if(bRuntime): print("runtime(ForceField_LJC.evalLJC.pre) [s]: ", time.time() - t0)
        cl_program.evalLJC( self.queue, global_size, local_size, *(kargs) )
        if bCopy:   cl.enqueue_copy( self.queue, FE, kargs[4] )
        if bFinish: self.queue.finish()
        if(bRuntime): print("runtime(ForceField_LJC.evalLJC) [s]: ", time.time() - t0)
        return FE
    
    def run_evalLJ_noPos(self, FE=None, local_size=(32,), bCopy=True, bFinish=True ):
        '''
        Compute Lennard-Jones forcefield without charges at grid points.

        Arguments:
            FE: np.ndarray or None. Array where output force field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to return the calculated electric field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns: np.ndarray if bCopy == True or None otherwise.
        '''

        if bRuntime: t0 = time.perf_counter()
        
        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        cl_program.evalLJ_noPos(self.queue, global_size, local_size, 
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_FE,
            self.nDim,
            self.lvec0   ,
            self.dlvec[0],
            self.dlvec[1],
            self.dlvec[2]
        )

        if bCopy: FE = self.downloadFF(FE)
        if bFinish: self.queue.finish()
        if bRuntime: print("runtime(ForceField_LJC.run_evalLJ_noPos) [s]: ", time.perf_counter() - t0)
        
        return FE

    def run_evalLJC_Q(self, FE=None, Qmix=0.0, local_size=(32,), bCopy=True, bFinish=True ):
        '''
        generate force-field
        '''
        if(bRuntime): t0 = time.time()
        if bCopy and (FE is None):
            FE = np.zeros( self.nDim[:3]+(4,), dtype=np.float32 )
            if(self.verbose>0): print("FE.shape", FE.shape, self.nDim)
        ntot = self.nDim[0]*self.nDim[1]*self.nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_poss,
            self.cl_FE,
            np.float32(Qmix),
        )
        if(bRuntime): print("runtime(ForceField_LJC.run_evalLJC_Q.pre) [s]: ", time.time() - t0)
        cl_program.evalLJC_Q( self.queue, global_size, local_size, *(kargs) )
        if bCopy:   cl.enqueue_copy( self.queue, FE, kargs[4] )
        if bFinish: self.queue.finish()
        if(bRuntime): print("runtime(ForceField_LJC.run_evalLJC_Q) [s]: ", time.time() - t0)
        return FE

    def run_evalLJC_QZs_noPos(self, FE=None, Qmix=0.0, local_size=(32,), bCopy=True, bFinish=True ):
        '''
        generate force-field
        '''
        if(bRuntime): t0 = time.time()
        if bCopy and (FE is None):
            ns = ( tuple(self.nDim[:3])+(4,) )
            FE = np.zeros( ns, dtype=np.float32 )
            if(self.verbose>0): print("FE.shape", FE.shape, self.nDim)
        ntot = self.nDim[0]*self.nDim[1]*self.nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_FE,
            self.nDim,
            self.lvec0   ,
            self.dlvec[0],
            self.dlvec[1],
            self.dlvec[2],
            self.Qs,
            self.QZs
        )
        if(bRuntime): print("runtime(ForceField_LJC.run_evalLJC_QZs_noPos.pre) [s]: ", time.time() - t0)
        cl_program.evalLJC_QZs_noPos( self.queue, global_size, local_size, *(kargs) )
        if bCopy:   cl.enqueue_copy( self.queue, FE, kargs[3] )
        if bFinish: self.queue.finish()
        if(bRuntime): print("runtime(ForceField_LJC.run_evalLJC_QZs_noPos) [s]: ", time.time() - t0)
        return FE

    def downloadFF(self, FE=None):
        '''
        Get force field array from device.

        Arguments:
            FE: np.ndarray or None. Array where output force field is copied to. If None,
                will be created automatically.
        '''

        # Get numpy array
        if FE:

            if not np.allclose(FE.shape, self.nDim):
                raise ValueError(f'FE array dimensions {FE.shape} do not match with '
                    f'force field dimensions {self.nDim}.')

            # Values are saved in Fortran order with the xyzw dimensions as the first index
            FE = FE.transpose(3, 0, 1, 2)
            if not FE.flags['F_CONTIGUOUS']:
                FE = np.asfortranarray(FE)

        else:
            FE = np.empty((self.nDim[3],) + tuple(self.nDim[:3]), dtype=np.float32, order='F')
            
        if self.verbose: print("FE.shape ", FE.shape)

        # Copy from device to host
        cl.enqueue_copy(self.queue, FE, self.cl_FE)
        self.queue.finish()

        # Transpose xyzw dimension back to last index
        FE = FE.transpose(1, 2, 3, 0)

        return FE

    def run_evalLJC_Q_noPos(self, FE=None, Qmix=0.0, local_size=(32,), bCopy=True, bFinish=True ):
        '''
        generate force-field
        '''
        if(bRuntime): t0 = time.time()
        if bCopy and (FE is None):
            ns = ( tuple(self.nDim[:3])+(4,) )
            FE = np.zeros( ns, dtype=np.float32 )
            if(self.verbose>0): print("FE.shape", FE.shape, self.nDim)
        ntot = self.nDim[0]*self.nDim[1]*self.nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_FE,
            self.nDim,
            self.lvec0   ,
            self.dlvec[0],
            self.dlvec[1],
            self.dlvec[2],
            np.float32(Qmix),
        )
        if(bRuntime): print("runtime(ForceField_LJC.run_evalLJC_Q_noPos.pre) [s]: ", time.time() - t0)
        cl_program.evalLJC_Q_noPos( self.queue, global_size, local_size, *(kargs) )
        if bCopy:   cl.enqueue_copy( self.queue, FE, self.cl_FE )
        if bFinish: self.queue.finish()
        if(bRuntime): print("runtime(ForceField_LJC.evalLJC_Q_noPos) [s]: ", time.time() - t0)
        return FE

    def run_evalLJC_Hartree(self, FE=None, local_size=(32,), bCopy=True, bFinish=True):
        '''
        Compute Lennard Jones force field at grid points and add to it the electrostatic force 
        from an electric field precomputed from a Hartree potential.

        Arguments:
            FE: np.ndarray or None. Array where output force field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to return the calculated forcefield to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns: np.ndarray if bCopy == True or None otherwise.
        '''

        if bRuntime: t0 = time.perf_counter()
            
        T = np.append(np.linalg.inv(self.dlvec[:, :3]).T.copy(), np.zeros((3, 1)), axis=1).astype(np.float32)

        if bRuntime: print("runtime(ForceField_LJC.run_evalLJC_Hartree.pre) [s]: ", time.perf_counter() - t0)

        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        cl_program.evalLJC_Hartree(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_Efield,
            self.cl_FE,
            self.nDim,
            self.lvec0,
            self.dlvec[0], self.dlvec[1], self.dlvec[2],
            T[0], T[1], T[2],
            self.Qs,
            self.QZs
        )

        if bCopy: FE = self.downloadFF(FE)
        if bFinish: self.queue.finish()
        if bRuntime: print("runtime(ForceField_LJC.run_evalLJC_Hartree) [s]: ", time.perf_counter() - t0)

        return FE

    def run_gradPotentialGrid(self, pot=None, E_field=None, h=None, local_size=(32,), bCopy=True, bFinish=True):
        '''
        Obtain electric field on the force field grid as the negative gradient of Hartree potential
        via centered difference.

        Arguments:
            pot: HartreePotential or None. Hartree potential to differentiate. If None, has to be initialized
                beforehand with prepareBuffers.
            E_field: np.ndarray or None. Array where output electric field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            h: float > 0.0 or None. Finite difference step size (one-sided) in angstroms. If None, the default
                value DEFAULT_FD_STEP is used.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to return the calculated electric field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns: np.ndarray if bCopy == True or None otherwise.
        '''

        if bRuntime: t0 = time.perf_counter()
        
        if pot:
            self.prepareBuffers(pot=pot)
        elif not self.pot:
            raise ValueError("Hartree potential not initialized on the device. "
                "Either initialize it with prepareBuffers or pass it here as a HartreePotential object.")
        
        if bCopy:
            E_field = E_field or np.empty(self.nDim, dtype=np.float32)
            if not np.allclose(E_field.shape, self.nDim):
                raise ValueError(f'E_field array dimensions {E_field.shape} do not match with '
                    f'force field dimensions {self.nDim}.')

        if not self.cl_Efield:
            self.prepareBuffers(E_field=True)

        h = h or DEFAULT_FD_STEP

        # Check if potential grid matches the force field grid and is orthogonal.
        # If it does, we don't need to do interpolation.
        matching_grid = (
            np.allclose(self.pot.shape, self.nDim[:3]) and
            (np.abs(self.pot.origin - self.lvec0[:3]) < 1e-3).all() and
            (np.abs(np.diag(self.pot.step) * self.pot.shape - np.diag(self.lvec[:, :3])) < 1e-3).all() and
            (self.pot.step == np.diag(np.diag(self.pot.step))).all()
        )
        if self.verbose > 0: print('Matching grid:', matching_grid)

        if bRuntime: print("runtime(ForceField_LJC.run_gradPotentialGrid.pre) [s]: ", time.perf_counter() - t0)

        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        if matching_grid:
            cl_program.gradPotential(self.queue, global_size, local_size,
                self.pot.cl_array,
                self.cl_Efield,
                np.append(self.pot.shape, 0).astype(np.int32),
                np.append(np.diag(self.pot.step), 0).astype(np.float32),
                np.int32(1)
            )
        else:
            T = np.append(np.linalg.inv(self.pot.step).T.copy(), np.zeros((3, 1)), axis=1).astype(np.float32)
            cl_program.gradPotentialGrid(self.queue, global_size, local_size,
                self.pot.cl_array,
                self.cl_Efield,
                np.append(self.pot.shape, 0).astype(np.int32),
                T[0], T[1], T[2],
                np.append(self.pot.origin, 0).astype(np.float32),
                self.nDim,
                self.dlvec[0], self.dlvec[1], self.dlvec[2],
                self.lvec0,
                np.array([h, h, h, 0.0], dtype=np.float32)
            )

        if bCopy: cl.enqueue_copy(self.queue, E_field, self.cl_Efield)
        if bFinish: self.queue.finish()
        if bRuntime: print("runtime(ForceField_LJC.run_gradPotentialGrid) [s]: ", time.perf_counter() - t0)

        return E_field

    def runRelaxStrokesDirect(self, Q, cl_FE, FE=None, local_size=(32,), nz=10 ):
        '''
        generate force-field
        '''
        if FE is None:
            ns = ( tuple(self.nDim[:3])+(4,) )
            FE    = np.zeros( ns, dtype=np.float32 )
            #FE     = np.empty( self.scan_dim+(4,), dtype=np.float32 )
            if(self.verbose>0): print("FE.shape", FE.shape, self.nDim)
        ntot = int( self.scan_dim[0]*self.scan_dim[1] ) 
        ntot=makeDivisibleUp(ntot,local_size[0])
        global_size = (ntot,) # TODO make sure divisible by local_size

        dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
        stiffness    = np.array( [-0.03,-0.03, -0.03,-1.0 ], dtype=np.float32 );
        dpos0        = np.array( [ 0.0 , 0.0 , -4.0 , 4.0 ], dtype=np.float32 );
        relax_params = np.array( [ 0.1 , 0.9 ,  0.02, 0.5 ], dtype=np.float32 );

        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_points,
            cl_FE,
            dTip,
            stiffness,
            dpos0,
            relax_params,
            np.float32(Q),
            np.int32(nz),
        )
        cl_program.relaxStrokesDirect( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, FE, kargs[4] )
        self.queue.finish()
        return FE

    def interp_pot(self, pot=None, array_out=None, rot=np.eye(3), rot_center=np.zeros(3),
            local_size=(32,), bCopy=True, bFinish=True):
        '''
        Interpolate Hartree potential on the force field grid with an optional rotation
        to the output grid coordinates.

        Arguments:
            pot: HartreePotential or None. Hartree potential to differentiate. If None, has to be
                initialized beforehand with prepareBuffers.
            array_out: np.ndarray, pyopencl.Buffer or None. Output array. Created automatically,
                if None. For bCopy==True it is a np.ndarray and for bCopy==False it is a
                pyopencl.Buffer.
            rot: np.ndarray of shape (3, 3). Rotation matrix to apply.
            rot_center: np.ndarray of shape (3,). Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to return the calculated electric field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns: np.ndarray if bCopy == True or pyopencl.Buffer otherwise.
        '''

        if bRuntime: t0 = time.perf_counter()

        if pot:
            self.prepareBuffers(pot=pot)
        elif not self.pot:
            raise ValueError("Hartree potential not initialized on the device. "
                "Either initialize it with prepareBuffers or pass it here as a HartreePotential object.")
        
        mf = cl.mem_flags
        if bCopy:
            array_out = array_out or np.empty(self.nDim[:3], dtype=np.float32)
            assert isinstance(array_out, np.ndarray), 'array_out should be a numpy array when bCopy==True'
            cl_array_out = cl.Buffer(self.ctx, mf.READ_WRITE, size=4*np.prod(self.nDim[:3]))
            if not np.allclose(array_out.shape, self.nDim[:3]):
                raise ValueError(f'array_out dimensions {array_out.shape} do not match with '
                    f'force field dimensions {self.nDim}.')
        else:
            if array_out:
                assert isinstance(array_out, cl.Buffer), 'array_out should be an pyopencl.Buffer when bCopy==False'
            else:
                array_out = cl.Buffer(self.ctx, mf.READ_WRITE, size=4*np.prod(self.nDim[:3]))
            cl_array_out = array_out

        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        T = np.append(np.linalg.inv(self.pot.step).T.copy(), np.zeros((3, 1)), axis=1).astype(np.float32)
        rot = np.append(rot, np.zeros((3, 1)), axis=1).astype(np.float32)

        if bRuntime: print("runtime(ForceField_LJC.interp_pot.pre) [s]: ", time.perf_counter() - t0)
        
        cl_program.interp_at(self.queue, global_size, local_size,
            self.pot.cl_array,
            cl_array_out,
            np.append(self.pot.shape, 0).astype(np.int32),
            T[0], T[1], T[2],
            np.append(self.pot.origin, 0).astype(np.float32),
            self.nDim,
            self.dlvec[0], self.dlvec[1], self.dlvec[2],
            self.lvec0,
            rot[0], rot[1], rot[2],
            np.append(rot_center, 0).astype(np.float32)
        )

        if bCopy: cl.enqueue_copy(self.queue, array_out, cl_array_out)
        if bFinish: self.queue.finish()
        if bRuntime: print("runtime(ForceField_LJC.interp_pot) [s]: ", time.perf_counter() - t0)

        return array_out

    def calc_force_fft(self, FE=None, rot=np.eye(3), rot_center=np.zeros(3), local_size=(32,),
            bCopy=True, bFinish=True):
        '''
        Calculate force field for LJ + Hartree convolved with tip density via FFT.

        Arguments:
            FE: np.ndarray or None. Array where output force field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            rot: np.ndarray of shape (3, 3). Rotation matrix applied to the atom coordinates.
            rot_center: np.ndarray of shape (3,). Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to copy the calculated forcefield field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns: np.ndarray if bCopy==True or None otherwise.
        '''

        if bRuntime: t0 = time.perf_counter()

        if (self.lvec[:, :3] != np.diag(np.diag(self.lvec[:, :3]))).any():
            raise NotImplementedError('Forcefield calculation via FFT for non-rectangular grids is not implemented. '
                'Note that the forcefield grid does not need to match the Hartree potential grid.')

        # Interpolate Hartree potential onto the correct grid
        pot_interp = self.interp_pot(rot=rot, rot_center=rot_center, local_size=local_size,
            bCopy=False, bFinish=bRuntime)

        if bRuntime: print("runtime(ForceField_LJC.calc_force_fft.interpolate) [s]: ", time.perf_counter() - t0)

        # Convolve Hartree potential and tip charge density
        E_cl = self.fft_conv.convolve(pot_interp, bCopy=False, bFinish=False)

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.calc_force_fft.convolution) [s]: ", time.perf_counter() - t0)

        # Take gradient to get electrostatic force field
        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        step = np.append(np.diag(self.dlvec[:, :3]), 0).astype(np.float32)
        cl_program.gradPotential(self.queue, global_size, local_size,
            E_cl,
            self.cl_FE,
            self.nDim,
            step,
            np.int32(0)
        )

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.calc_force_fft.gradient) [s]: ", time.perf_counter() - t0)

        # Add Lennard-Jones force
        local_size = (min(local_size[0], 64),)
        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        cl_program.addLJ(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_FE,
            self.nDim,
            self.lvec0,
            self.dlvec[0], self.dlvec[1], self.dlvec[2]
        )

        if bCopy: FE = self.downloadFF(FE)
        if bFinish or bRuntime: self.queue.finish()
        if bRuntime: print("runtime(ForceField_LJC.calc_force_fft) [s]: ", time.perf_counter() - t0)

        return FE

    def makeFF(self, atoms=None, cLJs=None, Qmix=0.0, FE=None, bRelease=True, bCopy=True, bFinish=True, bQZ=False):
        '''
        Generate force-field from given positions/charges (atoms), Lennard-Jones parameters (cLJs) etc.
        '''
        
        if(bRuntime): t0 = time.time()

        self.atoms = atoms
        cLJs = cLJs.astype(np.float32, copy=False)
        self.prepareBuffers(atoms, cLJs)
        if(bRuntime): print("runtime(ForceField_LJC.makeFF.pre) [s]: ", time.time() - t0)

        if self.cl_poss is not None:
            FF = self.run_evalLJC_Q( FE=FE, Qmix=Qmix, local_size=(32,), bCopy=bCopy, bFinish=bFinish )
        else:
            if np.allclose(self.atoms[:, -1], 0): # No charges
                FF = self.run_evalLJ_noPos()
            elif bQZ:
                FF = self.run_evalLJC_QZs_noPos( FE=FE, Qmix=Qmix, local_size=(32,), bCopy=bCopy, bFinish=bFinish )
            else:
                FF = self.run_evalLJC_Q_noPos( FE=FE, Qmix=Qmix, local_size=(32,), bCopy=bCopy, bFinish=bFinish )

        if(bRelease): self.tryReleaseBuffers()
        if(bRuntime): print("runtime(ForceField_LJC.makeFF.tot) [s]: ", time.time() - t0)

        return FF, atoms

    def makeFFHartree(self, atoms, cLJs, pot=None, rho=None, FE=None, rot=np.eye(3), rot_center=np.zeros(3),
            local_size=(32,), bRelease=True, bCopy=True, bFinish=True):
        '''
        Generate a force field from a list of atoms and a Hartree potential.

        Arguments:
            atoms: np.ndarray of shape (n_atoms, 3). xyz positions of atoms.
            cLJs: np.ndarray of shape (n_atoms, 2). Lennard-Jones interaction parameters for each atom.
            pot: HartreePotential or None. Hartree potential used for electrostatic interaction.
                If None, has to be initialized beforehand with prepareBuffers.
            rho: MultipoleTipDensity or None. Probe tip charge density. If None and one has not been
                set for the forcefield, point-charge electrostatics for the tip will be used instead.
            FE: np.ndarray or None. Array where output force field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            rot: np.ndarray of shape (3, 3). Rotation matrix applied to the atom coordinates.
            rot_center: np.ndarray of shape (3,). Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
            bRelease: Bool. Whether to delete data on device after computation is done.
            bCopy: Bool. Whether to copy the calculated forcefield field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns: np.ndarray if bCopy==True or None otherwise.
        '''

        if(bRuntime): t0 = time.perf_counter()

        if not hasattr(self, 'nDim') or not hasattr(self, 'lvec'):
            raise RuntimeError('Forcefield position is not initialized. Initialize with initSampling.')

        # Rotate atoms
        atoms = atoms - rot_center
        atoms = np.dot(atoms, rot.T)
        atoms += rot_center
        rot_ff = np.linalg.inv(rot) # Force field rotation is in opposite direction to atoms

        # Prepare data on device
        self.atoms = np.pad(atoms, ((0, 0), (0, 1)))
        self.prepareBuffers(self.atoms, cLJs, pot=pot, rho=rho)

        if(bRuntime): print("runtime(ForceField_LJC.makeFFHartree.pre) [s]: ", time.perf_counter() - t0)
        
        if rho == None and self.rho is None:
            if not np.allclose(rot, np.eye(3)):
                raise NotImplementedError('Force field calculation with rotation for Hartree potential with '
                    'point charges tip density is not implemented.')
            self.run_gradPotentialGrid(local_size=local_size, bCopy=False, bFinish=False)
            FF = self.run_evalLJC_Hartree(FE=FE, local_size=local_size, bCopy=bCopy, bFinish=bFinish)
        else:
            FF = self.calc_force_fft(rot=rot_ff, rot_center=rot_center, local_size=local_size,
                bCopy=bCopy, bFinish=bFinish)

        if(bRelease): self.tryReleaseBuffers()
        if(bRuntime): print("runtime(ForceField_LJC.makeFFHartree.tot) [s]: ", time.perf_counter() - t0)

        return FF

class AtomProcjetion:
    '''
        to generate reference output maps ( Ys )  in generator for Neural Network training
    '''
    Rpp     =  2.0   #  probe-particle radius
    zmin    = -3.0   #  minim position of pixels sampled in SphereMaps
    dzmax   =  2.0   #  maximum distance of atoms from sampling screen for Atomic Disk maps ( similar )
    dzmax_s = np.Inf #  maximum depth of vdW shell in Atomic Disks

    Rmax       =  10.0  #  Radial function of bonds&atoms potential  ; used in Bonds
    drStep     =   0.1  #  step dx (dr) for sampling of radial function; used in Bonds 
    elipticity =  0.5;  #  ration between major and minor semiaxi;   used in Bonds 

    # occlusion
    zmargin =  0.2   #  zmargin 
    tgMax   =  0.5   #  tangens of angle limiting occlusion for SphereCaps
    tgWidth =  0.1   #  tangens of angle for limiting rendered area for SphereCaps
    Rfunc   = None

    def __init__( self ):
        self.ctx   = oclu.ctx; 
        self.queue = oclu.queue

    def makeCoefsZR(self, Zs, ELEMENTS ):
        '''
        make atomic coeficients used e.g. in MultiMap
        '''
        na = len(Zs)
        coefs = np.zeros( (na,4), dtype=np.float32 )
        if(verbose>0): print("Zs", Zs)
        for i,ie in enumerate(Zs):
            coefs[i,0] = 1.0
            coefs[i,1] = ie
            coefs[i,2] = ELEMENTS[ie-1][6]
            coefs[i,3] = ELEMENTS[ie-1][7]
        if(verbose>0): print("coefs[:,2]", coefs[:,2])
        return coefs

    def prepareBuffers(self, atoms, prj_dim, coefs=None, bonds2atoms=None, Rfunc=None, elem_channels=None ):
        '''
        allocate GPU buffers
        '''
        if(verbose>0): print("AtomProcjetion.prepareBuffers prj_dim", prj_dim)
        self.prj_dim = prj_dim
        nbytes   =  0;
        self.nAtoms   = np.int32( len(atoms) ) 
        #print " initArgsLJC ", nAtoms
        mf       = cl.mem_flags
        self.cl_atoms = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes

        if (Rfunc is not None) or (self.Rfunc is not None):
            if Rfunc is None: Rfunc = self.Rfunc
            #print Rfunc
            self.Rfunc = Rfunc
            Rfunc = Rfunc.astype(np.float32,copy=False)
            self.cl_Rfunc = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=Rfunc );
            #exit(0);

        if bonds2atoms is not None:
            self.nBonds = np.int32(len(bonds2atoms))
            bondPoints = np.empty( (self.nBonds,8), dtype=np.float32 )
            bondPoints[ :,:4] = atoms[bonds2atoms[:,0]]
            bondPoints[ :,4:] = atoms[bonds2atoms[:,1]]
            self.bondPoints=bondPoints
            self.bonds2atoms=bonds2atoms
            self.cl_bondPoints = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=bondPoints ); nbytes+=bondPoints.nbytes

        if coefs is None:
            coefs = np.zeros( (self.nAtoms,4), dtype=np.float32 )
            coefs[:,0] = 1.0 # amplitude
            coefs[:,1] = 0.1 # width

        self.cl_coefs  = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=coefs  ); nbytes+=coefs.nbytes
        
        npostot = prj_dim[0] * prj_dim[1]
        
        bsz=np.dtype(np.float32).itemsize * npostot
        self.cl_poss  = cl.Buffer(self.ctx, mf.READ_ONLY , bsz*4           );   nbytes+=bsz*4  # float4
        self.cl_Eout  = cl.Buffer(self.ctx, mf.WRITE_ONLY, bsz*prj_dim[2]  );   nbytes+=bsz    # float

        self.cl_itypes  = cl.Buffer(self.ctx, mf.READ_ONLY, 200*np.dtype(np.int32).itemsize );   nbytes+=bsz    # float
        
        if elem_channels:
            elem_channels = np.array(elem_channels).astype(np.int32)
            self.cl_elem_channels = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=elem_channels ); nbytes+=elem_channels.nbytes

        if(verbose>0): print("AtomProcjetion.prepareBuffers.nbytes ", nbytes)

    def updateBuffers(self, atoms=None, coefs=None, poss=None ):
        '''
        upload data to GPU
        '''
        oclu.updateBuffer(atoms, self.cl_atoms )
        oclu.updateBuffer(coefs, self.cl_coefs  )
        oclu.updateBuffer(poss,  self.cl_poss  )

    def setAtomTypes(self, types, sel=[1,6,8]):
        '''
        setup selection of atomic types for SpheresType kernel and upload them to GPU
        '''
        self.nTypes   = np.int32( len(sel) )
        dct = { typ:i for i,typ in enumerate(sel) }
        itypes = np.ones( 200, dtype=np.int32); itypes[:]*=-1 
        for i,typ in enumerate(types):
            if typ in dct:
                itypes[i] = dct[typ]
        cl.enqueue_copy( self.queue, self.cl_itypes, itypes )
        return itypes, dct

    def releaseBuffers(self):
        '''
        deallocated all GPU buffers
        '''
        if(verbose>0): print(" AtomProjection.releaseBuffers ")
        self.cl_atoms.release()
        self.cl_coefs.release()
        self.cl_poss.release()
        self.cl_FE.release()

    def tryReleaseBuffers(self):
        '''
        deallocated all GPU buffers (those which exists)
        '''
        if(verbose>0): print(" AtomProjection.releaseBuffers ")
        try:
            self.cl_atoms.release()
        except:
            pass
        try:
            self.cl_coefs.release()
        except:
            pass
        try:
            self.cl_poss.release()
        except:
            pass
        try:
            self.cl_FE.release()
        except:
            pass

    def run_evalLorenz(self, poss=None,  Eout=None, local_size=(32,) ):
        '''
        kernel producing lorenzian function around each atom
        '''
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            if(verbose>0): print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
        )
        cl_program.evalLorenz( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[4] )
        self.queue.finish()
        return Eout

    def run_evaldisks(self, poss=None, Eout=None, tipRot=None, offset=0.0, local_size=(32,) ):
        '''
        kernel producing atomic disks with conical profile
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            if(verbose>0): print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.dzmax   ),
            np.float32( self.dzmax_s ),
            np.float32( offset ),
            np.float32( self.Rpp ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2]
        )
        cl_program.evalDisk( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[4] )
        self.queue.finish()
        return Eout

    def run_evaldisks_occlusion(self, poss=None, Eout=None, tipRot=None, local_size=(32,) ):
        '''
        kernel producing atomic disks occluded by higher nearby atoms
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            if(verbose>0): print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.Rpp     ),
            np.float32( self.zmin    ),
            np.float32( self.zmargin ),
            np.float32( self.dzmax   ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2]
        )
        cl_program.evalDisk_occlusion( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[4] )
        self.queue.finish()
        return Eout

    def run_evalSpheres(self, poss=None, Eout=None, tipRot=None, local_size=(32,) ):
        '''
        kernel producing van der Waals spheres
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            if(verbose>0): print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.Rpp   ),
            np.float32( self.zmin  ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2]
        )
        cl_program.evalSpheres( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[4] )
        self.queue.finish()
        return Eout

    def run_evalSphereCaps(self, poss=None, Eout=None, tipRot=None, local_size=(32,) ):
        '''
        kernel producing spherical caps (just to top most part of vdW sphere)
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            if(verbose>0): print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.Rpp   ),
            np.float32( self.zmin  ),
            np.float32( self.tgMax ),
            np.float32( self.tgWidth ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2]
        )
        cl_program.evalSphereCaps( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[4] )
        self.queue.finish()
        return Eout

    def run_evalQdisks(self, poss=None, Eout=None, tipRot=None, local_size=(32,) ):
        '''
        kernel producing atoms disks with positive and negative value encoding charge
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            if(verbose>0): print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.dzmax ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2]
        )
        cl_program.evalQDisk( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[3] )
        self.queue.finish()
        return Eout

    def run_evalMultiMapSpheres(self, poss=None, Eout=None, tipRot=None, bOccl=0, Rmin=1.4, Rstep=0.1, local_size=(32,) ):
        '''
         kernel to produce multiple channels of vdW Sphere maps each containing atoms with different vdW radius
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.Rpp   ),
            np.float32( self.zmin  ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2],
            np.int32(bOccl),
            np.int32( self.prj_dim[2] ),
            np.float32(Rmin),
            np.float32(Rstep)
        )
        cl_program.evalMultiMapSpheres( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[4] )
        self.queue.finish()
        return Eout

    def run_evalMultiMapSpheresElements(self, poss=None, Eout=None, tipRot=None, bOccl=0, Rmin=1.4, Rstep=0.1, local_size=(32,) ):
        '''
         kernel to produce multiple channels of vdW Sphere maps each containing atoms with different vdW radius
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_elem_channels,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.Rpp   ),
            np.float32( self.zmin  ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2],
            np.int32(bOccl),
            np.int32( self.prj_dim[2] ),
        )
        cl_program.evalMultiMapSpheresElements( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, self.cl_Eout )
        self.queue.finish()
        return Eout

    def run_evalSpheresType(self, poss=None, Eout=None, tipRot=None, bOccl=0,  local_size=(32,) ):
        '''
         kernel to produce multiple channels of vdW Sphere maps each coresponding to different atom type
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.nTypes,
            self.cl_atoms,
            self.cl_itypes,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.Rpp   ),
            np.float32( self.zmin  ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2],
            np.int32(bOccl),
        )
        cl_program.evalSpheresType( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[6] )
        self.queue.finish()
        return Eout

    def run_evalBondEllipses(self, poss=None, Eout=None, tipRot=None, bOccl=0,  local_size=(32,) ):
        '''
         kernel to produce multiple channels of vdW Sphere maps each coresponding to different atom type
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nBonds,
            self.cl_bondPoints,
            self.cl_poss,
            self.cl_Eout,
            self.cl_Rfunc,
            np.float32( self.drStep ),
            np.float32( self.Rmax    ),
            np.float32( self.elipticity    ),
            np.float32( self.zmin    ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2]
        )
        cl_program.evalBondEllipses( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[3] )
        self.queue.finish()
        return Eout

    def run_evalAtomRfunc(self, poss=None, Eout=None, tipRot=None, bOccl=0,  local_size=(32,) ):
        '''
         kernel to produce multiple channels of vdW Sphere maps each coresponding to different atom type
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            self.cl_Rfunc,
            np.float32( self.drStep  ),
            np.float32( self.Rmax    ),
            np.float32( self.zmin    ),
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2]
        )
        cl_program.evalAtomRfunc( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[4] )
        self.queue.finish()
        return Eout

    def run_evalCoulomb(self, poss=None, Eout=None, local_size=(32,) ):
        '''
        kernel producing coulomb potential and field
        '''
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            if(verbose>0): print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_poss,
            self.cl_Eout,
        )
        cl_program.evalCoulomb( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, Eout, kargs[3] )
        self.queue.finish()
        return Eout

    def run_evalHartreeGradient(self, pot, poss=None, Eout=None, h=None, rot=np.eye(3), rot_center=None,
            local_size=(32,)):
        '''
        Get electric field as the negative gradient of a Hartree potential.

        Arguments:
            pot: HartreePotential. Hartree potential to differentiate.
            poss: np.ndarray or None. Position grid for points to get the field at.
            Eout: np.ndarray or None. Output array. If None, will be created automatically.
            h: float > 0.0 or None. Finite difference step size (one-sided) in angstroms. If None, the default
                value DEFAULT_FD_STEP is used.
            rot: np.ndarray of shape (3, 3). Rotation matrix to apply to the position coordinates.
            rot_center: np.ndarray of shape (3,). Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
        '''

        if Eout is None:
            Eout = np.zeros(self.prj_dim[:2], dtype=np.float32)
            if(verbose>0): print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if(verbose>0): print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)

        
        global_size = (int(np.ceil(np.prod(self.prj_dim[:2]) / local_size[0]) * local_size[0]),)
        T = np.append(np.linalg.inv(pot.step).T.copy(), np.zeros((3, 1)), axis=1).astype(np.float32)
        rot = np.append(rot, np.zeros((3, 1)), axis=1).astype(np.float32)
        h = h or DEFAULT_FD_STEP

        cl_program.evalHartreeGradientZ(self.queue, global_size, local_size,
            pot.cl_array,
            self.cl_poss,
            self.cl_Eout,
            np.append(pot.shape, 0).astype(np.int32),
            T[0], T[1], T[2],
            np.append(pot.origin, 0).astype(np.float32),
            rot[0], rot[1], rot[2],
            np.append(rot_center, 0).astype(np.float32),
            np.float32(h),
        )
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()

        return Eout