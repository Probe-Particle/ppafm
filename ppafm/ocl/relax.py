#!/usr/bin/python


import numpy as np
import pyopencl as cl

from .. import common as PPU

# ========== Globals

cl_program = None
oclu = None

# fmt: off
DEFAULT_dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 )
DEFAULT_stiffness    = np.array( [-0.03,-0.03, -0.03,-1.0 ], dtype=np.float32 )
DEFAULT_dpos0        = np.array( [ 0.0 , 0.0 , -4.0 , 4.0 ], dtype=np.float32 )
DEFAULT_relax_params = np.array( [ 0.5 , 0.1 ,  0.02, 0.5 ], dtype=np.float32 )
# fmt: on

verbose = 0

# ========== Functions


def init(env):
    global cl_program
    global oclu
    cl_program = env.loadProgram(env.CL_PATH / "relax.cl")
    oclu = env


def mat3x3to4f(M):
    a = np.zeros(4, dtype=np.float32)
    a[0:3] = M[0]
    b = np.zeros(4, dtype=np.float32)
    b[0:3] = M[1]
    c = np.zeros(4, dtype=np.float32)
    c[0:3] = M[2]
    return (a, b, c)


def getInvCell(lvec):
    cell = lvec[1:4, 0:3]
    invCell = np.transpose(np.linalg.inv(cell))
    if verbose > 0:
        print(invCell)
    return mat3x3to4f(invCell)


def preparePoss(scan_dim, z0, start=(0.0, 0.0), end=(10.0, 10.0)):
    ys = np.linspace(start[0], end[0], scan_dim[0])
    xs = np.linspace(start[1], end[1], scan_dim[1])
    Xs, Ys = np.meshgrid(xs, ys)
    poss = np.zeros(Xs.shape + (4,), dtype=np.float32)
    poss[:, :, 0] = Ys
    poss[:, :, 1] = Xs
    poss[:, :, 2] = z0
    return poss


def preparePossRot(scan_dim, pos0, avec, bvec, start=(-5.0, -5.0), end=(5.0, 5.0)):
    xs = np.linspace(start[0], end[0], scan_dim[0])
    ys = np.linspace(start[1], end[1], scan_dim[1])
    As, Bs = np.meshgrid(xs, ys, indexing="ij")
    poss = np.zeros(As.shape + (4,), dtype=np.float32)
    poss[:, :, 0] = pos0[0] + As * avec[0] + Bs * bvec[0]
    poss[:, :, 1] = pos0[1] + As * avec[1] + Bs * bvec[1]
    poss[:, :, 2] = pos0[2] + As * avec[2] + Bs * bvec[2]
    return poss


def rotTip(rot, zstep, tipR0=[0.0, 0.0, 4.0]):
    dTip = np.zeros(4, dtype=np.float32)
    dTip[:3] = rot[2] * -zstep
    tipRot = mat3x3to4f(rot)
    tipRot[2][3] = -zstep

    dpos0Tip = np.zeros(4, dtype=np.float32)
    dpos0Tip[0] = tipR0[0]
    dpos0Tip[1] = tipR0[1]
    dpos0Tip[2] = -np.sqrt(tipR0[2] ** 2 - tipR0[0] ** 2 - tipR0[1] ** 2)
    dpos0Tip[3] = tipR0[2]

    dpos0 = np.zeros(4, dtype=np.float32)
    dpos0[:3] = np.dot(rot.transpose(), dpos0Tip[:3])
    dpos0[3] = tipR0[2]
    return dTip, tipRot, dpos0Tip, dpos0


## ============= Relax Class:


class RelaxedScanner:
    verbose = 0

    def __init__(self):
        self.queue = oclu.queue
        self.ctx = oclu.ctx
        self.stiffness = DEFAULT_stiffness.copy()
        self.relax_params = DEFAULT_relax_params.copy()

        self.zstep = 0.1  # step of tip approach in scan [Angstroem]
        self.start = (-5.0, -5.0)  # scan region start
        self.end = (5.0, 5.0)  # scan region end
        self.tipR0 = 4.0  # equlibirum distance of ProbeParticle form ancoring point

        self.surfFF = np.zeros(4, dtype=np.float32)

        self.cl_atoms = None
        self.cl_zMap = None
        self.cl_feMap = None

    def updateFEin(self, FEin_cl, bFinish=False):
        if verbose > 0:
            print(" updateFEin ", FEin_cl, self.cl_ImgIn, self.FEin_shape)
        if bFinish:
            self.queue.finish()
        cl.enqueue_copy(queue=self.queue, src=FEin_cl, dest=self.cl_ImgIn, offset=0, origin=(0, 0, 0), region=self.FEin_shape[:3])
        if bFinish:
            self.queue.finish()
        self.FEin_cl = FEin_cl

    def updateAtoms(self, atoms):
        if self.cl_atoms:
            self.cl_atoms.release()
        self.nAtoms = np.int32(len(atoms))
        self.cl_atoms = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=atoms)

    def prepareAuxMapBuffers(self, bZMap=False, bFEmap=False, atoms=None):
        nbytes = 0
        mf = cl.mem_flags
        fsize = np.dtype(np.float32).itemsize
        nxy = self.scan_dim[0] * self.scan_dim[1]
        if bZMap:
            if self.cl_zMap:
                self.cl_zMap.release()
            self.cl_zMap = cl.Buffer(self.ctx, mf.WRITE_ONLY, nxy * fsize)
            nbytes += nxy * fsize
        if bFEmap:
            if self.cl_feMap:
                self.cl_feMap.release()
            self.cl_feMap = cl.Buffer(self.ctx, mf.WRITE_ONLY, nxy * fsize * 4)
            nbytes += nxy * fsize * 4
        if atoms is not None:
            if self.cl_atoms:
                self.cl_atoms.release()
            self.nAtoms = np.int32(len(atoms))
            self.cl_atoms = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=atoms)
            nbytes += atoms.nbytes
        if self.verbose > 0:
            print("prepareAuxMapBuffers.nbytes: ", nbytes)

    def prepareBuffers(self, FEin_np=None, lvec=None, FEin_cl=None, FEin_shape=None, scan_dim=None, nDimConv=None, nDimConvOut=None, bZMap=False, bFEmap=False, atoms=None):
        nbytes = 0
        mf = cl.mem_flags

        if lvec is not None:
            self.lvec = lvec
            self.invCell = getInvCell(lvec)
        if FEin_np is not None:
            self.cl_ImgIn = cl.image_from_array(self.ctx, FEin_np, num_channels=4, mode="r")
            nbytes += FEin_np.nbytes  # TODO make this re-uploadable
            if self.verbose > 0:
                print("prepareBuffers made self.cl_ImgIn ", self.cl_ImgIn)
        else:
            if FEin_shape is not None:
                self.FEin_shape = FEin_shape
                self.image_format = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
                self.cl_ImgIn = cl.Image(self.ctx, mf.READ_ONLY, self.image_format, shape=FEin_shape[:3], pitches=None, hostbuf=None, is_array=False, buffer=None)
                if self.verbose > 0:
                    print("prepareBuffers made self.cl_ImgIn ", self.cl_ImgIn)
            if FEin_cl is not None:
                self.updateFEin(FEin_cl)
                self.FEin_cl = FEin_cl

        # see: https://stackoverflow.com/questions/39533635/pyopencl-3d-rgba-image-from-numpy-array
        if scan_dim is not None:
            self.scan_dim = scan_dim
            fsize = np.dtype(np.float32).itemsize
            f4size = fsize * 4
            nxy = self.scan_dim[0] * self.scan_dim[1]
            bsz = f4size * nxy
            self.cl_poss = cl.Buffer(self.ctx, mf.READ_ONLY, bsz)
            nbytes += bsz  # float4
            self.cl_FEout = cl.Buffer(self.ctx, mf.READ_WRITE, bsz * self.scan_dim[2])
            nbytes += bsz * self.scan_dim[2]
            self.cl_paths = cl.Buffer(self.ctx, mf.READ_WRITE, bsz * self.scan_dim[2])
            nbytes += bsz * self.scan_dim[2]
            if nDimConv is not None:
                self.nDimConv = nDimConv
                self.nDimConvOut = nDimConvOut
                self.cl_FEconv = cl.Buffer(self.ctx, mf.WRITE_ONLY, bsz * self.nDimConvOut)
                nbytes += bsz * self.nDimConvOut
                self.cl_WZconv = cl.Buffer(self.ctx, mf.READ_ONLY, fsize * self.nDimConv)
                nbytes += fsize * self.nDimConv
                self.FEconv = np.empty(
                    self.scan_dim[:2]
                    + (
                        self.nDimConvOut,
                        4,
                    ),
                    dtype=np.float32,
                )

        if bZMap:
            self.cl_zMap = cl.Buffer(self.ctx, mf.WRITE_ONLY, nxy * fsize)
            nbytes += nxy * fsize
        if bFEmap:
            self.cl_feMap = cl.Buffer(self.ctx, mf.WRITE_ONLY, nxy * fsize * 4)
            nbytes += nxy * fsize * 4
        if atoms is not None:
            self.updateAtoms(atoms)
            nbytes += atoms.nbytes

        if self.verbose > 0:
            print("prepareBuffers.nbytes: ", nbytes)

    def releaseBuffers(self):
        if self.verbose > 0:
            print("tryReleaseBuffers self.cl_ImgIn ", self.cl_ImgIn)
        self.cl_ImgIn.release()
        self.cl_poss.release()
        self.cl_FEout.release()
        if self.cl_zMap is not None:
            self.cl_zMap.release()
        if self.cl_feMap is not None:
            self.cl_feMap.release()
        if self.cl_atoms is not None:
            self.cl_atoms.release()

    def tryReleaseBuffers(self):
        if self.verbose > 0:
            print("tryReleaseBuffers self.cl_ImgIn ", self.cl_ImgIn)
        try:
            self.cl_ImgIn.release()
        except:
            pass
        try:
            self.cl_poss.release()
        except:
            pass
        try:
            self.cl_FEout.release()
        except:
            pass
        try:
            self.cl_zMap.release()
        except:
            pass
        try:
            self.cl_feMap.release()
        except:
            pass
        try:
            self.cl_atoms.release()
        except:
            pass

    def preparePosBasis(self, start=(-5.0, -5.0), end=(5.0, 5.0)):
        self.start = start
        self.end = end
        self.xs = np.linspace(start[0], end[0], self.scan_dim[0])
        self.ys = np.linspace(start[1], end[1], self.scan_dim[1])
        self.As, self.Bs = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.poss = np.zeros(self.As.shape + (4,), dtype=np.float32)

    def preparePossRot(self, pos0, avec, bvec):
        self.poss[:, :, 0] = pos0[0] + self.As * avec[0] + self.Bs * bvec[0]
        self.poss[:, :, 1] = pos0[1] + self.As * avec[1] + self.Bs * bvec[1]
        self.poss[:, :, 2] = pos0[2] + self.As * avec[2] + self.Bs * bvec[2]
        return self.poss

    def setScanRot(self, pos0, rot=None, zstep=None, tipR0=[0.0, 0.0, 4.0]):
        if rot is None:
            rot = np.eye(3)
        if zstep:
            self.zstep = zstep
        self.dTip, self.tipRot, self.dpos0Tip, self.dpos0 = rotTip(rot, self.zstep, tipR0)
        poss = self.preparePossRot(pos0, rot[0], rot[1])
        cl.enqueue_copy(self.queue, self.cl_poss, poss)
        return poss

    def updateBuffers(self, FEin=None, lvec=None, WZconv=None):
        if lvec is not None:
            self.invCell = getInvCell(lvec)
        if FEin is not None:
            region = FEin.shape[:3]
            region = region[::-1]
            if self.verbose > 0:
                print("region : ", region)
            cl.enqueue_copy(self.queue, self.cl_ImgIn, FEin, origin=(0, 0, 0), region=region)
        if WZconv is not None:
            cl.enqueue_copy(self.queue, self.cl_WZconv, WZconv)

    def downloadPaths(self):
        """
        Get probe particle path array from device.

        Returns:
            paths: np.ndarray of shape scan_dim + (3,). xyz positions of probe particle at all scan points.
        """

        # Make numpy array. Last axis is bigger by one because OCL aligns to multiples of 4 floats.
        paths = np.empty(self.scan_dim + (4,), dtype=np.float32, order="C")

        if self.verbose:
            print("paths.shape ", paths.shape)

        # Copy from device to host
        cl.enqueue_copy(self.queue, paths, self.cl_paths)
        self.queue.finish()

        # Get rid of extra column
        paths = paths[:, :, :, :3]

        # Add origin because the OCL kernel does not take it into account
        paths += self.lvec[0]

        return paths

    def run(self, FEout=None, FEin=None, lvec=None, nz=None):
        """
        calculate force on relaxing probe particle approaching from top (z-direction)
        """
        if nz is None:
            nz = self.scan_dim[2]
        self.updateBuffers(FEin=FEin, lvec=lvec)
        if FEout is None:
            FEout = np.empty(self.scan_dim + (4,), dtype=np.float32)
        # fmt: off
        cl_program.relaxStrokes(self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None,
            self.cl_ImgIn,
            self.cl_poss,
            self.cl_FEout,
            self.invCell[0],
            self.invCell[1],
            self.invCell[2],
            self.dTip,
            self.stiffness,
            self.dpos0,
            self.relax_params,
            np.int32(nz)
        )
        # fmt: on
        cl.enqueue_copy(self.queue, FEout, self.cl_FEout)
        self.queue.finish()
        return FEout

    def run_relaxStrokesTilted(self, FEout=None, FEin=None, lvec=None, nz=None, bCopy=True, bFinish=True):
        """
        calculate force on relaxing probe particle approaching from particular direction
        """
        if nz is None:
            nz = self.scan_dim[2]
        if bCopy and (FEout is None):
            FEout = np.empty(self.scan_dim + (4,), dtype=np.float32)
        self.updateBuffers(FEin=FEin, lvec=lvec)
        # fmt: off
        cl_program.relaxStrokesTilted(self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None,
            self.cl_ImgIn,
            self.cl_poss,
            self.cl_FEout,
            self.cl_paths,
            self.invCell[0],
            self.invCell[1],
            self.invCell[2],
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2],
            self.stiffness,
            self.dpos0Tip,
            self.relax_params,
            self.surfFF,
            np.int32(nz)
        )
        # fmt: on
        if bCopy:
            cl.enqueue_copy(self.queue, FEout, self.cl_FEout)
        if bFinish:
            self.queue.finish()
        return FEout

    def run_relaxStrokesTilted_convZ(self, FEconv=None, FEin=None, lvec=None, nz=None):
        """
        calculate force on relaxing probe particle approaching from particular direction
        """
        if nz is None:
            nz = self.scan_dim[2]
        if FEconv is None:
            if self.FEconv is not None:
                FEconv = self.FEconv
            else:
                FEconv = self.prepareFEConv()
        self.updateBuffers(FEin=FEin, lvec=lvec)
        # fmt: off
        cl_program.relaxStrokesTilted_convZ(self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None,
            self.cl_ImgIn,
            self.cl_poss,
            self.cl_WZconv,
            self.cl_FEconv,
            self.invCell[0],
            self.invCell[1],
            self.invCell[2],
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2],
            self.stiffness,
            self.dpos0Tip,
            self.relax_params,
            self.surfFF,
            np.int32(nz), np.int32(self.nDimConvOut),
        )
        # fmt: on
        cl.enqueue_copy(self.queue, FEconv, self.cl_FEconv)
        self.queue.finish()
        return FEconv

    def run_getFEinStrokes(self, FEout=None, FEconv=None, FEin=None, lvec=None, nz=None, WZconv=None, bDoConv=False):
        """
        un-relaxed sampling of FE values from input Force-field (cl_ImgIn) store to cl_FEout
        """
        if nz is None:
            nz = self.scan_dim[2]
        self.updateBuffers(FEin=FEin, lvec=lvec, WZconv=WZconv)
        # fmt: off
        cl_program.getFEinStrokes(self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None,
            self.cl_ImgIn,
            self.cl_poss,
            self.cl_FEout,
            self.invCell[0],
            self.invCell[1],
            self.invCell[2],
            self.dTip,
            self.dpos0,
            np.int32(nz)
        )
        # fmt: on
        if bDoConv:
            # This function is missing. Maybe this should be self.run_convolveZ?
            FEout = runZConv(self, FEconv=FEconv, nz=nz)
        else:
            if FEout is None:
                FEout = np.empty(self.scan_dim + (4,), dtype=np.float32)
            cl.enqueue_copy(self.queue, FEout, self.cl_FEout)
        self.queue.finish()
        return FEout

    def run_getFEinStrokesTilted(self, FEout=None, FEin=None, lvec=None, nz=None):
        """
        un-relaxed sampling of FE values from input Force-field (cl_ImgIn) store to cl_FEout
        operates in coordinates rotated by tipRot
        """
        if nz is None:
            nz = self.scan_dim[2]
        self.updateBuffers(FEin=FEin, lvec=lvec)
        # fmt: off
        cl_program.getFEinStrokesTilted(self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None,
            self.cl_ImgIn,
            self.cl_poss,
            self.cl_FEout,
            self.invCell[0],
            self.invCell[1],
            self.invCell[2],
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2],
            self.dTip,
            self.dpos0,
            np.int32(nz)
        )
        # fmt: on
        cl.enqueue_copy(self.queue, FEout, self.cl_FEout)
        self.queue.finish()
        return FEout

    def prepareFEConv(self):
        return np.empty(
            self.scan_dim[:2]
            + (
                self.nDimConvOut,
                4,
            ),
            dtype=np.float32,
        )

    def run_convolveZ(self, FEconv=None, nz=None):
        """
        convolve 3D forcefield in FEout with 1D weight mask WZconv
        """
        if nz is None:
            nz = self.scan_dim[2]
        if FEconv is None:
            if self.FEconv is not None:
                FEconv = self.FEconv
            else:
                FEconv = self.prepareFEConv()
        # fmt: off
        cl_program.convolveZ(self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None,
            self.cl_FEout,
            self.cl_FEconv,
            self.cl_WZconv,
            np.int32(nz), np.int32(self.nDimConvOut)
        )
        # fmt: on
        cl.enqueue_copy(self.queue, FEconv, self.cl_FEconv)
        self.queue.finish()
        return FEconv

    def run_izoZ(self, zMap=None, iso=0.0, nz=None):
        """
        get isosurface of input 3D field from top (z)
        used to generate HeightMap
        if cl_FEout is Forcefield it takes "z" where ( F(z) > iso )
        """
        if nz is None:
            nz = self.scan_dim[2]
        if zMap is None:
            zMap = np.empty(self.scan_dim[:2], dtype=np.float32)
        # fmt: off
        cl_program.izoZ( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None,
            self.cl_FEout,
            self.cl_zMap,
            np.int32(nz), np.float32(iso)
        )
        # fmt: on
        cl.enqueue_copy(self.queue, zMap, self.cl_zMap)
        self.queue.finish()
        return zMap

    def run_getZisoTilted(self, zMap=None, iso=0.0, nz=None):
        """
        get isosurface of input 3D field from given direction
        used to generate HeightMap
        operates in coordinates rotated by tipRot
        """
        if nz is None:
            nz = self.scan_dim[2]
        if zMap is None:
            zMap = np.empty(self.scan_dim[:2], dtype=np.float32)
        local_size = (1,)
        # fmt: off
        cl_program.getZisoTilted(self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), local_size,
            self.cl_ImgIn,
            self.cl_poss,
            self.cl_zMap,
            self.invCell[0],
            self.invCell[1],
            self.invCell[2],
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2],
            self.dTip,
            self.dpos0,
            np.int32(nz), np.float32( iso )
        )
        # fmt: on
        cl.enqueue_copy(self.queue, zMap, self.cl_zMap)
        self.queue.finish()
        return zMap

    def run_getZisoFETilted(self, zMap=None, feMap=None, iso=0.0, nz=None):
        """
        get isosurface of input 3D field from given direction
        get map of 3D volume FE (e.g. electrostatic field) maped on 2D isosurface
        used to generate ElectrostaticMap
        returns zMap, feMap
        operates in coordinates rotated by tipRot
        """
        if self.cl_atoms is None:
            raise ValueError("Atoms must be set before calculating the Electrostatic Map")
        if nz is None:
            nz = self.scan_dim[2]
        if zMap is None:
            zMap = np.empty(self.scan_dim[:2], dtype=np.float32)
        if feMap is None:
            feMap = np.empty(self.scan_dim[:2] + (4,), dtype=np.float32)
        local_size = (1,)
        # fmt: off
        cl_program.getZisoFETilted(self.queue, ( np.int32(self.scan_dim[0]*self.scan_dim[1]),), local_size,
            self.cl_ImgIn,
            self.cl_poss,
            self.cl_zMap,
            self.cl_feMap,
            self.nAtoms,
            self.cl_atoms,
            self.invCell[0], self.invCell[1], self.invCell[2],
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2],
            self.dTip,
            self.dpos0,
            np.int32(nz), np.float32( iso )
        )
        # fmt: on
        cl.enqueue_copy(self.queue, zMap, self.cl_zMap)
        cl.enqueue_copy(self.queue, feMap, self.cl_feMap)
        self.queue.finish()
        return zMap, feMap
