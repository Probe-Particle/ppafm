#!/usr/bin/python

import sys
import os
import pyopencl as cl
import numpy as np

import oclUtils as oclu

import common as PPU
#import cpp_utils as cpp_utils

# ========== Globals

cl_program = None
#cl_queue   = None 
#cl_context = None

#FE         = None
#FEout      = None

DEFAULT_dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
DEFAULT_stiffness    = np.array( [-0.03,-0.03, -0.03,-1.0 ], dtype=np.float32 );
DEFAULT_dpos0        = np.array( [ 0.0 , 0.0 , -4.0 , 4.0 ], dtype=np.float32 );
DEFAULT_relax_params = np.array( [ 0.1 , 0.9 ,  0.02, 0.5 ], dtype=np.float32 );

verbose = 0

# ========== Functions

'''
# this should be elsewhere - we do not want dependnece on GridUtils
def loadFEcl( Q = None ):
    import GridUtils as GU 
    E ,lvec, nDim, head = GU.loadXSF('ELJ_cl.xsf' ); FE = np.zeros( E.shape+(4,), dtype=np.float32 ); FE.shape; FE[:,:,:,3] = E
    Fx,lvec, nDim, head = GU.loadXSF('FLJx_cl.xsf'); FE[:,:,:,0] = Fx   
    Fy,lvec, nDim, head = GU.loadXSF('FLJy_cl.xsf'); FE[:,:,:,1] = Fy
    Fz,lvec, nDim, head = GU.loadXSF('FLJz_cl.xsf'); FE[:,:,:,2] = Fz    
    if Q is not None:
        E ,lvec, nDim, head = GU.loadXSF('Eel_cl.xsf' ); FE[:,:,:,3] += Q*E
        Fx,lvec, nDim, head = GU.loadXSF('Felx_cl.xsf'); FE[:,:,:,0] += Q*Fx   
        Fy,lvec, nDim, head = GU.loadXSF('Fely_cl.xsf'); FE[:,:,:,1] += Q*Fy
        Fz,lvec, nDim, head = GU.loadXSF('Felz_cl.xsf'); FE[:,:,:,2] += Q*Fz    
    return FE, lvec
'''

def init( cl_context=oclu.ctx):
    global cl_program
    cl_program  = oclu.loadProgram(oclu.CL_PATH+"/relax.cl")

def mat3x3to4f( M ):
    a = np.zeros( 4, dtype=np.float32); a[0:3] = M[0]
    b = np.zeros( 4, dtype=np.float32); b[0:3] = M[1]
    c = np.zeros( 4, dtype=np.float32); c[0:3] = M[2]
    return (a, b, c)

def getInvCell( lvec ):
    cell = lvec[1:4,0:3]
    invCell = np.transpose( np.linalg.inv(cell) )
    if(verbose>0): print invCell
    #invA = np.zeros( 4, dtype=np.float32); invA[0:3] = invCell[0]
    #invB = np.zeros( 4, dtype=np.float32); invB[0:3] = invCell[1]
    #invC = np.zeros( 4, dtype=np.float32); invC[0:3] = invCell[2]
    return mat3x3to4f( invCell )

def preparePoss( scan_dim, z0, start=(0.0,0.0), end=(10.0,10.0) ):
    #print "DEBUG preparePoss : ", scan_dim, z0, start, end
    ys    = np.linspace(start[0],end[0],scan_dim[0])
    xs    = np.linspace(start[1],end[1],scan_dim[1])
    Xs,Ys = np.meshgrid(xs,ys)
    poss  = np.zeros(Xs.shape+(4,), dtype=np.float32)
    poss[:,:,0] = Ys
    poss[:,:,1] = Xs
    poss[:,:,2] = z0
    #print "DEBUG: poss[:,:,0:2]: " , poss[:,:,0:2]
    return poss

def preparePossRot( scan_dim, pos0, avec, bvec, start=(-5.0,-5.0), end=(5.0,5.0) ):
    ys    = np.linspace(start[0],end[0],scan_dim[0])
    xs    = np.linspace(start[1],end[1],scan_dim[1])
    As,Bs = np.meshgrid(xs,ys)
    poss  = np.zeros(As.shape+(4,), dtype=np.float32)
    poss[:,:,0] = pos0[0] + As*avec[0] + Bs*bvec[0]
    poss[:,:,1] = pos0[1] + As*avec[1] + Bs*bvec[1]
    poss[:,:,2] = pos0[2] + As*avec[2] + Bs*bvec[2]
    #print "DEBUG: poss[:,:,0:2]: " , poss[:,:,0:2]
    return poss

def prepareBuffers( FE, scan_dim, ctx=oclu.ctx ):
    nbytes = 0
    print "prepareBuffers FE.shape", FE.shape
    mf       = cl.mem_flags
    cl_ImgIn = cl.image_from_array(ctx,FE,num_channels=4,mode='r');  nbytes+=FE.nbytes        # TODO make this re-uploadable
    bsz=np.dtype(np.float32).itemsize * 4 * scan_dim[0] * scan_dim[1]
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY , bsz                );   nbytes+=bsz              # float4
    cl_FEout = cl.Buffer(ctx, mf.WRITE_ONLY, bsz * scan_dim[2] );   nbytes+=bsz*scan_dim[2] # float4
    print "FFout.nbytes : ", bsz * scan_dim[2]
    print "prepareBuffers.nbytes: ", nbytes
    kargs = (cl_ImgIn, cl_poss, cl_FEout )
    return kargs

def releaseArgs( kargs ):
    kargs[0].release() # cl_ImgIn
    kargs[1].release() # cl_poss
    kargs[2].release() # cl_FEout

def relax( kargs, scan_dim, invCell, poss=None, FEin=None, FEout=None, dTip=DEFAULT_dTip, stiffness=DEFAULT_stiffness, dpos0=DEFAULT_dpos0, relax_params=DEFAULT_relax_params, queue=oclu.queue):
    nz = np.int32( scan_dim[2] )
    kargs = kargs  + ( invCell[0],invCell[1],invCell[2], dTip, stiffness, dpos0, relax_params, nz )
    if FEout is None:
        FEout = np.zeros( scan_dim+(4,), dtype=np.float32 )
        if(verbose>0): print "FEout.shape", FEout.shape, scan_dim
    if poss is not None:
        cl.enqueue_copy( queue, kargs[1], poss )
    if FEin is not None:
        region = FEin.shape[:3]; region = region[::-1]; 
        if(verbose>0): print "region : ", region
        cl.enqueue_copy( queue, kargs[0], FEin, origin=(0,0,0), region=region )
    #print kargs
    cl_program.relaxStrokes( queue, ( int(scan_dim[0]*scan_dim[1]),), None, *kargs )
    cl.enqueue_copy( queue, FEout, kargs[2] )
    queue.finish()
    return FEout

def saveResults():
    lvec_OUT = (
        [0.0,0.0,0.0],
        [10.0,0.0,0.0],
        [0.0,10.0,0.0],
        [0.0,0.0,6.0]
    )
    nd  = FEout.shape
    nd_ = (nd[2],nd[0],nd[1])
    print nd_
    Ftmp=np.zeros(nd_);
    Ftmp[:,:,:] = np.transpose( FEout[:,:,:,2], (2,0,1) ); GU.saveXSF( 'OutFz_cl.xsf',  Ftmp, lvec_OUT );

if __name__ == "__main__":
    prepareProgram()
    kargs, relaxShape = prepareBuffers()
    relax( kargs, relaxShape )
    saveResults()

## ============= Relax Class:

class RelaxedScanner:

    #verbose=0  # this is global for now

    def __init__( self ):
        self.queue  = oclu.queue
        self.ctx    = oclu.ctx
        self.stiffness    = DEFAULT_stiffness.copy()
        self.relax_params = DEFAULT_relax_params.copy()

        #self.pos0  = pos0; 
        self.zstep = 0.1
        self.start = (-5.0,-5.0)
        self.end   = ( 5.0, 5.0)
        self.tipR0 = 4.0

        self.surfFF = np.zeros(4,dtype=np.float32);

    def prepareBuffers(self, FEin, lvec, scan_dim=(100,100,20), nDimConv=None, nDimConvOut=None, bZMap=False, bFEmap=False, FE2in=None ):
        self.scan_dim = scan_dim
        self.invCell  = getInvCell(lvec)
        nbytes = 0
        #print "prepareBuffers FE.shape", FE.shape
        mf       = cl.mem_flags
        self.cl_ImgIn = cl.image_from_array(self.ctx,FEin,num_channels=4,mode='r');  nbytes+=FEin.nbytes        # TODO make this re-uploadable
        # see:    https://stackoverflow.com/questions/39533635/pyopencl-3d-rgba-image-from-numpy-array
        #img_format = cl.ImageFormat( cl.channel_order.RGBA, channel_type)
        #self.cl_ImgIn =  cl.Image(self.ctx, mf.READ_ONLY, img_format, shape=None, pitches=None, is_array=False, buffer=None)
        fsize  = np.dtype(np.float32).itemsize
        f4size = fsize * 4
        nxy    =  self.scan_dim[0] * self.scan_dim[1]
        bsz    = f4size * nxy
        self.cl_poss  = cl.Buffer(self.ctx, mf.READ_ONLY , bsz                    );   nbytes+=bsz                  # float4
        self.cl_FEout = cl.Buffer(self.ctx, mf.READ_WRITE, bsz * self.scan_dim[2] );   nbytes+=bsz*self.scan_dim[2]
        if nDimConv is not None:
            self.nDimConv    = nDimConv
            self.nDimConvOut = nDimConvOut
#            print "nDimConv %i nDimConvOut %i" %( nDimConv, nDimConvOut )
            self.cl_FEconv  = cl.Buffer(self.ctx, mf.WRITE_ONLY, bsz       * self.nDimConvOut ); nbytes += bsz  *self.nDimConvOut
            self.cl_WZconv  = cl.Buffer(self.ctx, mf.READ_ONLY,  fsize     * self.nDimConv    ); nbytes += fsize*self.nDimConv
        self.cl_zMap = None; self.cl_feMap=None; self.cl_ImgFE = None
        if bZMap:
#            print "nxy ", nxy
            self.cl_zMap    = cl.Buffer(self.ctx, mf.WRITE_ONLY, nxy*fsize   ); nbytes += nxy*fsize
        if bFEmap:
            self.cl_feMap  = cl.Buffer(self.ctx, mf.WRITE_ONLY, nxy*fsize*4  ); nbytes += nxy*fsize*4
            if FE2in is not None:
                self.cl_ImgFE = cl.image_from_array(self.ctx,FE2in,num_channels=4,mode='r');  nbytes+=FE2in.nbytes
        if(verbose>0): print "prepareBuffers.nbytes: ", nbytes

    def releaseBuffers(self):
        self.cl_ImgIn.release()
        self.cl_poss.release()
        self.cl_FEout.release()
        if self.cl_ImgFE is not None: self.cl_ImgFE.release()
        if self.cl_zMap  is not None: self.cl_zMap.release()
        if self.cl_feMap is not None: self.cl_feMap.release()

    def setScanRot(self, pos0, rot=None, start=(-5.0,-5.0), end=(5.0,5.0), zstep=0.1, tipR0=[0.0,0.0,4.0] ):
        if rot is None:
            rot = np.identity()
        #self.pos0=pos0; 
        self.zstep=zstep; self.start= start; self.end=end
        #pos0 += rot[2]*self.distAbove
        self.dTip     = np.zeros(4,dtype=np.float32); self.dTip [:3] = rot[2]*-zstep
        self.tipRot = mat3x3to4f( rot )
        self.tipRot[2][3] = -zstep

        self.dpos0Tip = np.zeros(4,dtype=np.float32);
        self.dpos0Tip[0]  =   tipR0[0];
        self.dpos0Tip[1]  =   tipR0[1];
        self.dpos0Tip[2]  =  -np.sqrt(tipR0[2]**2 - tipR0[0]**2 - tipR0[1]**2);
        self.dpos0Tip[3]  =   tipR0[2]

        self.dpos0    = np.zeros(4,dtype=np.float32); 
        #self.dpos0[:3] = rot[2]*-tipR0;  self.dpos0[3] = tipR0
        self.dpos0[:3]  = np.dot( rot, self.dpos0Tip[:3] );  self.dpos0[3] = tipR0[2]
        print " self.dpos0Tip: ", self.dpos0Tip, " self.dpos0 ", self.dpos0


        ys    = np.linspace(start[0],end[0],self.scan_dim[0])
        xs    = np.linspace(start[1],end[1],self.scan_dim[1])
        As,Bs = np.meshgrid(xs,ys)
        poss  = np.zeros(As.shape+(4,), dtype=np.float32)
        poss[:,:,0] = pos0[0] + As*rot[0,0] + Bs*rot[1,0]
        poss[:,:,1] = pos0[1] + As*rot[0,1] + Bs*rot[1,1]
        poss[:,:,2] = pos0[2] + As*rot[0,2] + Bs*rot[1,2]
        #print "DEBUG: poss[:,:,0:2]: " , poss[:,:,0:2]
        #poss[:,:,:] = pos0[None,None,:] + As[:,:,None] * rot[0][None,None,:]   + As[:,:,None] * rot[1][None,None,:]
        #self.pos0s = poss
        cl.enqueue_copy( self.queue, self.cl_poss, poss )
        return poss

    def updateBuffers(self, FEin=None, FE2in=None, lvec=None, WZconv=None ):
        #if FEout is None:    FEout = np.zeros( self.scan_dim+(4,), dtype=np.float32 )
        if lvec is not None: self.invCell = getInvCell(lvec)
        if FEin is not None:
            region = FEin.shape[:3]; region = region[::-1]; 
            if(verbose>0): print "region : ", region
            cl.enqueue_copy( self.queue, self.cl_ImgIn, FEin, origin=(0,0,0), region=region )
        if FE2in is not None:
            region = FE2in.shape[:3]; region = region[::-1]; 
            if(verbose>0): print "region : ", region
            cl.enqueue_copy( self.queue, self.cl_ImgFE, FE2in, origin=(0,0,0), region=region )
        if WZconv is not None:
            #print "WZconv: ", WZconv.dtype, WZconv
            #cl.enqueue_copy( self.queue, WZconv, self.cl_WZconv )
            cl.enqueue_copy( self.queue, self.cl_WZconv, WZconv )
            #self.queue.finish()

    def run(self, FEout=None, FEin=None, lvec=None, nz=None ):
        if nz is None: nz=self.scan_dim[2] 
        self.updateBuffers( FEin=FEin, lvec=lvec )
        if FEout is None:    FEout = np.empty( self.scan_dim+(4,), dtype=np.float32 )
        kargs = ( 
            self.cl_ImgIn, 
            self.cl_poss, 
            self.cl_FEout, 
            self.invCell[0], self.invCell[1], self.invCell[2], 
            self.dTip, 
            self.stiffness, 
            self.dpos0, 
            self.relax_params, 
            np.int32(nz) )
        #print kargs
        cl_program.relaxStrokes( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None, *kargs )
        cl.enqueue_copy( self.queue, FEout, self.cl_FEout )
        self.queue.finish()
        return FEout

    def run_relaxStrokesTilted(self, FEout=None, FEin=None, lvec=None, nz=None ):
        if nz is None: nz=self.scan_dim[2] 
        if FEout is None:    FEout = np.empty( self.scan_dim+(4,), dtype=np.float32 )
        self.updateBuffers( FEin=FEin, lvec=lvec )
        kargs = ( 
            self.cl_ImgIn, 
            self.cl_poss, 
            self.cl_FEout, 
            self.invCell[0], self.invCell[1], self.invCell[2], 
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2],
            self.stiffness, 
            self.dpos0Tip, 
            self.relax_params, 
            self.surfFF,
            np.int32(nz) )
        #print kargs
        cl_program.relaxStrokesTilted( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None, *kargs )
        cl.enqueue_copy( self.queue, FEout, self.cl_FEout )
        self.queue.finish()
        #print " !!!! FEout.shape", FEout.shape
        return FEout

    def run_getFEinStrokes(self, FEout=None, FEconv=None, FEin=None, lvec=None, nz=None, WZconv=None, bDoConv=False ):
        if nz is None: nz=self.scan_dim[2] 
        self.updateBuffers( FEin=FEin, lvec=lvec, WZconv=WZconv )
        kargs = ( 
            self.cl_ImgIn, 
            self.cl_poss, 
            self.cl_FEout, 
            self.invCell[0], self.invCell[1], self.invCell[2],
            self.dTip,
            self.dpos0, 
            np.int32(nz) )
        cl_program.getFEinStrokes( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None, *kargs )
        if bDoConv:
            FEout = runZConv(self, FEconv=FEconv, nz=nz )
        else:
            if FEout is None:    FEout = np.empty( self.scan_dim+(4,), dtype=np.float32 )
            cl.enqueue_copy( self.queue, FEout, self.cl_FEout )
        self.queue.finish()
        return FEout

    def run_getFEinStrokesTilted(self, FEout=None, FEin=None, lvec=None, nz=None ):
        if nz is None: nz=self.scan_dim[2] 
        self.updateBuffers( FEin=FEin, lvec=lvec )
        kargs = ( 
            self.cl_ImgIn, 
            self.cl_poss, 
            self.cl_FEout, 
            self.invCell[0], self.invCell[1], self.invCell[2], 
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2],
            self.dTip,
            self.dpos0, 
            np.int32(nz) )
        cl_program.getFEinStrokesTilted( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None, *kargs )
        cl.enqueue_copy( self.queue, FEout, self.cl_FEout )
        self.queue.finish()
        return FEout

    def run_convolveZ(self, FEconv=None, nz=None ):
        if nz is None: nz=self.scan_dim[2]
        #print " runZConv  nz %i nDimConv %i "  %(nz, self.nDimConvOut)
        if FEconv is None: FEconv = np.empty( self.scan_dim[:2]+(self.nDimConvOut,4,), dtype=np.float32 )
        kargs = ( 
            self.cl_FEout,
            self.cl_FEconv,
            self.cl_WZconv, 
            np.int32(nz), np.int32( self.nDimConvOut ) )
        cl_program.convolveZ( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None, *kargs )
        cl.enqueue_copy( self.queue, FEconv, self.cl_FEconv )
        self.queue.finish()
        return FEconv

    def run_izoZ(self, zMap=None, iso=0.0, nz=None ):
        if nz is None: nz=self.scan_dim[2]
        if zMap is None: zMap = np.empty( self.scan_dim[:2], dtype=np.float32 )
        kargs = ( 
            self.cl_FEout,
            self.cl_zMap, 
            np.int32(nz), np.float32( iso ) )
        cl_program.izoZ( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None, *kargs )
        cl.enqueue_copy( self.queue, zMap, self.cl_zMap )
        self.queue.finish()
        return zMap

    def run_getZisoTilted(self, zMap=None, iso=0.0, nz=None ):
        if nz is None: nz=self.scan_dim[2]
        if zMap is None: zMap = np.empty( self.scan_dim[:2], dtype=np.float32 )
        kargs = (
            self.cl_ImgIn,
            self.cl_poss,
            self.cl_zMap,
            self.invCell[0], self.invCell[1], self.invCell[2],
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2],
            self.dTip,
            self.dpos0,
            np.int32(nz), np.float32( iso ) )
        local_size = (1,)
        cl_program.getZisoTilted( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), local_size, *kargs )
        cl.enqueue_copy( self.queue, zMap, self.cl_zMap )
        self.queue.finish()
        return zMap

    def run_getZisoFETilted(self, zMap=None, feMap=None, iso=0.0, nz=None ):
        if nz is None: nz=self.scan_dim[2]
        if zMap is None:  zMap  = np.empty( self.scan_dim[:2], dtype=np.float32 )
        if feMap is None: feMap = np.empty( self.scan_dim[:2]+(4,), dtype=np.float32 )
        kargs = (
            self.cl_ImgIn,
            self.cl_ImgFE,
            self.cl_poss,
            self.cl_zMap,
            self.cl_feMap,
            self.invCell[0], self.invCell[1], self.invCell[2],
            self.tipRot[0],  self.tipRot[1],  self.tipRot[2],
            self.dTip,
            self.dpos0,
            np.int32(nz), np.float32( iso ) )
        local_size = (1,)
        cl_program.getZisoFETilted( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), local_size, *kargs )
        cl.enqueue_copy( self.queue,  zMap, self.cl_zMap  )
        cl.enqueue_copy( self.queue, feMap, self.cl_feMap )
        self.queue.finish()
        return zMap, feMap

