#!/usr/bin/python

import sys
import os
import pyopencl as cl
import numpy as np

import oclUtils as oclu
import GridUtils as GU
import common as PPU
#import cpp_utils as cpp_utils

# ========== Globals

cl_program = None
#cl_queue   = None 
#cl_context = None

FE         = None
FEout      = None

DEFAULT_dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
DEFAULT_stiffness    = np.array( [-0.03,-0.03, -0.03,-1.0 ], dtype=np.float32 );
DEFAULT_dpos0        = np.array( [ 0.0 , 0.0 , -4.0 , 4.0 ], dtype=np.float32 );
DEFAULT_relax_params = np.array( [ 0.1 , 0.9 ,  0.02, 0.5 ], dtype=np.float32 );

# ========== Functions

def loadFEcl( Q = None ):
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

def init( cl_context=oclu.ctx):
    global cl_program
    cl_program  = oclu.loadProgram(oclu.CL_PATH+"/relax.cl")

def getInvCell( lvec ):
    cell = lvec[1:4,0:3]
    invCell = np.transpose( np.linalg.inv(cell) )
    print invCell
    invA = np.zeros( 4, dtype=np.float32); invA[0:3] = invCell[0]
    invB = np.zeros( 4, dtype=np.float32); invB[0:3] = invCell[1]
    invC = np.zeros( 4, dtype=np.float32); invC[0:3] = invCell[2]
    return (invA, invB, invC)

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
        print "FEout.shape", FEout.shape, scan_dim
    if poss is not None:
        cl.enqueue_copy( queue, kargs[1], poss )
    if FEin is not None:
        region = FEin.shape[:3]; region = region[::-1]; print "region : ", region
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

    def __init__( self ):
        self.queue  = oclu.queue
        self.ctx    = oclu.ctx
        #self.ndim   = ( 100, 100, 20)
        #distAbove  = 7.5
        #islices    = [0,+2,+4,+6,+8]
        #self.relax_params = np.array( [ 0.1, 0.9, 0.1*0.2, 0.1*5.0 ], dtype=np.float32 );
        #self.dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
        #self.stiffness    = np.array( [ 0.24,0.24,0.0, 30.0    ], dtype=np.float32 ); stiffness/=-16.0217662;
        #self.dpos0        = np.array( [ 0.0,0.0,0.0,4.0        ], dtype=np.float32 ); 
        #self.dpos0[2]     = -np.sqrt( dpos0[3]**2 - dpos0[0]**2 + dpos0[1]**2 );
        self.stiffness    = DEFAULT_stiffness.copy()
        self.relax_params = DEFAULT_relax_params.copy()
        #self.dTip         = DEFAULT_dTip.copy()
        #self.dpos0        = DEFAULT_dpos0.copy()
        #print "dpos0 ", dpos0

    def prepareBuffers(self, FEin, lvec, scan_dim=(100,100,20) ):
        self.scan_dim = scan_dim
        self.invCell  = getInvCell(lvec)
        nbytes = 0
        #print "prepareBuffers FE.shape", FE.shape
        mf       = cl.mem_flags
        self.cl_ImgIn = cl.image_from_array(self.ctx,FEin,num_channels=4,mode='r');  nbytes+=FEin.nbytes        # TODO make this re-uploadable
        # see:    https://stackoverflow.com/questions/39533635/pyopencl-3d-rgba-image-from-numpy-array
        #img_format = cl.ImageFormat( cl.channel_order.RGBA, channel_type)
        #self.cl_ImgIn =  cl.Image(self.ctx, mf.READ_ONLY, img_format, shape=None, pitches=None, is_array=False, buffer=None)
        bsz=np.dtype(np.float32).itemsize * 4 * self.scan_dim[0] * self.scan_dim[1]
        self.cl_poss  = cl.Buffer(self.ctx, mf.READ_ONLY , bsz                     );  nbytes+=bsz                  # float4
        self.cl_FEout = cl.Buffer(self.ctx, mf.WRITE_ONLY, bsz * self.scan_dim[2] );   nbytes+=bsz*self.scan_dim[2] # float4
        #print "FFout.nbytes : ", bsz * scan_dim[2]
        print "prepareBuffers.nbytes: ", nbytes

    def releaseBuffers(self):
        self.cl_ImgIn.release()
        self.cl_poss.release()
        self.cl_FEout.release()

    def setScanRot(self, pos0, rot=None, start=(-5.0,-5.0), end=(5.0,5.0), zstep=0.1, tipR0=4.0 ):
        if rot is None:
            rot = np.identity()
        self.dTip =np.zeros(4,dtype=np.float32); self.dTip [:3] = rot[2]*-zstep
        self.dpos0=np.zeros(4,dtype=np.float32); self.dpos0[:3] = rot[2]*-tipR0;  self.dpos0[3] = tipR0
        ys    = np.linspace(start[0],end[0],self.scan_dim[0])
        xs    = np.linspace(start[1],end[1],self.scan_dim[1])
        As,Bs = np.meshgrid(xs,ys)
        poss  = np.zeros(As.shape+(4,), dtype=np.float32)
        poss[:,:,0] = pos0[0] + As*rot[0,0] + Bs*rot[1,0]
        poss[:,:,1] = pos0[1] + As*rot[0,1] + Bs*rot[1,1]
        poss[:,:,2] = pos0[2] + As*rot[0,2] + Bs*rot[1,2]
        #print "DEBUG: poss[:,:,0:2]: " , poss[:,:,0:2]
        #poss[:,:,:] = pos0[None,None,:] + As[:,:,None] * rot[0][None,None,:]   + As[:,:,None] * rot[1][None,None,:]
        #self.poss = poss
        cl.enqueue_copy( self.queue, self.cl_poss, poss )

    def run(self, FEout=None, FEin=None, lvec=None ):
        nz = np.int32( self.scan_dim[2] )
        kargs = ( 
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
            nz )
        if FEout is None:
            FEout = np.zeros( self.scan_dim+(4,), dtype=np.float32 )
            print "FEout.shape", FEout.shape, self.scan_dim
        if lvec is not None:
            self.invCell = getInvCell(lvec)
        if FEin is not None:
            region = FEin.shape[:3]; region = region[::-1]; print "region : ", region
            cl.enqueue_copy( self.queue, self.cl_ImgIn, FEin, origin=(0,0,0), region=region )
        #print kargs
        cl_program.relaxStrokes( self.queue, ( int(self.scan_dim[0]*self.scan_dim[1]),), None, *kargs )
        cl.enqueue_copy( self.queue, FEout, self.cl_FEout )
        self.queue.finish()
        return FEout



