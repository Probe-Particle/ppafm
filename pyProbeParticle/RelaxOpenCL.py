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

def preparePoss( relax_dim, z0, start=(0.0,0.0), end=(10.0,10.0) ):
    #print "DEBUG preparePoss : ", relax_dim, z0, start, end
    ys    = np.linspace(start[0],end[0],relax_dim[0])
    xs    = np.linspace(start[1],end[1],relax_dim[1])
    Xs,Ys = np.meshgrid(xs,ys)
    poss  = np.zeros(Xs.shape+(4,), dtype=np.float32)
    poss[:,:,0] = Ys
    poss[:,:,1] = Xs
    poss[:,:,2] = z0
    #print "DEBUG: poss[:,:,0:2]: " , poss[:,:,0:2]
    return poss

def preparePossRot( relax_dim, pos0, avec, bvec, start=(-5.0,-5.0), end=(5.0,5.0) ):
    ys    = np.linspace(start[0],end[0],relax_dim[0])
    xs    = np.linspace(start[1],end[1],relax_dim[1])
    As,Bs = np.meshgrid(xs,ys)
    poss  = np.zeros(As.shape+(4,), dtype=np.float32)
    poss[:,:,0] = pos0[0] + As*avec[0] + Bs*bvec[0]
    poss[:,:,1] = pos0[1] + As*avec[1] + Bs*bvec[1]
    poss[:,:,2] = pos0[2] + As*avec[2] + Bs*bvec[2]
    #print "DEBUG: poss[:,:,0:2]: " , poss[:,:,0:2]
    return poss

def prepareBuffers( FE, relax_dim, ctx=oclu.ctx ):
    nbytes = 0
    print "prepareBuffers FE.shape", FE.shape
    mf       = cl.mem_flags
    cl_ImgIn = cl.image_from_array(ctx,FE,num_channels=4,mode='r');  nbytes+=FE.nbytes        # TODO make this re-uploadable
    bsz=np.dtype(np.float32).itemsize * 4 * relax_dim[0] * relax_dim[1]
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY , bsz                );   nbytes+=bsz              # float4
    cl_FEout = cl.Buffer(ctx, mf.WRITE_ONLY, bsz * relax_dim[2] );   nbytes+=bsz*relax_dim[2] # float4
    print "FFout.nbytes : ", bsz * relax_dim[2]
    print "prepareBuffers.nbytes: ", nbytes
    kargs = (cl_ImgIn, cl_poss, cl_FEout )
    return kargs

def releaseArgs( kargs ):
    kargs[0].release() # cl_ImgIn
    kargs[1].release() # cl_poss
    kargs[2].release() # cl_FEout

def relax( kargs, relax_dim, invCell, poss=None, FEin=None, FEout=None, dTip=DEFAULT_dTip, stiffness=DEFAULT_stiffness, dpos0=DEFAULT_dpos0, relax_params=DEFAULT_relax_params, queue=oclu.queue):
    nz = np.int32( relax_dim[2] )
    kargs = kargs  + ( invCell[0],invCell[1],invCell[2], dTip, stiffness, dpos0, relax_params, nz )
    if FEout is None:
        FEout = np.zeros( relax_dim+(4,), dtype=np.float32 )
        print "FEout.shape", FEout.shape, relax_dim
    if poss is not None:
        cl.enqueue_copy( queue, kargs[1], poss )
    if FEin is not None:
        region = FEin.shape[:3]; region = region[::-1]; print "region : ", region
        cl.enqueue_copy( queue, kargs[0], FEin, origin=(0,0,0), region=region )
    #print kargs
    cl_program.relaxStrokes( queue, ( int(relax_dim[0]*relax_dim[1]),), None, *kargs )
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

