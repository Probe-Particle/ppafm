#!/usr/bin/python

import sys
import os
import time
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
DEFAULT_relax_params = np.array( [ 0.01 , 0.9 , 0.01, 0.3 ], dtype=np.float32 );

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

# --- prepare cl program

def init( cl_context=oclu.ctx):
    global cl_program
    #cl_program  = cl.Program(cl_context, CL_CODE).build()
    #cl_program  = cl.Program(cl_context, CL_CODE).build()
    cl_program  = oclu.loadProgram(oclu.CL_PATH+"/relax.cl")

def getInvCell( lvec ):
    cell = lvec[1:4,0:3]
    invCell = np.linalg.inv(cell)
    print invCell
    invA = np.zeros( 4, dtype=np.float32); invA[0:3] = invCell[0]
    invB = np.zeros( 4, dtype=np.float32); invB[0:3] = invCell[1]
    invC = np.zeros( 4, dtype=np.float32); invC[0:3] = invCell[2]
    return (invA, invB, invC)

def preparePoss( relax_dim, z0, start=(0.0,0.0), end=(10.0,10.0) ):
    xs    = np.linspace(start[0],end[1],relax_dim[0])
    ys    = np.linspace(start[0],end[1],relax_dim[1])
    Xs,Ys = np.meshgrid(xs,ys)
    poss  = np.zeros(Xs.shape+(4,), dtype=np.float32)
    poss[:,:,0] = Xs
    poss[:,:,1] = Ys
    poss[:,:,2] = z0
    return poss

def prepareBuffers( FE, relax_dim, ctx=oclu.ctx ):
    mf       = cl.mem_flags
    cl_ImgIn = cl.image_from_array(ctx,FE,num_channels=4,mode='r')               # TODO make this re-uploadable
    #cl_poss = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss   ) # float4
    bsz=np.dtype(np.float32).itemsize * 4 * relax_dim[0] * relax_dim[1]
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY , bsz                ) # float4
    cl_FEout = cl.Buffer(ctx, mf.WRITE_ONLY, bsz * relax_dim[2] ) # float4
    print "FFout.nbytes : ", bsz * relax_dim[2]
    kargs = (cl_ImgIn, cl_poss, cl_FEout )
    return kargs

def relax( kargs, relax_dim, invCell, poss=None, FEout=None, dTip=DEFAULT_dTip, stiffness=DEFAULT_stiffness, dpos0=DEFAULT_dpos0, relax_params=DEFAULT_relax_params, queue=oclu.queue):
    nz = np.int32( relax_dim[2] )
    kargs = kargs  + ( invCell[0],invCell[1],invCell[2], dTip, stiffness, dpos0, relax_params, nz )
    t1 = time.clock() 
    if FEout is None:
        FEout = np.zeros( relax_dim+(4,), dtype=np.float32 )
        #print "FFout.nbytes : ", FEout.nbytes
    if poss is not None:
        cl.enqueue_copy( queue, kargs[1], poss )
    #print kargs
    cl_program.relaxStrokes( queue, (relax_dim[0]*relax_dim[1],), None, *kargs )
    cl.enqueue_copy( queue, FEout, kargs[2] )
    queue.finish()
    t2 = time.clock(); print "relaxStrokes time %f [s]" %(t2-t1) 
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
    #import matplotlib.pyplot as plt
    #plt.show()

