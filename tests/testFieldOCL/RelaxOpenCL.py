#!/usr/bin/python

import sys
import os
import time
import pyopencl as cl
import numpy as np

sys.path.append("/home/prokop/git/ProbeParticleModel_OCL") 

from   pyProbeParticle import basUtils
from   pyProbeParticle import PPPlot 
import pyProbeParticle.GridUtils as GU
import pyProbeParticle.common as PPU
import pyProbeParticle.cpp_utils as cpp_utils

# ========== Globals

cl_program = None
cl_queue   = None 
cl_context = None

FE         = None
FEout      = None

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
    cell = lvec[1:4,0:3]
    invCell = np.linalg.inv(cell)
    print invCell
    invA = np.zeros( 4, dtype=np.float32); invA[0:3] = invCell[0]
    invB = np.zeros( 4, dtype=np.float32); invB[0:3] = invCell[1]
    invC = np.zeros( 4, dtype=np.float32); invC[0:3] = invCell[2]
    return FE, (invA, invB, invC)

# --- prepare cl program

def prepareProgram():
    global cl_program,cl_queue, cl_context
    THIS_FILE_PATH = os.path.dirname( os.path.realpath( __file__ ) )
    CL_PATH  = os.path.normpath( THIS_FILE_PATH  + '/../../cl' ); print CL_PATH 
    f        = open(CL_PATH+"/relax.cl", 'r')
    CL_CODE  = "".join( f.readlines() )
    plats    = cl.get_platforms()
    cl_context  = cl.Context(properties=[(cl.context_properties.PLATFORM, plats[0])], devices=None)  
    cl_queue    = cl.CommandQueue(cl_context)
    cl_program  = cl.Program(cl_context, CL_CODE).build()

def prepareBuffers():
    global FE,FEout
    #  --- prepare buffers

    dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
    stiffness    = np.array( [-0.03,-0.03, -0.03,-1.0 ], dtype=np.float32 );
    dpos0        = np.array( [ 0.0 , 0.0 ,  4.0 , 4.0 ], dtype=np.float32 );
    relax_params = np.array( [ 0.01 , 0.9 , 0.01, 0.3 ], dtype=np.float32 );

    nz   = 60

    xs    = np.linspace(0.0,10.0,100)
    ys    = np.linspace(0.0,10.0,100)
    Xs,Ys = np.meshgrid(xs,ys)
    poss  = np.zeros(Xs.shape+(4,), dtype=np.float32)
    poss[:,:,0] = Xs
    poss[:,:,1] = Ys
    poss[:,:,2] = 10.0
    FEout       = np.zeros( Xs.shape+(nz,4), dtype=np.float32)

    relaxShape = Xs.shape + (nz,)
    #print poss

    FE, (invA, invB, invC) = loadFEcl( Q = -0.2 )

    print "invA ", invA
    print "invB ", invB
    print "invC ", invC

    mf       = cl.mem_flags
    cl_ImgIn = cl.image_from_array(cl_context,FE,num_channels=4,mode='r')
    cl_poss  = cl.Buffer(cl_context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss   ) # float4
    cl_FEout = cl.Buffer(cl_context, mf.WRITE_ONLY                   , poss.nbytes*nz ) # float4
    
    kargs = (cl_ImgIn, cl_poss, cl_FEout, invA, invB, invC, dTip, stiffness, dpos0, relax_params, np.int32(nz) )
    return kargs, relaxShape

def relax( kargs, relaxShape ):
    t1 = time.clock() 
    #cl_program.getFEinPoints ( cl_queue, (len(ts),), None, *(cl_ImgIn, cl_poss, cl_FEout, invA, invB, invC ) )
    #cl_program.getFEinStrokes( cl_queue, (len(ts),), None, *(cl_ImgIn, cl_poss, cl_FEout, invA, invB, invC, dTip, np.int32(nz) ) )
    cl_program.relaxStrokes( cl_queue, (relaxShape[0]*relaxShape[1],), None, *kargs )
    #cl_program.test_kernel( cl_queue, (len(ts),), None, *(cl_poss, cl_vals) )
    #cl.enqueue_copy( cl_queue, FEout, cl_FEout )
    cl.enqueue_copy( cl_queue, FEout, kargs[2] )
    cl_queue.finish()
    t2 = time.clock(); print "relaxStrokes time %f [s]" %(t2-t1) 

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

