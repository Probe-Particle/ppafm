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
    print(invCell)
    invA = np.zeros( 4, dtype=np.float32); invA[0:3] = invCell[0]
    invB = np.zeros( 4, dtype=np.float32); invB[0:3] = invCell[1]
    invC = np.zeros( 4, dtype=np.float32); invC[0:3] = invCell[2]
    return FE, (invA, invB, invC)

# --- prepare cl program

THIS_FILE_PATH = os.path.dirname( os.path.realpath( __file__ ) )

CL_PATH  = os.path.normpath( THIS_FILE_PATH  + '/../../cl' ); print(CL_PATH) 
f        = open(CL_PATH+"/relax.cl", 'r')
CL_CODE  = "".join( f.readlines() )
plats    = cl.get_platforms()
ctx      = cl.Context(properties=[(cl.context_properties.PLATFORM, plats[0])], devices=None)  
queue    = cl.CommandQueue(ctx)
prg      = cl.Program(ctx, CL_CODE).build()
mf       = cl.mem_flags

#  --- prepare buffers

dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
stiffness    = np.array( [-0.03,-0.03, -0.03,-1.0 ], dtype=np.float32 );
dpos0        = np.array( [ 0.0 , 0.0 ,  4.0 , 4.0 ], dtype=np.float32 );
relax_params = np.array( [ 0.01 , 0.9 , 0.01, 0.3 ], dtype=np.float32 );

nz   = 60


'''
ts        = np.linspace(0.0,1.0,100)
poss      = np.zeros((len(ts),4), dtype=np.float32)
poss[:,0] =  0.0 + ts *  10.0
poss[:,1] =  0.0 + ts *  10.0
poss[:,2] = 10.0 + ts *   0.0
FEout     = np.zeros( (len(ts)*nz,4), dtype=np.float32)
'''

xs = np.linspace(0.0,10.0,100)
ys = np.linspace(0.0,10.0,100)
Xs,Ys = np.meshgrid(xs,ys)
poss = np.zeros(Xs.shape+(4,), dtype=np.float32)
poss[:,:,0] = Xs
poss[:,:,1] = Ys
poss[:,:,2] = 10.0
FEout       = np.zeros( Xs.shape+(nz,4), dtype=np.float32)

#print poss

FE, (invA, invB, invC) = loadFEcl( Q = -0.2 )

print("invA ", invA)
print("invB ", invB)
print("invC ", invC)

cl_ImgIn = cl.image_from_array(ctx,FE,num_channels=4,mode='r')
cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss   ) # float4
cl_FEout = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes*nz ) # float4

# --- execute
t1 = time.clock() 
#prg.getFEinPoints ( queue, (len(ts),), None, *(cl_ImgIn, cl_poss, cl_FEout, invA, invB, invC ) )
#prg.getFEinStrokes( queue, (len(ts),), None, *(cl_ImgIn, cl_poss, cl_FEout, invA, invB, invC, dTip, np.int32(nz) ) )
prg.relaxStrokes( queue, (Xs.size,), None, *(cl_ImgIn, cl_poss, cl_FEout, invA, invB, invC, dTip, stiffness, dpos0, relax_params, np.int32(nz) ) )

#prg.test_kernel( queue, (len(ts),), None, *(cl_poss, cl_vals) )
cl.enqueue_copy( queue, FEout, cl_FEout )
queue.finish()

t2 = time.clock(); print("relaxStrokes time %f [s]" %(t2-t1)) 

import matplotlib.pyplot as plt

#FEout[:,3].reshape((-1,nz))

'''
plt.figure(figsize=(10,10))
plt.imshow( FEout[:,:,0,2] )
'''

'''
vmin=-0.1
vmax=+0.1
plt.figure(figsize=(10,10))
plt.subplot(2,2,1); plt.imshow( FEout[:,0].reshape((-1,nz)), vmin=vmin, vmax=vmax ); plt.colorbar(); plt.title('Fx')
plt.subplot(2,2,2); plt.imshow( FEout[:,1].reshape((-1,nz)), vmin=vmin, vmax=vmax ); plt.colorbar(); plt.title('Fy')
plt.subplot(2,2,3); plt.imshow( FEout[:,2].reshape((-1,nz)), vmin=vmin, vmax=vmax ); plt.colorbar(); plt.title('Fz')
plt.subplot(2,2,4); plt.imshow( FEout[:,3].reshape((-1,nz)), vmin=vmin, vmax=vmax ); plt.colorbar(); plt.title('E')
'''

lvec_OUT = (
    [0.0,0.0,0.0],
    [10.0,0.0,0.0],
    [0.0,10.0,0.0],
    [0.0,0.0,6.0]
)

Ftmp=np.zeros((nz,)+Xs.shape);
Ftmp[:,:,:] = np.transpose( FEout[:,:,:,2], (2,0,1) ); GU.saveXSF( 'OutFz_cl.xsf',  Ftmp, lvec_OUT );


plt.show()



