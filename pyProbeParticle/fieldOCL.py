#!/usr/bin/python

import os
import pyopencl as cl
import numpy    as np 

# initialize OpenCL

def initCl():
    PACKAGE_PATH = os.path.dirname( os.path.realpath( __file__ ) ); print PACKAGE_PATH
    #CL_PATH  = os.path.normpath( PACKAGE_PATH + '../../cl/' )
    CL_PATH  = os.path.normpath( PACKAGE_PATH + '/../cl' )
    #CL_PATH = PACKAGE_PATH+"/cl/"
    print CL_PATH
    plats   = cl.get_platforms()
    ctx     = cl.Context(properties=[(cl.context_properties.PLATFORM, plats[0])], devices=None)       
    queue   = cl.CommandQueue(ctx)
    f       = open(CL_PATH+"/PP.cl", 'r')
    fstr    = "".join(f.readlines())
    program = cl.Program(ctx, fstr).build()
    return ctx,queue,program

ctx,queue,program = initCl()

def initArgsCoulomb(atoms, poss ):
    nAtoms     = np.int32( len(atoms) ) 
    mf         = cl.mem_flags
    cl_FE      = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   )
    cl_atoms   = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms )
    cl_poss    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  )
    kargs = ( nAtoms, cl_atoms, cl_poss, cl_FE )
    return kargs 
    	
def runCoulomb( kargs, nDim, local_size=(16,) ):
    print "run opencl kernel ..."
    global_size = (nDim[0]*nDim[1]*nDim[2],)
    #global_size = (1,);  local_size=(1,)
    FE          = np.zeros( nDim+(4,) , dtype=np.float32 )
    print "FE.shape",      FE.shape
    print "global_size: ", global_size
    print "local_size:  ", local_size
    #print kargs
    #program.evalCoulomb( queue, global_size, local_size, *(kargs))
    program.evalLJC( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy    ( queue, FE, kargs[3] );
    queue.finish()
    print "... opencl kernel DONE"
    #print FE[:,60,60,:]; print "================"
    #print FE[100,:,60,:]; print "================"
    #print FE[100,60,:,:]; print "================"
    return FE

def initArgsLJC(atoms,cLJs, poss ):
    nAtoms     = np.int32( len(atoms) ) 
    mf         = cl.mem_flags
    cl_atoms   = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms )
    cl_cLJs    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=cLJs  )
    cl_poss    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ) # float4
    cl_FE     = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes*2 ) # float8
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    return kargs 

def runLJC( kargs, nDim, local_size=(16,) ):
    print "run opencl kernel ..."
    global_size = (nDim[0]*nDim[1]*nDim[2],)
    #global_size = (1,);  local_size=(1,)
    FE          = np.zeros( nDim+(8,) , dtype=np.float32 ) # float8
    print "FE.shape",      FE.shape
    print "global_size: ", global_size
    print "local_size:  ", local_size
    print kargs
    program.evalLJC( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy( queue, FE, kargs[4] )
    queue.finish()
    #print FE[:,60,60,:]; print "================"
    #print FE[100,:,60,:]; print "================"
    #print FE[100,60,:,:]; print "================"
    print "... opencl kernel DONE"
    return FE

def getPos(lvec, nDim=None, step=(0.1,0.1,0.1) ):
    if nDim is None:
        nDim = (    int(np.linalg.norm(lvec[3,:])/step[2]),
                    int(np.linalg.norm(lvec[2,:])/step[1]),
                    int(np.linalg.norm(lvec[1,:])/step[0]))
    dCell = np.array( ( lvec[1,:]/nDim[2], lvec[2,:]/nDim[1], lvec[3,:]/nDim[0] ) ) 
    ABC   = np.mgrid[0:nDim[0],0:nDim[1],0:nDim[2]]
    print "nDim",nDim
    print "ABC[0].shape ", ABC[0].shape
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
    
def atoms2float4(atoms):
    atoms_   = np.zeros( (len(atoms[0]),4), dtype=np.float32)
    atoms_[:,0] = np.array( atoms[1] )
    atoms_[:,1] = np.array( atoms[2] )
    atoms_[:,2] = np.array( atoms[3] )
    atoms_[:,3] = np.array( atoms[4] )
    return atoms_
    
def xyzq2float4(xyzs,qs):
    atoms_      = np.zeros( (len(qs),4), dtype=np.float32)
    atoms_[:,:3] = xyzs[:,:]
    atoms_[:, 3] = qs[:]      
    return atoms_

def CLJ2float2(C6s,C12s):
    cLJs      = np.zeros( (len(C6s),2), dtype=np.float32)
    cLJs[:,0] = C6s
    cLJs[:,1] = C12s      
    return cLJs
	
	
	

