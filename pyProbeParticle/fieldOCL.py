#!/usr/bin/python

import os
import pyopencl as cl
import numpy    as np 

import oclUtils as oclu

cl_program = None

def init():
    global cl_program
    cl_program = oclu.loadProgram(oclu.CL_PATH+"/FF.cl")

def initArgsCoulomb( atoms, poss, ctx=oclu.ctx ):
    nAtoms     = np.int32( len(atoms) ) 
    mf         = cl.mem_flags
    cl_FE      = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   )
    cl_atoms   = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms )
    cl_poss    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  )
    kargs = ( nAtoms, cl_atoms, cl_poss, cl_FE )
    return kargs 
    	
def runCoulomb( kargs, nDim, local_size=(16,), ctx=oclu.ctx, queue=oclu.queue  ):
    global_size = (nDim[0]*nDim[1]*nDim[2],)
    FE          = np.zeros( nDim+(4,) , dtype=np.float32 )
    cl_program.evalLJC ( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy    ( queue, FE, kargs[3] );
    queue.finish()
    return FE

def initArgsLJC(atoms,cLJs, poss, ctx=oclu.ctx, queue=oclu.queue ):
    nAtoms   = np.int32( len(atoms) ) 
    mf       = cl.mem_flags
    cl_atoms = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms )
    cl_cLJs  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=cLJs  )
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ) # float4
    cl_FE    = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes*2 ) # float8
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    return kargs

def initArgsMorse(atoms,REAs, poss, ctx=oclu.ctx, queue=oclu.queue ):
    nAtoms   = np.int32( len(atoms) ) 
    print " initArgsMorse ", nAtoms
    mf       = cl.mem_flags
    cl_atoms = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms )
    cl_REAs  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=REAs  )
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ) # float4
    cl_FE    = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   ) # float4
    kargs = ( nAtoms, cl_atoms, cl_REAs, cl_poss, cl_FE )
    return kargs

def updateArgsLJC( kargs_old, atoms=None, cLJs=None, poss=None, ctx=oclu.ctx, queue=oclu.queue ):
    mf       = cl.mem_flags
    if atoms is not None:
        nAtoms   = np.int32( len(atoms) )
        if (kargs_old[0] != nAtoms):
            print " NOT IMPLEMENTED :  kargs_old[0] != nAtoms"; exit()
        else:
            cl_atoms=kargs_old[1]
            cl.enqueue_copy( queue, cl_atoms, atoms )
    else:
        cl_atoms=kargs_old[1]
    if cLJs is not None:
        print " NOT IMPLEMENTED : new cLJs"; exit()
    else:
        cl_cLJs=kargs_old[2]
    if poss is not None:
        print " NOT IMPLEMENTED : new poss"; exit()
    else:
        cl_poss=kargs_old[3]
    cl_FE=kargs_old[4]
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    return kargs

def updateArgsMorse( kargs_old=None, atoms=None, REAs=None, poss=None, ctx=oclu.ctx, queue=oclu.queue ):
    mf       = cl.mem_flags

    if kargs_old is None:
        return initArgsMorse( atoms, REAs, poss, ctx=ctx, queue=queue )
    else:
        if atoms is not None:
            nAtoms   = np.int32( len(atoms) )
            if (kargs_old[0] != nAtoms):
                print " NOT IMPLEMENTED :  kargs_old[0] != nAtoms"; exit()
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
            #print " NOT IMPLEMENTED : new poss"; exit()
        else:
            cl_poss=kargs_old[3]

        cl_FE=kargs_old[4]
        kargs = ( nAtoms, cl_atoms, cl_cREAs, cl_poss, cl_FE )
        return kargs

def runLJC( kargs, nDim, local_size=(32,), queue=oclu.queue ):
    global_size = (nDim[0]*nDim[1]*nDim[2],)
    FE          = np.zeros( nDim+(8,) , dtype=np.float32 ) # float8
    cl_program.evalLJC( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy( queue, FE, kargs[4] )
    queue.finish()
    return FE

def makeDivisibleUp( num, divisor ):
    rest = num % divisor;
    if rest > 0: num += (divisor-rest)
    return num

def runMorse( kargs, nDim, local_size=(32,), queue=oclu.queue ):
#def runMorse( kargs, nDim, local_size=(1,), queue=oclu.queue ):
#def runMorse( kargs, nDim, local_size=None, queue=oclu.queue ):
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,)
    FE          = np.zeros( nDim+(4,) , dtype=np.float32 ) # float4
    cl_program.evalMorse( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy( queue, FE, kargs[4] )
    queue.finish()
    return FE

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
    atoms_      = np.zeros( (len(qs),4), dtype=np.float32)
    atoms_[:,:3] = xyzs[:,:]
    atoms_[:, 3] = qs[:]      
    return atoms_

#def REA2float4(REAs):
#    clREAs      = np.zeros( (len(qs),4), dtype=np.float32)
#    clREAs[:,:3] = REAs[:,:]
#    return clREAs

def CLJ2float2(C6s,C12s):
    cLJs      = np.zeros( (len(C6s),2), dtype=np.float32)
    cLJs[:,0] = C6s
    cLJs[:,1] = C12s      
    return cLJs
	
	
	

