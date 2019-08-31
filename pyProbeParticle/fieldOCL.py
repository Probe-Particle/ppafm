#!/usr/bin/python

import os
import pyopencl as cl
import numpy    as np 

import oclUtils as oclu

cl_program = None

def init():
    global cl_program
    cl_program = oclu.loadProgram(oclu.CL_PATH+"/FF.cl")

# TODO: this is clearly candidate form Object Oriented desing
#
#  Class FFCalculator:
#       init()
#       update()
#       run()

verbose = 0

# ========= init Args 

def initArgsCoulomb( atoms, poss, ctx=oclu.ctx ):
    nbytes     =  0;
    nAtoms     = np.int32( len(atoms) ) 
    mf         = cl.mem_flags
    cl_atoms   = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
    cl_poss    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes
    cl_FE      = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   ); nbytes+=poss.nbytes
    kargs = ( nAtoms, cl_atoms, cl_poss, cl_FE )
    if(verbose>0):print "initArgsCoulomb.nbytes ", nbytes
    return kargs 

def initArgsLJC( atoms, cLJs, poss, ctx=oclu.ctx, queue=oclu.queue ):
    nbytes     =  0;
    nAtoms   = np.int32( len(atoms) ) 
    #print " initArgsLJC ", nAtoms
    mf       = cl.mem_flags
    cl_atoms = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
    cl_cLJs  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=cLJs  ); nbytes+=cLJs.nbytes
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes   # float4
    cl_FE    = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes*2 ); nbytes+=poss.nbytes*2 # float8
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    if(verbose>0):print "initArgsLJC.nbytes ", nbytes
    return kargs

def initArgsLJ(atoms,cLJs, poss, ctx=oclu.ctx, queue=oclu.queue ):
    nbytes     =  0;
    nAtoms   = np.int32( len(atoms) ) 
    #print "initArgsLJ ", nAtoms
    mf       = cl.mem_flags
    cl_atoms = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
    cl_cLJs  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=cLJs  ); nbytes+=cLJs.nbytes
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes   # float4
    cl_FE    = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   ); nbytes+=poss.nbytes   # float4
    kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
    if(verbose>0):print "initArgsLJ.nbytes ", nbytes
    return kargs

def initArgsMorse(atoms,REAs, poss, ctx=oclu.ctx, queue=oclu.queue ):
    nbytes     =  0;
    nAtoms   = np.int32( len(atoms) ) 
    #print "initArgsMorse ", nAtoms
    mf       = cl.mem_flags
    cl_atoms = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
    cl_REAs  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=REAs  ); nbytes+=REAs.nbytes
    cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes # float4
    cl_FE    = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes   ); nbytes+=poss.nbytes # float4
    kargs = ( nAtoms, cl_atoms, cl_REAs, cl_poss, cl_FE )
    if(verbose>0):print "initArgsMorse.nbytes ", nbytes
    return kargs

def releaseArgs( kargs ):
    for karg in kargs[1:]:
        karg.release()
    #kargs[1].release() # cl_atoms
    #kargs[2].release() # cl_cLJs
    #kargs[3].release() # cl_poss
    #kargs[4].release() # cl_FE

# ========= Update Args 

def updateArgsLJC( kargs_old, atoms=None, cLJs=None, poss=None, ctx=oclu.ctx, queue=oclu.queue ):
    mf       = cl.mem_flags
    if kargs_old is None:
        
        return initArgsLJC( atoms, cLJs, poss, ctx=ctx, queue=queue )
    else:
        if atoms is not None:
            nAtoms   = np.int32( len(atoms) )
            if (kargs_old[0] != nAtoms):
                if(verbose>0): print " kargs_old[0] != nAtoms; TRY only"#; exit()
                return initArgsLJC( atoms, cLJs, poss, ctx=ctx, queue=queue )
                #print " NOT IMPLEMENTED :  kargs_old[0] != nAtoms"; exit()
            else:
                cl_atoms=kargs_old[1]
                cl.enqueue_copy( queue, cl_atoms, atoms )
        else:
            cl_atoms=kargs_old[1]
        if cLJs is not None:
            cl_cLJs=kargs_old[2]
            cl.enqueue_copy( queue, cl_cLJs, cLJs )
            #print " NOT IMPLEMENTED : new cLJs"; exit()
        else:
            cl_cLJs=kargs_old[2]
        if poss is not None:
            cl_poss=kargs_old[3]
            cl.enqueue_copy( queue, cl_poss, poss )
            #print " NOT IMPLEMENTED : new poss"; exit()

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
                if(verbose>0): print " kargs_old[0] != nAtoms; TRY only"#; exit()
                return initArgsMorse( atoms, REAs, poss, ctx=ctx, queue=queue )
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

def updateArgsLJ( kargs_old, atoms=None, cLJs=None, poss=None, ctx=oclu.ctx, queue=oclu.queue ):
    mf       = cl.mem_flags
    if kargs_old is None:
        return initArgsLJ( atoms, cLJs, poss, ctx=ctx, queue=queue )
    else:
        if atoms is not None:
            nAtoms   = np.int32( len(atoms) )
            if (kargs_old[0] != nAtoms):
                if(verbose>0): print " kargs_old[0] != nAtoms; TRY only"#; exit()
                return initArgsLJ( atoms, cLJs, poss, ctx=ctx, queue=queue )
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

def runCoulomb( kargs, nDim, local_size=(32,), ctx=oclu.ctx, queue=oclu.queue  ):
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,) 
    FE          = np.zeros( nDim+(4,) , dtype=np.float32 )
    cl_program.evalLJC ( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy    ( queue, FE, kargs[3] );
    queue.finish()
    return FE

def runLJC( kargs, nDim, local_size=(32,), queue=oclu.queue ):
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,) # TODO make sure divisible by local_size
    #print "global_size:", global_size
    FE = np.zeros( nDim+(8,), dtype=np.float32 ) # float8
    if(verbose>0): print "FE.shape ", FE.shape
    cl_program.evalLJC( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy( queue, FE, kargs[4] )
    queue.finish()
    return FE

def runLJ( kargs, nDim, local_size=(32,), queue=oclu.queue ):  # slowed down, because of problems with the field far away
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,)
    #print "global_size:", global_size
    FE          = np.zeros( nDim+(4,) , dtype=np.float32 ) # float4
    cl_program.evalLJ( queue, global_size, local_size, *(kargs))
    cl.enqueue_copy( queue, FE, kargs[4] )
    queue.finish()
    return FE

def runMorse( kargs, nDim, local_size=(32,), queue=oclu.queue ):
#def runMorse( kargs, nDim, local_size=(1,), queue=oclu.queue ):
#def runMorse( kargs, nDim, local_size=None, queue=oclu.queue ):
    ntot = nDim[0]*nDim[1]*nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
    global_size = (ntot,)
    #print "global_size:", global_size
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
        int(round(pixPerAngstrome * np.sqrt(np.dot(lvec[3],lvec[3])) ))
    ])
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

#def REA2float4(REAs):
#    clREAs      = np.zeros( (len(qs),4), dtype=np.float32)
#    clREAs[:,:3] = REAs[:,:]
#    return clREAs

def CLJ2float2(C6s,C12s):
    cLJs      = np.zeros( (len(C6s),2), dtype=np.float32)
    cLJs[:,0] = C6s
    cLJs[:,1] = C12s
    return cLJs

# ========= classes

class ForceField_LJC:
    '''
        to evaluate ForceField on GPU
    '''
    #verbose=0  # this is global for now

    def __init__( self ):
        self.ctx   = oclu.ctx; 
        self.queue = oclu.queue

    def prepareBuffers(self, atoms, cLJs, poss=None, bDirect=False, nz=20 ):
        '''
        allocate all necessary buffers in GPU memory
        '''
        nbytes   =  0;
        self.nAtoms   = np.int32( len(atoms) ) 
        mf       = cl.mem_flags
        self.cl_atoms = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=atoms ); nbytes+=atoms.nbytes
        self.cl_cLJs  = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=cLJs  ); nbytes+=cLJs.nbytes
        if poss is not None:
            self.nDim = poss.shape
            self.cl_poss  = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes   # float4
        if(not bDirect):
            self.cl_FE    = cl.Buffer(self.ctx, mf.WRITE_ONLY                   , poss.nbytes*2  ); nbytes+=poss.nbytes*2 # float8
        #else:
        #    self.cl_FE    = cl.Buffer(self.ctx, mf.WRITE_ONLY                   , poss.nbytes*nz ); nbytes+=poss.nbytes*nz # float8
        if(verbose>0): print "initArgsLJC.nbytes ", nbytes

    def updateBuffers(self, atoms=None, cLJs=None, poss=None ):
        '''
        update content of all buffers
        '''
        oclu.updateBuffer(atoms, self.cl_atoms )
        oclu.updateBuffer(cLJs,  self.cl_cLJs  )
        oclu.updateBuffer(poss,  self.cl_poss  )

    def releaseBuffers(self):
        '''
        release all buffers
        '''
        self.cl_atoms.release()
        self.cl_cLJs.release()
        self.cl_poss.release()
        self.cl_FE.release()

    def run(self, FE=None, local_size=(32,) ):
        '''
        generate force-field
        '''
        if FE is None:
            FE = np.zeros( self.nDim[:3]+(8,), dtype=np.float32 )
            if(verbose>0): print "FE.shape", FE.shape, self.nDim
        ntot = self.nDim[0]*self.nDim[1]*self.nDim[2]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
        #print "self.nAtoms ", self.nAtoms
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_poss,
            self.cl_FE,
        )
        cl_program.evalLJC( self.queue, global_size, local_size, *(kargs) )
        cl.enqueue_copy( self.queue, FE, kargs[4] )
        self.queue.finish()
        return FE

    def runRelaxStrokesDirect(self, Q, cl_FE, FE=None, local_size=(32,), nz=10 ):
        '''
        generate force-field
        '''
        if FE is None:
            #FE    = np.zeros( self.nDim[:3]+(4,), dtype=np.float32 )
            FE     = np.empty( self.scan_dim+(4,), dtype=np.float32 )
            if(verbose>0): print "FE.shape", FE.shape, self.nDim
        ntot = int( self.scan_dim[0]*self.scan_dim[1] ) 
        ntot=makeDivisibleUp(ntot,local_size[0])
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
        #print "self.nAtoms ", self.nAtoms

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

    def makeFF(self, xyzs, qs, cLJs, bonds2atoms=None, poss=None, nDim=None, lvec=None, pixPerAngstrome=10 ):
        '''
        generate force-field from given posions(xyzs), chagres(qs), Leanrd-Jones parameters (cLJs) etc.
        '''
        if poss is None:
            if nDim is None:
                nDim = genFFSampling( lvec, pixPerAngstrome=pixPerAngstrome )
            poss  = getposs( lvec, nDim )
        #xyzs,Zs,enames,qs = basUtils.loadAtomsLines( atom_lines )
        #natoms0 = len(Zs)
        #if( npbc is not None ):
        #    Zs, xyzs, qs = PPU.PBCAtoms3D( Zs, xyzs, qs, lvec[1:], npbc=npbc )

        atoms = xyzq2float4(xyzs,qs);      self.atoms = atoms

        cLJs  = cLJs.astype(np.float32);   
        self.prepareBuffers(atoms, cLJs, poss )
        FF = self.run()
        self.releaseBuffers()
        return FF, atoms


class AtomProcjetion:
    '''
        to generate reference output maps ( Ys )  in generator for Neural Network training
    '''
    Rpp     =  2.0   #  probe-particle radius
    zmin    = -3.0   #  minim position of pixels sampled in SphereMaps
    dzmax   =  2.0   #  maximum distance of atoms from sampling screen for Atomic Disk maps ( similar )
    #beta    = -1.0  #  decay of exponetial radial function (used e.g. for Bonds)

    Rmax       =  10.0  #  Radial function of bonds&atoms potential  ; used in Bonds
    drStep     =   0.1  #  step dx (dr) for sampling of radial function; used in Bonds 
    elipticity =  0.5;  #  ration between major and minor semiaxi;   used in Bonds 

    # occlusion
    zmargin =  0.2   #  zmargin 
    tgMax   =  0.5   #  tangens of angle limiting occlusion for SphereCaps
    tgWidth =  0.1   #  tangens of angle for limiting rendered area for SphereCaps

    def __init__( self ):
        self.ctx   = oclu.ctx; 
        self.queue = oclu.queue

    def makeCoefsZR(self, Zs, ELEMENTS ):
        '''
        make atomic coeficients used e.g. in MultiMap
        '''
        na = len(Zs)
        coefs = np.zeros( (na,4), dtype=np.float32 )
        if(verbose>0): print "Zs", Zs
        for i,ie in enumerate(Zs):
            coefs[i,0] = 1.0
            coefs[i,1] = ie
            coefs[i,2] = ELEMENTS[ie-1][6]
            coefs[i,3] = ELEMENTS[ie-1][7]
            #coefs[i,3] = ie
        if(verbose>0): print "coefs[:,2]", coefs[:,2]
        return coefs

    def prepareBuffers(self, atoms, prj_dim, coefs=None, bonds2atoms=None, Rfunc=None ):
        '''
        allocate GPU buffers
        '''
        if(verbose>0): print "AtomProcjetion.prepareBuffers prj_dim", prj_dim
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
        #self.cl_poss  = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ); nbytes+=poss.nbytes   # float4
        #self.cl_Eout  = cl.Buffer(self.ctx, mf.WRITE_ONLY                   , poss.nbytes/4 ); nbytes+=poss.nbytes/4 # float

        npostot = prj_dim[0] * prj_dim[1]
        
        bsz=np.dtype(np.float32).itemsize * npostot
        #if(verbose>0): print prj_dim, npostot, " nbytes : = ", bsz*4
        self.cl_poss  = cl.Buffer(self.ctx, mf.READ_ONLY , bsz*4           );   nbytes+=bsz*4  # float4
        self.cl_Eout  = cl.Buffer(self.ctx, mf.WRITE_ONLY, bsz*prj_dim[2]  );   nbytes+=bsz    # float

        self.cl_itypes  = cl.Buffer(self.ctx, mf.READ_ONLY, 200*np.dtype(np.int32).itemsize );   nbytes+=bsz    # float
        #self.cl_MultiMap  = cl.Buffer(self.ctx, mf.WRITE_ONLY, bsz*8     );   nbytes+=bsz    # float

        #kargs = ( nAtoms, cl_atoms, cl_cLJs, cl_poss, cl_FE )
        if(verbose>0): print "AtomProcjetion.prepareBuffers.nbytes ", nbytes
        #return kargs

    def updateBuffers(self, atoms=None, coefs=None, poss=None ):
        '''
        upload data to GPU
        '''
        #print "updateBuffers poss.shape: ", poss.shape
        oclu.updateBuffer(atoms, self.cl_atoms )
        oclu.updateBuffer(coefs, self.cl_coefs  )
        oclu.updateBuffer(poss,  self.cl_poss  )

    def setAtomTypes(self, types, sel=[1,6,8]):
        '''
        setup selection of atomic types for SpheresType kernel and upload them to GPU
        '''
        #print types
        self.nTypes   = np.int32( len(sel) )
        dct = { typ:i for i,typ in enumerate(sel) }
        itypes = np.ones( 200, dtype=np.int32); itypes[:]*=-1 
        #print dct
        #print dct
        #itypes = [ dct[typ] for i,typ in enumerate(types) if typ in dct ]
        for i,typ in enumerate(types):
            if typ in dct:
                itypes[i] = dct[typ]
        #itypes = np.array( itypes, dtype=np.int32)
        #for ii,i in enumerate(types):
        #    itypes[i] = ii
        #print itypes
        cl.enqueue_copy( self.queue, self.cl_itypes, itypes )
        return itypes, dct

    def releaseBuffers(self):
        '''
        deallocated all GPU buffers
        '''
        self.cl_atoms.release()
        self.cl_coefs.release()
        self.cl_poss.release()
        self.cl_FE.release()

    def tryReleaseBuffers(self):
        '''
        deallocated all GPU buffers (those which exists)
        '''
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
            if(verbose>0): print "FE.shape", Eout.shape, self.nDim
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
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

    def run_evaldisks(self, poss=None, Eout=None, tipRot=None, local_size=(32,) ):
        '''
        kernel producing atomic disks with conical profile
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            if(verbose>0): print "FE.shape", Eout.shape, self.nDim
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32( self.dzmax   ),
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
            if(verbose>0): print "FE.shape", Eout.shape, self.nDim
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
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
            if(verbose>0): print "FE.shape", Eout.shape, self.nDim
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
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
            if(verbose>0): print "FE.shape", Eout.shape, self.nDim
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
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
            if(verbose>0): print "FE.shape", Eout.shape, self.nDim
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
        kargs = (  
            self.nAtoms,
            self.cl_atoms,
            #self.cl_coefs,
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
            #if(verbose>0): 
            #print "FE.shape", Eout.shape
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
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

    def run_evalSpheresType(self, poss=None, Eout=None, tipRot=None, bOccl=0,  local_size=(32,) ):
        '''
         kernel to produce multiple channels of vdW Sphere maps each coresponding to different atom type
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            #if(verbose>0): 
            #print "FE.shape", Eout.shape
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
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
            #if(verbose>0): 
            #print "FE.shape", Eout.shape
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
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
        #print "Eout: \n", Eout
        #print " run_evalBondEllipses DONE "
        return Eout

    def run_evalAtomRfunc(self, poss=None, Eout=None, tipRot=None, bOccl=0,  local_size=(32,) ):
        '''
         kernel to produce multiple channels of vdW Sphere maps each coresponding to different atom type
        '''
        if tipRot is not None:
            self.tipRot=tipRot
        if Eout is None:
            Eout = np.zeros( self.prj_dim, dtype=np.float32 )
            #if(verbose>0): 
            #print "FE.shape", Eout.shape
        if poss is not None:
            if(verbose>0): print "poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype
            oclu.updateBuffer(poss, self.cl_poss )
        ntot = self.prj_dim[0]*self.prj_dim[1]; ntot=makeDivisibleUp(ntot,local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,) # TODO make sure divisible by local_size
        #print "global_size:", global_size
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
        #print "Eout: \n", Eout
        #print " run_evalBondEllipses DONE "
        return Eout