
import os
import sys
import __main__ as main
import numpy as np
#import GridUtils as GU
#sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")

import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT

bDebug = False

# ===============================================================================================================
#      General utility functions
# ===============================================================================================================

def makeTransformMat( ns, lvec, angle=0.0, rot=None ):
    nx,ny,nz=ns
    #nz,ny,nx=ns
    if rot is None:
        rot = GU.rot3DFormAngle(angle)
    lvec = lvec  + 0.  # copy
    lvec[0,:]*=1./nx
    lvec[1,:]*=1./ny
    lvec[2,:]*=1./nz
    mat = np.dot( lvec, rot )
    if bDebug:
        print("mat ", mat)
    return mat

def shiftHalfAxis( X, d, n, ax=1 ):
    shift = n//2;
    if( n%2 != 0 ):  shift += 1
    X -= shift
    return np.roll( X, shift, axis=ax), shift

def normalizeGridWf( F ):
    q = (F**2).sum()
    return F/np.sqrt(q)

# ===============================================================================================================
#      Functions to construct Tip(Cavity) Field on grid from analytic functions (spherical harmonics, multipoles)
# ===============================================================================================================

def getMGrid2D(nDim, dd):
    'returns coordinate arrays X, Y, Z'
    (dx, dy) = dd
    XY = np.mgrid[0:nDim[0],0:nDim[1]].astype(float)
    yshift = nDim[1]//2;  yshift_ = yshift;
    xshift = nDim[0]//2;  xshift_ = xshift;
    if( nDim[1]%2 != 0 ):  yshift_ += 1.0
    if( nDim[0]%2 != 0 ):  xshift_ += 1.0
    X = XY[0] - xshift_;
    Y = XY[1] - yshift_;
    Y = dy * np.roll( Y, yshift, axis=1)
    X = dx * np.roll( X, xshift, axis=0)
    #print X[:,0]
    return X, Y, (xshift, yshift)

def getMGrid3D( nDim, dd ):
    'returns coordinate arrays X, Y, Z'
    (dx,dy,dz) = dd
    (nx,ny,nz) = nDim[:3]
    print (dd)
    #print (nDim)
    XYZ = np.mgrid[0:nx,0:ny,0:nz].astype(float)
    X,xshift = shiftHalfAxis( XYZ[0], dx, nx, ax=0 )
    Y,yshift = shiftHalfAxis( XYZ[1], dy, ny, ax=1 )
    #Z,zshift = shiftHalfAxis( XYZ[2], dz, nz, ax=2 )
    Z = XYZ[2]
    return X*dx, Y*dy, Z*dz, (xshift, yshift, 0)

def makeTipField2D( sh, dd, z=10.0, sigma=1.0, multipole_dict={'s':1.0} ):
    Vtip = np.zeros( sh[:2] )
    #X,Y,shifts  = getMGrid2D( sh, dd )
    Y,X,shifts  = getMGrid2D( sh, dd )
    #X *= dd[0]; Y *= dd[1];  # this is already done in getMGrid
    #print "Z = ", z
    #print "(xmax,ymax) = ", X[-1,-1],Y[-1,-1],  X[0,0],Y[0,0]
    radial = 1/np.sqrt( X**2 + Y**2  + z**2 + sigma**2  ) 
    #print "radial ", radial[:,radial.shape[1]/2]
    if multipole_dict is not None:    # multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        Vtip = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.items():
            Vtip += radial * coef * fFFT.getSphericalHarmonic( X, Y, z, kind=kind, tilt=0 )
    else:
        Vtip = radial
    #Vtip = X
    #Vtip = radial
    #print "Vtip ", Vtip[:,Vtip.shape[1]/2]
    return Vtip, shifts

def makeTipField3D( sh, dd, z0=10.0, sigma=1.0, multipole_dict={'s':1.0} ):
    Vtip = np.zeros( sh )
    X,Y,Z,shifts  = getMGrid3D( sh, dd )
    Z += z0
    #X *= dd[0]; Y *= dd[1];  # this is already done in getMGrid
    #print "Z = ", z
    #print "(xmax,ymax) = ", X[-1,-1],Y[-1,-1],  X[0,0],Y[0,0]
    radial = 1/np.sqrt( X**2 + Y**2  + Z**2 + sigma**2 ) 
    #print "radial ", radial[:,radial.shape[1]/2]
    if multipole_dict is not None:    # multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        Vtip = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.items():
            Vtip += radial * coef * fFFT.getSphericalHarmonic( X, Y, Z, kind=kind, tilt=0 )
    else:
        Vtip = radial
    #print "Vtip.shape     ", Vtip.shape
    Vtip = Vtip.transpose((2,1,0)).copy()
    #print " -> Vtip.shape ", Vtip.shape
    #print "shifts ", shifts
    return Vtip, shifts

def convFFT(F1,F2):
    return np.fft.ifftn( np.fft.fftn(F1) * np.fft.fftn(F2) )

# ==========================================================================
#      Functions to project trasition densities on a common grid (Canvas)
# ==========================================================================

'''
# ---- DEPRECATED
def photonMap2D( rhoTrans, tipDict, lvec, z=10.0, sigma=1.0, multipole_dict={'s':1.0}):
    sh = rhoTrans.shape
    #print "shape: ", sh
    #rho  = np.zeros( (sh[:2]) )
    rho  = np.sum(  rhoTrans, axis=2) 
    #Vtip = np.zeros( (sh[:2]) )
    #print lvec
    dx   = lvec[3][2]/sh[0]; #print lvec[3][2],sh[0]
    dy   = lvec[2][1]/sh[1]; #print lvec[2][1],sh[1]
    dd = (dx,dy)
    #print " (dx,dy) ", dd
    Vtip, shifts = makeTipField2D( sh[:2], dd, z=z, sigma=sigma, multipole_dict=multipole_dict )
    renorm = 1./( (Vtip**2).sum() * (rho**2).sum() )
    phmap  = convFFT(Vtip,rho).real * renorm
    #print "rho  ", rho.shape  
    #print "Vtip ", Vtip.shape  
    Vtip = np.roll( Vtip, -shifts[1], axis=1)
    Vtip = np.roll( Vtip, -shifts[0], axis=0)
    return phmap, Vtip, rho , dd
'''

def evalGridStep2D( sh, lvec ):
    #print( "lvec", lvec )
    return (
        lvec[3][2]/sh[0],
        lvec[2][1]/sh[1]
    )

def photonMap2D_stamp( rhos, lvecs, z=10.0, sigma=1.0, multipole_dict={'s':1.0}, rots=[0.0], poss=[ [0.0,0.0] ], coefs=[ [1.0,0.0] ], ncanv=(300,300),byCenter=False ):
    
    dd_canv = evalGridStep2D( rhos[0].shape, lvecs[0] )
    dtype=np.complex128
    if isinstance(coefs[0], float): dtype=np.float64    
    canvas = np.zeros  ( ncanv, dtype=dtype ) 
    
    for i in range(len(poss)):
        
        coef = coefs[i]
        rho = np.sum    (  rhos[i], axis=2  )
        rho = rho.astype( dtype                 ) 
        ddi = evalGridStep2D( rhos[i].shape, lvecs[i] )
        #pos = [ poss[i][0]+ncanv[0]*0.5*dd_canv[0],   poss[i][1]+ncanv[1]*0.5*dd_canv[1]  ]
        pos  = poss[i][:2]    ; print( "pos ", pos )

        pos   = np.array(pos)/np.array(dd_canv)
        dd_fac=( ddi[0]/dd_canv[0], ddi[1]/dd_canv[1] )

        # ToDo : problem if the two grids does not have the same samplig
        if isinstance(coef, float):
            #print("GU.stampToGrid2D()") 
            GU.stampToGrid2D( canvas, rho, pos, rots[i], dd=dd_fac, coef=coef, byCenter=byCenter )
        else:
            #print("GU.stampToGrid2D_complex()")
            coef = complex( coef[0], coef[1] )
            GU.stampToGrid2D_complex( canvas, rho, pos, rots[i], dd=dd_fac, coef=coef, byCenter=byCenter)

    #canvas = np.zeros((300,300))

    Vtip, shifts = makeTipField2D( ncanv[:2], dd_canv, z=z, sigma=sigma, multipole_dict=multipole_dict )
    #renorm = 1./( (Vtip**2).sum() * (rho.real**2+rho.imag**2).sum() )
    renorm = 1./( (Vtip**2).sum() )
    phmap  = convFFT(Vtip,canvas) * renorm
    #print "rho  ", rho.shape  
    #print "Vtip ", Vtip.shape  
    Vtip = np.roll( Vtip, (-shifts[1]), axis=1)
    Vtip = np.roll( Vtip, (-shifts[0]), axis=0)
    return phmap, Vtip, canvas, dd_canv

def photonMap2D_stamp_old( rhoTrans, lvec, z=10.0, sigma=1.0, multipole_dict={'s':1.0}, rots=[0.0], poss=[ [0.0,0.0] ], coefs=[ [1.0,0.0] ], ncanv=(300,300),byCenter=False ):
    sh = rhoTrans.shape
    #print "shape: ", sh
    #print lvec
    dx   = lvec[3][2]/sh[0]; #print lvec[3][2],sh[0]
    dy   = lvec[2][1]/sh[1]; #print lvec[2][1],sh[1]
    dd = (dx,dy)
    #print " (dx,dy) ", dd

    dtype=np.complex128
    if isinstance(coefs[0], float): dtype=np.float64    
    #canvas = np.zeros( ncanv[::-1], dtype=dtype )
    rho    = np.sum  (  rhoTrans, axis=2 )
    rho    = rho.astype( dtype ) 
    canvas = np.zeros( ncanv, dtype=dtype ) 
    
    #rho    = np.ascontiguousarray( rho,    dtype=np.complex128 )
    #canvas = np.ascontiguousarray( canvas, dtype=np.complex128 )

    for i in range(len(poss)):
        pos = [ poss[i][0]+ncanv[0]*0.5*dx,   poss[i][1]+ncanv[1]*0.5*dy  ]
        coef = coefs[i]
        if isinstance(coef, float):
            #print("GU.stampToGrid2D()") 
            GU.stampToGrid2D( canvas, rho, pos, rots[i], dd=dd, coef=coef, byCenter=byCenter )
        else:
            #print("GU.stampToGrid2D_complex()")
            coef = complex( coef[0], coef[1] )
            GU.stampToGrid2D_complex( canvas, rho, pos, rots[i], dd=dd, coef=coef, byCenter=byCenter)

    #canvas = np.zeros((300,300))

    Vtip, shifts = makeTipField2D( ncanv[:2], dd, z=z, sigma=sigma, multipole_dict=multipole_dict )
    #renorm = 1./( (Vtip**2).sum() * (rho.real**2+rho.imag**2).sum() )
    renorm = 1./( (Vtip**2).sum() )
    phmap  = convFFT(Vtip,canvas) * renorm
    #print "rho  ", rho.shape  
    #print "Vtip ", Vtip.shape  
    Vtip = np.roll( Vtip, (-shifts[1]), axis=1)
    Vtip = np.roll( Vtip, (-shifts[0]), axis=0)
    return phmap, Vtip, canvas, dd


def photonMap3D_stamp( rhoTrans, lvec, z=10.0, sigma=1.0, multipole_dict={'s':1.0}, rots=[0.0], poss=[ [0.0,0.0] ], coefs=[ [1.0,0.0] ], ncanv=(300,300), byCenter=False ):
    sh  = rhoTrans.shape
    if len(ncanv)<3:
        ncanv = ( (ncanv[0],ncanv[1],sh[2]) )
    dtype=np.complex128
    if isinstance(coefs[0], float): dtype=np.float64    
    canvas = np.zeros( ncanv[::-1],        dtype=dtype )
    rho    = rhoTrans.transpose((2,1,0)).astype( dtype ).copy()
    #print "shape: stamp ", rho.shape, " canvas ", canvas.shape
    #print lvec
    dx   = lvec[3][2]/sh[0]; #print lvec[3][2],sh[0]
    dy   = lvec[2][1]/sh[1]; #print lvec[2][1],sh[1]
    dz   = lvec[1][0]/sh[2]; #print lvec[1][0],sh[2]
    dd = (dx,dy,dz);    #print " dd ", dd
 
    for i in range(len(poss)):
        pos_ = poss[i]
        pos_z = 0.0
        if len(pos_)>2: pos_z=pos_[2]
        pos = [ pos_[0]+ncanv[0]*0.5*dx,   pos_[1]+ncanv[1]*0.5*dy, pos_z ]
        coef = coefs[i]
        if isinstance(coef, float):
            #print("GU.stampToGrid3D()") 
            GU.stampToGrid3D( canvas, rho, pos, rots[i], dd=dd, coef=coef, byCenter=byCenter )
        else:
            #print("GU.stampToGrid3D_complex()")
            coef = complex( coef[0], coef[1] )
            GU.stampToGrid3D_complex( canvas, rho, pos, rots[i], dd=dd, coef=coef, byCenter=byCenter )

    Vtip, shifts = makeTipField3D( ncanv, dd, z0=z, sigma=sigma, multipole_dict=multipole_dict )
    #renorm = 1./( (Vtip**2).sum() * (rho.real**2+rho.imag**2).sum() )
    renorm = 1./( (Vtip**2).sum() )
    phmap  = convFFT(Vtip,canvas) * renorm
    #print "rho  ", rho.shape  
    #print "Vtip ", Vtip.shape  
    #Vtip = np.roll( Vtip, -shifts[2], axis=2)
    Vtip = np.roll( Vtip, -shifts[1], axis=1)
    Vtip = np.roll( Vtip, -shifts[0], axis=2)
    return phmap, Vtip, canvas, dd

# ================================================================
#          Functions to Solve System of Couplet exciton
# ================================================================

def prepareRhoTransForCoumpling( rhoTrans, nsub=None ):
    if nsub is not None: # down-sample ?
        #print( "rhoTrans.shape ", rhoTrans.shape ) 
        ndim1 = rhoTrans.shape
        ndim2 = (ndim1[0]//nsub,ndim1[1]//nsub,ndim1[2]//nsub) 
        #(nDim[0]//subsamp,nDim[1]//subsamp,nDim[2]//subsamp)
        if bDebug:
            sum1 = (rhoTrans**2).sum()
        rho = GU.downSample3D( rhoTrans, ndim=ndim2 )
        if bDebug:
            #print rhoTrans.shape
            sum2  = (rho**2).sum()
            #print(sum2, sum1)
            print(sum2/(ndim2[0]*ndim2[1]*ndim2[2]), sum1/(ndim1[0]*ndim1[1]*ndim1[2]))
            GU.saveXSF("rhoTrans_down.xsf", rhoTrans, lvec )
            #exit()
    else:
        rho = rhoTrans
    #sh = rhoTrans.shape
    #dV         = (lvec[0,0]*lvec[1,1]*lvec[2,2])/(sh[0]*sh[1]*sh[2])  
    return rho

def hackHamiltoian( H ):
    # ABAB
    #    H[1,2]*=0; H[2,1]*=0; H[2,3]*=0; H[3,2]*=0; H[3,0]*=0; H[0,3]*=0; H[0,1]*=0; H[1,0]*=0; #H[0,0]*=0.999; H[2,2]*=0.999
    # AAAB
    #    H[0,3]*=0;  H[3,0]*=0;  H[1,3]*=0;  H[3,1]*=0; H[2,3]*=0; H[3,2]*=0; #H[3,3]*=0.999
    # AABB
    #    H[2,0]*=0; H[0,2]*=0; H[1,2]*=0; H[2,1]*=0; H[0,3]*=0; H[3,0]*=0; H[3,1]*=0; H[1,3]*=0; #H[0,0]*=0.999;H[2,2]*=0.999;
    return

def assembleExcitonHamiltonian( rhos, poss, latMats, Ediags, byCenter=False ):
    coulomb_const = 14.3996   # [eV*A/e^2]  # https://en.wikipedia.org/wiki/Coulomb_constant
    prefactor     = coulomb_const #/(dV*dV)
    n = len(poss)
    H = np.eye(n)
    for i in range(n): 
        H[i,i]*=Ediags[i]
    for i in range(n):
        lat1 = latMats[i]   #;print(rot1)
        p1   = poss[i]   #;print("p1 shape:",p1.shape)
        rho1 = rhos[i]
        ns1  = rho1.shape
        if byCenter: p1 = p1 + lat1[0,:]*(ns1[0]*-0.5) + lat1[1,:]*(ns1[1]*-0.5)
        for j in range(i):
            #print "eval H[%i,%i] " %(i,j)
            #dpos = poss[i] - poss[j]
            lat2 = latMats[j]
            p2   = poss[j]
            rho2 = rhos[j]
            ns2  = rho2.shape
            if byCenter: p2 = p2 + lat2[0,:]*(ns2[0]*-0.5) + lat2[1,:]*(ns2[1]*-0.5)
            #if bDebug:
            GU.setDebugFileName( "coulombGrid_%03i_%03i_.xyz" %(i,j) )
            eij = GU.coulombGrid_brute( rho1, rho2, pos1=p1, pos2=p2, lat1=lat1, lat2=lat2 )
            eij *= prefactor
            H[i,j]=eij
            H[j,i]=eij
    print("H  = \n", H)
    return H

def solveExcitonHamliltonian( H ):
    es,vs = np.linalg.eig(H)
    #print("eigenvalues    ", es)
    #print("eigenvectors \n", vs)
    #print("!!! ordering Eigen-pairs ")
    idx = np.argsort(es)
    es  = es[idx]
    vs  = vs.transpose()
    vs  = vs[idx]
    print("eigenvalues    ", es)
    print("eigenvectors \n", vs)
    #    for i,v in enumerate(vs):
    #        print("E[%i]=%g" %(i,es[i]), " v=",v, " |v|=",(v**2).sum())
    return es,vs

def solveExcitonSystem( rhoTranss, lvecs, poss, rots, nSub=None, byCenter=False, Ediags=1.0, hackHfunc=hackHamiltoian, bMultipole=True ):
    '''
    Solve coupled excitonic system according to :
    https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Book%3A_Time_Dependent_Quantum_Mechanics_and_Spectroscopy_(Tokmakoff)/15%3A_Energy_and_Charge_Transfer/15.03%3A_Excitons_in_Molecular_Aggregates
    '''
    n = len(poss)
    poss = np.array(poss)
    if isinstance(Ediags, float ):
        Ediags=[Ediags]*n

    if not isinstance(rhoTranss,list):
        rho = prepareRhoTransForCoumpling( rhoTranss, nsub=nSub )
        rhos = [ rho ] * n
    else:
        rhos = []
        for i in range(n):
            rhos.append(  prepareRhoTransForCoumpling( rhoTranss[i], nsub=nSub ) )
    latMats = []
    for i in range(n):
        lvec = lvecs[i]
        lvec = np.array(lvec[1:][::-1,::-1])
        latMat = makeTransformMat( rhos[i].shape, lvec, rots[i] )
        latMats.append( latMat  )
        if bMultipole:
            mpol_coefs = GU.evalMultipole( rhos[i], rot=latMat )
            print("Mol[%i] multipoles coefs: " %i, mpol_coefs )

    H = assembleExcitonHamiltonian( rhos, poss, latMats, Ediags, byCenter=byCenter )
    if hackHfunc is not None: 
        hackHfunc( H )
    es,vs = solveExcitonHamliltonian( H )
    #    print(" <<<<!!!!! DEBUG : solveExcitonSystem() DONE .... this is WIP, do not take seriously ")
    return es,vs,H

# ================================================================
#          Utilities for building systems
# ================================================================

def makePreset_row( n, dx=5.0, ang=0.0 ): 
    rots  = [ ang ] * n
    poss  = [ [(i-0.5*(n-1))*dx,0.0,0.0] for i in range(n) ]
    return poss, rots

def makePreset_cycle( n, R=10.0, ang0=0.0 ):
    dang = np.pi*2/n
    rots=[]; poss=[]
    for i in range(n):
        a=dang*i 
        poss.append( [-np.cos(a)*R ,np.sin(a)*R,0] )
        rots.append( -a+ang0 )
    return poss, rots

def makePreset_arr1( m,n, R=10.0 ):
    dang = np.pi/2
    rots=[]; poss=[]
    for i in range(m):
        for j in range(n):
            ii=(i-(m-1)/2.)
            jj=(j-(n-1)/2.)
            poss.append( [ii*R ,jj*R,0] )
            rots.append((j%2+((i+1)%2 * (n+1)%2))*dang )
    return poss, rots


def combinator(oents):
    '''
    This function finds all possible combinations of of excited states on molecule  
    '''
    oents=np.array(oents)   ; #print(oents)
    isx=np.argsort(oents)   ; #print(isx)
    ents=oents[isx]   #; print(ents)

    funiqs=np.unique(ents, True,False, False) #indices of first uniqs
    funiqs=funiqs[1]  #; print(funiqs)

    nuniqs=np.unique(ents, False, False, True) #numbers of uniqs
    nuniqs=nuniqs[1]        #; print(nuniqs)
    tuniqs=np.copy(nuniqs)  #; print(tuniqs) # factorization coefs

    for i in range(len(nuniqs)-1): #calculate total number of combinations
        tuniqs[-i-2]=nuniqs[-i-2]*tuniqs[-i-1]

    combos=np.zeros((tuniqs[0],len(nuniqs)),dtype=int)

    for i in range(tuniqs[0]):
        comb=i
        for j in range(len(nuniqs)-1):
            combos[i,j]=comb//tuniqs[j+1]
            comb-=tuniqs[j+1]*(comb//tuniqs[j+1])
        combos[i,-1]=comb

    print("Combinations for various molecules:")
    print(combos)

    # --- make 2D list of permutation index to reorder any property
    inds = []
    (ni,nj)=np.shape(combos)
    for i in range(ni):
        inds.append( [  isx[ funiqs[j]+combos[i,j] ] for j in range(nj) ] )
    return inds

def applyCombinator( lst, inds ):
    out = []
    for js in inds:
        out.append( [  lst[ j ] for j in js ] )
    return out
