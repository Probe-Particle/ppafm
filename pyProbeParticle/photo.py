
"""
# =============================================================================
#                          Theoretical Background
# =============================================================================

pyLightSTM Photo Module
----------------------

This module implements core functionality for simulating tip-enhanced electro-luminescence
in scanning probe microscopy. The simulation combines two key physical processes:

1. Optical Coupling:
   - Interaction between molecular transition densities and tip cavity field
   - Field modeled using multipole expansion of spherical harmonics
   - Coupling computed efficiently using FFT-based convolution

2. Quantum Mechanical Coupling:
   - Implements exciton coupling between multiple molecules
   - Solves eigenvalue problem for coupled system Hamiltonian
   - Accounts for position-dependent interactions and molecular orientations

The implementation uses grid-based calculations with efficient FFT methods for
convolutions and supports both 2D and 3D geometries. The modular design allows
separate handling of optical and quantum mechanical aspects while maintaining
physical accuracy.

Key Features:
- Multipole expansion for tip fields
- FFT-accelerated convolutions
- Coupled exciton system solver
- Support for molecular aggregates
- Flexible grid and coordinate handling

References:
1. Time Dependent Quantum Mechanics and Spectroscopy (Tokmakoff)
2. Excitons in Molecular Aggregates (LibreTexts)
"""

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
    """Creates transformation matrix for molecular orientations.
    
    Args:
        ns (tuple): Grid dimensions (nx, ny, nz)
        lvec (ndarray): Lattice vectors defining molecular orientation
        angle (float): Rotation angle in radians
        rot (ndarray, optional): Pre-computed rotation matrix
        
    Returns:
        ndarray: Transformation matrix for coordinate mapping
    """
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
    """Generate 2D coordinate grid for calculations.
    
    Args:
        nDim (tuple): Grid dimensions (nx, ny)
        dd (tuple): Grid spacing (dx, dy)
        
    Returns:
        tuple: X and Y coordinate arrays and grid shifts (xshift, yshift)
    """
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
    return X, Y, (xshift, yshift)

def getMGrid3D( nDim, dd ):
    """Generate 3D coordinate grid for calculations.
    
    Args:
        nDim (tuple): Grid dimensions (nx, ny, nz)
        dd (tuple): Grid spacing (dx, dy, dz)
        
    Returns:
        tuple: X, Y, Z coordinate arrays and grid shifts (xshift, yshift, zshift)
    """
    'returns coordinate arrays X, Y, Z'
    (dx,dy,dz) = dd
    (nx,ny,nz) = nDim[:3]
    # print ( "dd ", dd)
    XYZ = np.mgrid[0:nx,0:ny,0:nz].astype(float)
    X,xshift = shiftHalfAxis( XYZ[0], dx, nx, ax=0 )
    Y,yshift = shiftHalfAxis( XYZ[1], dy, ny, ax=1 )
    #Z,zshift = shiftHalfAxis( XYZ[2], dz, nz, ax=2 )
    Z = XYZ[2]
    return X*dx, Y*dy, Z*dz, (xshift, yshift, 0)

'''
def makeTipField2D( sh, dd, z=10.0, sigma=1.0, multipole_dict={'s':1.0}, bSTM=False, b3D=False ):
    Vtip = np.zeros( sh[:2] )
    Y,X,shifts  = getMGrid2D( sh, dd )
    radial = 1/np.sqrt( X**2 + Y**2  + z**2 + sigma**2  ) 
    if multipole_dict is not None:    # multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        Vtip = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.items():
            Vtip += radial * coef * fFFT.getSphericalHarmonic( X, Y, z, kind=kind, tilt=0 )
    else:
        Vtip = radial
    return Vtip, shifts
'''

def makeTipField( sh, dd, z0=10.0, sigma=1.0, multipole_dict={'s':1.0}, b3D=False, bSTM=False, beta=1.0 ):
    """Construct tip field using multipole expansion.
    
    Generates either optical cavity field or STM tunneling field based on bSTM flag.
    For optical field: Uses multipole expansion with spherical harmonics.
    For STM field: Uses exponential decay with beta parameter.
    
    Args:
        sh (tuple): Grid shape
        dd (tuple): Grid spacing
        z0 (float): Tip height
        sigma (float): Regularization parameter for field divergence
        multipole_dict (dict): Multipole coefficients {orbital: coefficient}
        b3D (bool): If True, generate 3D field
        bSTM (bool): If True, generate STM tunneling field instead of optical field
        beta (float): Decay parameter for STM tunneling
        
    Returns:
        ndarray: Generated tip field and grid shifts
    """
    #Vtip = np.zeros( sh )
    if b3D:
        X,Y,Z,shifts  = getMGrid3D( sh,     dd )
        Z += z0
    else:
        Y,X,shifts    = getMGrid2D( sh[:2], dd[:2] )
        Z = z0
    if bSTM:
        radial = np.exp( -beta * np.sqrt( X**2 + Y**2  + Z**2 )  )
    else:
        radial = 1/np.sqrt( X**2 + Y**2  + Z**2 + sigma**2 ) 
    if multipole_dict is not None:    # multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        Vtip = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.items():
            Vtip += radial * coef * fFFT.getSphericalHarmonic( X, Y, Z, kind=kind, tilt=0 )
    else:
        Vtip = radial
    if b3D:
        Vtip = Vtip.transpose((2,1,0)).copy()
    return Vtip, shifts

def convFFT(F1,F2, bNormalize=False):
    result = np.fft.ifftn( np.fft.fftn(F1) * np.fft.fftn(F2) )
    if bNormalize:
        sh = result.shape
        result = result / ( sh[0]*sh[1] )
    return result

# ==========================================================================
#      Functions to project trasition densities on a common grid (Canvas)
# ==========================================================================

def evalGridStep2D( sh, lvec ):
    return (
        lvec[3][2]/sh[0],
        lvec[2][1]/sh[1]
    )

def evalGridStep3D( sh, lvec ):
    return (
        lvec[3][2]/sh[0],
        lvec[2][1]/sh[1],
        lvec[1][0]/sh[2],    
    )

def photonMap2D_stamp( rhos, lvecs, Vtip, dd_canv, rots=[0.0], poss=[ [0.0,0.0] ], coefs=[ [1.0,0.0] ], byCenter=False, bComplex=False ):
    """Generate 2D photon map by convolving transition densities with tip field.
    
    Computes optical coupling between molecular transitions and tip cavity field
    using FFT-based convolution.
    
    Args:
        rhos (list): Transition densities for each molecule
        lvecs (list): Lattice vectors for each molecule
        Vtip (ndarray): Tip field
        dd_canv (tuple): Grid spacing for canvas
        rots (list): Rotation angles for each molecule
        poss (list): Positions of molecules
        coefs (list): Coefficients for each molecule's contribution
        byCenter (bool): If True, positions are relative to molecular centers
        bComplex (bool): If True, return complex values
        
    Returns:
        tuple: Total photon map and individual molecular contributions
    """
    ncanv = Vtip.shape
    dd_canv = np.array( dd_canv )
    if bComplex:
        dtype=np.complex128
    else:
        dtype=np.float64    
    canvas = np.zeros( ncanv, dtype=dtype )
    for i in range(len(poss)):
        coef = coefs[i]
        rho  = np.sum    (  rhos[i], axis=2  )
        rho  = rho.astype( dtype                 ) 
        ddi  = np.array(  evalGridStep2D( rhos[i].shape, lvecs[i] ) )
        pos  = np.array( poss[i][:2] ) 
        pos    /= dd_canv
        dd_fac = ddi/dd_canv
        if not isinstance(coef, float):
            coef = complex( coef[0], coef[1] )
        GU.stampToGrid2D( canvas, rho, pos, rots[i], dd=dd_fac, coef=coef, byCenter=byCenter, bComplex=bComplex)
    phmap  = convFFT(Vtip,canvas)
    return phmap, canvas

def photonMap3D_stamp( rhos, lvecs, Vtip, dd_canv, rots=[0.0], poss=[ [0.0,0.0] ], coefs=[ [1.0,0.0] ], byCenter=False ):
    """Generate 3D photon map by convolving transition densities with tip field.
    
    Computes optical coupling between molecular transitions and tip cavity field
    using FFT-based convolution.
    
    Args:
        rhos (list): Transition densities for each molecule
        lvecs (list): Lattice vectors for each molecule
        Vtip (ndarray): Tip field
        dd_canv (tuple): Grid spacing for canvas
        rots (list): Rotation angles for each molecule
        poss (list): Positions of molecules
        coefs (list): Coefficients for each molecule's contribution
        byCenter (bool): If True, positions are relative to molecular centers
        
    Returns:
        tuple: Total photon map and individual molecular contributions
    """
    ncanv = Vtip.shape
    dd_canv = np.array( dd_canv )
    if len(ncanv)<3:
        ncanv = ( (rhos[0].shape[2],ncanv[0],ncanv[1] ) )
    dtype=np.complex128
    if isinstance(coefs[0], float): dtype=np.float64    
    canvas = np.zeros( ncanv, dtype=dtype )
    #print( "Vtip.shape ", Vtip.shape, " canvas.shape ", canvas.shape  )
    for i in range(len(poss)):
        coef = coefs[i]
        rho    = rhos[i].transpose((2,1,0)).astype( dtype ).copy()
        ddi = evalGridStep3D( rhos[i].shape, lvecs[i] )
        dd_fac = ddi/dd_canv 
        pos  = np.array( poss[i] ) 
        pos    /= dd_canv
        coef = coefs[i]
        if not isinstance(coef, float):
            coef = complex( coef[0], coef[1] )
        GU.stampToGrid3D( canvas, rho, pos, rots[i], dd=dd_fac, coef=coef, byCenter=byCenter )
    #phmap  = convFFT(Vtip,canvas)   # WARRNING : FFT should not be done in z-direction
    phmap   = np.zeros( canvas.shape[1:], dtype=np.complex128  )
    for i in range( canvas.shape[0] ):
        phmap  += convFFT( Vtip[i,:,:],canvas[i,:,:]) 
    return phmap, canvas

# ================================================================
#          Functions to Solve System of Couplet exciton
# ================================================================

def prepareRhoTransForCoumpling( rhoTrans, nsub=None, lvec=None ):
    """Prepare transition density for exciton coupling.
    
    Down-samples transition density if nsub is provided.
    
    Args:
        rhoTrans (ndarray): Transition density
        nsub (int, optional): Down-sampling factor
        lvec (ndarray, optional): Lattice vector
        
    Returns:
        ndarray: Prepared transition density
    """
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
    """Modify Hamiltonian matrix.
    
    Applies user-defined modifications to the Hamiltonian matrix.
    
    Args:
        H (ndarray): Hamiltonian matrix
        
    Returns:
        ndarray: Modified Hamiltonian matrix
    """
    # ABAB
    #    H[1,2]*=0; H[2,1]*=0; H[2,3]*=0; H[3,2]*=0; H[3,0]*=0; H[0,3]*=0; H[0,1]*=0; H[1,0]*=0; #H[0,0]*=0.999; H[2,2]*=0.999
    # AAAB
    #    H[0,3]*=0;  H[3,0]*=0;  H[1,3]*=0;  H[3,1]*=0; H[2,3]*=0; H[3,2]*=0; #H[3,3]*=0.999
    # AABB
    #    H[2,0]*=0; H[0,2]*=0; H[1,2]*=0; H[2,1]*=0; H[0,3]*=0; H[3,0]*=0; H[3,1]*=0; H[1,3]*=0; #H[0,0]*=0.999;H[2,2]*=0.999;
    return

def assembleExcitonHamiltonian( rhos, poss, latMats, Ediags, byCenter=False ):
    """Assemble exciton Hamiltonian matrix.
    
    Computes coupling between molecular transitions using Coulomb interaction.
    
    Args:
        rhos (list): Transition densities for each molecule
        poss (list): Positions of molecules
        latMats (list): Lattice vectors for each molecule
        Ediags (list): Diagonal energies for each molecule
        byCenter (bool): If True, positions are relative to molecular centers
        
    Returns:
        ndarray: Assembled Hamiltonian matrix
    """
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
            if bDebug:
                GU.setDebugFileName( "coulombGrid_%03i_%03i_.xyz" %(i,j) )
            eij = GU.coulombGrid_brute( rho1, rho2, pos1=p1, pos2=p2, lat1=lat1, lat2=lat2 )
            eij *= prefactor
            H[i,j]=eij
            H[j,i]=eij
    print("H  = \n", H)
    return H

def solveExcitonHamliltonian( H ):
    """Solve exciton Hamiltonian eigenvalue problem.
    
    Computes eigenvalues and eigenvectors of the Hamiltonian matrix.
    
    Args:
        H (ndarray): Hamiltonian matrix
        
    Returns:
        tuple: Eigenvalues and eigenvectors
    """
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
    """Solve coupled exciton system for molecular aggregate.
    
    Implements quantum mechanical coupling between molecules according to:
    https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Book%3A_Time_Dependent_Quantum_Mechanics_and_Spectroscopy_(Tokmakoff)/15%3A_Energy_and_Charge_Transfer/15.03%3A_Excitons_in_Molecular_Aggregates
    
    Args:
        rhoTranss (list/ndarray): Transition densities for each molecule
        lvecs (list): Lattice vectors defining molecular orientations
        poss (list): Positions of molecules
        rots (list): Rotation angles for each molecule
        nSub (int, optional): Number of subsystems
        byCenter (bool): If True, positions are relative to molecular centers
        Ediags (float/list): Diagonal energies for each molecule
        hackHfunc (callable): Function to modify Hamiltonian
        bMultipole (bool): If True, compute multipole coefficients
        
    Returns:
        tuple: Eigenvalues, eigenvectors, and Hamiltonian matrix
    """
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
    """Create preset row configuration.
    
    Generates positions and rotation angles for a row of molecules.
    
    Args:
        n (int): Number of molecules
        dx (float): Distance between molecules
        ang (float): Rotation angle
        
    Returns:
        tuple: Positions and rotation angles
    """
    rots  = [ ang ] * n
    poss  = [ [(i-0.5*(n-1))*dx,0.0,0.0] for i in range(n) ]
    return poss, rots

def makePreset_cycle( n, R=10.0, ang0=0.0 ):
    """Create preset cycle configuration.
    
    Generates positions and rotation angles for a cycle of molecules.
    
    Args:
        n (int): Number of molecules
        R (float): Radius of cycle
        ang0 (float): Initial rotation angle
        
    Returns:
        tuple: Positions and rotation angles
    """
    dang = np.pi*2/n
    rots=[]; poss=[]
    for i in range(n):
        a=dang*i 
        poss.append( [-np.cos(a)*R ,np.sin(a)*R,0] )
        rots.append( -a+ang0 )
    return poss, rots

def makePreset_arr1( m,n, R=10.0 ):
    """Create preset array configuration.
    
    Generates positions and rotation angles for a 2D array of molecules.
    
    Args:
        m (int): Number of rows
        n (int): Number of columns
        R (float): Distance between molecules
        
    Returns:
        tuple: Positions and rotation angles
    """
    dang = np.pi/2
    rots=[]; poss=[]
    for i in range(m):
        for j in range(n):
            ii=(i-(m-1)/2.)
            jj=(j-(n-1)/2.)
            poss.append( [ii*R ,jj*R,0] )
            rots.append((j%2+((i+1)%2 * (n+1)%2))*dang )
    return poss, rots


def combinator(oents,subsys=False):
    """Find all possible combinations of orthogonal excited states.
    
    Args:
        oents (list): Orthogonal excited states
        subsys (bool): If True, consider subsystems
        
    Returns:
        ndarray: Combinations of excited states
    """
    oents=np.array(oents)   ; #print(oents)
    isx=np.argsort(oents)   ; #print(isx)
    ents=oents[isx]   #; print(ents)

    funiqs=np.unique(ents, True,False, False) #indices of first uniqs
    funiqs=funiqs[1]  #; print(funiqs)

    nuniqs=np.unique(ents, False, False, True) #numbers of uniqs
    nuniqs=nuniqs[1]        #; print(nuniqs)
    if subsys:
        nuniqs=nuniqs+1        #this creates the possibility of empty exciton state for each molecule
    tuniqs=np.copy(nuniqs)  #; print(tuniqs) # factorization coefs

    for i in range(len(nuniqs)-1): #calculate total number of combinations
        tuniqs[-i-2]=nuniqs[-i-2]*tuniqs[-i-1]

    combos=np.zeros((tuniqs[0],len(nuniqs)),dtype=int)

    for i in range(tuniqs[0]):
        comb=i
        for j in range(len(nuniqs)-1):
            combos[i,j]=comb//tuniqs[j+1]
            comb-=tuniqs[j+1]*(comb//tuniqs[j+1])
            if subsys:
                if combos[i,j] >= nuniqs[j]-1: #this is for when the molecule will be missing
                    combos[i,j]=-1

        combos[i,-1]=comb
        if subsys:
            if comb >= nuniqs[-1]-1: #this is for when the molecule will be missing
                combos[i,-1]=-1
        

    print("Combinations for various molecules:")
    print(combos)

    # --- make 2D list of permutation index to reorder any property
    #inds = []
    #(ni,nj)=np.shape(combos)
    #for i in range(ni):
    #    inds.append( [  isx[ funiqs[j]+combos[i,j] ] for j in range(nj) ] )
    (ni,nj)=np.shape(combos)
    inds = np.zeros((ni,nj))
    for i in range(ni):
        for j in range(nj):
            if combos[i,j] == -1:
                inds[i,j]=-1
            else:
                inds[i,j]=  (isx[ funiqs[j]+combos[i,j] ])


    return inds

def combine( lst, inds ):
    """Combine elements of a list based on indices.
    
    Args:
        lst (list): List of elements
        inds (ndarray): Indices for combination
        
    Returns:
        list: Combined elements
    """
    out = []
    for j in inds:
        if j !=-1:
            out.append(lst[int(j)])
    return out

def applyCombinator( lst, inds ):
    """Apply combinator to a list of elements.
    
    Args:
        lst (list): List of elements
        inds (ndarray): Indices for combination
        
    Returns:
        list: Combined elements
    """
    out = []
    for js in inds:
        out.append( combine( lst, js ) )
        '''
        outl = []
        for j in js:
            if j !=-1:
                outl.append(lst[int(j)])
        out.append( outl )
        '''
    return out

