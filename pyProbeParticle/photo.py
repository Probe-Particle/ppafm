
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

_exc_dbg_env = os.environ.get('PP_EXCITON_DEBUG', '')
bExcitonDebug = (_exc_dbg_env != '') and (_exc_dbg_env != '0') and (_exc_dbg_env.lower() != 'false')

_exc_dump_pair_env = os.environ.get('PP_EXCITON_DUMP_PAIR', '').strip()
_exc_dump_all_env  = os.environ.get('PP_EXCITON_DUMP_ALL', '').strip()
_exc_dump_all = (_exc_dump_all_env != '') and (_exc_dump_all_env != '0') and (_exc_dump_all_env.lower() != 'false')
_exc_dump_pair = None
if _exc_dump_pair_env != '':
    _pp = _exc_dump_pair_env.replace(';', ',').replace(' ', '').split(',')
    if (len(_pp) >= 2) and (_pp[0].lstrip('-').isdigit()) and (_pp[1].lstrip('-').isdigit()):
        _exc_dump_pair = (int(_pp[0]), int(_pp[1]))

_exc_report_multipoles_env = os.environ.get('PP_EXCITON_REPORT_MULTIPOLES', '').strip()
_exc_report_multipoles = (_exc_report_multipoles_env != '') and (_exc_report_multipoles_env != '0') and (_exc_report_multipoles_env.lower() != 'false')

_exc_report_sij_multipoles_env = os.environ.get('PP_EXCITON_REPORT_SIJ_MULTIPOLES', '').strip()
_exc_report_sij_multipoles = (_exc_report_sij_multipoles_env != '') and (_exc_report_sij_multipoles_env != '0') and (_exc_report_sij_multipoles_env.lower() != 'false')

def _gridStats( name, rho, lat=None ):
    rho = np.asarray(rho)
    s1  = rho.sum()
    s2  = (rho*rho).sum()
    mn  = rho.min()
    mx  = rho.max()
    amx = np.abs(rho).max()
    print(f"[EXCITON-DBG] {name}: shape={rho.shape} min={mn:.6g} max={mx:.6g} maxabs={amx:.6g} sum={s1:.6g} sumsq={s2:.6g}")
    if lat is not None:
        lat = np.asarray(lat)
        dv = abs(np.dot(lat[0], np.cross(lat[1], lat[2])))
        n0 = np.linalg.norm(lat[0])
        n1 = np.linalg.norm(lat[1])
        n2 = np.linalg.norm(lat[2])
        print(f"[EXCITON-DBG] {name}: step_norms_A=({n0:.6g},{n1:.6g},{n2:.6g}) dV_A3={dv:.6g}")

def _dump_rho_pair_xyz( fname, rho1, rho2, pos1, pos2, lat1, lat2 ):
    rho1 = np.asarray(rho1)
    rho2 = np.asarray(rho2)
    lat1 = np.asarray(lat1)
    lat2 = np.asarray(lat2)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    n1 = int(rho1.size)
    n2 = int(rho2.size)
    with open(fname, 'w') as f:
        f.write(f"{n1+n2+1}\n")
        f.write(f"# ns1={n1} ns2={n2}   (He=rho1 grid, Ne=rho2 grid)\n")
        f.write("U 0.0 0.0 0.0\n")

        nx, ny, nz = rho1.shape
        ix, iy, iz = np.indices((nx, ny, nz), dtype=float)
        v = np.stack([ix+0.5, iy+0.5, iz+0.5], axis=-1).reshape(-1, 3)
        p = v @ lat1 + pos1[None, :]
        q = rho1.reshape(-1)
        for (x, y, z), qq in zip(p, q):
            f.write(f"He  {x:.6f} {y:.6f} {z:.6f}   {qq:.15g}\n")

        nx, ny, nz = rho2.shape
        ix, iy, iz = np.indices((nx, ny, nz), dtype=float)
        v = np.stack([ix+0.5, iy+0.5, iz+0.5], axis=-1).reshape(-1, 3)
        p = v @ lat2 + pos2[None, :]
        q = rho2.reshape(-1)
        for (x, y, z), qq in zip(p, q):
            f.write(f"Ne  {x:.6f} {y:.6f} {z:.6f}   {qq:.15g}\n")

def _unitvec( v ):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-30:
        return v*0.0
    return v/n

def _Vdd( mu1, mu2, R ):
    R = np.asarray(R, dtype=float)
    r = np.linalg.norm(R)
    if r < 1e-12:
        return 0.0
    rhat = R / r
    return (np.dot(mu1, mu2) - 3.0*np.dot(mu1, rhat)*np.dot(mu2, rhat)) / (r**3)

def _grad_phi_Q( Q, R ):
    R = np.asarray(R, dtype=float)
    r = np.linalg.norm(R)
    if r < 1e-12:
        return np.zeros(3)
    S = float(R @ Q @ R)
    return (Q @ R) / (r**5) - 2.5 * S * R / (r**7)

def _VmuQ( mu, Q, R ):
    return float(np.dot(mu, _grad_phi_Q(Q, R)))

def _multipoles_from_density( rho, pos, lat, byCenter=False ):
    rho = np.asarray(rho)
    pos = np.asarray(pos, dtype=float)
    lat = np.asarray(lat, dtype=float)
    nx, ny, nz = rho.shape
    pos0 = pos.copy()
    if byCenter:
        pos0 = pos0 + lat[0]*(nx*-0.5) + lat[1]*(ny*-0.5)

    ix, iy, iz = np.indices((nx, ny, nz), dtype=float)
    v = np.stack([ix+0.5, iy+0.5, iz+0.5], axis=-1).reshape(-1, 3)
    r = v @ lat + pos0[None, :]
    q = rho.reshape(-1)

    qsum = q.sum()
    dip  = (q[:, None] * r).sum(axis=0)

    r0   = r.mean(axis=0)
    rc   = r - r0[None, :]
    dip_c = (q[:, None] * rc).sum(axis=0)

    r2 = (rc*rc).sum(axis=1)
    Q = np.zeros((3, 3))
    Q[0, 0] = (q*(3.0*rc[:, 0]*rc[:, 0] - r2)).sum()
    Q[1, 1] = (q*(3.0*rc[:, 1]*rc[:, 1] - r2)).sum()
    Q[2, 2] = (q*(3.0*rc[:, 2]*rc[:, 2] - r2)).sum()
    Q[0, 1] = (q*(3.0*rc[:, 0]*rc[:, 1])).sum(); Q[1, 0] = Q[0, 1]
    Q[0, 2] = (q*(3.0*rc[:, 0]*rc[:, 2])).sum(); Q[2, 0] = Q[0, 2]
    Q[1, 2] = (q*(3.0*rc[:, 1]*rc[:, 2])).sum(); Q[2, 1] = Q[1, 2]

    return qsum, dip, dip_c, Q, r0

def _downsample3D_blocksum( Fin, nsub ):
    sh = Fin.shape
    ndim2 = (sh[0]//nsub, sh[1]//nsub, sh[2]//nsub)
    sh2 = (ndim2[0]*nsub, ndim2[1]*nsub, ndim2[2]*nsub)
    F = np.asarray(Fin[:sh2[0], :sh2[1], :sh2[2]], order='C')
    F = F.reshape(ndim2[0], nsub, ndim2[1], nsub, ndim2[2], nsub)
    return F.sum(axis=(1,3,5))

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
    """Calculate 2D grid spacing from lattice vectors and shape.
    
    Args:
        sh: Shape (nx, ny, nz) of the 3D density array
        lvec: Lattice vectors [origin, X-vec, Y-vec, Z-vec]
        
    Returns:
        tuple: (dx, dy) grid spacing in Ångströms
    """
    # Use full vector lengths, not individual components
    # sh[0] = nx (points along X), sh[1] = ny (points along Y)
    import numpy as np
    return (
        np.linalg.norm(lvec[1][:2]) / sh[0],  # X-vector length / nx
        np.linalg.norm(lvec[2][:2]) / sh[1]   # Y-vector length / ny
    )

def evalGridStep3D( sh, lvec ):
    """Calculate 3D grid spacing from lattice vectors and shape.
    
    Args:
        sh: Shape (nx, ny, nz) of the 3D density array
        lvec: Lattice vectors [origin, X-vec, Y-vec, Z-vec]
        
    Returns:
        tuple: (dx, dy, dz) grid spacing in Ångströms
    """
    import numpy as np
    return (
        np.linalg.norm(lvec[1]) / sh[0],  # X-vector length / nx
        np.linalg.norm(lvec[2]) / sh[1],  # Y-vector length / ny
        np.linalg.norm(lvec[3]) / sh[2]   # Z-vector length / nz
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
    
    # Debug flag check (import params if available)
    import importlib.util
    debug_dims = False
    if importlib.util.find_spec("photonMap"):
        import photonMap
        debug_dims = getattr(photonMap, "params", {}).get("debug_dims", False)
    
    for i in range(len(poss)):
        coef = coefs[i]
        rho  = np.sum    (  rhos[i], axis=2  )
        rho  = rho.astype( dtype                 ) 
        # Calculate grid spacing BEFORE transpose
        ddi  = np.array(  evalGridStep2D( rhos[i].shape, lvecs[i] ) )
        # Transpose to match matplotlib imshow convention: (nx,ny) -> (ny,nx)
        # imshow treats first dimension as rows (Y) and second as columns (X)
        rho  = rho.T
        # Also swap the grid spacing to match transposed array
        ddi  = ddi[::-1]  # Swap (dx, dy) to (dy, dx)
        pos  = np.array( poss[i][:2] ) 
        pos    /= dd_canv
        dd_fac = ddi/dd_canv
        
        if debug_dims:
            print(f"[DEBUG] photonMap2D_stamp molecule {i}:")
            print(f"[DEBUG]   rho (2D after Z-sum).shape: {rho.shape}")
            print(f"[DEBUG]   ddi (cube grid spacing): {ddi}")
            print(f"[DEBUG]   dd_canv (canvas grid spacing): {dd_canv}")
            print(f"[DEBUG]   dd_fac (scaling factor): {dd_fac}")
            print(f"[DEBUG]   Stamped size on canvas: {rho.shape * dd_fac}")
        
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
        
        # CRITICAL FIX: Negate rotation because transpose((2,1,0)) swaps X and Z axes
        # This causes the rotation to be inverted relative to the box
        rot_angle = -rots[i]  # Negate to match box rotation direction
        
        if 'debug_dims' in globals() or (
            'photonMap' in sys.modules and getattr(sys.modules['photonMap'], 'params', {}).get('debug_dims', False)):
            print(f"[DEBUG] photonMap3D_stamp molecule {i}: Original rotation={rots[i]:.3f} rad ({np.degrees(rots[i]):.1f}°)")
            print(f"[DEBUG] photonMap3D_stamp molecule {i}: Negated rotation={rot_angle:.3f} rad ({np.degrees(rot_angle):.1f}°) to compensate for transpose")
            
        GU.stampToGrid3D( canvas, rho, pos, rot_angle, dd=dd_fac, coef=coef, byCenter=byCenter )
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
        if bExcitonDebug:
            _gridStats( "rhoTrans_in", rhoTrans )
        if bDebug:
            sum1 = (rhoTrans**2).sum()
        rho = _downsample3D_blocksum( rhoTrans, nsub )
        if bExcitonDebug:
            _gridStats( "rhoTrans_down_blocksum", rho )
        if bDebug:
            sum2  = (rho**2).sum()
            print(sum2/(ndim2[0]*ndim2[1]*ndim2[2]), sum1/(ndim1[0]*ndim1[1]*ndim1[2]))
            GU.saveXSF("rhoTrans_down.xsf", rhoTrans, lvec )
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

def assembleExcitonHamiltonian( rhos, poss, latMats, Ediags, byCenter=False, dipoles=None ):
    """Assemble exciton Hamiltonian matrix.
    
    Computes coupling between molecular transitions using Coulomb interaction.
    
    Args:
        rhos (list): Transition densities for each molecule
        poss (list): Positions of molecules
        latMats (list): Lattice vectors for each molecule
        Ediags (list): Diagonal energies for each molecule
        byCenter (bool): If True, positions are relative to molecular centers
        dipoles (list): Optional dipole vectors for debug output
        
    Returns:
        ndarray: Assembled Hamiltonian matrix
    """
    coulomb_const = 14.3996   # [eV*A/e^2]  # https://en.wikipedia.org/wiki/Coulomb_constant
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
            dpos = p1 - p2
            _do_dump = False
            if _exc_dump_all:
                _do_dump = True
            elif _exc_dump_pair is not None:
                _a, _b = _exc_dump_pair
                _do_dump = ((i == _a) and (j == _b)) or ((i == _b) and (j == _a))

            if np.dot(dpos, dpos) < 1e-12:
                eij = 0.0
                if _do_dump:
                    fname = "coulombGrid_%03i_%03i_.xyz" % (i, j)
                    if bExcitonDebug:
                        print(f"[EXCITON-DBG] dumping (on-site) sample points for H[{i},{j}] to {fname} (coupling forced to zero)")
                    _dump_rho_pair_xyz( fname, rho1, rho2, p1, p2, lat1, lat2 )
            else:
                if _do_dump:
                    GU.setDebugFileName( "coulombGrid_%03i_%03i_.xyz" %(i,j) )
                    if bExcitonDebug:
                        print(f"[EXCITON-DBG] dumping coulomb sample points for H[{i},{j}] to coulombGrid_{i:03d}_{j:03d}_.xyz")
                else:
                    GU.setDebugFileName( "" )
                if bExcitonDebug:
                    dv1 = abs(np.dot(lat1[0], np.cross(lat1[1], lat1[2])))
                    dv2 = abs(np.dot(lat2[0], np.cross(lat2[1], lat2[2])))
                    print(f"[EXCITON-DBG] H[{i},{j}] dR_A={np.linalg.norm(dpos):.6g} dV1_A3={dv1:.6g} dV2_A3={dv2:.6g}")
                eij = GU.coulombGrid_brute( rho1, rho2, pos1=p1, pos2=p2, lat1=lat1, lat2=lat2 )
                if bExcitonDebug:
                    print(f"[EXCITON-DBG] H[{i},{j}] eij_raw_e2_over_A={eij:.6g} eij_eV={eij*coulomb_const:.6g}")
            eij *= coulomb_const
            if not np.isfinite(eij):
                eij = 0.0
            H[i,j]=eij
            H[j,i]=eij
    print("H  = \n", H)
    
    # Comprehensive debug: print FULL NxN table with coordinates and dipoles
    if bExcitonDebug and dipoles is not None:
        print("\n" + "="*100)
        print("COMPLETE FRENKEL HAMILTONIAN ANALYSIS - ALL ELEMENTS WITH COORDINATES AND DIPOLES")
        print("="*100)
        print(f"{'H[i,j]':8s} | {'pos_i (x,y,z) Ang':24s} | {'pos_j (x,y,z) Ang':24s} | {'dR Ang':8s} | {'dipole_i (x,y,z)':24s} | {'dipole_j (x,y,z)':24s} | {'angle':7s} | {'H_ij meV':12s}")
        print("-"*100)
        for i in range(n):
            for j in range(n):
                pi = np.array(poss[i])
                pj = np.array(poss[j])
                di = dipoles[i] if dipoles is not None else np.zeros(3)
                dj = dipoles[j] if dipoles is not None else np.zeros(3)
                dR = np.linalg.norm(pi - pj)
                
                # Calculate angle between dipoles
                di_mag, dj_mag = np.linalg.norm(di), np.linalg.norm(dj)
                if di_mag > 1e-10 and dj_mag > 1e-10:
                    cos_angle = np.dot(di/di_mag, dj/dj_mag)
                    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                else:
                    angle_deg = 0.0
                
                H_meV = H[i,j] * 1000  # Convert to meV
                
                pos_i_str = f"({pi[0]:+7.2f},{pi[1]:+7.2f},{pi[2]:+5.2f})"
                pos_j_str = f"({pj[0]:+7.2f},{pj[1]:+7.2f},{pj[2]:+5.2f})"
                dip_i_str = f"({di[0]:+6.3f},{di[1]:+6.3f},{di[2]:+6.3f})"
                dip_j_str = f"({dj[0]:+6.3f},{dj[1]:+6.3f},{dj[2]:+6.3f})"
                
                if i == j:
                    print(f"H[{i},{j}]    | {pos_i_str:24s} | {'--- DIAGONAL ---':24s} | {'---':8s} | {dip_i_str:24s} | {'---':24s} | {'---':7s} | {H_meV:+12.4f}")
                else:
                    print(f"H[{i},{j}]    | {pos_i_str:24s} | {pos_j_str:24s} | {dR:8.3f} | {dip_i_str:24s} | {dip_j_str:24s} | {angle_deg:6.1f}° | {H_meV:+12.4f}")
        print("="*100)
        
        # Also print the raw Hamiltonian matrix in meV
        print("\nHamiltonian matrix in meV:")
        print("      ", end="")
        for j in range(n):
            print(f"   [{j}]      ", end="")
        print()
        for i in range(n):
            print(f"[{i}]  ", end="")
            for j in range(n):
                print(f"{H[i,j]*1000:+10.3f} ", end="")
            print()
        print()
    
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
    dipoles = []  # Store dipole vectors for debug
    _mp_qs   = []
    _mp_mus  = []
    _mp_Qs   = []
    _mp_r0s  = []
    for i in range(n):
        lvec = np.array(lvecs[i][1:])
        latMat = makeTransformMat( rhos[i].shape, lvec, rots[i] )
        latMats.append( latMat  )
        if bExcitonDebug:
            _gridStats( f"Mol[{i}] rho", rhos[i], lat=latMat )
        if bMultipole:
            mpol_coefs = GU.evalMultipole( rhos[i], rot=latMat )
            print("Mol[%i] multipoles coefs: " %i, mpol_coefs )
        # Always extract dipole for debug analysis
        mpol = GU.evalMultipole( rhos[i], rot=latMat )
        dipoles.append( mpol[1:4].copy() )  # px, py, pz

        qsum, dip_raw, dip_c, Q, r0 = _multipoles_from_density( rhos[i], poss[i], latMat, byCenter=byCenter )
        _mp_qs .append( qsum )
        _mp_mus.append( dip_c )
        _mp_Qs .append( Q )
        _mp_r0s.append( r0 )
    
    # Debug: Print dipole orientations and predict coupling patterns
    if bExcitonDebug:
        print("\n" + "="*70)
        print("DIPOLE ORIENTATION ANALYSIS")
        print("="*70)
        for i in range(n):
            d = dipoles[i]
            dmag = np.linalg.norm(d)
            if dmag > 1e-10:
                dhat = d / dmag
            else:
                dhat = np.array([0.,0.,0.])
            print(f"State[{i}] pos={poss[i][:2]} rot={rots[i]*180/np.pi:.1f}deg")
            print(f"         dipole=({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}) |d|={dmag:.4f}")
            print(f"         dir=({dhat[0]:+.3f}, {dhat[1]:+.3f}, {dhat[2]:+.3f})")
        
        print("\n" + "-"*70)
        print("DIPOLE-DIPOLE COUPLING PREDICTION (for off-diagonal H elements)")
        print("-"*70)
        for i in range(n):
            for j in range(i):
                di, dj = dipoles[i], dipoles[j]
                di_mag, dj_mag = np.linalg.norm(di), np.linalg.norm(dj)
                if di_mag > 1e-10 and dj_mag > 1e-10:
                    di_hat, dj_hat = di/di_mag, dj/dj_mag
                    cos_angle = np.dot(di_hat, dj_hat)
                    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                    # Dipole-dipole: parallel=strong, perpendicular=weak
                    if abs(cos_angle) > 0.7:
                        orient = "PARALLEL" if cos_angle > 0 else "ANTIPARALLEL"
                        expect = "STRONG"
                    elif abs(cos_angle) < 0.3:
                        orient = "PERPENDICULAR"
                        expect = "WEAK/ZERO"
                    else:
                        orient = "OBLIQUE"
                        expect = "MODERATE"
                    print(f"H[{i},{j}]: angle={angle_deg:5.1f}deg cos={cos_angle:+.3f} -> {orient:12s} expect {expect}")
                else:
                    print(f"H[{i},{j}]: one dipole is zero")
        print("="*70 + "\n")
        
        # Save XYZ file with molecular positions and dipole vectors for visualization
        xyz_fname = "exciton_dipoles_debug.xyz"
        with open(xyz_fname, 'w') as f:
            # Write positions as atoms, dipole endpoints as separate atoms
            n_atoms = n * 3  # position + dipole start + dipole end
            f.write(f"{n_atoms}\n")
            f.write("Exciton dipoles: C=position, N=dipole_start, O=dipole_end (dipole scaled for visibility)\n")
            dipole_scale = 5.0  # Scale factor for visualization
            for i in range(n):
                p = poss[i]
                d = dipoles[i]
                dmag = np.linalg.norm(d)
                if dmag > 1e-10:
                    d_scaled = d / dmag * dipole_scale
                else:
                    d_scaled = np.zeros(3)
                # Position marker
                f.write(f"C  {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
                # Dipole start (at position)
                f.write(f"N  {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
                # Dipole end (position + scaled dipole)
                end = p + d_scaled
                f.write(f"O  {end[0]:12.6f} {end[1]:12.6f} {end[2]:12.6f}\n")
        print(f"[DEBUG] Saved dipole positions to: {xyz_fname}")
        
        # Also save a detailed CSV with all data
        csv_fname = "exciton_dipoles_debug.csv"
        with open(csv_fname, 'w') as f:
            f.write("state,pos_x,pos_y,pos_z,rot_deg,dipole_x,dipole_y,dipole_z,dipole_mag,dir_x,dir_y,dir_z\n")
            for i in range(n):
                p = poss[i]
                d = dipoles[i]
                dmag = np.linalg.norm(d)
                dhat = d / dmag if dmag > 1e-10 else np.zeros(3)
                rot_deg = rots[i] * 180 / np.pi
                f.write(f"{i},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{rot_deg:.1f},{d[0]:.6f},{d[1]:.6f},{d[2]:.6f},{dmag:.6f},{dhat[0]:.6f},{dhat[1]:.6f},{dhat[2]:.6f}\n")
        print(f"[DEBUG] Saved dipole data to: {csv_fname}")
        
        # Analyze why "perpendicular" couplings might not be exactly zero
        print("\n" + "="*70)
        print("ANALYSIS: WHY PERPENDICULAR COUPLINGS ARE NOT EXACTLY ZERO")
        print("="*70)
        print("The Coulomb coupling is NOT just dipole-dipole interaction!")
        print("Full Coulomb integral includes ALL multipole contributions:")
        print("  V = sum_ij q_i * q_j / |r_i - r_j|")
        print("")
        print("Breakdown of contributions:")
        for i in range(n):
            mpol = GU.evalMultipole(rhos[i], rot=latMats[i])
            q = mpol[0]  # monopole (should be ~0 for transition density)
            d = mpol[1:4]  # dipole
            Q = mpol[4:10]  # quadrupole
            print(f"State[{i}]: monopole={q:.2e}, |dipole|={np.linalg.norm(d):.4f}, |quadrupole|={np.linalg.norm(Q):.4f}")
        print("")
        print("Non-zero 'perpendicular' couplings arise from:")
        print("  1. Quadrupole-quadrupole interactions (always present)")
        print("  2. Dipole-quadrupole cross-terms")
        print("  3. Finite spatial extent of charge distributions")
        print("  4. Small numerical deviations from perfect 90 degrees")
        print("="*70 + "\n")
    
    H = assembleExcitonHamiltonian( rhos, poss, latMats, Ediags, byCenter=byCenter, dipoles=dipoles )
    if hackHfunc is not None: 
        hackHfunc( H )

    if _exc_report_sij_multipoles:
        coulomb_const = 14.3996
        print("\n" + "="*120)
        print("PAIR MULTIPOLE DECOMPOSITION (approx; from centered dipole/quadrupole of each transition density)")
        print("="*120)
        
        # First: show per-state dipole analysis with off-axis components
        print("\n--- PER-STATE DIPOLE ANALYSIS (off-axis components from cube file) ---")
        for i in range(n):
            mu_i = np.array(_mp_mus[i])
            mu_mag = np.linalg.norm(mu_i)
            main_idx = np.argmax(np.abs(mu_i))
            main_val = mu_i[main_idx]
            axis_names = ['X', 'Y', 'Z']
            print(f"  State {i}: μ = ({mu_i[0]:+.4f}, {mu_i[1]:+.4f}, {mu_i[2]:+.4f})  |μ| = {mu_mag:.4f}")
            print(f"           Main axis: {axis_names[main_idx]} = {main_val:+.4f}")
            off_axis_info = []
            for k in range(3):
                if k != main_idx and abs(mu_i[k]) > 1e-10:
                    pct = 100 * abs(mu_i[k]) / abs(main_val)
                    off_axis_info.append(f"{axis_names[k]}={mu_i[k]:+.4f} ({pct:.2f}%)")
            if off_axis_info:
                print(f"           Off-axis:  {', '.join(off_axis_info)}  <-- from cube file, causes Vdd≠0 for ⊥ dipoles!")
            print()
        
        # Pair analysis with detailed Vdd breakdown
        print("--- PAIR COUPLING ANALYSIS ---")
        print(f"{'pair':6s} | {'angle':7s} | {'μi·μj':10s} | {'μi·R̂':8s} | {'μj·R̂':8s} | {'(μi·R̂)(μj·R̂)':14s} | {'Vdd(meV)':10s} | {'H_ij(meV)':10s}")
        print("-"*120)
        for i in range(n):
            for j in range(i):
                R = np.array(_mp_r0s[i]) - np.array(_mp_r0s[j])
                r = float(np.linalg.norm(R))
                if r < 1e-12:
                    continue
                Rhat = R / r
                mu_i = np.array(_mp_mus[i])
                mu_j = np.array(_mp_mus[j])
                
                # Compute angle between dipoles
                mi_mag = np.linalg.norm(mu_i)
                mj_mag = np.linalg.norm(mu_j)
                if mi_mag > 1e-10 and mj_mag > 1e-10:
                    cos_ang = np.dot(mu_i, mu_j) / (mi_mag * mj_mag)
                    angle_deg = np.arccos(np.clip(cos_ang, -1, 1)) * 180 / np.pi
                else:
                    angle_deg = 0.0
                
                # Compute Vdd terms
                mu_i_dot_mu_j = np.dot(mu_i, mu_j)
                mu_i_dot_Rhat = np.dot(mu_i, Rhat)
                mu_j_dot_Rhat = np.dot(mu_j, Rhat)
                cross_term = mu_i_dot_Rhat * mu_j_dot_Rhat
                
                vdd = _Vdd(mu_i, mu_j, R) * coulomb_const
                vmuq = (_VmuQ(mu_i, _mp_Qs[j], R) + _VmuQ(mu_j, _mp_Qs[i], -R)) * coulomb_const
                
                # Flag perpendicular pairs with non-zero Vdd
                perp_flag = ""
                if 85 < angle_deg < 95 and abs(vdd*1000) > 0.01:
                    perp_flag = " <-- ⊥ but Vdd≠0!"
                
                print(f"{i:02d},{j:02d}  | {angle_deg:6.1f}° | {mu_i_dot_mu_j:+10.4f} | {mu_i_dot_Rhat:+8.4f} | {mu_j_dot_Rhat:+8.4f} | {cross_term:+14.6f} | {vdd*1000:+10.3f} | {H[i,j]*1000:+10.3f}{perp_flag}")
        
        print("-"*120)
        print("NOTE: For Vdd=0, you need BOTH: (1) μi·μj=0 (perpendicular) AND (2) (μi·R̂)(μj·R̂)=0")
        print("      The off-axis dipole components (from cube file asymmetry) make condition (2) fail!")
        print("="*120 + "\n")

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

