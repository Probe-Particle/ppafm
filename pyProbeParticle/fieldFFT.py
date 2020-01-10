#!/usr/bin/env python

import numpy as np
from . import GridUtils
   
def getSampleDimensions(lvec):
    'returns lvec without the first row'
    return np.matrix(lvec[1:])
    
def getSize(inp_axis, dims, sampleSize):
    'returns size of data set in dimension inp_axis \
    together with the length element in the given dimension'
    axes = {'x':0, 'y':1, 'z':2} # !!!
    if inp_axis in list(axes.keys()): axis = axes[inp_axis]
    size = np.linalg.norm(sampleSize[axis])   
    return size, size/(dims[axis] - 1)

def getMGrid(dims, dd):
    'returns coordinate arrays X, Y, Z'
    (dx, dy, dz) = dd
    nDim = [dims[2], dims[1], dims[0]]
    XYZ = np.mgrid[0:nDim[0],0:nDim[1],0:nDim[2]].astype(float)
    
    xshift = nDim[2]/2;  xshift_ = xshift;
    yshift = nDim[1]/2;  yshift_ = yshift;
    zshift = nDim[0]/2;  zshift_ = zshift;
    
    if( nDim[2]%2 != 0 ):  xshift_ += 1.0
    if( nDim[1]%2 != 0 ):  yshift_ += 1.0
    if( nDim[0]%2 != 0 ):  zshift_ += 1.0
    
    X = dx*np.roll( XYZ[2] - xshift_, int(xshift), axis=2)
    Y = dy*np.roll( XYZ[1] - yshift_, int(yshift), axis=1)
    Z = dz*np.roll( XYZ[0] - zshift_, int(zshift), axis=0)
    return X, Y, Z

def getSphericalHarmonic( X, Y, Z, kind='dz2' ):
    # TODO: renormalization should be probaby here
    if    kind=='s':
        print('Spherical harmonic: s')
        return 1.0
    # p-functions
    elif  kind=='px':
        print('Spherical harmonic: px')
        return X
    elif  kind=='py':
        print('Spherical harmonic: py')
        return Y
    elif  kind=='pz':
        print('Spherical harmonic: pz')
        return Z
    # d-functions
    if    kind=='dz2' :
        print('Spherical harmonic: dz2')
        return 0.25*(2*Z**2 - X**2 - Y**2) #quadrupole normalized to get 3 times the quadrpole in the standard (cartesian) tensor normalization of Qzz. Also, 3D integral of rho_dz2(x,y,z)*(z/sigma)**2 gives 1 in the normalization use here.
    elif    kind=='dx2' :
        print('Spherical harmonic: dx2')
        return 0.25*(2*X**2 - Y**2 - Z**2)
    elif    kind=='dy2' :
        print('Spherical harmonic: dy2')
        return 0.25*(2*Y**2 - X**2 - Z**2)
    elif    kind=='dxy' :
        print('Spherical harmonic: dxy')
        return X*Y
    elif    kind=='dxz' :
        print('Spherical harmonic: dxz')
        return X*Z
    elif    kind=='dyz' :
        print('Spherical harmonic: dyz')
        return Y*Z
    else:
        return 0.0

'''
def getProbeDensity(sampleSize, X, Y, Z, sigma, dd ):
    'returns probe particle potential'
    mat = getNormalizedBasisMatrix(sampleSize).getT()
    rx = X*mat[0, 0] + Y*mat[0, 1] + Z*mat[0, 2]
    ry = X*mat[1, 0] + Y*mat[1, 1] + Z*mat[1, 2]
    rz = X*mat[2, 0] + Y*mat[2, 1] + Z*mat[2, 2]
    rquad = rx**2 + ry**2 + rz**2
    rho = np.exp( -(rquad)/(1*sigma**2) )
    rho_sum = np.sum(rho)*np.abs(np.linalg.det(mat))*dd[0]*dd[1]*dd[2]
    rho = rho / rho_sum
    return rho
'''
    
def getProbeDensity( sampleSize, X, Y, Z, dd, sigma=0.7, multipole_dict=None ):
    'returns probe particle potential'
    mat = getNormalizedBasisMatrix(sampleSize).getT()
    rx = X*mat[0, 0] + Y*mat[0, 1] + Z*mat[0, 2]
    ry = X*mat[1, 0] + Y*mat[1, 1] + Z*mat[1, 2]
    rz = X*mat[2, 0] + Y*mat[2, 1] + Z*mat[2, 2]
    rquad  = rx**2 + ry**2 + rz**2
    radial       = np.exp( -(rquad)/(2*sigma**2) )
    radial_renom = np.sum(radial)*np.abs(np.linalg.det(mat))*dd[0]*dd[1]*dd[2]  # TODO analytical renormalization may save some time ?
    radial      /= radial_renom
    if multipole_dict is not None:    # multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        rho = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.items():
            rho += radial * coef * getSphericalHarmonic( rx/sigma, ry/sigma, rz/sigma, kind=kind )
    else:
        rho = radial
    return rho
   
def getSkewNormalBasis(sampleSize):
    'returns normalized basis vectors pertaining to the skew basis'
    ax = sampleSize[0]/(np.linalg.norm(sampleSize[0]))
    ay = sampleSize[1]/(np.linalg.norm(sampleSize[1]))
    az = sampleSize[2]/(np.linalg.norm(sampleSize[2]))
    ax = np.copy(ax.flat)
    ay = np.copy(ay.flat)
    az = np.copy(az.flat)
    return ax, ay, az

def getForces(V, rho, sampleSize, dims, dd, X, Y, Z):
    'returns forces for all axes, calculation performed \
    in orthogonal coordinates, but results are expressed in skew coord.'
    LmatInv = getNormalizedBasisMatrix(sampleSize).getI()
    detLmatInv = np.abs(np.linalg.det(LmatInv))
    VFFT = np.fft.fftn(V)
    rhoFFT = np.fft.fftn(rho) 
    derConvFFT = 2*(np.pi)*1j*VFFT*rhoFFT
    # det(Lmat) = 1 / det(LmatInv) !!!
    derConvFFT = derConvFFT * (dd[0]*dd[1]*dd[2]) / (detLmatInv)   
    # dd = (dx, dy, dz) !!!
    dzetax = 1/(dims[0]*dd[0]*dd[0])
    dzetay = 1/(dims[1]*dd[1]*dd[1])
    dzetaz = 1/(dims[2]*dd[2]*dd[2])
    zeta = [0, 0, 0]
    for axis in range(3):
        zeta[axis]  = LmatInv[axis,0]*dzetax*X 
        zeta[axis] += LmatInv[axis,1]*dzetay*Y
        zeta[axis] += LmatInv[axis,2]*dzetaz*Z   
    forceSkewFFTx = zeta[0]*derConvFFT
    forceSkewFFTy = zeta[1]*derConvFFT
    forceSkewFFTz = zeta[2]*derConvFFT
    forceSkewx = np.real(np.fft.ifftn(forceSkewFFTx))
    forceSkewy = np.real(np.fft.ifftn(forceSkewFFTy))
    forceSkewz = np.real(np.fft.ifftn(forceSkewFFTz))
    return forceSkewx, forceSkewy, forceSkewz    

def getNormalizedBasisMatrix(sampleSize):
    'returns transformation matrix from OG basis to skew basis'
    ax, ay, az = getSkewNormalBasis(sampleSize)
    Lmat = [ax, ay, az]
    return np.matrix(Lmat)
  
def printMetadata(sampleSize, dims, dd, xsize, ysize, zsize, V, rho):
    first_col = 30    
    sec_col = 25
    print('basis transformation matrix:'.rjust(first_col))
    print('sampleSize = \n', sampleSize)
    print('Lmat = \n', getNormalizedBasisMatrix(sampleSize))
    print('number of data points:'.rjust(first_col), ' dims'.rjust(sec_col), \
    ' = %s' % list(dims))
    print('specimen size:'.rjust(first_col), '(xsize, ysize, zsize)'.rjust(sec_col), \
    ' = (%s, %s, %s)' % (xsize, ysize, zsize))
    print('elementary lengths:'.rjust(first_col), '(dx, dy, dz)'.rjust(sec_col), \
    ' = (%.5f, %.5f, %.5f)' % dd)
    print('V potential:'.rjust(first_col), '(max, min)'.rjust(sec_col), \
    ' = (%s, %s)' % (V.max(), V.min()))
    print(''.rjust(first_col), 'V.shape'.rjust(sec_col), ' = %s' % list(V.shape))
    print('probe potential:'.rjust(first_col), '(max, min)'.rjust(sec_col), \
    ' = (%s, %s)' % (rho.max(), rho.min()))
    print(''.rjust(first_col), 'rho.shape'.rjust(sec_col), ' = %s' % list(rho.shape))
    
def exportPotential(rho, rho_data='rho_data'):
    filerho = open(rho_data, 'w')
    dimRho = rho.shape
    filerho.write(str(dimRho[0]) + " " + str(dimRho[1]) + " " + str(dimRho[2]) + '\n')
    for line in rho.flat:
        filerho.write("%s \n" % line)
        #filerho.write(rho)
    filerho.close()

def potential2forces( V, lvec, nDim, sigma = 0.7, rho=None, multipole=None):
    print('--- Preprocessing ---')
    sampleSize = getSampleDimensions( lvec )
    dims = (nDim[2], nDim[1], nDim[0])
    xsize, dx = getSize('x', dims, sampleSize)
    ysize, dy = getSize('y', dims, sampleSize)
    zsize, dz = getSize('z', dims, sampleSize)
    dd = (dx, dy, dz)
    X, Y, Z = getMGrid(dims, dd)
    if rho == None:
        print('--- Get Probe Density ---')
        rho = getProbeDensity(sampleSize, X, Y, Z, dd, sigma=sigma, multipole_dict=multipole)
        #GridUtils.saveXSF("rho_tip.xsf", rho, lvec)
    else:
        rho[:,:,:] = rho[::-1,::-1,::-1].copy()
    print('--- Get Forces ---')
    Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)
    print('Fz.max(), Fz.min() = ', Fz.max(), Fz.min())
    return Fx,Fy,Fz

def Average_surf( Val_surf, W_surf, W_tip ):
    '''
                Int_r Val_surf(r+R)  W_tip(r) W_sample(r+R)     W_tip) * (Val_surf W_sample)
     <F>(R) = -----------------------------------------  = -----------------------------; where * means convolution
                Int_r W_tip(r) W_sample(r+R)                     W_tip * W_sample
    '''
    print("Forward FFT ") 
    kE_tip   = np.fft.fftn( W_tip[::-1,::-1,::-1]    )  # W_tip
    kE_surf  = np.fft.fftn( W_surf   )                  # W_sample
    kFE_surf = np.fft.fftn( W_surf * Val_surf  )        # (Val_surf W_surf)

    del Val_surf; del W_surf; del W_tip

    kE  = kE_tip *  kE_surf
    kFE = kE_tip * kFE_surf

    del kE_tip; del kE_surf; del kFE_surf

    print("Backward FFT ") 

    E  = np.fft.ifftn(kE)
    FE = np.fft.ifftn(kFE)

    del kE; del kFE
    return (FE/E).real;

def Average_tip( Val_tip, W_surf, W_tip ):
    '''
                Int_r Val_tip(r)  W_tip(r) W_sample(r+R)    (Val_tip W_tip) * W_sample
     <F>(R) = -----------------------------------------  = -----------------------------; where * means convolution
                Int_r W_surf(r) W_sample(r+R)                     W_tip * W_sample
    '''
    print("Forward FFT ") 
    kE_tip   = np.fft.fftn( W_tip[::-1,::-1,::-1]    )                               # W_tip
    kE_surf  = np.fft.fftn( W_surf   )                                               # W_sample
    kFE_tip  = np.fft.fftn( W_tip[::-1,::-1,::-1] * (-1)*Val_tip[::-1,::-1,::-1]  )  # (Val_tip W_tip)

    del Val_tip; del W_surf; del W_tip

    kE  = kE_tip  *  kE_surf
    kFE = kE_surf *  kFE_tip

    del kE_tip; del kE_surf; del kFE_tip

    print("Backward FFT ") 

    E  = np.fft.ifftn(kE)
    FE = np.fft.ifftn(kFE)

    del kE; del kFE
    return (FE/E).real;

