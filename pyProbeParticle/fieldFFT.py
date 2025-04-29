#!/usr/bin/env python

import numpy as np

import gc

from . import GridUtils as GU

verbose = 0


def fieldInfo( F, label="FieldInfo: min max av: " ):
    print(label, np.min(F), np.max(F), np.average(F))
   
def getSampleDimensions(lvec):
    'returns lvec without the first row'
    return np.matrix(lvec[1:])
    
def getSize(inp_axis, dims, sampleSize):
    'returns size of data set in dimension inp_axis  together with the length element in the given dimension'
    axes = {'x':0, 'y':1, 'z':2} # !!!
    if inp_axis in list(axes.keys()): axis = axes[inp_axis]
    size = np.linalg.norm(sampleSize[axis])   
    #return size, size/(dims[axis] - 1)  # This was walid befor we cut replicated last slice in .xsf
    return size, size/dims[axis]

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
    
    X = dx*np.roll( XYZ[2] - xshift_, xshift, axis=2)
    Y = dy*np.roll( XYZ[1] - yshift_, yshift, axis=1)
    Z = dz*np.roll( XYZ[0] - zshift_, zshift, axis=0)
    return X, Y, Z

def rotZX( Z, X, tilt = 0.0 ):
    ca = np.cos(tilt)
    sa = np.sin(tilt)
    return (ca*Z -sa*X), (sa*Z +ca*X) 

def getSphericalHarmonic( X, Y, Z, kind='dz2', tilt = 0.0 ):
    Z,X = rotZX(Z,X, tilt=tilt)
    invrR = 1/np.sqrt( X**2 + Y**2 + Z**2 + 1e-8 )
    # TODO: renormalization should be probaby here
    if    kind=='s':
        if(verbose>0): print('Spherical harmonic: s')
        return 1.0
    # p-functions
    elif  kind=='px':
        if(verbose>0): print('Spherical harmonic: px')
        return X*invrR
    elif  kind=='py':
        if(verbose>0): print('Spherical harmonic: py')
        return Y*invrR
    elif  kind=='pz':
        if(verbose>0): print('Spherical harmonic: pz')
        return Z*invrR
    # d-functions
    elif    kind=='dz2' :
        if(verbose>0): print('Spherical harmonic: dz2')
        return 0.25*(2*Z**2 - X**2 - Y**2)*(invrR**2) #quadrupole normalized to get 3 times the quadrpole in the standard (cartesian) tensor normalization of Qzz. Also, 3D integral of rho_dz2(x,y,z)*(z/sigma)**2 gives 1 in the normalization use here.
    elif    kind=='dx2' :
        if(verbose>0): print('Spherical harmonic: dz2')
        return 0.25*(2*X**2 - Y**2 - Z**2)*(invrR**2) 
    elif    kind=='dy2' :
        if(verbose>0): print('Spherical harmonic: dz2')
        return 0.25*(2*Y**2 - X**2 - Z**2)*(invrR**2) 
    elif    kind=='dx2' :
        if(verbose>0): print('Spherical harmonic: dx2')
        return 0.25*(2*X**2 - Y**2 - Z**2)*(invrR**2)
    elif    kind=='dy2' :
        if(verbose>0): print('Spherical harmonic: dy2')
        return 0.25*(2*Y**2 - X**2 - Z**2)*(invrR**2)
    elif    kind=='dxy' :
        if(verbose>0): print('Spherical harmonic: dxy')
        return X*Y*(invrR**2)
    elif    kind=='dxz' :
        if(verbose>0): print('Spherical harmonic: dxz')
        return X*Z*(invrR**2)
    elif    kind=='dyz' :
        if(verbose>0): print('Spherical harmonic: dyz')
        return Y*Z*(invrR**2)
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

def getSampleSizes( lvec, nDim ):
    sampleSize = getSampleDimensions( lvec )
    dims = (nDim[2], nDim[1], nDim[0])
    xsize, dx = getSize('x', dims, sampleSize)
    ysize, dy = getSize('y', dims, sampleSize)
    zsize, dz = getSize('z', dims, sampleSize)
    dd = (dx, dy, dz)
    return sampleSize, dd, dims


def getMultiploleGrid( lvec, nDim, sigma=0.7, multipole_dict=None, tilt=0.0 ):
    '''
    sampleSize = getSampleDimensions( lvec )
    dims = (nDim[2], nDim[1], nDim[0])
    xsize, dx = getSize('x', dims, sampleSize)
    ysize, dy = getSize('y', dims, sampleSize)
    zsize, dz = getSize('z', dims, sampleSize)
    dd = (dx, dy, dz)
    '''
    sampleSize, dd, dims = getSampleSizes( lvec, nDim )
    X, Y, Z = getMGrid(dims, dd)
    rho = getMultiplole( sampleSize, X, Y, Z, dd, sigma=sigma, multipole_dict=multipole_dict, tilt=tilt )
    return rho

def getMultiplole( sampleSize, X, Y, Z, dd, sigma=0.7, multipole_dict=None, tilt=0.0 ):
    'returns probe particle potential'
    if(verbose>0): print("sigma: ", sigma); #exit()
    mat = getNormalizedBasisMatrix(sampleSize).getT()
    rx = X*mat[0, 0] + Y*mat[0, 1] + Z*mat[0, 2]
    ry = X*mat[1, 0] + Y*mat[1, 1] + Z*mat[1, 2]
    rz = X*mat[2, 0] + Y*mat[2, 1] + Z*mat[2, 2]
    r2  = rx**2 + ry**2 + rz**2
    radial       = 1/( sigma**2  +  r2  )
    #radial_renom = np.sum(radial)*np.abs(np.linalg.det(mat))*dd[0]*dd[1]*dd[2]  # TODO analytical renormalization may save some time ?
    #radial      /= radial_renom
    invR = 1/np.sqrt(r2)
    rx*=invR
    ry*=invR
    ry*=invR
    if multipole_dict is not None:    # multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        rho = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.items():
            rho += radial * coef * getSphericalHarmonic( rx, ry, rz, kind=kind, tilt=tilt )
    else:
        rho = radial
    return rho


def getProbeDensity( sampleSize, X, Y, Z, dd, sigma=0.7, multipole_dict=None, tilt=0.0 ):
    'returns probe particle potential'
    if(verbose>0): print("sigma: ", sigma); #exit()
    mat = getNormalizedBasisMatrix(sampleSize).getT()
    
    rx = X*mat[0, 0] + Y*mat[0, 1] + Z*mat[0, 2]
    ry = X*mat[1, 0] + Y*mat[1, 1] + Z*mat[1, 2]
    rz = X*mat[2, 0] + Y*mat[2, 1] + Z*mat[2, 2]
    rquad  = rx**2 + ry**2 + rz**2
    
    #rquad = X**2 + Y**2 + Z**2
    radial       = np.exp( -(rquad)/(2*sigma**2) )
    radial_renom = np.sum(radial)*np.abs(np.linalg.det(mat))*dd[0]*dd[1]*dd[2]  # TODO analytical renormalization may save some time ?
    radial      /= radial_renom
    if multipole_dict is not None:    # multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        rho = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.items():
            #rho += radial * coef * getSphericalHarmonic( X/sigma, Y/sigma, Z/sigma, kind=kind, tilt=tilt )
            rho += radial * coef * getSphericalHarmonic( rx/sigma, ry/sigma, rz/sigma, kind=kind, tilt=tilt )
    else:
        rho = radial
    return rho

def addCoreDensity( rho, val, x,y,z, sigma=0.1 ):
    rquad        = x**2 + y**2 + z**2
    radial       = np.exp( -rquad/(2*sigma**2) )
    radial_renom = np.sum(radial)  # TODO analytical renormalization may save some time ?
    radial      *= val/float(radial_renom)
    print("radial.sum()", radial.sum(), val)
    rho         += radial

def addCoreDensities( atoms, valElDict, rho, lvec, sigma=0.1 ):
    '''
    sampleSize = getSampleDimensions( lvec )
    nDim = rho.shape 
    dims = (nDim[2], nDim[1], nDim[0])
    xsize, dx = getSize('x', dims, sampleSize)
    ysize, dy = getSize('y', dims, sampleSize)
    zsize, dz = getSize('z', dims, sampleSize)
    dd = (dx, dy, dz)
    '''
    sampleSize, dd, dims = getSampleSizes( lvec, nDim )

    X, Y, Z = getMGrid(dims, dd)

    mat = getNormalizedBasisMatrix(sampleSize).getT()
    
    dV = np.abs(np.linalg.det(mat))*dd[0]*dd[1]*dd[2]
    rx = X*mat[0, 0] + Y*mat[0, 1] + Z*mat[0, 2]
    ry = X*mat[1, 0] + Y*mat[1, 1] + Z*mat[1, 2]
    rz = X*mat[2, 0] + Y*mat[2, 1] + Z*mat[2, 2]

    for ia in range(len(atoms[0])):
        nel = valElDict[atoms[0,ia]]
        xyz = atoms[1:4,ia]
        addCoreDensity( rho, -nel/dV, rx-xyz[0],ry-xyz[1],rz-xyz[2], sigma=sigma )

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

def conv3DFFT( F2, F1 ):
    return  np.fft.ifftn( np.fft.fftn(F1) * np.fft.fftn(F2) )

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
    if(verbose>0): print("Ftrans :   ", detLmatInv, zeta[0].sum(), zeta[1].sum(), zeta[2].sum())
    if(verbose>0): print("derConvFFT ", derConvFFT.sum(),derConvFFT.min(),derConvFFT.max())
    forceSkewFFTx = zeta[0]*derConvFFT
    forceSkewFFTy = zeta[1]*derConvFFT
    forceSkewFFTz = zeta[2]*derConvFFT
    forceSkewx = np.real(np.fft.ifftn(forceSkewFFTx))
    forceSkewy = np.real(np.fft.ifftn(forceSkewFFTy))
    forceSkewz = np.real(np.fft.ifftn(forceSkewFFTz))
    return forceSkewx, forceSkewy, forceSkewz    


def getForceTransform(sampleSize, dims, dd, X, Y, Z ):
    LmatInv     = getNormalizedBasisMatrix(sampleSize).getI()
    detLmatInv  = np.abs(np.linalg.det(LmatInv))
    dzetax = 1/(dims[0]*dd[0]*dd[0])
    dzetay = 1/(dims[1]*dd[1]*dd[1])
    dzetaz = 1/(dims[2]*dd[2]*dd[2])
    zeta = [0, 0, 0]
    for axis in range(3):
        zeta[axis]  = LmatInv[axis,0]*dzetax*X 
        zeta[axis] += LmatInv[axis,1]*dzetay*Y
        zeta[axis] += LmatInv[axis,2]*dzetaz*Z  
    if(verbose>0): print("Ftrans :   ", zeta[0].sum(), zeta[1].sum(), zeta[2].sum())
    return zeta[0],zeta[1],zeta[2], detLmatInv

'''
def getForceEnergy(V, rho, sampleSize, dims, dd, X, Y, Z):
    'returns forces for all axes, calculation performed \
    in orthogonal coordinates, but results are expressed in skew coord.'
    zeta, detLmatInv = getForceTransform(sampleSize, dims, dd)
    VFFT          = np.fft.fftn(V)
    rhoFFT        = np.fft.fftn(rho) 
    convFFT       = VFFT*rhoFFT;          del VFFT,rhoFFT
    E             = forceSkewx = np.real(np.fft.ifftn(convFFT))
    derConvFFT    = 2*(np.pi)*1j*convFFT * (dd[0]*dd[1]*dd[2]) / (detLmatInv)   
    forceSkewx = np.real(np.fft.ifftn(zeta[0]*derConvFFT))
    forceSkewy = np.real(np.fft.ifftn(zeta[1]*derConvFFT))
    forceSkewz = np.real(np.fft.ifftn(zeta[2]*derConvFFT))
    return forceSkewx, forceSkewy, forceSkewz, E     
'''

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
    print('number of data points:'.rjust(first_col), ' dims'.rjust(sec_col), ' = %s' % list(dims))
    print('specimen size:'.rjust(first_col), '(xsize, ysize, zsize)'.rjust(sec_col), ' = (%s, %s, %s)' % (xsize, ysize, zsize))
    print('elementary lengths:'.rjust(first_col), '(dx, dy, dz)'.rjust(sec_col), ' = (%.5f, %.5f, %.5f)' % dd)
    print('V potential:'.rjust(first_col), '(max, min)'.rjust(sec_col), ' = (%s, %s)' % (V.max(), V.min()))
    print(''.rjust(first_col), 'V.shape'.rjust(sec_col), ' = %s' % list(V.shape))
    print('probe potential:'.rjust(first_col), '(max, min)'.rjust(sec_col), ' = (%s, %s)' % (rho.max(), rho.min()))
    print(''.rjust(first_col), 'rho.shape'.rjust(sec_col), ' = %s' % list(rho.shape))

def exportPotential(rho, rho_data='rho_data'):
    filerho = open(rho_data, 'w')
    dimRho = rho.shape
    filerho.write(str(dimRho[0]) + " " + str(dimRho[1]) + " " + str(dimRho[2]) + '\n')
    for line in rho.flat:
        filerho.write("%s \n" % line)
        #filerho.write(rho)
    filerho.close()

def potential2forces( V, lvec, nDim, sigma = 0.7, rho=None, multipole=None, tilt=0.0 ):
    fieldInfo( V, label="fieldInfo V " )
    if(verbose>0): print('--- Preprocessing ---')
    '''
    sampleSize = getSampleDimensions( lvec )
    dims = (nDim[2], nDim[1], nDim[0])
    xsize, dx = getSize('x', dims, sampleSize)
    ysize, dy = getSize('y', dims, sampleSize)
    zsize, dz = getSize('z', dims, sampleSize)
    dd = (dx, dy, dz)
    '''
    sampleSize, dd, dims = getSampleSizes( lvec, nDim )
    X, Y, Z = getMGrid(dims, dd)
    fieldInfo( Z, label="fieldInfo Z " )
    if rho == None:
        if(verbose>0): print('--- Get Probe Density ---')
        rho = getProbeDensity(sampleSize, X, Y, Z, dd, sigma=sigma, multipole_dict=multipole, tilt=tilt)
    else:
        rho[:,:,:] = rho[::-1,::-1,::-1].copy()
    fieldInfo( rho, label="fieldInfo rho " )
    #GU.saveXSF( "DEBUG_rho.xsf", rho, lvec )
    if(verbose>0): print('--- Get Forces ---')
    Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)
    fieldInfo( Fz, label="fieldInfo Fz " )
    if(verbose>0): print('Fz.max(), Fz.min() = ', Fz.max(), Fz.min())
    return Fx,Fy,Fz


def potential2forces_mem( V, lvec, nDim, sigma = 0.7, rho=None, multipole=None, doForce=True, doPot=False, deleteV=True, tilt=0.0 ):
    print('--- Preprocessing ---')
    '''
    sampleSize = getSampleDimensions( lvec )
    dims = (nDim[2], nDim[1], nDim[0])
    xsize, dx = getSize('x', dims, sampleSize)
    ysize, dy = getSize('y', dims, sampleSize)
    zsize, dz = getSize('z', dims, sampleSize)
    dd = (dx, dy, dz)
    '''
    sampleSize, dd, dims = getSampleSizes( lvec, nDim )
    #potential2forces_mem: dims  (92, 91, 60)
    #potential2forces_mem: dims  (49.219764194907896, 48.690656801482945, 32.288778971493564)
    if(verbose>0): print("potential2forces_mem: dims ", dims) 
    if(verbose>0): print("potential2forces_mem: dims ", dd) 

    if(verbose>0): print('--- X, Y, Z ---')
    X, Y, Z = getMGrid(dims, dd)
    if rho is None:
        if(verbose>0): print('--- Get Probe Density ---')
        rho = getProbeDensity(sampleSize, X, Y, Z, dd, sigma=sigma, multipole_dict=multipole, tilt=tilt )
        '''
        NO NEED TO RENORMALIZE : This is already density
        print "rho.abs().sum() ", np.sum(np.abs(rho))
        scQ2dens = 1.0/GU.dens2Q_CHGCARxsf(rho,lvec)
        print " scQ2dens ", scQ2dens
        print "rho.abs().sum()*scQ2dens ", np.sum(np.abs(rho*scQ2dens))
        print "rho.abs().sum()/scQ2dens ", np.sum(np.abs(rho/scQ2dens))
        exit()
        '''
        GU.saveXSF( "rhoTip.xsf", rho, lvec )

    else:
        if(verbose>0): print("rho backward (rho[::-1,::-1,::-1]) ")
        rho[:,:,:] = rho[::-1,::-1,::-1].copy()
        #GU.saveXSF( "rho.xsf",   rho, lvec )
    if doForce:
        if(verbose>0): print('--- prepare Force transforms ---')
        zetaX,zetaY,zetaZ,detLmatInv = getForceTransform(sampleSize, dims, dd, X, Y, Z )
    del X,Y,Z
    E=None;Fx=None;Fy=None;Fz=None;
    if(verbose>0): print('--- forward FFT ---')
    #VFFT      =  
    #if deleteV: del V
    #rhoFFT    =    del rho
    #convFFT   = VFFT*rhoFFT;        del VFFT,rhoFFT
    gc.collect()
    convFFT    = np.fft.fftn(V) * np.fft.fftn(rho);
    if deleteV: del V
    gc.collect()
    #cRenorm = detLmatInv/(dd[0]*dd[1]*dd[2]); print " cRenorm ", cRenorm, 1/cRenorm, detLmatInv, dd[0], dd[1], dd[2]   #  det(invM)/(dx*dy*dz) = 1/( dx*dy*dz*det(M) ) =   N/( det(M)*det(M) ) 
    dV = (dd[0]*dd[1]*dd[2])/detLmatInv; print(" dV ", dV)
    if doPot:
        if(verbose>0): print('--- Get Potential ---')
        #E         = np.real(np.fft.ifftn(convFFT))
        #E          = np.real(np.fft.ifftn(convFFT))
        E         = np.real(np.fft.ifftn(convFFT)) * dV
        #E         = np.real(np.fft.ifftn(convFFT * (dd[0]*dd[1]*dd[2]) / (detLmatInv) ) )
    if doForce:
        if(verbose>0): print('--- Get Forces ---')
        convFFT  *= 2*np.pi*1j  *dV             # * (dd[0]*dd[1]*dd[2])/(detLmatInv)   
        if(verbose>0): print("derConvFFT ", convFFT.sum(),convFFT.min(),convFFT.max())
        Fx        = np.real(np.fft.ifftn(zetaX*convFFT)); del zetaX; gc.collect()
        Fy        = np.real(np.fft.ifftn(zetaY*convFFT)); del zetaY; gc.collect()
        Fz        = np.real(np.fft.ifftn(zetaZ*convFFT)); del zetaZ; gc.collect()
    if(verbose>0): print('Fz.max(), Fz.min() = ', Fz.max(), Fz.min())
    return Fx,Fy,Fz, E

def convolveFFT( F1, F2, lvec, nDim ):
    print('--- Preprocessing ---')
    #sampleSize = getSampleDimensions( lvec )
    #dims = (nDim[2], nDim[1], nDim[0])
    sampleSize, dd, dims = getSampleSizes( lvec, nDim )
    LmatInv     = getNormalizedBasisMatrix(sampleSize).getI()
    detLmatInv  = np.abs(np.linalg.det(LmatInv))
    F12w        = np.fft.fftn(F1) * np.fft.fftn(F2)
    gc.collect()
    dV          = (dd[0]*dd[1]*dd[2])/detLmatInv; print(" dV ", dV)
    F12         = np.real(np.fft.ifftn(F12w)) * dV
    return F12

def Average_surf( Val_surf, W_surf, W_tip ):
    '''
                Int_r Val_surf(r+R)  W_tip(r) W_sample(r+R)     W_tip) * (Val_surf W_sample)
     <F>(R) = -----------------------------------------  = -----------------------------; where * means convolution
                Int_r W_tip(r) W_sample(r+R)                     W_tip * W_sample
    '''
    if(verbose>0): print("Forward FFT ") 
    kE_tip   = np.fft.fftn( W_tip[::-1,::-1,::-1]    )  # W_tip
    kE_surf  = np.fft.fftn( W_surf   )                  # W_sample
    kFE_surf = np.fft.fftn( W_surf * Val_surf  )        # (Val_surf W_surf)

    del Val_surf; del W_surf; del W_tip

    kE  = kE_tip *  kE_surf
    kFE = kE_tip * kFE_surf

    del kE_tip; del kE_surf; del kFE_surf

    if(verbose>0): print("Backward FFT ") 

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
    if(verbose>0): print("Forward FFT ") 
    kE_tip   = np.fft.fftn( W_tip[::-1,::-1,::-1]    )                               # W_tip
    kE_surf  = np.fft.fftn( W_surf   )                                               # W_sample
    kFE_tip  = np.fft.fftn( W_tip[::-1,::-1,::-1] * (-1)*Val_tip[::-1,::-1,::-1]  )  # (Val_tip W_tip)

    del Val_tip; del W_surf; del W_tip

    kE  = kE_tip  *  kE_surf
    kFE = kE_surf *  kFE_tip

    del kE_tip; del kE_surf; del kFE_tip

    if(verbose>0): print("Backward FFT ") 

    E  = np.fft.ifftn(kE)
    FE = np.fft.ifftn(kFE)

    del kE; del kFE
    return (FE/E).real;

