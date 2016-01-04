#!/usr/bin/env python

import numpy as np
   
def getSampleDimensions(lvec):
	'returns lvec without the first row'
	return np.matrix(lvec[1:])
    
def getSize(inp_axis, dims, sampleSize):
	'returns size of data set in dimension inp_axis \
	together with the length element in the given dimension'
	axes = {'x':0, 'y':1, 'z':2} # !!!
	if inp_axis in axes.keys(): axis = axes[inp_axis]
	size = np.linalg.norm(sampleSize[axis])   
	return size, size/(dims[axis] - 1)

def getMGrid(dims, dd):
	'returns coordinate arrays X, Y, Z'
	(dx, dy, dz) = dd
	nDim = [dims[2], dims[1], dims[0]]
	XYZ = np.mgrid[0:nDim[0],0:nDim[1],0:nDim[2]].astype(float)
	X = dx*np.roll( XYZ[2] - nDim[2]/2 -1, nDim[2]/2 , axis=2)
	Y = dy*np.roll( XYZ[1] - nDim[1]/2 -1, nDim[1]/2 , axis=1)
	Z = dz*np.roll( XYZ[0] - nDim[0]/2 -1, nDim[0]/2 , axis=0)
	return X, Y, Z

def getSphericalHarmonic( X, Y, Z, kind='dz2' ):
	# TODO: renormalization should be probaby here
	if    kind=='s':
		return 1.0
	# p-functions
	elif  kind=='px':
		return Z
	elif  kind=='py':
		return Y
	elif  kind=='pz':
		return Z
	# d-functions
	if    kind=='dz2' :
		return X**2 + Y**2 - 2*Z**2
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
	
def getProbeDensity(sampleSize, X, Y, Z, sigma, dd, multipole_dict=None ):
	'returns probe particle potential'
	mat = getNormalizedBasisMatrix(sampleSize).getT()
	rx = X*mat[0, 0] + Y*mat[0, 1] + Z*mat[0, 2]
	ry = X*mat[1, 0] + Y*mat[1, 1] + Z*mat[1, 2]
	rz = X*mat[2, 0] + Y*mat[2, 1] + Z*mat[2, 2]
	rquad  = rx**2 + ry**2 + rz**2
	radial       = np.exp( -rquad/( sigma**2 ) )
	radial_renom = np.sum(radial)*np.abs(np.linalg.det(mat))*dd[0]*dd[1]*dd[2]  # TODO analytical renormalization may save some time ?
	radial      /= radial_renom
	if multipole_dict is not None:	# multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
		rho = np.zeros( shape(radial) )
		for kind, coef in multipole_dict.iteritems():
			rho += radial * coef * getSphericalHarmonic( X, Y, Z, kind=kind )    # TODO renormalization should be probaby inside getSphericalHarmonic if possible ?
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
	print 'basis transformation matrix:'.rjust(first_col)
	print 'sampleSize = \n', sampleSize
	print 'Lmat = \n', getNormalizedBasisMatrix(sampleSize)
	print 'number of data points:'.rjust(first_col), ' dims'.rjust(sec_col), \
	' = %s' % list(dims)
	print 'specimen size:'.rjust(first_col), '(xsize, ysize, zsize)'.rjust(sec_col), \
	' = (%s, %s, %s)' % (xsize, ysize, zsize)
	print 'elementary lengths:'.rjust(first_col), '(dx, dy, dz)'.rjust(sec_col), \
	' = (%.5f, %.5f, %.5f)' % dd
	print 'V potential:'.rjust(first_col), '(max, min)'.rjust(sec_col), \
	' = (%s, %s)' % (V.max(), V.min())
	print ''.rjust(first_col), 'V.shape'.rjust(sec_col), ' = %s' % list(V.shape)
	print 'probe potential:'.rjust(first_col), '(max, min)'.rjust(sec_col), \
	' = (%s, %s)' % (rho.max(), rho.min())
	print ''.rjust(first_col), 'rho.shape'.rjust(sec_col), ' = %s' % list(rho.shape)
    
def exportPotential(rho, rho_data='rho_data'):
	filerho = open(rho_data, 'w')
	dimRho = rho.shape
	filerho.write(str(dimRho[0]) + " " + str(dimRho[1]) + " " + str(dimRho[2]) + '\n')
	for line in rho.flat:
		filerho.write("%s \n" % line)
		#filerho.write(rho)
	filerho.close()

def potential2forces( V, lvec, nDim, sigma = 1.0 ):
	print '--- Preprocessing ---'
	sampleSize = getSampleDimensions( lvec )
	dims = (nDim[2], nDim[1], nDim[0])
	xsize, dx = getSize('x', dims, sampleSize)
	ysize, dy = getSize('y', dims, sampleSize)
	zsize, dz = getSize('z', dims, sampleSize)
	dd = (dx, dy, dz)
	X, Y, Z = getMGrid(dims, dd)
	print '--- Get Probe Density ---'
	rho = getProbeDensity(sampleSize, X, Y, Z, sigma, dd)
	print '--- Get Forces ---'
	Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)
	print 'Fx.max(), Fx.min() = ', Fx.max(), Fx.min()
	return Fx,Fy,Fz


