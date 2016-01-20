#!/usr/bin/python

from pylab import *
import GridUtils as GU
from STHM_Utils import *
from os import *

sigma  = 1.0 # [ Angstroem ] 
V0     = 2.7 # [ eV ]



V = None
'''
if path.isfile("LOCPOT.npy"):
	print "loading binary"
	V = np.load("LOCPOT.npy")
	nDim = shape(V)
	print "nDim: " , nDim
else:
	print "loading xsf (ASCII)"
	V,lvec, nDim, head = loadXSF('LOCPOT.xsf')
	V = V[50:70]
	np.save("LOCPOT", V )
'''

print " loading: "
V,lvec, nDim, head = GU.loadCUBE('hartree.cube')

print nDim
print lvec

zsz = lvec[3,2]
ysz = lvec[2,1]
xsz = lvec[1,0]

print "xsz,ysz,zsz: ", xsz,ysz,zsz

dz     = zsz/(nDim[0]-1)
dy     = ysz/(nDim[1]-1)
dx     = xsz/(nDim[2]-1)
extent = ( -xsz/2, xsz/2,  -ysz/2, ysz/2 )

print "dz,dy,dx: ",dz,dy,dx 
ilist = range(  40,50,2) 
#ilist = range(  0,20,2) 

V = V - V0

XYZ = mgrid[0:nDim[0],0:nDim[1],0:nDim[2]].astype(float)

X = dx*roll( XYZ[2] - nDim[2]/2 -1, nDim[2]/2 , axis=2)
Y = dy*roll( XYZ[1] - nDim[1]/2 -1, nDim[1]/2 , axis=1)
Z = dz*roll( XYZ[0] - nDim[0]/2 -1, nDim[0]/2 , axis=0)

rho = exp( -( X**2 + Y**2 + Z**2 )/(sigma**2) )
rho = rho/sum(rho)

imshow( rho[0] ,extent=extent)

print "rho max min :",rho.max(), rho.min()
print "V   max min :",V.max(), V.min()

print " FFT "
Vw   = fftn(V)
rhow = fftn(rho)

print " iFFT "
Vo  = Vw*rhow
V   = real( ifftn(Vo)   )
Fx  = real( ifftn(Vo*X*1j*2*pi/(dx*dx*nDim[2]) ) )
Fy  = real( ifftn(Vo*Y*1j*2*pi/(dy*dy*nDim[1]) ) )
Fz  = real( ifftn(Vo*Z*1j*2*pi/(dz*dz*nDim[0]) ) )

Fx_check = (V[:,:,1:] - V[:,:,:-1])/dx
Fx_diff  = 0.5*(Fx[:,:,1:]+Fx[:,:,:-1]) - Fx_check

print "max :",Fx.max(), Fx_check.max(), Fx_diff.max()
print "min :",Fx.min(), Fx_check.min(), Fx_diff.min()


print " plotting"

plotWithAtoms( Fx, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True  )
plotWithAtoms( Fy, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=False )
plotWithAtoms( Fz, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=False )

print " saving "
GU.saveXSF('Fx.xsf', Fx, lvec, head)
GU.saveXSF('Fy.xsf', Fy, lvec, head)
GU.saveXSF('Fz.xsf', Fz, lvec, head)

#plotWithAtoms( Fx_check, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )
#plotWithAtoms( Fx_diff, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )

show()

