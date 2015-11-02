#!/usr/bin/python
import numpy as np
#import matplotlib.pyplot as plt
#from xsfutil import *
#from STHM_Utils import *
import GridUtils as GU
from libFFTfin import *

# --- initialization ---

sigma  = 1.0 # [ Angstroem ] 

print '--- Data Loading ---'

V, lvec, nDim, head = GU.loadXSF('LOCPOT.xsf')
print '--- Preprocessing ---'

sampleSize = getSampleDimensions(lvec)
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

print "--- Saving ---"

GU.saveXSF('FFel_x.xsf', Fx, lvec, head)
GU.saveXSF('FFel_y.xsf' , Fy, lvec, head)
GU.saveXSF('FFel_z.xsf' , Fz, lvec, head)
    
#show()

