import numpy as np
import matplotlib.pyplot as plt
from xsfutil import *
from STHM_Utils import *
from libFFTfin import *

# --- initialization ---

sigma  = 1.0 # [ Angstroem ] 
V0     = 2.7 # [ eV ]
V = None

# --- data loading ---
print '--- Data Loading ---'

V, lvec, nDim, head = loadXSF('LOCPOT.xsf')

# --- preprocessing ---
print '--- Preprocessing ---'

sampleSize = getSampleDimensions(lvec)
dims = (nDim[2], nDim[1], nDim[0])

xsize, dx = getSize('x', dims, sampleSize)
ysize, dy = getSize('y', dims, sampleSize)
zsize, dz = getSize('z', dims, sampleSize)

dd = (dx, dy, dz)

extent = ( -xsize/2, xsize/2,  -ysize/2, ysize/2 )
ilist = range(50,70,2) 

V = V - V0
X, Y, Z = getMGrid(dims, dd)

# --- probe particle potential ---

rho = getProbePotential(sampleSize, X, Y, Z, sigma, dd)

# --- print various quantities

printMetadata(sampleSize, dims, dd, xsize, ysize, zsize, V, rho)    

imshow(rho[0], extent=extent)
plt.colorbar()
plt.show()

# --- get forces ---
print '--- Get Forces ---'

Fx, Fy, Fz = getForces(V, rho, sampleSize, dims, dd, X, Y, Z)
print 'Fx.max(), Fx.min() = ', Fx.max(), Fx.min()

# --- Plotting ---
print "--- Plotting ---"

plotWithAtoms( Fx, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )
plotWithAtoms( Fy, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )
plotWithAtoms( Fz, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )

# --- Saving ---
print "--- Saving ---"

saveXSF('Fx.xsf', head, lvec, Fx)
saveXSF('Fy.xsf', head, lvec, Fy)
saveXSF('Fz.xsf', head, lvec, Fz)

# --- rho file ---
print '--- rho file ---'

#exportPotential(rho)
    
show()

