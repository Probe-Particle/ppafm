
import sys 
import numpy as np
from xsfutil import *
#import matplotlib.pyplot as plt
#from STHM_Utils import *
from libFFTfin import *

'''
 how to run:
>> python /home/prokop/git/ProbeParticleModel/code/makeFFel.py

'''

# ============ SETUP

sigma  = 1.0 # [ Angstroem ] 
V0     = 2.7 # [ eV ]
V      = None

fname = 'LOCPOT.xsf'

if len(sys.argv) > 1:
	fname = sys.argv[1]
if len(sys.argv) > 2:
	sigma = float(sys.argv[2])

# ============ data loading ---
print '--- Data Loading ---'

V, lvec, nDim, head = loadXSF( fname )

# ============ preprocessing ---
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

# ============ probe particle potential ---

rho = getProbePotential(sampleSize, X, Y, Z, sigma, dd)

# ============ print various quantities

printMetadata(sampleSize, dims, dd, xsize, ysize, zsize, V, rho)    

#imshow(rho[0], extent=extent)
#plt.colorbar()
#plt.show()

# --- get forces ---
print '--- Get Forces ---'

Fx, Fy, Fz = getForces(V, rho, sampleSize, dims, dd, X, Y, Z)
print 'Fx.max(), Fx.min() = ', Fx.max(), Fx.min()

# --- Plotting ---
#print "--- Plotting ---"
#plotWithAtoms( Fx, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )
#plotWithAtoms( Fy, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )
#plotWithAtoms( Fz, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )

# --- Saving ---
print "--- Saving ---"

saveXSF('FFel_x.xsf', head, lvec, Fx)
saveXSF('FFel_y.xsf', head, lvec, Fy)
saveXSF('FFel_z.xsf', head, lvec, Fz)

# --- rho file ---
#print '--- rho file ---'

#exportPotential(rho)
    
# show()

