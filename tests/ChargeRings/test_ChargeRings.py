import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
#from pyProbeParticle import atomicUtils as au
import matplotlib.pyplot as plt







ps    = [[0.0,0.0,0.0],]
Qtips = [0.0]

R = 5.0
nsite = 3
phis = np.linspace(0,2*np.pi,nsite, endpoint=False); print(phis)
spos = np.zeros((3,3));
spos[:,1] = np.cos(phis)*R
spos[:,2] = np.sin(phis)*R

Esite = [ -1.0, -1.0, -1.0 ]

Qsites = chr.solveSiteOccupancies( ps, Qtips, spos, Esite, E_mu=0.0, cCouling=1.0, niter=100, tol=1e-6, dt=0.1 )

print( " Qsites ", Qsites )