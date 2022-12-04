#!/usr/bin/python
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys
import __main__ as main
import numpy as np
import matplotlib.pyplot as plt
#import GridUtils as GU

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
import ppafm                as PPU
import ppafm.GridUtils      as GU
import ppafm.fieldFFT       as fFFT
from optparse import OptionParser



def maskFunc( x, x0, x1 ):
    sc = 1/(x1-x0)
    x = (x-x0)*sc
    mask = (3 - 2 * x)*x*x
    mask[x<0] = 1
    mask[x>1] = 0
    return mask



# ======== Main

parser = OptionParser()
parser.add_option( "-s", "--sample", action="store", type="string", default="CHGCAR.xsf", help="sample 3D data-file (.xsf)")

(options, args) = parser.parse_args()

print(">>> Loading sample from ", options.sample, " ... ")
rho, lvec, nDim, head = GU.loadXSF( options.sample )

rho_low = rho.copy()
Ycut = 1e-1
rho_low[rho_low>Ycut] = Ycut

'''
#X,Y,Z = getPos( lvec, nDim=nDim )
X,Y,Z = fFFT.getMGrid( nDim, [1./nDim[0],1./nDim[0],1./nDim[0]])
print  " X.min,max ", X.max(), X.min()," Y.min,max ", Y.max()," Z.min,max ", Y.min(), Z.max(), Z.min()
R = np.sqrt( X**2 + Y**2 + Z**2 )    # ToDo - for non-orthogonal cells
print "R.max() ", R.max()
rho_fft  = np.fft.fftn(rho)
#mask = maskFunc(R, 0.2, 0.25 )
mask = maskFunc(R, 0.1, 0.125 )
print "mask.min(), mask.max() ", mask.min(), mask.max()
rho_low = np.fft.ifftn( rho_fft * mask )
'''

dRho = rho - rho_low

print(" rho.max(), rho_low.max(), dRho.max() ", rho.max(), rho_low.max(), dRho.max())

namestr = options.sample+"_resudual.xsf"
print(">>> Saving  residual to  ", namestr, " ... ")
GU.saveXSF( namestr, dRho, lvec, head=head )

namestr = options.sample+"_LowPass.xsf"
print(">>> Saving  LowPass to  ", namestr, " ... ")
GU.saveXSF( namestr, rho_low, lvec, head=head )


#Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)
