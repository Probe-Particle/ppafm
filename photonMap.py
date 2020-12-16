#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys
import __main__ as main
import numpy as np
import matplotlib.pyplot as plt
#import GridUtils as GU
#sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")

import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT
from optparse import OptionParser

# ======== Main

parser = OptionParser()
parser.add_option( "-H", "--homo",   action="store", type="string", default="homo.xsf", help="orbital of electron hole;    3D data-file (.xsf,.cube)")
parser.add_option( "-L", "--lumo",   action="store", type="string", default="lumo.xsf", help="orbital of excited electron; 3D data-file (.xsf,.cube)")
parser.add_option( "-R", "--radius", action="store", type="float",  default="1.0", help="tip radius")
parser.add_option( "-z", "--ztip", action="store", type="float",  default="5.0", help="tip above substrate")
parser.add_option( "-t", "--tip",    action="store", type="string", default="s",   help="tip compositon s,px,py,pz,d...")

#parser.add_option( "-o", "--output", action="store", type="string", default="pauli", help="output 3D data-file (.xsf)")

(options, args) = parser.parse_args()

#rho1, lvec1, nDim1, head1 = GU.loadXSF("./pyridine/CHGCAR.xsf")
#rho2, lvec2, nDim2, head2 = GU.loadXSF("./CO_/CHGCAR.xsf")

print( ">>> Loading HOMO from ", options.homo, " ... " )
homo, lvecH, nDimH, headH = GU.loadCUBE( options.homo )
print( ">>> Loading LUMO from ", options.lumo, " ... " )
lumo, lvecL, nDimL, headL = GU.loadCUBE( options.lumo )

print( "HOMO.sum2() ", np.sum(homo**2), " LUMO.sum2() ", np.sum(lumo**2) )

tip =  { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }

rhoTrans = homo*lumo

#print( ">>> Evaluating convolution E(R) = A*Integral_r ( rho_tip^B(r-R) * rho_sample^B(r) ) using FFT ... " )
#Fx,Fy,Fz,E = fFFT.potential2forces_mem( rhoTrans, lvecH, nDimH, rho=tip, doForce=True, doPot=True, deleteV=False )

if __name__ == "__main__":
    import matplotlib as plt

    GU.saveXSF( "HOMO.xsf", homo, lvecH, head=headH )
    GU.saveXSF( "LUMO.xsf", lumo, lvecL, head=headL )
    GU.saveXSF( "rhoTrans_HOMO_LUMO.xsf", rhoTrans, lvecH, head=headH )

    # Density Overlap Model
    #if options.energy:
    #    GU.saveXSF( "E"+namestr+".xsf", E*(PQ*-1.0), lvecS, head=headS )
    #GU.saveXSF( "FF"+namestr+"_x.xsf", Fx*PQ,       lvecS, head=headS )
    #GU.saveXSF( "FF"+namestr+"_y.xsf", Fy*PQ,       lvecS, head=headS )
    #GU.saveXSF( "FF"+namestr+"_z.xsf", Fz*PQ,       lvecS, head=headS )

    #Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)

