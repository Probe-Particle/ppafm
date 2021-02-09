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

# ======== Functions

def handleAECCAR( fname, lvec, rho ):
    if "AECCAR" in fname:
        V = np.abs( np.linalg.det(lvec[1:]) )
        rho /= V

def handleNegativeDensity( rho ):
    Q = rho.sum()
    rho[rho<0] = 0
    rho *= ( Q/rho.sum() )

# ======== Main

parser = OptionParser()
parser.add_option( "-s", "--sample", action="store", type="string", default="CHGCAR.xsf", help="sample 3D data-file (.xsf)")
parser.add_option( "-t", "--tip",    action="store", type="string", default="./tip/CHGCAR.xsf", help="tip 3D data-file (.xsf)")
parser.add_option( "-o", "--output", action="store", type="string", default="pauli", help="output 3D data-file (.xsf)")
parser.add_option( "-B", "--Bpower", action="store", type="float", default="-1.0", help="exponent B in formula E = A*Integral( rho_tip^B * rho_sample^B ); NOTE: negative value equivalent to B=1 ")
parser.add_option( "-A", "--Apauli", action="store", type="float", default="1.0", help="prefactor A in formula E = A*Integral( rho_tip^B * rho_sample^B ); NOTE: default A=1 since re-scaling done in relax_scan_PVE.py")
parser.add_option( "-E", "--energy",     action="store_true",            help="Compue potential energ y(not just Force)", default=False)
parser.add_option( "--saveDebugXsfs",        action="store_true",  help="save auxuliary xsf files for debugging", default=False )
parser.add_option( "--densityMayBeNegative", action="store_false", help="input desnity files from DFT may contain negative voxels, lets handle them properly", default=True )

(options, args) = parser.parse_args()

#rho1, lvec1, nDim1, head1 = GU.loadXSF("./pyridine/CHGCAR.xsf")
#rho2, lvec2, nDim2, head2 = GU.loadXSF("./CO_/CHGCAR.xsf")

print(">>> Loading sample from ", options.sample, " ... ")
rhoS, lvecS, nDimS, headS = GU.loadXSF( options.sample )
print(">>> Loading tip from ", options.tip, " ... ")
rhoT, lvecT, nDimT, headT = GU.loadXSF( options.tip    )

if np.any( nDimS != nDimT ): raise Exception( "Tip and Sample grids has different dimensions! - sample: "+str(nDimS)+" tip: "+str(nDimT) )
if np.any( lvecS != lvecT ): raise Exception( "Tip and Sample grids has different shap! - sample: "+str(lvecS )+" tip: "+str(lvecT) )

# -------- Check basics
# -- does it still work if we change cell size ?
#lvecS[1,0]=10.0
#lvecS[2,1]=25.0
#lvecS[3,2]=30.0
print("lvecS ", lvecS[1:,:])
V  = np.linalg.det( lvecS[1:,:] )
N = nDimS[0]*nDimS[1]*nDimS[2]
dV = (V/N)  # volume of one voxel
#cRAs[:,0] *= dV    # Debugging
print("V ",V," N ",N," dV ",dV)
Qtip = rhoT.sum(); Qsam = rhoS.sum()
print("Total Charge Tip %g Sample %g [unnorm] " %(Qtip,Qsam)); 
print("Total Charge Tip %g Sample %g [norm  ] " %(Qtip*dV,Qsam*dV)); 
#exit()

# -------- Example Trial values : Unitary Homogenous Potential And Density  
#rhoT[:,:,:] = 1.0/V     #  Constant electron density  1 electron/cell_volume [e/A^3]
#rhoS[:,:,:] = 1.0       #  Constant potential         1 [eV]



handleAECCAR( options.sample, lvecS, rhoS )
handleAECCAR( options.tip,    lvecT, rhoT )

if options.Bpower > 0.0:
    B = options.Bpower
    print(">>> computing rho^B where B = ", B)
    #print " rhoS.min,max ",rhoS.min(), rhoS.max(), " rhoT.min,max ",rhoT.min(), rhoT.max()
    # NOTE: due to round-off error the density from DFT code is often negative in some voxels which produce NaNs after exponentiation; we need to correct this 
    if options.densityMayBeNegative:
        handleNegativeDensity( rhoS )
        handleNegativeDensity( rhoT )
    #print " rhoS.min,max ",rhoS.min(), rhoS.max(), " rhoT.min,max ",rhoT.min(), rhoT.max()
    rhoS[:,:,:] = rhoS[:,:,:]**B
    rhoT[:,:,:] = rhoT[:,:,:]**B
    if options.saveDebugXsfs:
        GU.saveXSF( "sample_density_pow_%03.3f.xsf" %B, rhoS, lvecS, head=headS )
        GU.saveXSF( "tip_density_pow_%03.3f.xsf" %B, rhoT, lvecT, head=headT )

print(">>> Evaluating convolution E(R) = A*Integral_r ( rho_tip^B(r-R) * rho_sample^B(r) ) using FFT ... ")
Fx,Fy,Fz,E = fFFT.potential2forces_mem( rhoS, lvecS, nDimS, rho=rhoT, doForce=True, doPot=True, deleteV=True )

print(" E samples : ", E[0,0,0],    E[50,50,50]   , np.mean(E))
#print " E samples : ", E[0,0,0]/N,  E[50,50,50]/N , np.mean(E)/N
print(" E samples : ", E[0,0,0]*dV, E[50,50,50]*dV, np.mean(E)*dV)
#exit()

PQ = options.Apauli

namestr = options.output
print(">>> Saving result of convolution to FF_",namestr,"_?.xsf ... ")

# Density Overlap Model
if options.energy:
    GU.saveXSF( "E"+namestr+".xsf", E*(PQ*-1.0), lvecS, head=headS )
GU.saveXSF( "FF"+namestr+"_x.xsf", Fx*PQ,       lvecS, head=headS )
GU.saveXSF( "FF"+namestr+"_y.xsf", Fy*PQ,       lvecS, head=headS )
GU.saveXSF( "FF"+namestr+"_z.xsf", Fz*PQ,       lvecS, head=headS )

#Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)

