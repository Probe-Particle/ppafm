#!/usr/bin/python
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

from optparse import OptionParser

import __main__ as main
import matplotlib.pyplot as plt
import numpy as np

import ppafm as PPU
import ppafm.fieldFFT as fFFT
from ppafm import io

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
parser.add_option( "-f","--data_format", action="store" , type="string", help="Specify the output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
parser.add_option( "--saveDebugXsfs",        action="store_true",  help="save auxuliary xsf files for debugging", default=False )
parser.add_option( "--densityMayBeNegative", action="store_false", help="input desnity files from DFT may contain negative voxels, lets handle them properly", default=True )

(options, args) = parser.parse_args()

print(">>> Loading sample from ", options.sample, " ... ")
rhoS, lvecS, nDimS, headS = io.loadXSF( options.sample )
print(">>> Loading tip from ", options.tip, " ... ")
rhoT, lvecT, nDimT, headT = io.loadXSF( options.tip    )

if np.any( nDimS != nDimT ): raise Exception( "Tip and Sample grids has different dimensions! - sample: "+str(nDimS)+" tip: "+str(nDimT) )
if np.any( lvecS != lvecT ): raise Exception( "Tip and Sample grids has different shap! - sample: "+str(lvecS )+" tip: "+str(lvecT) )

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
        io.save_scal_field( "sample_density_pow_%03.3f.xsf" %B, rhoS, lvecS, data_format=options.data_format, head=headS )
        io.save_scal_field( "tip_density_pow_%03.3f.xsf" %B, rhoT, lvecT, data_format=options.data_format, head=headT )

print(">>> Evaluating convolution E(R) = A*Integral_r ( rho_tip^B(r-R) * rho_sample^B(r) ) using FFT ... ")
Fx,Fy,Fz,E = fFFT.potential2forces_mem( rhoS, lvecS, nDimS, rho=rhoT, doForce=True, doPot=True, deleteV=True )

PQ = options.Apauli

namestr = options.output
print(">>> Saving result of convolution to FF_",namestr,"_?.xsf ... ")

# Density Overlap Model
if options.energy:
    io.save_scal_field( "E"+namestr, E*PQ, lvecS, data_format=options.data_format, head=headS )
FF = io.packVecGrid(Fx*PQ,Fy*PQ,Fz*PQ)
io.save_vec_field( "FF"+namestr, FF, lvecS, data_format=options.data_format, head=headS )

#Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)
