#!/usr/bin/python
import numpy as np

import ppafm as PPU
import ppafm.fieldFFT as fFFT
from ppafm import io


def handleAECCAR( fname, lvec, rho ):
    if "AECCAR" in fname:
        V = np.abs( np.linalg.det(lvec[1:]) )
        rho /= V

def handleNegativeDensity( rho ):
    Q = rho.sum()
    rho[rho<0] = 0
    rho *= ( Q/rho.sum() )

def main():

    parser = PPU.CLIParser(
        description='Calculate the density overlap integral for Pauli force field in the full-density based model. '
        'The integral has two parameters A and B, and is of form A*Integral( rho_tip^B * rho_sample^B )'
    )

    parser.add_argument('-s', '--sample', action='store', required=True, help='Path to sample charge density (.xsf).')
    parser.add_argument('-t', '--tip',    action='store', required=True, help='Path to tip charge density (.xsf).')
    parser.add_argument('-o', '--output', action='store', default='pauli', help='Name of output energy/force files.')
    parser.add_arguments(['output_format', 'energy', 'Apauli', 'Bpauli'])
    parser.add_argument('--saveDebugXsfs', action='store_true', help='Save auxiliary xsf files for debugging.')
    parser.add_argument('--no_negative_check', action='store_true',
                        help='Input density files may contain negative voxels. This is handled by default by setting negative values to zero '
                        'and rescaling the density so that the total charge is conserved. Setting this option disables the check.'
                        )

    args = parser.parse_args()

    print(">>> Loading sample from ", args.sample, " ... ")
    rhoS, lvecS, nDimS, headS = io.loadXSF( args.sample )
    print(">>> Loading tip from ", args.tip, " ... ")
    rhoT, lvecT, nDimT, headT = io.loadXSF( args.tip    )

    if np.any( nDimS != nDimT ): raise Exception( "Tip and Sample grids have different dimensions! - sample: "+str(nDimS)+" tip: "+str(nDimT) )
    if np.any( lvecS != lvecT ): raise Exception( "Tip and Sample grids have different shapes! - sample: "+str(lvecS )+" tip: "+str(lvecT) )

    handleAECCAR( args.sample, lvecS, rhoS )
    handleAECCAR( args.tip,    lvecT, rhoT )

    if args.Bpauli > 0.0:
        B = args.Bpauli
        print(">>> computing rho^B where B = ", B)
        # NOTE: due to round-off error the density from DFT code is often negative in some voxels which produce NaNs after exponentiation; we need to correct this
        if not args.no_negative_check:
            handleNegativeDensity( rhoS )
            handleNegativeDensity( rhoT )
        rhoS[:,:,:] = rhoS[:,:,:]**B
        rhoT[:,:,:] = rhoT[:,:,:]**B
        if args.saveDebugXsfs:
            io.save_scal_field( "sample_density_pow_%03.3f.xsf" %B, rhoS, lvecS, data_format=args.output_format, head=headS )
            io.save_scal_field( "tip_density_pow_%03.3f.xsf" %B, rhoT, lvecT, data_format=args.output_format, head=headT )

    print(">>> Evaluating convolution E(R) = A*Integral_r ( rho_tip^B(r-R) * rho_sample^B(r) ) using FFT ... ")
    Fx,Fy,Fz,E = fFFT.potential2forces_mem( rhoS, lvecS, nDimS, rho=rhoT, doForce=True, doPot=True, deleteV=True )

    PQ = args.Apauli

    namestr = args.output
    print(">>> Saving result of convolution to FF_",namestr,"_?.xsf ... ")

    # Density Overlap Model
    if args.energy:
        io.save_scal_field( "E"+namestr, E*PQ, lvecS, data_format=args.output_format, head=headS )
    FF = io.packVecGrid(Fx*PQ,Fy*PQ,Fz*PQ)
    io.save_vec_field( "FF"+namestr, FF, lvecS, data_format=args.output_format, head=headS )


if __name__ == "__main__":
    main()
