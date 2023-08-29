#!/usr/bin/python -u

import os

import numpy as np

import ppafm as PPU
import ppafm.core as PPC
import ppafm.cpp_utils as cpp_utils
import ppafm.GridUtils as GU
import ppafm.HighLevel as PPH
from ppafm import io


def rotVec( v, a ):
    ca=np.cos(a);      sa=np.sin(a)
    x=ca*v[0]-sa*v[1]; y=sa*v[0]+ca*v[1];
    v[0]=x; v[1]=y;
    return v

def rotFF( Fx,Fy, a ):
    ca=np.cos(a);      sa=np.sin(a)
    Fx_=ca*Fx-sa*Fy; Fy_=sa*Fx+ca*Fy;
    return Fx_,Fy_


def main():

    parser = PPU.CLIParser(
        description='Perform a scan, relaxing the probe particle in a precalculated force field. '
            'The generated force field is saved to Q{charge}K{klat}/OutFz.xsf.'
    )

    parser.add_arguments(['klat', 'krange', 'charge', 'qrange', 'Vbias', 'Vrange', 'Apauli', 'output_format'])
    parser.add_argument("--noLJ", action="store_true", help="Load Pauli and vdW force fields from separate files")
    parser.add_argument("-b", "--boltzmann", action="store_true", help="Calculate forces with boltzmann particle")
    parser.add_argument("--bI", action="store_true", help="Calculate current between boltzmann particle and tip")
    parser.add_argument("--pos", action="store_true", help="Save probe particle positions")
    parser.add_argument("--disp", action="store_true", help="Save probe particle displacements")
    parser.add_argument("--bDebugFFtot", action="store_true", help="Store total force field for debugging")
    parser.add_argument("--vib", action="store", type=int, default=-1, help="Map PP vibration eigenmodes; 0-just eigenvals; 1-3 eigenvecs")
    parser.add_argument("--tipspline", action="store", type=str, help="File where spline is stored")
    parser.add_argument("--rotate", action="store", type=float, default=0.0, help="Rotates sampling in xy-plane")
    parser.add_argument("--pol_t", action="store", type=float, default=1.0, help="Scaling factor for tip polarization")
    parser.add_argument("--pol_s", action="store", type=float, default=1.0, help="Scaling factor for sample polarization")
    args = parser.parse_args()

    opt_dict = vars(args)
    PPU.loadParams('params.ini')
    PPU.apply_options(opt_dict)

    # =============== Setup

    # Ks
    if opt_dict['krange'] is not None:
        Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], int(opt_dict['krange'][2]) )
    elif opt_dict['klat'] is not None:
        Ks = [opt_dict['klat']]
    elif PPU.params['stiffness'][0] > 0.0:
        Ks = [PPU.params['stiffness'][0]]
    else:
        Ks = [PPU.params['klat']]

    # Qs
    charged_system = False
    if opt_dict['qrange'] is not None:
        Qs = np.linspace( opt_dict['qrange'][0], opt_dict['qrange'][1], int(opt_dict['qrange'][2]) )
    elif opt_dict['charge'] is not None:
        Qs = [ opt_dict['charge'] ]
    else:
        Qs = [ PPU.params['charge'] ]
    for iq,Q in enumerate(Qs):
        if ( abs(Q) > 1e-7):
            charged_system = True

    # Vkpfm
    applied_bias = False
    if opt_dict['Vrange'] is not None:
        Vs = np.linspace( opt_dict['Vrange'][0], opt_dict['Vrange'][1], int(opt_dict['Vrange'][2]) )
    elif opt_dict['Vbias'] is not None:
        Vs = [ opt_dict['Vbias'] ]
    else:
        Vs = [0.0]
    for iV,Vx in enumerate(Vs):
        if ( abs(Vx) > 1e-7):
            applied_bias = True

    if applied_bias:
        print("Vs   =", Vs)
    print("Ks   =", Ks)
    print("Qs   =", Qs)

    print(" ============= RUN  ")

    FFvdW = FFpauli = FFel = FFboltz = FFkpfm_t0sV = FFkpfm_tVs0 = None

    if args.noLJ:

        print("Apauli", PPU.params["Apauli"])

        print("Loading Pauli force field from FFpauli_{x,y,z}")
        FFpauli, lvec, nDim, atomic_info_or_head = io.load_vec_field("FFpauli", data_format=args.output_format)
        FFpauli[0,:,:,:], FFpauli[1,:,:,:] = rotFF( FFpauli[0,:,:,:], FFpauli[1,:,:,:], opt_dict['rotate'] )

        print("Loading vdW force field from FFvdW_{x,y,z}")
        FFvdW, lvec, nDim, atomic_info_or_head = io.load_vec_field("FFvdW", data_format=args.output_format)
        FFvdW[0,:,:,:], FFvdW[1,:,:,:] = rotFF( FFvdW[0,:,:,:], FFvdW[1,:,:,:], opt_dict['rotate'] )

    else:
        print("Loading Lennard-Jones force field from FFLJ_{x,y,z}")
        FFvdW, lvec, nDim, atomic_info_or_head = io.load_vec_field("FFLJ", data_format=args.output_format)
        FFvdW[0,:,:,:], FFvdW[1,:,:,:] = rotFF( FFvdW[0,:,:,:], FFvdW[1,:,:,:], opt_dict['rotate'] )

    if charged_system:
        print("Loading electrostatic force field from FFel_{x,y,z}")
        FFel, lvec, nDim, atomic_info_or_head = io.load_vec_field("FFel", data_format=args.output_format)
        FFel[0,:,:,:], FFel[1,:,:,:] = rotFF( FFel[0,:,:,:], FFel[1,:,:,:], opt_dict['rotate'] )

    if args.boltzmann or args.bI:
        print("Loading Boltzmann force field from FFboltz_{x,y,z}")
        FFboltz, lvec, nDim, atomic_info_or_head = io.load_vec_field("FFboltz", data_format=args.output_format)
        FFboltz[0,:,:,:], FFboltz[1,:,:,:] = rotFF( FFboltz[0,:,:,:], FFboltz[1,:,:,:], opt_dict['rotate'] )

    if applied_bias:
        print("Loading electrostatic contribution from applied bias from FFkpfm_t0sV_{x,y,z} and FFkpfm_tVs0_{x,y,z}")
        FFkpfm_t0sV, lvec, nDim, atomic_info_or_head = io.load_vec_field("FFkpfm_t0sV", data_format=args.output_format)
        FFkpfm_tVs0, lvec, nDim, atomic_info_or_head = io.load_vec_field("FFkpfm_tVs0", data_format=args.output_format)

        FFkpfm_t0sV[0,:,:,:], FFkpfm_t0sV[1,:,:,:] = rotFF( FFkpfm_t0sV[0,:,:,:], FFkpfm_t0sV[1,:,:,:], opt_dict['rotate'] )
        FFkpfm_tVs0[0,:,:,:], FFkpfm_tVs0[1,:,:,:] = rotFF( FFkpfm_tVs0[0,:,:,:], FFkpfm_tVs0[1,:,:,:], opt_dict['rotate'] )

        FFkpfm_t0sV = FFkpfm_t0sV*opt_dict['pol_s']
        FFkpfm_tVs0 = FFkpfm_tVs0*opt_dict['pol_t']

    lvec[1,:] = rotVec( lvec[1,:], opt_dict['rotate'] )
    lvec[2,:] = rotVec( lvec[2,:], opt_dict['rotate'] )
    print(lvec)
    PPU.lvec2params( lvec )

    for iq,Q in enumerate( Qs ):
        PPU.params['charge'] = Q

        for ik,K in enumerate( Ks ):
            PPU.params['klat'] = K

            for iv,Vx in enumerate( Vs ):
                PPU.params['Vbias'] = Vx

                dirname = f"Q{Q:1.2f}K{K:1.2f}"
                if applied_bias:
                    dirname += f"V{Vx:1.2f}"
                print(" Relaxed_scan for ", dirname)
                if not os.path.exists( dirname ):
                    os.makedirs( dirname )

                # Run relaxation
                fzs, PPpos, PPdisp, lvecScan = PPH.perform_relaxation(lvec, FFvdW, FFel=FFel, FFpauli=FFpauli, FFboltz=FFboltz,
                                                                      FFkpfm_t0sV=FFkpfm_t0sV, FFkpfm_tVs0=FFkpfm_tVs0, tipspline=args.tipspline, bFFtotDebug=args.bDebugFFtot)

                data_info = {'lvec': lvecScan, 'data_format': args.output_format, 'head': atomic_info_or_head, 'atomic_info': atomic_info_or_head}
                if PPU.params['tiltedScan']:
                    io.save_vec_field( dirname+'/OutF', fzs, **data_info)
                else:
                    io.save_scal_field( dirname+'/OutFz', fzs, **data_info)

                if opt_dict['vib'] >= 0:
                    which = opt_dict['vib']
                    print(f" === Computing eigenvectors of dynamical matrix: which={which} ddisp={PPU.params['ddisp']}")
                    xTips, yTips, zTips, lvecScan = PPU.prepareScanGrids()
                    rTips = np.array(np.meshgrid(xTips, yTips, zTips)).transpose(3,1,2,0).copy()
                    evals,evecs = PPC.stiffnessMatrix( rTips.reshape((-1,3)), PPpos.reshape((-1,3)), which=which, ddisp=PPU.params['ddisp'] )
                    io.save_vec_field( dirname+'/eigvalKs', evals.reshape( rTips.shape ), **data_info)
                    if which > 0: io.save_vec_field( dirname+'/eigvecK1', evecs[0].reshape( rTips.shape ), **data_info)
                    if which > 1: io.save_vec_field( dirname+'/eigvecK2', evecs[1].reshape( rTips.shape ), **data_info)
                    if which > 2: io.save_vec_field( dirname+'/eigvecK3', evecs[2].reshape( rTips.shape ), **data_info)

                if opt_dict['disp']:
                    io.save_vec_field( dirname+'/PPdisp', PPdisp, **data_info)

                if opt_dict['pos']:
                    io.save_vec_field( dirname+'/PPpos', PPpos, **data_info)

                if args.bI:
                    print("Calculating current from tip to the Boltzmann particle:")
                    I_in, lvec, nDim, atomic_info_or_head = io.load_scal_field('I_boltzmann', data_format=args.output_format)
                    I_out = GU.interpolate_cartesian( I_in, PPpos, cell=lvec[1:,:], result=None )
                    del I_in
                    io.save_scal_field(dirname+'/OutI_boltzmann', I_out, **data_info)

if __name__ == "__main__":
    main()
