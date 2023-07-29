#!/usr/bin/python -u

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import ppafm as PPU
import ppafm.atomicUtils as au
import ppafm.cpp_utils as cpp_utils
import ppafm.HighLevel as PPH
import ppafm.PPPlot as PPPlot
from ppafm import elements, io

import matplotlib as mpl;  mpl.use('Agg'); print("plot WITHOUT Xserver"); # this makes it run without Xserver (e.g. on supercomputer) # see http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server


def main():

    parser = PPU.CLIParser(
        description='Plot results for a scan with a specified charge, amplitude, and spring constant. '
            'Images are saved in folder Q{charge}K{klat}/Amp{Amplitude}.'
    )
    parser.add_arguments(['output_format', 'Amplitude', 'arange', 'klat', 'krange', 'charge', 'qrange', 'Vbias', 'Vrange', 'noPBC'])
    parser.add_argument( "--iets", action="store", type=float, help="Mass [a.u.]; Bias offset [eV]; Peak width [eV] ", nargs=3 )
    parser.add_argument( "--LCPD_maps", action="store_true", help="Print LCPD maps")
    parser.add_argument("--z0", action="store", type=float, default=0.0, help="Height of the topmost layer of metallic substrate for E to V conversion (Ang)")
    parser.add_argument("--V0", action="store", type=float, default=0.0, help="Empirical LCPD maxima shift due to mesoscopic workfunction diference")

    parser.add_argument( "--df",       action="store_true", help="Plot images for dfz " )
    parser.add_argument( "--save_df" , action="store_true", help="Save frequency shift as df.xsf " )
    parser.add_argument( "--Laplace",  action="store_true", help="Plot Laplace-filtered images and save them " )
    parser.add_argument( "--pos",      action="store_true", help="Save probe particle positions" )
    parser.add_argument( "--atoms",    action="store_true", help="Plot atoms to images" )
    parser.add_argument( "--bonds",    action="store_true", help="Plot bonds to images" )
    parser.add_argument( "--cbar",     action="store_true", help="Plot colorbars to images" )
    parser.add_argument( "--WSxM",     action="store_true", help="Save frequency shift into WsXM *.dat files" )
    parser.add_argument( "--bI",       action="store_true", help="Plot images for Boltzmann current" )

    args = parser.parse_args()
    opt_dict = vars(args)

    PPU.loadParams( 'params.ini' )
    PPU.apply_options(opt_dict)

    if opt_dict['Laplace']:
        from scipy.ndimage import laplace

    print(" >> OVEWRITING SETTINGS by command line arguments  ")
    # Ks
    if opt_dict['krange'] is not None:
        Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], int(opt_dict['krange'][2]) )
    elif opt_dict['klat'] is not None:
        Ks = [ opt_dict['klat'] ]
    elif PPU.params['stiffness'][0] > 0.0:
        Ks = [PPU.params['stiffness'][0]]
    else:
        Ks = [ PPU.params['klat']]
    # Qs
    if opt_dict['qrange'] is not None:
        Qs = np.linspace( opt_dict['qrange'][0], opt_dict['qrange'][1], int(opt_dict['qrange'][2]) )
    elif opt_dict['charge'] is not None:
        Qs = [ opt_dict['charge'] ]
    else:
        Qs = [ PPU.params['charge'] ]
    # Amps
    if opt_dict['arange'] is not None:
        Amps = np.linspace( opt_dict['arange'][0], opt_dict['arange'][1], int(opt_dict['arange'][2]) )
    elif opt_dict['Amplitude'] is not None:
        Amps = [ opt_dict['Amplitude'] ]
    else:
        Amps = [ PPU.params['Amplitude'] ]
    # Applied biases
    applied_bias = False
    if opt_dict['Vrange'] is not None:
        Vs = np.linspace( opt_dict['Vrange'][0], opt_dict['Vrange'][1], int(opt_dict['Vrange'][2]) )
    elif opt_dict['Vbias'] is not None:
        Vs = [ opt_dict['Vbias'] ]
    else:
        Vs = [0.0]
    for iV,Vx in enumerate(Vs):
        if ( abs(Vx) > 1e-7):
            applied_bias=True

    if (applied_bias == True):
        print("Vs   =", Vs)
    print("Ks   =", Ks)
    print("Qs   =", Qs)
    print("Amps =", Amps)
    print(" ============= RUN  ")

    dz  = PPU.params['scanStep'][2]
    xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )
    extent = ( xTips[0], xTips[-1], yTips[0], yTips[-1] )

    atoms_str=""
    atoms = None
    bonds = None
    FFparams = None
    if opt_dict['atoms'] or opt_dict['bonds']:
        speciesFile=None
        if os.path.isfile( 'atomtypes.ini' ):
            speciesFile='atomtypes.ini'
        FFparams=PPU.loadSpecies( speciesFile )
        atoms_str="_atoms"
        xyzs, Zs, qs, _ = io.loadXYZ('input_plot.xyz')
        atoms = [list(Zs), list(xyzs[:, 0]), list(xyzs[:, 1]), list(xyzs[:, 2]), list(qs)]
        FFparams            = PPU.loadSpecies( )
        elem_dict           = PPU.getFFdict(FFparams)
        iZs,Rs,Qs_tmp=PPU.parseAtoms(atoms, elem_dict, autogeom = False, PBC = PPU.params['PBC'] )
        atom_colors = au.getAtomColors(iZs,FFparams=FFparams)
        Rs=Rs.transpose().copy()
        atoms= [iZs,Rs[0],Rs[1],Rs[2],atom_colors]
    if opt_dict['bonds']:
        bonds = au.findBonds(atoms,iZs,1.0,FFparams=FFparams)
    atomSize = 0.15

    cbar_str =""
    if opt_dict['cbar']:
        cbar_str="_cbar"

    for iq,Q in enumerate( Qs ):
        for ik,K in enumerate( Ks ):
            dirname = "Q%1.2fK%1.2f" %(Q,K)
            for iv,Vx in enumerate( Vs ):
                if applied_bias:
                    dirname = "Q%1.2fK%1.2fV%1.2f" %(Q,K,Vx)
                if opt_dict['pos']:
                    try:
                        PPpos, lvec, nDim , atomic_info_or_head = io.load_vec_field(dirname+'/PPpos' ,data_format=args.output_format)
                        print(" plotting PPpos : ")
                        PPPlot.plotDistortions(
                            dirname+"/xy"+atoms_str+cbar_str, PPpos[:,:,:,0], PPpos[:,:,:,1], slices = list(range( 0, len(PPpos))), BG=PPpos[:,:,:,2],
                            extent=extent, atoms=atoms, bonds=bonds, atomSize=atomSize, markersize=2.0, cbar=opt_dict['cbar']
                        )
                        del PPpos
                    except:
                        print("error: ", sys.exc_info())
                        print("cannot load : " + ( dirname+'/PPpos_?.' + args.output_format ))
                if opt_dict['iets'] is not None:
                    try :
                        eigvalK, lvec, nDim = io.load_vec_field( dirname+'/eigvalKs' ,data_format=args.output_format)
                        M  = opt_dict['iets'][0]
                        E0 = opt_dict['iets'][1]
                        w  = opt_dict['iets'][2]
                        print(" plotting IETS M=%f V=%f w=%f " %(M,E0,w))
                        hbar       = 6.58211951440e-16 # [eV.s]
                        aumass     = 1.66053904020e-27 # [kg]
                        eVA2_to_Nm = 16.0217662        # [eV/A^2] / [N/m]
                        Evib = hbar * np.sqrt( ( eVA2_to_Nm * eigvalK )/( M * aumass ) )
                        IETS = PPH.symGauss(Evib[:,:,:,0], E0, w) + PPH.symGauss(Evib[:,:,:,1], E0, w) + PPH.symGauss(Evib[:,:,:,2], E0, w)
                        PPPlot.plotImages( dirname+"/IETS"+atoms_str+cbar_str, IETS, slices = list(range(0,len(IETS))), zs=zTips, extent=extent, atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar'] )
                        PPPlot.plotImages( dirname+"/Evib"+atoms_str+cbar_str, Evib[:,:,:,0], slices = list(range(0,len(IETS))), zs=zTips, extent=extent, atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar'] )
                        PPPlot.plotImages( dirname+"/Kvib"+atoms_str+cbar_str, 16.0217662 * eigvalK[:,:,:,0], slices = list(range(0,len(IETS))), zs=zTips, extent=extent, atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar'] )
                        del eigvalK; del Evib; del IETS
                    except:
                        print("error: ", sys.exc_info())
                        print("cannot load : ", dirname+'/PPpos_?.' + args.output_format)
                if (  opt_dict['df'] or opt_dict['save_df'] or opt_dict['WSxM']  ):
                    try :
                        for iA,Amp in enumerate( Amps ):
                            PPU.params['Amplitude'] = Amp
                            AmpStr = "/Amp%2.2f" %Amp
                            print("Amp= ",AmpStr)
                            dirNameAmp = dirname+AmpStr
                            if not os.path.exists( dirNameAmp ):
                                os.makedirs( dirNameAmp )
                            if PPU.params['tiltedScan']:
                                Fout, lvec, nDim , atomic_info_or_head = io.load_vec_field(dirname+'/OutF' , data_format=args.output_format)
                                dfs = PPU.Fz2df_tilt( Fout, PPU.params['scanTilt'], k0 = PPU.params['kCantilever'], f0=PPU.params['f0Cantilever'], n=int(Amp/dz) )
                            else:
                                fzs, lvec, nDim , atomic_info_or_head = io.load_scal_field(dirname+'/OutFz' , data_format=args.output_format)
                                if (applied_bias):
                                    Rtip = PPU.params['Rtip']
                                    for iz,z in enumerate( zTips ):
                                        fzs[iz,:,:] = fzs[iz,:,:] - np.pi*PPU.params['permit']*((Rtip*Rtip)/((z-args.z0)*(z+Rtip)))*(Vx-args.V0)*(Vx-args.V0)
                                dfs = PPU.Fz2df( fzs, dz = dz, k0 = PPU.params['kCantilever'], f0=PPU.params['f0Cantilever'], n=int(Amp/dz) )
                            if opt_dict['save_df']:
                                io.save_scal_field(dirNameAmp+'/df', dfs, lvec,data_format=args.output_format , head = atomic_info_or_head , atomic_info = atomic_info_or_head)
                            if opt_dict['df']:
                                print(" plotting df : ")
                                PPPlot.plotImages(
                                    dirNameAmp+"/df"+atoms_str+cbar_str, dfs,  slices = list(range( 0, len(dfs))), zs=zTips+PPU.params['Amplitude']/2.0,
                                    extent=extent,cmap=PPU.params['colorscale'], atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar']
                                )
                            if opt_dict['Laplace']:
                                print("plotting Laplace-filtered df : ")
                                df_LaplaceFiltered = dfs.copy()
                                laplace( dfs, output = df_LaplaceFiltered )
                                io.save_scal_field(dirNameAmp+'/df_laplace', df_LaplaceFiltered, lvec,data_format=args.output_format , head = atomic_info_or_head , atomic_info = atomic_info_or_head)
                                PPPlot.plotImages(
                                    dirNameAmp+"/df_laplace"+atoms_str+cbar_str, df_LaplaceFiltered, slices = list(range( 0, len(dfs))), zs=zTips+PPU.params['Amplitude']/2.0,
                                    extent=extent,cmap=PPU.params['colorscale'], atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar']
                                )
                            if opt_dict['WSxM']:
                                print(" printing df into WSxM files :")
                                io.saveWSxM_3D( dirNameAmp+"/df" , dfs , extent , slices=None)
                            if  opt_dict['LCPD_maps']:
                                if (iv == 0):
                                    LCPD_b = - dfs
                                if (iv == (Vs.shape[0]-1)):
                                    LCPD_b = (LCPD_b + dfs)/(2*Vx)

                                if (iv == 0):
                                    LCPD_a = dfs
                                if (Vx == 0.0):
                                    LCPD_a = LCPD_a - 2*dfs
                                if (iv == (Vs.shape[0]-1)):
                                    LCPD_a = (LCPD_a + dfs)/(2*Vx**2)
                            del dfs
                        del fzs
                    except:
                        print("error: ", sys.exc_info())
                        print("cannot load : ",dirname+'/OutFz.'+args.output_format)
                if opt_dict['bI']:
                    try:
                        I, lvec, nDim , atomic_info_or_head = io.load_scal_field(dirname+'/OutI_boltzmann', data_format=args.output_format )
                        print(" plotting Boltzmann current: ")
                        PPPlot.plotImages( dirname+"/OutI"+atoms_str+cbar_str, I,  slices = list(range( 0,len(I))), zs=zTips, extent=extent, atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar'] )
                        del I
                    except:
                        print("error: ", sys.exc_info())
                        print("cannot load : " + (dirname+'/OutI_boltzmann.'+args.output_format ))
            if  opt_dict['LCPD_maps']:
                LCPD = -LCPD_b/(2*LCPD_a)
                PPPlot.plotImages(
                    "./LCPD"+atoms_str+cbar_str, LCPD,  slices = list(range( 0, len(LCPD))), zs=zTips+PPU.params['Amplitude']/2.0,
                    extent=extent,cmap=PPU.params['colorscale_kpfm'], atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar'], symetric_map=True ,V0=args.V0
                )
                PPPlot.plotImages(
                    "./_Asym-LCPD"+atoms_str+cbar_str, LCPD,  slices = list(range( 0, len(LCPD))), zs=zTips+PPU.params['Amplitude']/2.0,
                    extent=extent,cmap=PPU.params['colorscale_kpfm'], atoms=atoms, bonds=bonds, atomSize=atomSize, cbar=opt_dict['cbar'], symetric_map=False
                )
                io.save_scal_field('./LCDP_HzperV', LCPD, lvec,data_format=args.output_format , head = atomic_info_or_head , atomic_info = atomic_info_or_head)
                if opt_dict['WSxM']:
                    print(" printing LCPD_b into WSxM files :")
                    io.saveWSxM_3D( "./LCPD"+atoms_str , LCPD , extent , slices=None)


    print(" ***** ALL DONE ***** ")

if __name__ == "__main__":
    main()
