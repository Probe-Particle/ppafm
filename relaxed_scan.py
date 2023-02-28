#!/usr/bin/python -u

import os

import numpy as np

import ppafm as PPU
import ppafm.core as PPC
import ppafm.cpp_utils as cpp_utils
import ppafm.GridUtils as GU
import ppafm.HighLevel as PPH

#import matplotlib.pyplot as plt



#import PPPlot 		# we do not want to make it dempendent on matplotlib
print("Amplitude ", PPU.params['Amplitude'])

def rotVec( v, a ):
    ca=np.cos(a);      sa=np.sin(a)
    x=ca*v[0]-sa*v[1]; y=sa*v[0]+ca*v[1];
    v[0]=x; v[1]=y;
    return v

def rotFF( Fx,Fy, a ):
    ca=np.cos(a);      sa=np.sin(a)
    Fx_=ca*Fx-sa*Fy; Fy_=sa*Fx+ca*Fy;
    return Fx_,Fy_


# =============== arguments definition

if __name__=="__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option( "-k", "--klat",  action="store", type="float", help="tip stiffenss [N/m]" )
    parser.add_option( "--krange", action="store", type="float", help="tip stiffenss range (min,max,n) [N/m]", nargs=3)
    parser.add_option( "-q","--charge",       action="store", type="float", help="tip charge [e]" )
    parser.add_option( "--qrange", action="store", type="float", help="tip charge range (min,max,n) [e]", nargs=3)
    parser.add_option( "-b", "--boltzmann" ,action="store_true", default=False, help="calculate forces with boltzmann particle" )
    parser.add_option( "--bI" ,action="store_true", default=False, help="calculate current between boltzmann particle and tip" )
    parser.add_option( "--pos",       action="store_true", default=False, help="save probe particle positions" )
    parser.add_option( "--disp",      action="store_true", default=False, help="save probe particle displacements")
    parser.add_option( "--vib",       action="store", type="int", default=-1, help="map PP vibration eigenmodes; 0-just eigenvals; 1-3 eigenvecs" )
    parser.add_option( "--tipspline", action="store", type="string", help="file where spline is stored", default=None )
    parser.add_option( "--rotate", action="store", type="float", help="rotates sampling in xy-plane", default=0.0 )
    parser.add_option("-f","--data_format" , action="store" , type="string",help="Specify the input/output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
    parser.add_option( "-V","--Vbias",       action="store", type="float", help="Aplied bias [V]" )
    parser.add_option( "--Vrange",       action="store", type="float", help="Bias range [V]", nargs=3 )
    parser.add_option( "--pol_t", action="store", type="float", default=1.0, help="scaling factor for tip polarization")
    parser.add_option( "--pol_s", action="store", type="float", default=1.0, help="scaling factor for sample polarization")
    (options, args) = parser.parse_args()
    opt_dict = vars(options)
    PPU.loadParams( 'params.ini' )
    PPU.apply_options(opt_dict)
    # =============== Setup

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
    charged_system=False
    if opt_dict['qrange'] is not None:
        Qs = np.linspace( opt_dict['qrange'][0], opt_dict['qrange'][1], int(opt_dict['qrange'][2]) )
    elif opt_dict['charge'] is not None:
        Qs = [ opt_dict['charge'] ]
    else:
        Qs = [ PPU.params['charge'] ]
    for iq,Q in enumerate(Qs):
        if ( abs(Q) > 1e-7):
            charged_system=True
    # Vkpfm
    aplied_bias=False
    if opt_dict['Vrange'] is not None:
        Vs = np.linspace( opt_dict['Vrange'][0], opt_dict['Vrange'][1], int(opt_dict['Vrange'][2]) )
    elif opt_dict['Vbias'] is not None:
        Vs = [ opt_dict['Vbias'] ]
    else:
        Vs = [0.0]
    for iV,Vx in enumerate(Vs):
        if ( abs(Vx) > 1e-7):
            aplied_bias=True

    if (aplied_bias == True):
        print("Vs   =", Vs)
    print("Ks   =", Ks)
    print("Qs   =", Qs)
    #print "Amps =", Amps

    print(" ============= RUN  ")
    FFLJ, FFel, FFboltz,FFkpfm_t0sV, FFkpfm_tVs0=None,None,None,None,None
    if ( charged_system == True):
        print(" load Electrostatic Force-field ")
        FFel, lvec, nDim, headORatoms = GU.load_vec_field( "FFel" ,data_format=options.data_format)
        FFel[0,:,:,:],FFel[1,:,:,:] = rotFF( FFel[0,:,:,:],FFel[1,:,:,:], opt_dict['rotate'] )
    if (options.boltzmann  or options.bI) :
        print(" load Boltzmann Force-field ")
        FFboltz, lvec, nDim = GU.load_vec_field( "FFboltz", data_format=options.data_format)
        FFboltz[0,:,:,:],FFboltz[1,:,:,:] = rotFF( FFboltz[0,:,:,:],FFboltz[1,:,:,:], opt_dict['rotate'] )
    if  ( aplied_bias == True):
        print(" load Electrostatic contribution from aplied bias")
        FFkpfm_t0sV, lvec, nDim, headORatoms = GU.load_vec_field( "FFkpfm_t0sV" ,data_format=options.data_format)
        FFkpfm_tVs0, lvec, nDim, headORatoms = GU.load_vec_field( "FFkpfm_tVs0" ,data_format=options.data_format)

        FFkpfm_t0sV[0,:,:,:],FFkpfm_t0sV[1,:,:,:] = rotFF( FFkpfm_t0sV[0,:,:,:],FFkpfm_t0sV[1,:,:,:], opt_dict['rotate'] )
        FFkpfm_tVs0[0,:,:,:],FFkpfm_tVs0[1,:,:,:] = rotFF( FFkpfm_tVs0[0,:,:,:],FFkpfm_tVs0[1,:,:,:], opt_dict['rotate'] )

        FFkpfm_t0sV = FFkpfm_t0sV*opt_dict['pol_s']
        FFkpfm_tVs0 = FFkpfm_tVs0*opt_dict['pol_t']
    print(" load Lenard-Jones Force-field ")
    FFLJ, lvec, nDim, headORatoms = GU.load_vec_field( "FFLJ" , data_format=options.data_format)
    FFLJ[0,:,:,:],FFLJ[1,:,:,:] = rotFF( FFLJ[0,:,:,:],FFLJ[1,:,:,:], opt_dict['rotate'] )
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
                if aplied_bias:
                        dirname = "Q%1.2fK%1.2fV%1.2f" %(Q,K,Vx)
                else:
                        dirname = "Q%1.2fK%1.2f" %(Q,K)
                dirname = "Q%1.2fK%1.2f" %(Q,K)
                print(" relaxed_scan for ", dirname)
                if not os.path.exists( dirname ):
                    os.makedirs( dirname )
                fzs,PPpos,PPdisp,lvecScan=PPH.perform_relaxation(lvec, FFLJ, FFel, FFboltz,options.tipspline)
                #PPC.setTip( kSpring = np.array((K,K,0.0))/-PPU.eVA_Nm )
                #Fs,rPPs,rTips = PPH.relaxedScan3D( xTips, yTips, zTips )
                #GU.save_scal_field( dirname+'/OutFz', Fs[:,:,:,2], lvecScan, data_format=data_format )
                if PPU.params['tiltedScan']:
                    GU.save_vec_field( dirname+'/OutF', fzs, lvecScan, data_format=options.data_format, head = headORatoms, at_array = headORatoms )
                else:
                    GU.save_scal_field( dirname+'/OutFz', fzs, lvecScan, data_format=options.data_format, head = headORatoms, at_array = headORatoms )
                if opt_dict['vib'] >= 0:
                    which = opt_dict['vib']
                    print(" === computing eigenvectors of dynamical matix which=%i ddisp=%f" %(which,PPU.params['ddisp']))
                    xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )
                    rTips = np.array(np.meshgrid(xTips,yTips,zTips)).transpose(3,1,2,0).copy()
                    evals,evecs = PPC.stiffnessMatrix( rTips.reshape((-1,3)), PPpos.reshape((-1,3)), which=which, ddisp=PPU.params['ddisp'] )
                    GU.save_vec_field( dirname+'/eigvalKs', evals   .reshape( rTips.shape ), lvecScan, data_format=data_format, head= headORatoms, at_array = headORatoms )
                    if which > 0: GU.save_vec_field( dirname+'/eigvecK1', evecs[0].reshape( rTips.shape ), lvecScan, data_format=data_format, head = headORatoms, at_array = headORatoms )
                    if which > 1: GU.save_vec_field( dirname+'/eigvecK2', evecs[1].reshape( rTips.shape ), lvecScan, data_format=data_format, head = headORatoms, at_array = headORatoms )
                    if which > 2: GU.save_vec_field( dirname+'/eigvecK3', evecs[2].reshape( rTips.shape ), lvecScan, data_format=data_format, head = headORatoms, at_array = headORatoms )
                    #print "SHAPE", PPpos.shape, xTips.shape, yTips.shape, zTips.shape
                if opt_dict['disp']:
                    GU.save_vec_field( dirname+'/PPdisp', PPdisp, lvecScan,data_format=options.data_format, head = headORatoms, at_array = headORatoms)
                if opt_dict['pos']:
                    GU.save_vec_field(dirname+'/PPpos', PPpos, lvecScan, data_format=options.data_format, head = headORatoms, at_array = headORatoms )
                if options.bI:
                    print("Calculating current from tip to the Boltzmann particle:")
                    I_in, lvec, nDim, headORatoms = GU.load_scal_field('I_boltzmann',
                    data_format=iptions.data_format)
                    I_out = GU.interpolate_cartesian( I_in, PPpos, cell=lvec[1:,:], result=None )
                    del I_in;
                    GU.save_scal_field(dirname+'/OutI_boltzmann', I_out, lvecScan,  data_format=options.data_format, head = headORatoms, at_array = headORatoms)
