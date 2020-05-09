#!/usr/bin/python -u

import os
import numpy as np
#import matplotlib.pyplot as plt
import sys

import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.HighLevel      as PPH
import pyProbeParticle.cpp_utils      as cpp_utils

#import PPPlot         # we do not want to make it dempendent on matplotlib
print("Amplitude ", PPU.params['Amplitude'])

# =============== arguments definition


if __name__=="__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option( "--pos",       action="store_true", default=False, help="save probe particle positions" )
    parser.add_option( "--disp",      action="store_true", default=False, help="save probe particle displacements")
    parser.add_option( "--tipspline", action="store", type="string", help="file where spline is stored", default=None )
    parser.add_option("-f","--data_format" , action="store" , type="string",help="Specify the input/output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
    (options, args) = parser.parse_args()
    opt_dict = vars(options)
    # =============== Setup

    FFparams=None
    if os.path.isfile( 'atomtypes.ini' ):
        print(">> LOADING LOCAL atomtypes.ini")
        FFparams=PPU.loadSpecies( 'atomtypes.ini' )
    else:
        FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )

    PPU.loadParams( 'params.ini',FFparams=FFparams )
    print(opt_dict)
    charged_system=False
    KC =  PPU.params['Cklat']
    QC =  PPU.params['Ccharge']
    KO =  PPU.params['Oklat']
    QO =  PPU.params['Ocharge']
    if (abs(QC)>1e-5 or abs(QO)>1e-5):
        charged_system=True
    charged_system=True
    FFLJC, FFel, FFboltzC=None,None,None
    FFLJO, FFelTip,FFboltzO=None,None,None
    #PPPlot.params = PPU.params             # now we dont use PPPlot here
    if ( charged_system == True):
        print(" load Electrostatic Force-field ")
        FFel, lvec, nDim = GU.load_vec_field( "FFel" ,data_format=options.data_format)
        if (PPU.params['tip'] != None) and (PPU.params['tip'] != 'None') and (PPU.params['tip'] != "None"):
            print("DEBUG: PPU.params['tip']",PPU.params['tip'])
    FFelTip, lvec, nDim = GU.load_vec_field( "FFelTip" ,data_format=options.data_format)
    print(" load Lenard-Jones Force-field ")
    FFLJC, lvec, nDim = GU.load_vec_field( "FFLJC" , data_format=options.data_format)
    FFLJO, lvec, nDim = GU.load_vec_field( "FFLJO" , data_format=options.data_format)
    PPU.lvec2params( lvec )


    dirname = "Qo%1.2fQc%1.2fK%1.2f" %(QO,QC,KO)
    print(" relaxed_scan for ", dirname)
    if not os.path.exists( dirname ):
        os.makedirs( dirname )
    if (PPU.params['tip'] == None) or (PPU.params['tip'] == 'None') or (PPU.params['tip'] == "None"):
        fzs,PPpos,PPdisp,lvecScan=PPH.perform_relaxation(lvec, FFLJC,
        FFLJO,FFel,FFTip=FFel[:,:,:,2].copy()   ,FFboltz=None)
    else:
        fzs,PPpos,PPdisp,lvecScan=PPH.perform_relaxation(lvec, FFLJC,
        FFLJO,FFel,FFTip=FFelTip[:,:,:,2].copy(),FFboltz=None,tipspline=options.tipspline)


    GU.save_scal_field( dirname+'/OutFz', fzs, lvecScan,
                        data_format=options.data_format )
    if opt_dict['disp']:
        GU.save_vec_field( dirname+'/PPdisp', PPdisp,
                           lvecScan,data_format=options.data_format)
    if opt_dict['pos']:
        GU.save_vec_field(dirname+'/PPpos', PPpos, lvecScan,
                          data_format=options.data_format )

