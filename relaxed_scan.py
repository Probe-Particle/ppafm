#!/usr/bin/python -u

import os
import numpy as np
#import matplotlib.pyplot as plt
import sys

import pyProbeParticle                as PPU     
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.HighLevel      as PPH
import pyProbeParticle.cpp_utils      as cpp_utils

#import PPPlot 		# we do not want to make it dempendent on matplotlib
print "Amplitude ", PPU.params['Amplitude']

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
    	print ">> LOADING LOCAL atomtypes.ini"  
        FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
    else:
        FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
    
    PPU.loadParams( 'params.ini',FFparams=FFparams )
    print opt_dict
    charged_system=False
    KC =  PPU.params['Cklat']
    KO =  PPU.params['Oklat']
    QC =  PPU.params['Ocharge'] 
    QO =  PPU.params['Ccharge'] 
    if (abs(QC)>1e-5):
        charged_system=True
    FFLJC, FFel, FFboltzC=None,None,None 
    FFLJO, FFboltzO=None,None 
    #PPPlot.params = PPU.params 			# now we dont use PPPlot here
    if ( charged_system == True):
        print " load Electrostatic Force-field "
        FFel, lvec, nDim = GU.load_vec_field( "FFel" ,data_format=options.data_format)
    print " load Lenard-Jones Force-field "
    FFLJC, lvec, nDim = GU.load_vec_field( "FFLJC" , data_format=options.data_format)
    FFLJO, lvec, nDim = GU.load_vec_field( "FFLJO" , data_format=options.data_format)
    PPU.lvec2params( lvec )


    dirname = "Q%1.2fK%1.2f" %(QC,KC)
    print " relaxed_scan for ", dirname
    if not os.path.exists( dirname ):
    	os.makedirs( dirname )
    fzs,PPpos,PPdisp,lvecScan=PPH.perform_relaxation(lvec, FFLJC,
    FFLJO,FFel,None,options.tipspline)
    GU.save_scal_field( dirname+'/OutFz', fzs, lvecScan,
                        data_format=options.data_format )
    if opt_dict['disp']:
        GU.save_vec_field( dirname+'/PPdisp', PPdisp,
                           lvecScan,data_format=options.data_format)
    if opt_dict['pos']:
        GU.save_vec_field(dirname+'/PPpos', PPpos, lvecScan,
                          data_format=options.data_format )

