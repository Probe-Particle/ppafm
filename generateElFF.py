#!/usr/bin/python
import sys
import numpy as np
import os
import __main__ as main


import pyProbeParticle                as PPU
#from   pyProbeParticle            import elements   
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT
import pyProbeParticle.HighLevel      as PPH
import pyProbeParticle.cpp_utils      as cpp_utils
import pyProbeParticle.basUtils       as BU

if __name__=="__main__":
    HELP_MSG="""Use this program in the following way:
    """+os.path.basename(main.__file__) +""" -i <filename> [ --sigma <value> ]
    Supported file fromats are:
       * cube
       * xsf """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option( "-i", "--input", action="store", type="string", help="format of input file")
    parser.add_option( "--tip_dens", action="store", type="string", default=None, help="tip enisty file (.xsf)" )
    #parser.add_option( "--sub_core",  action="store_true",  help="subtract core density", default=False )
    parser.add_option( "--Rcore",   default=PPU.params["Rcore"],    action="store", type="float", help="Width of nuclear charge density blob to achieve charge neutrality [Angstroem]" )
    parser.add_option( "-t", "--tip", action="store", type="string", help="tip model (multipole) {s,pz,dz2,..}", default=None)
    parser.add_option( "--tilt", action="store", type="float", help="tilt of tip electrostatic field (radians)", default=0 )
    parser.add_option( "-E", "--energy", action="store_true",  help="pbc False", default=False)
    parser.add_option("--noPBC", action="store_false",  help="pbc False",dest="PBC", default=None)
    parser.add_option( "-w", "--sigma", action="store", type="float",help="gaussian width for convolution in Electrostatics [Angstroem]", default=None)
    parser.add_option("-f","--data_format" , action="store" , type="string", help="Specify the output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
    (options, args) = parser.parse_args()

    #print "options.tip_dens ", options.tip_dens;  exit() 

    if options.input is None:
        sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MSG)
    opt_dict = vars(options)

    if os.path.isfile( 'params.ini' ):
        FFparams=PPU.loadParams( 'params.ini' ) 
    else:
        print ">> LOADING default params.ini >> 's' ="  
        FFparams = PPU.loadParams( cpp_utils.PACKAGE_PATH+'/defaults/params.ini' )
    #PPU.loadParams( 'params.ini' )
    PPU.apply_options(opt_dict)    

    if os.path.isfile( 'atomtypes.ini' ):
        print ">> LOADING LOCAL atomtypes.ini"  
        FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
    else:
        FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )

    if options.Rcore > 0.0:  # We do it here, in case it crash we don't want to wait for all the huge density files to load
        if options.tip_dens is None: raise Exception( " Rcore>0 but no tip density provided ! " )
        valElDict        = PPH.loadValenceElectronDict()
        Rs_tip,elems_tip = PPH.getAtomsWhichTouchPBCcell( options.tip_dens, Rcut=options.Rcore )

    atoms_samp,nDim_samp,lvec_samp = BU.loadGeometry( options.input, params=PPU.params )
    head_samp                      = BU.primcoords2Xsf( atoms_samp[0], [atoms_samp[1],atoms_samp[2],atoms_samp[3]], lvec_samp )

    V=None
    if(options.input.lower().endswith(".xsf") ):
        print ">>> loading Hartree potential from  ",options.input,"..."
        print "Use loadXSF"
        V, lvec, nDim, head = GU.loadXSF(options.input)
    elif(options.input.lower().endswith(".cube") ):
        print " loading Hartree potential from ",options.input,"..."
        print "Use loadCUBE"
        V, lvec, nDim, head = GU.loadCUBE(options.input)
    
    if PPU.params['tip']==".py":
        #import tip
        execfile("tip.py")
        print tipMultipole
        PPU.params['tip'] = tipMultipole
        print " PPU.params['tip'] ", PPU.params['tip']

    if options.tip_dens is not None:
        '''
        ###  NO NEED TO RENORMALIZE : fieldFFT already works with density
        rho_tip, lvec_tip, nDim_tip, head_tip = GU.loadXSF( options.tip_dens )
        rho_tip *= GU.dens2Q_CHGCARxsf(rho_tip, lvec_tip)
        PPU.params['tip'] = rho_tip
        print " dens_tip check_sum Q =  ", np.sum( rho_tip )
        '''
        print ">>> loading tip density from ",options.tip_dens,"..."
        rho_tip, lvec_tip, nDim_tip, head_tip = GU.loadXSF( options.tip_dens )

        if options.Rcore > 0.0:
            print ">>> subtracting core densities from rho_tip ... "
            #subtractCoreDensities( rho_tip, lvec_tip, fname=options.tip_dens, valElDict=valElDict, Rcore=options.Rcore )
            PPH.subtractCoreDensities( rho_tip, lvec_tip, elems=elems_tip, Rs=Rs_tip, valElDict=valElDict, Rcore=options.Rcore, head=head_tip )

        PPU.params['tip'] = rho_tip

    print ">>> calculating electrostatic forcefiled with FFT convolution as Eel(R) = Integral( rho_tip(r-R) V_sample(r) ) ... "
    #FFel,Eel=PPH.computeElFF(V,lvec,nDim,PPU.params['tip'],Fmax=10.0,computeVpot=options.energy,Vmax=10, tilt=opt_dict['tilt'] )
    FFel,Eel=PPH.computeElFF(V,lvec,nDim,PPU.params['tip'],computeVpot=options.energy , tilt=opt_dict['tilt'] )
    
    print ">>> saving electrostatic forcefiled ... "
    
    GU.save_vec_field('FFel',FFel,lvec_samp ,data_format=options.data_format, head=head_samp)
    if options.energy:
        GU.save_scal_field( 'Eel', Eel, lvec_samp, data_format=options.data_format)
    del FFel,V;
    
    
    
    
