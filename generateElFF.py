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

valElDict = {
1:1.0, 2:2.0, 
3:1.0, 4:2.0, 5:3.0, 6:4.0, 7:5.0, 8:6.0, 
9:7.0, 10:8.0, 11:1.0, 12:2.0, 13:3.0, 14:4.0, 15:5.0, 16:6.0, 17:7.0, 18:8.0,
35:7.0, 36:8.0,
53:7.0,  54:8.0
}

if __name__=="__main__":
    HELP_MSG="""Use this program in the following way:
    """+os.path.basename(main.__file__) +""" -i <filename> [ --sigma <value> ]
    Supported file fromats are:
       * cube
       * xsf """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option( "-i", "--input", action="store", type="string", help="format of input file")
    parser.add_option( "--tip_dens", action="store", type="string",  help="tip enisty file (.xsf)")
    parser.add_option( "--sub_core",  action="store_true",  help="subtract core density", default=False )
    parser.add_option( "-t", "--tip", action="store", type="string", help="tip model (multipole) {s,pz,dz2,..}", default=None)
    parser.add_option( "--tilt", action="store", type="float", help="tilt of tip electrostatic field (radians)", default=0 )
    parser.add_option( "-E", "--energy", action="store_true",  help="pbc False", default=False)
    parser.add_option("--noPBC", action="store_false",  help="pbc False",dest="PBC", default=None)
    parser.add_option( "-w", "--sigma", action="store", type="float",help="gaussian width for convolution in Electrostatics [Angstroem]", default=None)
    parser.add_option("-f","--data_format" , action="store" , type="string", help="Specify the output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
    (options, args) = parser.parse_args()
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

    V=None
    if(options.input.lower().endswith(".xsf") ):
        print " loading Hartree potential from disk "
        print "Use loadXSF"
        V, lvec, nDim, head = GU.loadXSF(options.input)
    elif(options.input.lower().endswith(".cube") ):
        print " loading Hartree potential from disk "
        print "Use loadCUBE"
        V, lvec, nDim, head = GU.loadCUBE(options.input)
    
    if PPU.params['tip']==".py":
        #import tip
        execfile("tip.py")
        print tipMultipole
        PPU.params['tip'] = tipMultipole
        print " PPU.params['tip'] ", PPU.params['tip']

    if options.tip_dens:
        '''
        ###  NO NEED TO RENORMALIZE : fieldFFT already works with density
        rho_tip, lvec_tip, nDim_tip, head_tip = GU.loadXSF( options.tip_dens )
        rho_tip *= GU.dens2Q_CHGCARxsf(rho_tip, lvec_tip)
        PPU.params['tip'] = rho_tip
        print " dens_tip check_sum Q =  ", np.sum( rho_tip )
        '''
        rho_tip, lvec_tip, nDim_tip, head_tip = GU.loadXSF( options.tip_dens )

        #if "AECCAR" in options.tip_dens:
        #    Vol = np.abs( np.linalg.det(lvec_tip[1:]) )
        #    rho_, lvec_, nDim_, head_ = GU.loadXSF( "../tip/AECCAR0.xsf" )
        #    rho_tip += rho_
        #    rho_tip /= Vol

        if options.sub_core:
            import pyProbeParticle.basUtils                as BU
            atoms,nDim,lvec     = BU.loadGeometry( options.tip_dens, params=PPU.params )
            #valElDict = { iZ:float(iZ) for iZ in atoms[0] }
            #print valElDict
            atoms_ = np.array(atoms)

            dV = np.abs(np.linalg.det(lvec_tip[1:]))/(nDim_tip[0]*nDim_tip[1]*nDim_tip[2])
            print "sum(RHO), Nelec",  rho_tip.sum(), dV, rho_tip.sum()*dV
            fFFT.addCoreDensities( atoms_, valElDict, rho_tip, lvec_tip, sigma=0.1 )
            print "sum(RHO), Nelec",  rho_tip.sum(), dV, rho_tip.sum()*dV
            #exit()
        PPU.params['tip'] = rho_tip

    #FFel,Eel=PPH.computeElFF(V,lvec,nDim,PPU.params['tip'],Fmax=10.0,computeVpot=options.energy,Vmax=10, tilt=opt_dict['tilt'] )
    FFel,Vel=PPH.computeElFF(V,lvec,nDim,PPU.params['tip'],computeVpot=options.energy , tilt=opt_dict['tilt'] )
    
    print " saving electrostatic forcefiled "
    
    if options.data_format=='xsf':
        import pyProbeParticle.basUtils  as BU
        atoms,nDim,lvec   = BU.loadGeometry( options.input, params=PPU.params )
        head              = BU.primcoords2Xsf( atoms[0], [atoms[1],atoms[2],atoms[3]], lvec )
        #print "atoms: ", atoms
        print "head:  ", head
    else:
        import pyProbeParticle.basUtils  as BU
        atoms,nDim,lvec   = BU.loadGeometry( options.input, params=PPU.params )
        head              = [ atoms[0], [atoms[1],atoms[2],atoms[3]], lvec ]
        #print "atoms: ", atoms
        print "head:  ", head
        
    GU.save_vec_field('FFel',FFel,lvec,data_format=options.data_format, head=head)
    if options.energy :
        GU.save_scal_field( 'Vel', Vel, lvec, data_format=options.data_format)
    del FFel,V;
    
    
    
    
