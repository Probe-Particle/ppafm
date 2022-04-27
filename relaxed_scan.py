#!/usr/bin/python3 -u

import os
import numpy as np
import sys

import pyProbeParticle                as PPU     
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.core           as PPC
import pyProbeParticle.HighLevel      as PPH

# =============== arguments definition

from optparse import OptionParser
parser = OptionParser()
parser.add_option( "-k",       action="store"     , type="float",  help="tip stiffenss [N/m]" )
parser.add_option( "--krange", action="store"     , type="float",  help="tip stiffenss range (min,max,n) [N/m]", nargs=3)
parser.add_option( "-q",       action="store"     , type="float",  help="tip charge [e]" )
parser.add_option( "--qrange", action="store"     , type="float",  help="tip charge range (min,max,n) [e]", nargs=3)
parser.add_option( "-b", "--boltzmann"            , action="store_true", default=False, help="calculate forces with boltzmann particle" )
parser.add_option( "--bI"     ,action="store_true", default=False, help="calculate current between boltzmann particle and tip" )

parser.add_option( "--tip_base",       action="store_true", default=False, help="interpolates F_z field in the position of the tip_base" )
parser.add_option( "--tipKPFM",        action="store", type="float", help="file STH like KPFM because of PP movements, store here heighest sample metal atom /z/ position", nargs=1 )

parser.add_option( "--pos",       action="store_true",      default=False, help="save probe particle positions" )
parser.add_option( "--vib",       action="store" , type="int", default=-1, help="map PP vibration eigenmodes; 0-just eigenvals; 1-3 eigenvecs" )
parser.add_option( "--disp",      action="store_true",      default=False, help="save probe particle displacements")
parser.add_option( "--tipspline", action="store" , type="string", help="file where spline is stored", default=None )
parser.add_option( "--npy" , action="store_true" , help="load and save fields in npy instead of xsf", default=False)

parser.add_option( "--stm" , action="store_true" , help="load and save data for the PPSTM code"     , default=False)


(options, args) = parser.parse_args()
opt_dict = vars(options)

data_format ="npy" if options.npy else "xsf"

# =============== Setup

PPU.loadParams( 'params.ini' )

print(opt_dict)
# Ks
Bds = False # Boolean double stiffness - possible double stifnness in params.ini
if opt_dict['krange'] is not None:
    Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], int( opt_dict['krange'][2] ) )
elif opt_dict['k'] is not None:
    Ks = [ opt_dict['k'] ]
else:
    Ks = [ PPU.params['stiffness'][0] ] ; Bds = True # Boolean double stiffness - possible double stifnness in params.ini
# Qs

charged_system=False
if opt_dict['qrange'] is not None:
    Qs = np.linspace( opt_dict['qrange'][0], opt_dict['qrange'][1], int( opt_dict['qrange'][2] ) )
elif opt_dict['q'] is not None:
    Qs = [ opt_dict['q'] ]
else:
    Qs = [ PPU.params['charge'] ]

for iq,Q in enumerate(Qs):
    if ( abs(Q) > 1e-7):
        charged_system=True

if options.tipspline is not None :
    try:
        S = np.genfromtxt(options.tipspline )
        print(" loading tip spline from "+options.tipspline)
        xs   = S[:,0].copy();  print("xs: ",   xs)
        ydys = S[:,1:].copy(); print("ydys: ", ydys)
        PPC.setTipSpline( xs, ydys )
        #Ks   = [0.0]
    except:
        print("cannot load tip spline from "+options.tipspline)
        sys.exit()

tip_base=options.tip_base
if not tip_base:
    tip_base = True if ((PPU.params["tip_base"][0]  != 'None') and (PPU.params["tip_base"][0] != None)) else False
KPFM   = True if ((PPU.params["KPFM"][0]  != 'None') or (PPU.params["KPFM"][0] != None)) else False
KPFM2  = False
KPFM_O = True if ((PPU.params["KPFM_O"] == 'Yes') or (PPU.params["KPFM_O"] == "Yes") or (PPU.params["KPFM_O"] == 'yes') or (PPU.params["KPFM_O"] == "yes")) else False
KPFM_C = True if ((PPU.params["KPFM_C"][0] == 'Yes') or (PPU.params["KPFM_C"][0] == "Yes") or (PPU.params["KPFM_C"][0]  == 'yes') or (PPU.params["KPFM_C"][0] == "yes")) else False

print("DEBUG: KPFM", KPFM, "KPFM2", KPFM2, "KPFM_O", KPFM_O, "KPFM_C", KPFM_C)


print("Ks   =", Ks) 
print("Qs   =", Qs) 
print("tip_base =", tip_base)

print(" ============= RUN  ")

if ( charged_system == True):
        print(" load Electrostatic Force-field ")
        FFel, lvec, nDim = GU.load_vec_field( "FFel" ,data_format=data_format)

if (options.boltzmann  or options.bI) :
        print(" load Boltzmann Force-field ")
        FFboltz, lvec, nDim = GU.load_vec_field( "FFboltz", data_format=data_format)

if KPFM_C:
        print ("calculating alpha and beta for KPFM_C")
        alpha = float(PPU.params["KPFM_C"][1])/np.sqrt( PPU.params["r0Probe"][0]**2 + PPU.params["r0Probe"][1]**2 + PPU.params["r0Probe"][2]**2 )
        beta  = alpha*float(PPU.params["KPFM_C"][2])
        print ("alpha", alpha)
        print ("beta" , beta)

print(" load Lenard-Jones Force-field ")
FFLJ, lvec, nDim = GU.load_vec_field( "FFLJ" , data_format=data_format)
PPU.lvec2params( lvec )
PPC.setFF( FFLJ )

xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )

for iq,Q in enumerate( Qs ):
    if ( charged_system == True):
        FF = FFLJ + FFel * Q
    else:
        FF = FFLJ
    if options.boltzmann :
        FF += FFboltz
    PPC.setFF_Fpointer( FF )
    for ik,K in enumerate( Ks ):
        dirname = "Q%1.2fK%1.2f" %(Q,K)
        print(" relaxed_scan for ", dirname)
        if not os.path.exists( dirname ):
            os.makedirs( dirname )
        if Bds:
            PPC.setTip( kSpring = np.array((PPU.params['stiffness'][0],PPU.params['stiffness'][1],0.0))/-PPU.eVA_Nm  )
        else:
            PPC.setTip( kSpring = np.array((K,K,0.0))/-PPU.eVA_Nm  )
        Fs,rPPs,rTips = PPH.relaxedScan3D( xTips, yTips, zTips )
        GU.save_scal_field( dirname+'/OutFz', Fs[:,:,:,2], lvecScan, data_format=data_format )
        if opt_dict['vib'] >= 0:
            which = opt_dict['vib']
            print(" === computing eigenvectors of dynamical matix which=%i ddisp=%f" %(which,PPU.params['ddisp']))
            evals,evecs = PPC.stiffnessMatrix( rTips.reshape((-1,3)), rPPs.reshape((-1,3)), which=which, ddisp=PPU.params['ddisp'] )
            GU.save_vec_field( dirname+'/eigvalKs', evals   .reshape( rTips.shape ), lvecScan, data_format=data_format )
            if which > 0: GU.save_vec_field( dirname+'/eigvecK1', evecs[0].reshape( rTips.shape ), lvecScan, data_format=data_format )
            if which > 1: GU.save_vec_field( dirname+'/eigvecK2', evecs[1].reshape( rTips.shape ), lvecScan, data_format=data_format )
            if which > 2: GU.save_vec_field( dirname+'/eigvecK3', evecs[2].reshape( rTips.shape ), lvecScan, data_format=data_format )
        #print "SHAPE", PPpos.shape, xTips.shape, yTips.shape, zTips.shape
        if opt_dict['disp']:
            GU.save_vec_field( dirname+'/PPdisp', rPPs-rTips+PPU.params['r0Probe'][0], lvecScan, data_format=data_format )
        if ( opt_dict['pos'] or opt_dict['stm']):
            GU.save_vec_field( dirname+'/PPpos', rPPs, lvecScan, data_format=data_format ) 
            # Please do not change this procedure, especialy the lvecScan - it is important for the STM calculations!
        if options.bI:
            print("Calculating current from tip to the Boltzmann particle:")
            I_in, lvec, nDim = GU.load_scal_field('I_boltzmann', data_format=data_format)
            I_out = GU.interpolate_cartesian( I_in, rPPs, cell=lvec[1:,:], result=None ) 
            del I_in;
            GU.save_scal_field( dirname+'/OutI_boltzmann', I_out, lvecScan, data_format=data_format)
        if tip_base:
            print("Interpolating FFel_tip_z in position of the tip_base. Beware, this is higher than the PP.")
            Ftip_in, lvec, nDim = GU.load_scal_field('FFel_tip', data_format=data_format)
            Ftip_out = GU.interpolate_cartesian( Ftip_in, rTips, cell=lvec[1:,:], result=None ) 
            del Ftip_in;
            GU.save_scal_field( './OutFzTip_base', Ftip_out, lvecScan, data_format=data_format)
            tip_base = False
        if KPFM:
            print("Interpolating FFel_KPFM in position of the tip_base. Beware, this is higher than the PP. the z-shift from the tip-base is: ", PPU.params["KPFM"][2])
            Ftip_in, lvec, nDim = GU.load_scal_field('FFel_kpfm', data_format=data_format)
            rTMP = rTips; rTMP[:,:,:,2]+=float(PPU.params["KPFM"][2]) ; # shifting z from the tip-base
            Ftip_out = float(PPU.params["KPFM"][3])*GU.interpolate_cartesian( Ftip_in, rTMP, cell=lvec[1:,:], result=None ) 
            GU.save_scal_field( './OutEz_KPFM', Ftip_out, lvecScan, data_format=data_format);
            KPFM = False; KPFM2 = True; del Ftip_out, rTMP;
        if KPFM2 and (KPFM_O):
            print("Interpolating FFel_KPFM in position of the PP")
            Ftip_out = float(PPU.params["KPFM"][3])*GU.interpolate_cartesian( Ftip_in, rPPs, cell=lvec[1:,:], result=None ) 
            GU.save_scal_field( dirname+'/OutEz_KPFM_O', Ftip_out, lvecScan, data_format=data_format)
            del Ftip_out;
        if KPFM2 and (KPFM_C):
            print("Interpolating FFel_KPFM in position of the imaginary 2ndPP-C")
            rTMP = (1-beta)*rTips + beta*rPPs ; rTMP[:,:,:,2] = (1-alpha)*rTips[:,:,:,2] + alpha*rPPs[:,:,:,2] ; # x,y 1st term & z 2nd term
            Ftip_out = float(PPU.params["KPFM"][3])*GU.interpolate_cartesian( Ftip_in, rTMP, cell=lvec[1:,:], result=None ) 
            GU.save_scal_field( dirname+'/OutEz_KPFM_C', Ftip_out, lvecScan, data_format=data_format)
            del Ftip_out, rTMP;
        if opt_dict['tipKPFM'] is not None:
            print("Trying to kickout something about KPFM if there are changes in positions")
            tip_kpfm= (rTips[:,:,:,2]-opt_dict['tipKPFM'])/(rPPs[:,:,:,2]-opt_dict['tipKPFM'])
            GU.save_scal_field( dirname+'/tip_KPFM', tip_kpfm, lvecScan, data_format=data_format)
            del tip_kpfm;

        # the rest is done in plot_results.py; For df, go to plot_results.py

print(" ***** ALL DONE ***** ")

#plt.show()
