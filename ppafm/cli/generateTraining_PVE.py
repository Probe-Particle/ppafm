#!/usr/bin/python -u

import os

import numpy as np

import ppafm as PPU
import ppafm.core as PPC
import ppafm.cpp_utils as cpp_utils
import ppafm.fieldFFT as fFFT
import ppafm.HighLevel as PPH
from ppafm import io

#file_format = "cube"
file_format = "xsf"

# =============== arguments definition

PPU.loadParams( 'params.ini' )

if os.path.isfile( 'atomtypes.ini' ):
    print(">> LOADING LOCAL atomtypes.ini")
    FFparams=PPU.loadSpecies('atomtypes.ini')
else:
    FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH / 'defaults' / 'atomtypes.ini' )

elem_dict   = PPU.getFFdict(FFparams); # print elem_dict
iPP         = PPU.atom2iZ( PPU.params['probeType'], elem_dict )

# -- load CO tip
drho_tip,lvec_dt, ndim_dt, atomic_info_or_head = io.load_scal_field( "drho_tip",data_format=file_format)
rho_tip ,lvec_t,  ndim_t , atomic_info_or_head = io.load_scal_field( "rho_tip" ,data_format=file_format)

#PPU      .params['gridN'] = ndim_t
PPU      .params['gridN'] = ndim_t[::-1];
PPU.params['gridA'] = lvec_t[1]; PPU.params['gridB'] = lvec_t[2]; PPU.params['gridC'] = lvec_t[3] # must be before parseAtoms
print(PPU.params['gridN'],        PPU.params['gridA'],           PPU.params['gridB'],           PPU.params['gridC'])

FF,V                = PPH.prepareArrays( None, False )

print("FFLJ.shape",FF.shape)
PPC.setFF_shape( np.shape(FF), lvec_t )

base_dir = os.getcwd()
paths=["out1","out2"]


for path in paths:

    os.chdir( path )

    # === load data

    atoms,nDim,lvec     = io.loadGeometry( "V.xsf", params=PPU.params )

    # === generate FF vdW

    iZs,Rs,Qs           = PPU.parseAtoms(atoms, elem_dict, autogeom=False, PBC = PPU.params['PBC'] )

    FF[:,:,:,:] = 0
    cLJs = PPU.getAtomsLJ( iPP, iZs, FFparams ); # print "cLJs",cLJs; np.savetxt("cLJs_3D.dat", cLJs);  exit()
    PPC.getVdWFF( Rs, cLJs )       # THE MAIN STUFF HERE

    # === generate FF Pauli

    rho1,lvec1, ndim1, atomic_info_or_head = io.load_scal_field( "rho",data_format=file_format)

    #print "rho1.shape, FF.shape ", rho1.shape, FF.shape
    #exit()

    Fx,Fy,Fz,E = fFFT.potential2forces_mem( rho1, lvec1, rho1.shape, rho=rho_tip, doForce=True, doPot=False, deleteV=True )
    FF[:,:,:,0] = Fx*PPU.params['Apauli']
    FF[:,:,:,1] = Fy*PPU.params['Apauli']
    FF[:,:,:,2] = Fz*PPU.params['Apauli']
    del Fx; del Fy; del Fz; del E;

    # === generate FF Electrostatic

    V_samp, lvec1, ndim1, atomic_info_or_head  = io.load_scal_field( "V",data_format=file_format)
    Fx,Fy,Fz,E = fFFT.potential2forces_mem( V_samp, lvec1, V_samp.shape, rho=drho_tip, doForce=True, doPot=False, deleteV=True )
    FF[:,:,:,0] = Fx*PPU.params['charge']
    FF[:,:,:,1] = Fy*PPU.params['charge']
    FF[:,:,:,2] = Fz*PPU.params['charge']
    del Fx; del Fy; del Fz; del E;

    # === relaxed scan

    #fzs,PPpos,PPdisp,lvecScan=PPH.perform_relaxation(lvec, FFvdW, FFel=FFel, FFpauli=FFpauli, FFboltz=FFboltz,tipspline=options.tipspline, bFFtotDebug=options.bDebugFFtot)
    xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )
    PPC.setTip( kSpring = np.array((PPU.params['klat'],PPU.params['klat'],0.0))/-PPU.eVA_Nm )
    fzs,PPpos = PPH.relaxedScan3D( xTips, yTips, zTips )

    io.save_scal_field( 'OutFz', fzs, lvecScan, data_format=file_format, head = atomic_info_or_head , atomic_info = atomic_info_or_head )

    os.chdir( base_dir )
