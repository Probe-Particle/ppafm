#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases


import sys
import os
import shutil
import time
import random
import matplotlib;
import numpy as np
from enum import Enum

import matplotlib as mpl;  mpl.use('Agg'); print("plot WITHOUT Xserver");
import matplotlib.pyplot as plt

#sys.path.append("/home/prokop/git/ProbeParticleModel_OCL") 
#import pyProbeParticle.GridUtils as GU

#from   pyProbeParticle import basUtils
#from   pyProbeParticle import PPPlot 
import pyProbeParticle.GridUtils as GU
#import pyProbeParticle.common    as PPU
#import pyProbeParticle.cpp_utils as cpp_utils

import pyopencl as cl
#import pyProbeParticle.oclUtils     as oclu 
#import pyProbeParticle.fieldOCL     as FFcl 
#import pyProbeParticle.RelaxOpenCL  as oclr
import pyProbeParticle.ocl.HighLevel as hl

hl.FFcl.init()
hl.oclr.init()

# ==== Setup

# --- Input files
dirNames  = ["out0"]
#dirNames  = ["out0", "out1", "out2", "out3", "out4", "out5", "out6" ] 
#geomFileNames = ["out0/pos.xyz", "out1/pos.xyz", "out2/pos.xyz", "out3/pos.xyz", "out4/pos.xyz", "out5/pos.xyz", "out6/pos.xyz" ]

# --- ForceField
pixPerAngstrome = 10
iZPP = 8
Q    = -0.1;
bPBC = True
lvec = np.array([
    [ 0.0,  0.0,  0.0],
    [19.0,  0.0,  0.0],
    [ 0.0, 20.0,  0.0],
    [ 0.0,  0.0, 21.0]
])

# --- Relaxation

relax_dim   = ( 100, 100, 20)

distAbove = 7.5
islices   = [0,+2,+4,+6,+8]

relax_params = np.array( [ 0.1,0.9,0.1*0.2,0.1*5.0], dtype=np.float32 );
dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
stiffness    = np.array( [0.24,0.24,0.0, 30.0     ], dtype=np.float32 ); stiffness/=-16.0217662;
dpos0        = np.array([0.0,0.0,0.0,4.0], dtype=np.float32 ); 
dpos0[2]     = -np.sqrt( dpos0[3]**2 - dpos0[0]**2 + dpos0[1]**2 ); 
print("dpos0 ", dpos0)

# === Main

if __name__ == "__main__":
    rotations = hl.PPU.genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, 8) )
    #print "rotations: ", rotations

    typeParams = hl.loadSpecies('atomtypes.ini')
    invCell    = hl.oclr.getInvCell(lvec)
    ff_dim     = hl.genFFSampling(lvec, pixPerAngstrome );     print("ff_dim ", ff_dim)
    poss       = hl.FFcl.getposs( lvec, ff_dim );              print("poss.shape ", poss.shape)

    for dirName in dirNames:
        print(" ==================", dirName)
        #t1ff = time.clock();
        atom_lines = open( dirName+"/pos.xyz" ).readlines()
        FF, atoms, natoms0 =  hl.makeFF_LJC( poss, atom_lines, typeParams, iZPP, lvec, npbc=(1,1,1) )
        FEin  = FF[:,:,:,:4] + Q*FF[:,:,:,4:];   del FF
        #Tff = time.clock()-t1ff;
        #GU.saveXSF( dirName+'/Fin_z.xsf',  FEin[:,:,:,2], lvec ); 

        #print "FEin.shape ", FEin.shape;
        relax_args  = hl.oclr.prepareBuffers( FEin, relax_dim )
        #cl.enqueue_copy( oclr.oclu.queue, relax_args[0], FEin, origin=(0,0,0), region=FEin.shape[:3][::-1] )

        for irot,rot in enumerate(rotations):
            # print "rotation #",irot, rot
            subDirName = dirName + ("/rot%02i" %irot)
            if os.path.isdir(subDirName):
                #os.rm(subDirName)
                shutil.rmtree(subDirName)
            os.makedirs( subDirName )

            #t1relax = time.clock();
            #imax,xdirmax  = PPU.maxAlongDir(atoms[:natoms0], rot[2] )
            #pos0 = atoms[imax,:3] + rot[2] * distAbove;
            pos0  = hl.posAboveTopAtom( atoms[:natoms0], rot[2], distAbove=distAbove )
            FEout = hl.scanFromRotation( relax_args, pos0, rot, invCell, ndim=relax_dim, span=(10.0,10.0), tipR0=4.0, zstep=0.1, stiffness=stiffness, relax_params=relax_params, debugXYZ=(subDirName + "/debugPos0.xyz",atom_lines) )
            #Trelax = time.clock() - t1relax;

            #writeDebugXYZ( subDirName + "/debugPosRlaxed.xyz", atom_lines, FEout[::10,::10,:,:].reshape(-1,4) )
            #GU.saveXSF( geomFileName+'_Fout_z.xsf',  FEout[:,:,:,2], lvec );
            #print "FEout.shape ", FEout.shape
            #np.save( subDirName+'/Fout_z.npy', FEout[:,:,:,2] )
            #t1plot = time.clock();
            for isl in islices:
                plt.imshow( FEout[:,:,isl,2] )
                plt.savefig( subDirName+( "/FoutZ%03i.png" %isl ), bbox_inches="tight"  ); 
                plt.close()
            #Tplot = time.clock()-t1plot;

            #Ttot = time.clock()-t1tot;
            #print "Timing[s] Ttot %f Tff %f Trelax %f Tprepare %f Tplot %f " %(Ttot, Tff, Trelax, Tprepare, Tplot)
        
        hl.oclr.releaseArgs(relax_args)

#plt.show()

