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

from   pyProbeParticle import basUtils
import pyProbeParticle.common    as PPU
import pyProbeParticle.GridUtils as GU

import pyopencl as cl
import pyProbeParticle.ocl.HighLevel as hl

# ==== Setup

# --- Input files
dirNames  = ["out0"]

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
npbc = (1,1,1)
#npbc = None
scan_dim   = ( 100, 100, 20)
distAbove = 7.5
islices   = [0,+2,+4,+6,+8,+10,+12,+14,+16]

relax_params = np.array( [ 0.1,0.9,0.1*0.2,0.1*5.0], dtype=np.float32 );
stiffness    = np.array( [0.24,0.24,0.0, 30.0     ], dtype=np.float32 ); stiffness/=-16.0217662;

# === Main

if __name__ == "__main__":

    FFcl = hl.FFcl; oclr = hl.oclr;
    FFcl.init()
    oclr.init()

    rotations = hl.PPU.genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, 8) )
    #print "rotations: ", rotations

    typeParams = hl.loadSpecies('atomtypes.ini')
    ff_dim     = hl.genFFSampling(lvec, pixPerAngstrome );  print("ff_dim ", ff_dim)
    poss       = FFcl.getposs( lvec, ff_dim );              print("poss.shape ", poss.shape)

    forcefield = FFcl.ForceField_LJC()

    scanner = oclr.RelaxedScanner()
    scanner.relax_params = relax_params
    scanner.stiffness    = stiffness

    for dirName in dirNames:
        print(" ==================", dirName)
        t1ff = time.clock();
        atom_lines = open( dirName+"/pos.xyz" ).readlines()
        #FF, atoms, natoms0 =  hl.makeFF_LJC( poss, atom_lines, typeParams, iZPP, lvec, npbc=(1,1,1) )
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( atom_lines )
        natoms0 = len(Zs)
        if( npbc is not None ):
            #Zs, xyzs, qs = PPU.PBCAtoms( Zs, xyzs, qs, avec=self.lvec[1], bvec=self.lvec[2] )
            Zs, xyzs, qs = PPU.PBCAtoms3D( Zs, xyzs, qs, lvec[1:], npbc=npbc )
        cLJs  = PPU.getAtomsLJ( iZPP, Zs, typeParams ).astype(np.float32)
        #FF = forcefield.makeFF( xyzs, qs, cLJs, lvec=lvec )
        FF,atoms = forcefield.makeFF( xyzs, qs, cLJs, poss=poss )

        #forcefield.updateBuffers(atoms,CLs)  # TODO : we can save copying of poss here
        #FF = forcefield..run()
        #forcefield..releaseBuffers()

        FEin  = FF[:,:,:,:4] + Q*FF[:,:,:,4:];   del FF
        Tff = time.clock()-t1ff;   print("Tff %f [s]" %Tff)
        #GU.saveXSF( dirName+'/Fin_z.xsf',  FEin[:,:,:,2], lvec ); 

        scanner.prepareBuffers( FEin, lvec, scan_dim=scan_dim )

        for irot,rot in enumerate(rotations):
            # print "rotation #",irot, rot
            subDirName = dirName + ("/rot%02i" %irot)
            if os.path.isdir(subDirName):
                #os.rm(subDirName)
                shutil.rmtree(subDirName)
            os.makedirs( subDirName )

            t1scan = time.clock();
            pos0  = hl.posAboveTopAtom( atoms[:natoms0], rot[2], distAbove=distAbove )
            scanner.setScanRot( pos0, rot=rot, start=(-10.0,-10.0), end=(10.0,10.0) )
            FEout = scanner.run()
            Tscan = time.clock()-t1scan;  print("Tscan %f [s]" %Tscan)

            for isl in islices:
                plt.imshow( FEout[:,:,isl,2] )
                plt.savefig( subDirName+( "/FoutZ%03i.png" %isl ), bbox_inches="tight"  ); 
                plt.close()

        scanner.releaseBuffers()

#plt.show()

