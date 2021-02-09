#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases


import sys
import os
import time
import random
import matplotlib;
import numpy as np
from enum import Enum

import matplotlib as mpl;  mpl.use('Agg'); print("plot WITHOUT Xserver");
import matplotlib.pyplot as plt

#sys.path.append("/home/prokop/git/ProbeParticleModel_OCL") 
#import pyProbeParticle.GridUtils as GU

from   pyProbeParticle import basUtils
from   pyProbeParticle import PPPlot 
import pyProbeParticle.GridUtils as GU
import pyProbeParticle.common    as PPU
import pyProbeParticle.cpp_utils as cpp_utils

import pyopencl as cl
import pyProbeParticle.oclUtils    as oclu 
import pyProbeParticle.fieldOCL    as FFcl 
import pyProbeParticle.RelaxOpenCL as oclr

Modes     = Enum( 'Modes',    'LJel LJel_pbc LJQ' )
DataViews = Enum( 'DataViews','FFin FFout df FFel FFpl' )

FFcl.init()
oclr.init()

# ==== Setup

#geomFileNames = ["out0/pos.xyz", "out1/pos.xyz", "out2/pos.xyz", "out3/pos.xyz", "out4/pos.xyz", "out5/pos.xyz", "out6/pos.xyz" ]
geomFileNames = ["out0/pos.xyz" ]
relax_params = np.array([0.1,0.9,0.1*0.2,0.1*5.0], dtype=np.float32 );

mode = Modes.LJQ.name
Q    = 0.0;
iZPP = 8

#bPBC = False
bPBC = True

step = np.array( [ 0.1, 0.1, 0.1 ] )
rmin = np.array( [ 0.0, 0.0, 0.0 ] )
rmax = np.array( [ 20.0, 20.0, 20.0 ] )

rSliceAbove = np.array( [ 0.0, 0.0, 7.0 ] )
islices = [-2,0,+2,+4,+6,+8]

stiffness    = np.array([0.24,0.24,0.0, 30.0 ], dtype=np.float32 ); 
stiffness/=-16.0217662;
print("stiffness ", stiffness)

dpos0    = np.array([0.0,0.0,0.0,4.0], dtype=np.float32 ); 
dpos0[2] = -np.sqrt( dpos0[3]**2 - dpos0[0]**2 + dpos0[1]**2 ); 
print("dpos0 ", dpos0)

# === Functions

def maxAlongDir(atoms, hdir):
    #print atoms[:,:3]
    xdir = np.dot( atoms[:,:3], hdir[:,None] )
    #print xdir
    imin = np.argmax(xdir)
    return imin, xdir[imin][0]

#def getOptSlice( atoms, hdir ):
#    

def loadSpecies(fname):
    try:
        with open(fname, 'r') as f:  
            str_Species = f.read(); 
    except:
        print("defaul atomtypes.ini")
        with open(cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini', 'r') as f:  
            str_Species = f.read();
    str_Species = "\n".join( "\t".join( l.split()[:5] )  for l in str_Species.split('\n')  )
    print("str_Species")
    print(str_Species)
    return PPU.loadSpeciesLines( str_Species.split('\n') )

def updateFF_LJC( ff_args, iZPP, xyzs, Zs, qs, typeParams, pbcnx=0, func_runFF=FFcl.runLJC ):
    atoms   = FFcl.xyzq2float4( xyzs, qs )
    cLJs_   = PPU.getAtomsLJ( iZPP, Zs, typeParams )
    cLJs    = cLJs_.astype(np.float32)
    ff_args = FFcl.updateArgsLJC( ff_args, atoms, cLJs, poss )
    ff_nDim = poss.shape
    return func_runFF( ff_args, ff_nDim )

def updateFF_Morse( ff_args, iZPP, xyzs, Zs, qs, typeParams, pbcnx=0, func_runFF=FFcl.runLJ, alphaFac=1.0 ):
    atoms   = FFcl.xyzq2float4(xyzs,qs);
    REAs    = PPU.getAtomsREA( iZPP, Zs, typeParams, alphaFac=alphaFac )
    REAs    = REAs.astype(np.float32)
    ff_args = FFcl.updateArgsMorse( ff_args, atoms, REAs, poss ) 

def evalFFatoms_LJC( atoms, cLJs, poss, func_runFF=FFcl.runLJC ):
    ff_args = FFcl.initArgsLJC( atoms, cLJs, poss )
    ff_nDim = poss.shape[:3]
    FF      = func_runFF( ff_args, ff_nDim )
    FFcl.releaseArgs(ff_args)
    return FF

def evalFF_LJC( iZPP, xyzs, Zs, qs, poss, typeParams, func_runFF=FFcl.runLJC ):
    atoms   = FFcl.xyzq2float4(xyzs,qs);
    cLJs_   = PPU.getAtomsLJ( iZPP, Zs, typeParams )
    cLJs    = cLJs_.astype(np.float32)
    return evalFF_LJC( atoms, cLJs, poss, func_runFF=FFcl.runLJC )

# === Main

if __name__ == "__main__":
    typeParams = loadSpecies('atomtypes.ini')
    lvec       = np.genfromtxt('cel.lvs') 
    lvec       = np.insert( lvec, 0, 0.0, axis=0); 
    print("lvec ", lvec)
    invCell = oclr.getInvCell(lvec)
    print("invCell ", invCell)
    ff_nDim       = np.array([
                int(round(10*(lvec[1][0]+lvec[1][1]))),
                int(round(10*(lvec[2][0]+lvec[2][1]))),
                int(round(10*(lvec[3][2]           )))
            ])
    print("ff_nDim ", ff_nDim)
    poss       = FFcl.getposs( lvec, ff_nDim )
    print("poss.shape", poss.shape)

    relax_dim  = tuple( ((rmax-rmin)/step).astype(np.int32) )
    relax_poss = oclr.preparePoss( relax_dim, z0=rmax[2], start=rmin, end=rmax )
    #relax_args  = oclr.prepareBuffers( FEin, relax_dim )   # TODO

    for geomFileName in geomFileNames:
        t1tot = time.clock()
        print(" ==================", geomFileName)
        t1prepare = time.clock();
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( open( geomFileName ).readlines() )

        if(bPBC):
            Zs, xyzs, qs = PPU.PBCAtoms( Zs, xyzs, qs, avec=lvec[1], bvec=lvec[2] )
            #Zs, xyzs, qs = PBCAtoms3D( Zs, xyzs, qs, lvec[1:], npbc=[1,1,1] )
        atoms = FFcl.xyzq2float4(xyzs,qs)
        
        hdir  = np.array([0.0,0.0,1.0])
        imax,xdirmax  = maxAlongDir(atoms, hdir)
        izslice = int( round( ( rmax[2] - xdirmax - rSliceAbove[2] )/-oclr.DEFAULT_dTip[2] ) )
        print("izslice ", izslice, " xdirmax ", xdirmax, rmax[2], rmax[2] - xdirmax - rSliceAbove[2])

        cLJs  = PPU.getAtomsLJ( iZPP, Zs, typeParams ).astype(np.float32)
        Tprepare = time.clock()-t1prepare;

        t1ff = time.clock();
        FF    = evalFFatoms_LJC( atoms, cLJs, poss, func_runFF=FFcl.runLJC )
        FEin =  FF[:,:,:,:4] + Q*FF[:,:,:,4:] 
        Tff = time.clock()-t1ff;
        #GU.saveXSF( geomFileName+'_Fin_z.xsf',  FEin[:,:,:,2], lvec );
        np.save( geomFileName+'_Fin_z.npy', FEin[:,:,:,2] )
        
        print("FEin.shape ", FEin.shape);

        t1relax = time.clock();
        region = FEin.shape[:3]
        relax_args  = oclr.prepareBuffers( FEin, relax_dim )
        cl.enqueue_copy( oclr.oclu.queue, relax_args[0], FEin, origin=(0,0,0), region=region)
        FEout = oclr.relax( relax_args, relax_dim, invCell, poss=relax_poss, dpos0=dpos0, stiffness=stiffness, relax_params=relax_params  )
        Trelax = time.clock() - t1relax;
        
        #GU.saveXSF( geomFileName+'_Fout_z.xsf',  FEout[:,:,:,2], lvec );
        np.save( geomFileName+'_Fout_z.npy', FEout[:,:,:,2] )
        t1plot = time.clock();
        for isl in islices:
            isl += izslice
            plt.imshow( FEout[:,:,isl,2] )
            plt.savefig( geomFileName+("_FoutZ%03i.png" %isl ), bbox_inches="tight"  ); 
            plt.close()
        Tplot = time.clock()-t1plot;

        Ttot = time.clock()-t1tot;
        print("Timing[s] Ttot %f Tff %f Trelax %f Tprepare %f Tplot %f " %(Ttot, Tff, Trelax, Tprepare, Tplot))

#plt.show()

