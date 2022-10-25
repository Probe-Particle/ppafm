#!/usr/bin/python

import numpy as np
import os
import pyopencl as cl

from . import basUtils
from . import common    as PPU
from . import oclUtils    as oclu 
from . import fieldOCL    as FFcl 
from . import RelaxOpenCL as oclr

# ============= Functions

verbose = 0

def genFFSampling( lvec, pixPerAngstrome=10 ):
    ff_nDim = np.array([
        int(round(pixPerAngstrome*(lvec[1][0]+lvec[1][1]))),
        int(round(pixPerAngstrome*(lvec[2][0]+lvec[2][1]))),
        int(round(pixPerAngstrome*(lvec[3][2]           )))
    ])
    return ff_nDim

def loadSpecies(fname):
    try:
        with open(fname, 'r') as f:  
            str_Species = f.read(); 
    except:
        if(verbose>0): print("defaul atomtypes.ini")
        fpath = os.path.dirname( os.path.realpath( __file__ ) ) + '/defaults/atomtypes.ini'
        print("loadSpecies from : ", fpath)
        with open(fpath, 'r') as f:
            str_Species = f.read();
    str_Species = "\n".join( "\t".join( l.split()[:5] )  for l in str_Species.split('\n')  )
    return PPU.loadSpeciesLines( str_Species.split('\n') )

def evalFFatoms_LJC( atoms, cLJs, poss ):
    ff_args = FFcl.initArgsLJC( atoms, cLJs, poss )
    ff_nDim = poss.shape[:3]
    FF      = FFcl.runLJC( ff_args, ff_nDim )
    FFcl.releaseArgs(ff_args)
    return FF

def posAboveTopAtom( atoms, hdir, distAbove = 7.5 ):
    imax,xdirmax  = PPU.maxAlongDir( atoms, hdir )
    return atoms[imax,:3] + hdir*distAbove

def writeDebugXYZ( fname, lines, poss ):
    fout  = open(fname,"w")
    natom = int(lines[0])
    npos  = len(poss)
    fout.write( "%i\n" %(natom + npos) )
    fout.write( "\n" )
    for line in lines[2:natom+1]:
        fout.write( line )
    for pos in poss:
        fout.write( "He %f %f %f\n" %(pos[0], pos[1], pos[2]) )
    fout.write( "\n" )

def makeFF_LJC( poss, atom_lines, typeParams, iZPP, lvec, npbc=(1,1,1) ):
    xyzs,Zs,enames,qs = basUtils.loadAtomsLines( atom_lines )
    natoms0 = len(Zs)
    if( npbc is not None ):
        Zs, xyzs, qs = PPU.PBCAtoms3D( Zs, xyzs, qs, lvec[1:], npbc=npbc )
    atoms = FFcl.xyzq2float4(xyzs,qs)
    cLJs  = PPU.getAtomsLJ( iZPP, Zs, typeParams ).astype(np.float32)
    FF    = evalFFatoms_LJC( atoms, cLJs, poss )
    return FF, atoms, natoms0

def scanFromRotation( relax_args, pos0, rot, invCell, ndim=( 100, 100, 20), span=(10.0,10.0), tipR0=4.0, zstep=0.1, distAbove= 7.5, stiffness=oclr.DEFAULT_stiffness, relax_params=oclr.DEFAULT_relax_params, debugXYZ=None ):
    dTip =np.zeros(4,dtype=np.float32); dTip [:3] = rot[2]*-zstep
    dpos0=np.zeros(4,dtype=np.float32); dpos0[:3] = rot[2]*-tipR0;  dpos0[3] = tipR0
    relax_poss = oclr.preparePossRot( ndim, pos0, rot[0], rot[1], start=(-span[0],-span[1]), end=span )
    if debugXYZ is not None:
        fname = debugXYZ[0]
        atom_lines    = debugXYZ[1]
        writeDebugXYZ( fname, atom_lines, relax_poss[::10,::10,:].reshape(-1,4) )
    FEout = oclr.relax( relax_args, ndim, invCell, poss=relax_poss, dTip=dTip, dpos0=dpos0, stiffness=stiffness, relax_params=relax_params  )
    return FEout

