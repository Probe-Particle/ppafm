#!/usr/bin/python

import sys
import time

import numpy as np
import pyopencl as cl

sys.path.append("/home/prokop/git/ProbeParticleModel_OCL")

import ppafm.common as PPU
import ppafm.cpp_utils as cpp_utils
import ppafm.ocl.field as FFcl
from ppafm import PPPlot, io


def loadInput( ):
    FFparams          = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
    xyzs,Zs,enames,qs = io.loadAtomsNP( 'input_wrap.xyz' )
    lvec              = np.genfromtxt('cel.lvs')

    Zs, xyzs, qs = PPU.PBCAtoms( Zs, xyzs, qs, avec=lvec[1], bvec=lvec[2] )

    t1    = time.clock()
    cLJs_ = PPU.getAtomsLJ     ( 8, Zs, FFparams ); #print "C6_,C12_",C6_,C12_
    t2    = time.clock(); print("getAtomsLJ time %f [s]" %(t2-t1))

    atoms_  = FFcl.xyzq2float4(xyzs,qs); #print atoms_
    cLJs_.astype(np.float32)

def getposs( ):
    X,Y,Z   = FFcl.getPos( lvec );
    X.shape
    FFcl.XYZ2float4(X,Y,Z)

t1 = time.clock()
kargs = FFcl.initArgsLJC(atoms_,cLJs,poss)
FE    = FFcl.runLJC( kargs, nDim )
t2 = time.clock()
print("OpenCL kernell time: %f [s]" %(t2-t1))

PPPlot.checkVecField(FE)
Ftmp=np.zeros(nDim);
Ftmp[:,:,:] = FE[:,:,:,0]; io.saveXSF( 'ELJ_cl.xsf',  Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,1]; io.saveXSF( 'FLJx_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,2]; io.saveXSF( 'FLJy_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,3]; io.saveXSF( 'FLJz_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,4]; io.saveXSF( 'Eel_cl.xsf',  Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,5]; io.saveXSF( 'Felx_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,6]; io.saveXSF( 'Fely_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,7]; io.saveXSF( 'Felz_cl.xsf', Ftmp, lvec );


print("==== ALL DONE === ")
