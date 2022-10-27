#!/usr/bin/python

import sys
import pyopencl as cl
import numpy    as np 
import time

sys.path.append("/home/prokop/git/ProbeParticleModel_OCL") 

from   pyProbeParticle import basUtils
from   pyProbeParticle import PPPlot 
import pyProbeParticle.GridUtils as GU
import pyProbeParticle.common as PPU
import pyProbeParticle.cpp_utils as cpp_utils

import pyProbeParticle.ocl.field as FFcl 

def loadInput( ):
    FFparams          = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
    xyzs,Zs,enames,qs = basUtils.loadAtomsNP( 'input_wrap.xyz' )
    lvec              = np.genfromtxt('cel.lvs')

    Zs, xyzs, qs = PPU.PBCAtoms( Zs, xyzs, qs, avec=lvec[1], bvec=lvec[2] )

    t1    = time.clock() 
    cLJs_ = PPU.getAtomsLJ     ( 8, Zs, FFparams ); #print "C6_,C12_",C6_,C12_
    t2    = time.clock(); print("getAtomsLJ time %f [s]" %(t2-t1)) 

    atoms_  = FFcl.xyzq2float4(xyzs,qs); #print atoms_
    cLJs    = cLJs_.astype(np.float32)

def getposs( ):
    X,Y,Z   = FFcl.getPos( lvec ); 
    nDim    = X.shape
    poss    = FFcl.XYZ2float4(X,Y,Z)

t1 = time.clock() 
kargs = FFcl.initArgsLJC(atoms_,cLJs,poss)
FE    = FFcl.runLJC( kargs, nDim )
t2 = time.clock()
print("OpenCL kernell time: %f [s]" %(t2-t1)) 

PPPlot.checkVecField(FE)
Ftmp=np.zeros(nDim);
Ftmp[:,:,:] = FE[:,:,:,0]; GU.saveXSF( 'ELJ_cl.xsf',  Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,1]; GU.saveXSF( 'FLJx_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,2]; GU.saveXSF( 'FLJy_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,3]; GU.saveXSF( 'FLJz_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,4]; GU.saveXSF( 'Eel_cl.xsf',  Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,5]; GU.saveXSF( 'Felx_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,6]; GU.saveXSF( 'Fely_cl.xsf', Ftmp, lvec );
Ftmp[:,:,:] = FE[:,:,:,7]; GU.saveXSF( 'Felz_cl.xsf', Ftmp, lvec );


print("==== ALL DONE === ")

