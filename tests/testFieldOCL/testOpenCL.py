#!/usr/bin/python

import sys
import pyopencl as cl
import numpy    as np 
import time

sys.path.append("/home/prokop/git/ProbeParticleModel_OCL") 
import pyProbeParticle.fieldOCL as FFcl 
from   pyProbeParticle import basUtils
from   pyProbeParticle import PPPlot 
import pyProbeParticle.GridUtils as GU

def getCoulombNP(X,Y,Z,atoms):
    E = np.zeros(X.shape)
    for i in range(len(atoms[0])):
        x = atoms[1][i]
        y = atoms[2][i]
        z = atoms[3][i]
        q = atoms[4][i]
        print "atom %i (%f,%f,%f) %f" %(i,x,y,z,q)
        dX = X-x;
        dY = Y-y;
        dZ = Z-z;
        R  = np.sqrt( dX**2 + dY**2 + dZ**2 + 1e-8)
        E += q/R
    return E

atoms    = basUtils.loadAtoms( 'input.xyz' )
#print atoms
lvec = np.genfromtxt('cel.lvs')
#print lvec

atoms_ = FFcl.atoms2float4(atoms)
X,Y,Z  = FFcl.getPos( lvec ); 
nDim = X.shape
poss   = FFcl.XYZ2float4(X,Y,Z)

#E = getCoulombNP(X,Y,Z,atoms)
#GU.saveXSF( 'V_np.xsf', E, lvec );

t1 = time.clock() 
#FFcl.initCL()
kargs = FFcl.initArgs(atoms_,poss)
#FE   = FFcl.run( kargs, global_size=(X.size), local_size=(16) )
FE    = FFcl.run( kargs, nDim )
t2 = time.clock()
print "OpenCL kernell time: %f [s]" %(t2-t1) 

print "saving ... "

PPPlot.checkVecField(FE)
E =np.zeros(nDim); E[:,:,:] = FE[:,:,:,0]
Fx=np.zeros(nDim); Fx[:,:,:] = FE[:,:,:,1]
Fy=np.zeros(nDim); Fy[:,:,:] = FE[:,:,:,2]
Fz=np.zeros(nDim); Fz[:,:,:] = FE[:,:,:,3]

GU.saveXSF( 'E_cl.xsf',  E, lvec );
GU.saveXSF( 'Fx_cl.xsf', Fx, lvec );
GU.saveXSF( 'Fy_cl.xsf', Fy, lvec );
GU.saveXSF( 'Fz_cl.xsf', Fz, lvec );

print "==== ALL DONE === "

