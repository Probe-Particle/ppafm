#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

LWD = '/home/prokop/git/ProbeParticleModel/code' 

print(" # ========== make & load  ProbeParticle C++ library ") 

def makeclean( ):
    import os
    [ os.remove(f) for f in os.listdir(".") if f.endswith(".so") ]
    [ os.remove(f) for f in os.listdir(".") if f.endswith(".o") ]
    [ os.remove(f) for f in os.listdir(".") if f.endswith(".pyc") ]

CWD = os.getcwd()
os.chdir(LWD);       print(" >> WORKDIR: ", os.getcwd())
makeclean( )
sys.path.insert(0, "./")
import GridUtils as GU
import ProbeParticle as PP
os.chdir(CWD);  print(" >> WORKDIR: ", os.getcwd())

print(" ============= RUN  ")

F,lvec,nDim,head=GU.loadXSF('LOCPOT.xsf')

F4 = 0.25*( F + F[:,:,::-1] + F[:,::-1,:] + F[:,::-1,::-1] )

#GU.saveXSF('LOCPOT_4sym.xsf', GU.XSF_HEAD_DEFAULT, lvec, F4 )
GU.saveXSF('LOCPOT_4sym.xsf', head, lvec, F4 )

plt.show()




