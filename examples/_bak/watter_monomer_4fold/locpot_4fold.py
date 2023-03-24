#!/usr/bin/python

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

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
import ProbeParticle as PP

from ppafm import io

os.chdir(CWD);  print(" >> WORKDIR: ", os.getcwd())

print(" ============= RUN  ")

F,lvec,nDim,head=io.loadXSF('LOCPOT.xsf')

F4 = 0.25*( F + F[:,:,::-1] + F[:,::-1,:] + F[:,::-1,::-1] )

io.saveXSF('LOCPOT_4sym.xsf', head, lvec, F4 )

plt.show()
