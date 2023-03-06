#!/usr/bin/python

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

print(" # ========== make & load C++ library ")

LWD = '/home/prokop/git/ProbeParticleModel/code'

def makeclean( ):
	import os
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".so") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".o") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".pyc") ]

CWD = os.getcwd()
os.chdir(LWD);       print(" >> WORKDIR: ", os.getcwd())
makeclean( )
sys.path.insert(0, ".")
from ppafm.io import loadXSF

os.chdir(CWD);  print(" >> WORKDIR: ", os.getcwd())

print(" ============= RUN  ")

Fz,lvec,nDim,head=loadXSF('Fz.xsf')

nslice = min( len( Fz ), 10 )

for i in range(nslice):
	plt.figure()
	plt.imshow( Fz[i,:,:], origin='upper', interpolation='nearest' )

plt.show()
