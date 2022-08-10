#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

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
import GridUtils as gu
os.chdir(CWD);  print(" >> WORKDIR: ", os.getcwd())

print(" ============= RUN  ")

Fz,lvec,nDim,head=gu.loadXSF('Fz.xsf')

nslice = min( len( Fz ), 10 ) 

for i in range(nslice):
	plt.figure()
	plt.imshow( Fz[i,:,:], origin='upper', interpolation='nearest' )

plt.show()




