#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

print " # ========== make & load  ProbeParticle C++ library " 

#PYSOLR_PATH = '~/git/ProbeParticleModel/code'
#PYSOLR_PATH = '/home/prokop/git/ProbeParticleModel/code'
#if not PYSOLR_PATH in sys.path:
#    sys.path.append(ENGINE_PATH)

def makeclean( ):
	import os
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".so") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".o") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".pyc") ]



os.chdir('../');       print " >> WORKDIR: ", os.getcwd()
makeclean( )
#sys.path.insert(0, "../")
sys.path.insert(0, "./")
import KosmoSuiteCpp as KS
import GridUtils as gu
os.chdir('examples');  print " >> WORKDIR: ", os.getcwd()

print " ============= RUN  "

PP.loadParams( 'params.ini' )

Fz,lvec,nDim,head=gu.loadXSF('Fz.xsf')

nslices = len( Fz ) 

for i in range(nslice):
	figure()
	plt.imshow( FF[i,:,:,2], origin='image', interpolation='nearest' )

plt.show()




