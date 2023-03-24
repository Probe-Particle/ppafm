#!/usr/bin/python


import matplotlib.pyplot as plt
import numpy as np

from ppafm import io

print(" ============= RUN  ")

Fz,lvec,nDim,head=io.loadXSF('Fz.xsf')

nslice = min( len( Fz ), 10 )

for i in range(nslice):
	plt.figure()
	plt.imshow( Fz[i,:,:], origin='upper', interpolation='nearest' )

plt.show()
