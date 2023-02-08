#!/usr/bin/python

import sys

import matplotlib.pyplot as plt
import numpy as np

'''

TODO:

determine z-position from approach curve ( slightly below minimum )


'''




sys.path.append("/home/prokop/git/ProbeParticleModel")

import ppafm.GridUtils as GU

E,lvec,nDim,head =  GU.loadXSF("OutFz.xsf")

E = E[15,:,:]


plt.figure(figsize=(10,15))
plt.subplot(3,2,1); plt.imshow( E , cmap="gray"); plt.colorbar()


Fx = E[:-1,:  ] - E[1:, :]; Fx=Fx[:  ,:-1]+Fx[ :,1:]
Fy = E[:  ,:-1] - E[ :,1:]; Fy=Fy[:-1,:  ]+Fy[1:, :]

Kx = Fx[:-1,:  ] - Fx[1:, :];  Kx=Kx[:  ,:-1]+Kx[ :,1:]
Ky = Fy[:  ,:-1] - Fy[ :,1:];  Ky=Ky[:-1,:  ]+Ky[1:, :]

plt.subplot(3,2,3); plt.imshow( Fx ); plt.colorbar()
plt.subplot(3,2,4); plt.imshow( Fy ); plt.colorbar()

Nulls = 1/(1e-8+Fx*Fx  + Fy*Fy)[1:,1:]


#Nulls = Fx*Fx  + Fy*Fy

plt.subplot(3,2,5); plt.imshow( Nulls ); plt.colorbar()

#concave = Nulls.flat >
#concave = np.logical_or(concave,Kx.flat < 0)
#concave = np.logical_or(concave,Ky.flat < 0)

kmin = 1e-4
concave = np.logical_or(Ky.flat < kmin,Kx.flat < kmin)

nullConcave = Nulls.copy()

npix = 8
#nullConcave[::npix,::npix] = np.NaN
minNull = 1e+7
concave = np.logical_or(concave, nullConcave.flat < minNull )

nullConcave.flat[concave] = 0;


plt.subplot(3,2,6); plt.imshow( nullConcave ); plt.colorbar()

plt.show()
