#!/usr/bin/python

import numpy as np
import sys
import matplotlib.pyplot as plt



'''

TODO:
determine z-position from approach curve ( slightly below minimum )

'''

sys.path.append("/home/prokop/git/ProbeParticleModel")

import pyProbeParticle.GridUtils as GU
import pyProbeParticle.atomfit as AF

E,lvec,nDim,head =  GU.loadXSF("OutFz.xsf")

E = E[15,:,:]

Fy = E[:-1,:  ] - E[1:, :]; Fy=Fy[:  ,:-1]+Fy[ :,1:]
Fx = E[:  ,:-1] - E[ :,1:]; Fx=Fx[:-1,:  ]+Fx[1:, :]

F = np.empty(Fx.shape+(2,))

F[:,:,0] =  Fx
F[:,:,1] =  Fy

F[:,:,:]*=1.0

#del Fx,Fy

'''
pos = np.array([
    [5.0,5.0],
    [5.0,6.0],
    [6.0,5.0],
    [6.0,6.0]
])
'''

Xs,Ys = np.meshgrid(np.linspace(0.0,16.0,16),np.linspace(0.0,16.0,16))
pos = np.empty(Xs.shape+(2,))
pos[:,:,0]=Xs + 2.0
pos[:,:,1]=Ys + 2.0
pos=pos.reshape(-1,2).copy()
print(pos)
print(pos.shape)

dpix=[0.1,0.1]
npix=F.shape
AF.setParams (1.0, 1.5 )
AF.setGridFF (F, dpix )
vel,force=AF.setAtoms  (pos)
AF.relaxAtoms(1000,0.5,0.9,-1.0)

#plt.imshow(Fx*Fx + Fy*Fy, extent=(0,npix[0]*dpix[0],0,npix[1]*dpix[1]))
extent=(0,npix[0]*dpix[0],0,npix[1]*dpix[1])
plt.imshow(E, extent=extent)

#plt.imshow(Fx, extent=(0,npix[0]*dpix[0],0,npix[1]*dpix[1]))
#plt.imshow(E, extent=(0,npix[0]*dpix[0],0,npix[1]*dpix[1]))
plt.plot( pos[:,0],pos[:,1], ".w" )

plt.xlim(extent[0],extent[1])
plt.ylim(extent[2],extent[3])

plt.show()

