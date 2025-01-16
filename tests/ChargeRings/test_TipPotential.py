import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt


import TipMultipole as tmul

def makeCircle( n=10, R=1.0, p0=(0.0,0.0,0.0), axs=(0,1,2) ):
    phis  = np.linspace(0,2*np.pi,n, endpoint=False)
    ps    = np.zeros((n,3))
    ps[:,axs[0]] = p0[axs[0]] + np.cos(phis)*R
    ps[:,axs[1]] = p0[axs[1]] + np.sin(phis)*R
    ps[:,axs[2]] = p0[axs[2]]
    return ps, phis

# Energy of states on the sites


Rtip   = 1.0
VBias  = 0.1
Rcirc  = 0.0
phiRot = 0.0
Qzz    = 0.0
Q0     = 1.0
L      = 10.0
npix   = 100
z_tip  = 6.0
#zV0    = -2.5

zV0    = -2.5
zQd    = 0.0

# Setup system geometry
# distribute sites on a circle
Esite0 = [0.0,0.0,0.0]
nsite = len(Esite0)
#phis = np.linspace(0,2*np.pi,nsite, endpoint=False)
#spos = np.zeros((3,3))
#spos[:,0] = np.cos(phis)*Rcirc
#spos[:,1] = np.sin(phis)*Rcirc

spos, phis = makeCircle( nsite, R=Rcirc, p0=(0.0,0.0,zQd) )
print( " Qdpos :", spos )
# rotation of multipoles on sites
rots = chr.makeRotMats( phis + phiRot, nsite )
# Setup site multipoles
mpols = np.zeros((3,10))
mpols[:,4] = Qzz
mpols[:,0] = Q0

circ1,_ = makeCircle( 16, R=Rtip, axs=(0,2,1), p0=(0.0,0.0,      z_tip) )
circ2,_ = makeCircle( 16, R=Rtip, axs=(0,2,1), p0=(0.0,0.0,2*zV0-z_tip) )

# --------- 1D scan tip trajectory
# --------- Setup scanning grid ( tip positions and charges )
extent     = [-L,L,-L,L]
ps,Xs,Ys   = chr.makePosXY(n=npix, L=L, p0=(0.0,0.0,0.0), axs=(0,2,1) )
#Vtip, Ttip = tmul.compute_site_energies_and_hopping_mirror( ps, spos, siteRots=None, mpols=None, Esite0=Esite0, VBias=VBias, Rtip=Rtip, beta=1.0, zV0=zV0 )
#Vtip, Vtip_ = tmul.compute_site_energies_and_hopping_mirror_2( ps, spos,  Esite0=Esite0, VBias=VBias, Rtip=Rtip, beta=1.0, zV0=zV0 )
#Vtip  = Vtip .reshape(npix,npix,3)[0]

# Vtip, Vtip_ =  tmul.compute_V_mirror( np.array([0.0,0.0,z_tip]), ps, VBias=VBias, Rtip=Rtip, zV0=zV0 )
# Vtip  = Vtip .reshape(npix,npix)
# Vtip_ = Vtip_.reshape(npix,npix)
# print("Vtip.shape, Vtip_.shape ", Vtip.shape, Vtip_.shape)

Vtip = tmul.compute_V_mirror( np.array([0.0,0.0,z_tip]), ps, VBias=VBias, Rtip=Rtip, zV0=zV0 )
Vtip = Vtip .reshape(npix,npix)

Esites = tmul.compute_site_energies( ps, spos, VBias, Rtip, zV0=zV0 )
Esites = Esites.reshape(npix,npix,3)

# plt.figure()
# plt.imshow( ps.reshape(npix,npix,3)[:,:,2], extent=extent, cmap='bwr' ) 
# plt.colorbar()

plt.figure()
#plt.subplot(1,2,1); 

# --------- Plot current along the 1D tip trajectory
#plt.figure(figsize=(5,5))
#plt.subplot(2,3,3+1);
plt.title('tip Potnetial')
im = plt.imshow( Vtip , extent=extent, cmap='bwr', origin='lower', vmin=-VBias, vmax=VBias) 
#plt.colorbar()
contour_levels = np.linspace( -VBias, VBias, 25)  # Define levels for the contours
contour_plot   = plt.contour( Xs, Ys, Vtip, levels=contour_levels, colors='k', linewidths=0.5)
cbar=plt.colorbar(im)
cbar.add_lines(contour_plot)
plt.plot(circ1[:,0], circ1[:,2], ':k')
plt.plot(circ2[:,0], circ2[:,2], ':k')
plt.grid()

plt.figure()
plt.imshow( Esites[:,:,0], extent=extent, origin='lower', cmap='bwr', vmin=-VBias, vmax=VBias) 
#plt.imshow( Vtip-VBias, extent=extent, cmap='bwr' ) 
plt.plot(spos[:,0], spos[:,2], '+g')
plt.axhline(zV0, ls='--',c='k', label='mirror surface')
plt.axhline(zQd, ls='--',c='g', label='Qdot height')
plt.gca().set_xticks(np.arange(extent[0], extent[1] + 1, 1))  # Gridlines every 1 unit in x
plt.gca().set_yticks(np.arange(extent[2], extent[3] + 1, 1))  # Gridlines every 1 unit in y
# plt.plot(circ1[:,0], circ1[:,2], ':k')
# plt.plot(circ2[:,0], circ2[:,2], ':k')
plt.colorbar()
plt.grid()

plt.savefig("test_TipPotential_mirror.png")


plt.show()

