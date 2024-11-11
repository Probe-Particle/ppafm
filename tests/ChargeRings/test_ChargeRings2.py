import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt

def makePosXY(n=100, L=10.0, z0=5.0):
    x = np.linspace(-L,L,n)
    y = np.linspace(-L,L,n)
    Xs,Ys = np.meshgrid(x,y)
    ps = np.zeros((n*n,3))
    ps[:,0] = Xs.flatten()
    ps[:,1] = Ys.flatten()
    ps[:,2] = z0
    return ps

# Setup system geometry
R = 5.0
nsite = 3
phis = np.linspace(0,2*np.pi,nsite, endpoint=False)
spos = np.zeros((3,3))
spos[:,0] = np.cos(phis)*R
spos[:,1] = np.sin(phis)*R
Esite = [-1.0, -1.0, -1.0]

# Setup multipoles
rot = np.zeros((3,3,3))
mpols = np.zeros((3,10))
phi2 = phis + 0.3
ca = np.cos(phi2)
sa = np.sin(phi2)
rot[:,0,0] = ca
rot[:,1,1] = ca
rot[:,0,1] = -sa
rot[:,1,0] = sa
rot[:,2,2] = 1.0
mpols[:,4] = 10.0
mpols[:,0] = 1.0

# Initialize global parameters
chr.initRingParams(spos, Esite, rot=rot, MultiPoles=mpols, E_Fermi=0.0, cCouling=0.03, Q_tip=0.48, )

# Setup scanning grid
L      = 20.0
npix   = 400
extent = [-L,L,-L,L]
ps = makePosXY(n=npix, L=L, z0=5.0)
Qtips = np.ones(len(ps))*0.48

# Calculate site occupancies
Qsites = chr.solveSiteOccupancies(ps, Qtips)
Qsites = Qsites.reshape((npix,npix,nsite))

# Calculate STM map
I_stm = chr.getSTM_map(ps, Qtips, Qsites.reshape(-1,nsite), decay=1.0)
I_stm = I_stm.reshape((npix,npix))

# Plot results
plt.figure(figsize=(20,5))
plt.subplot(1,5,1)
plt.imshow(np.sum(Qsites,axis=2), origin="lower", extent=extent)
plt.plot(spos[:,0], spos[:,1], 'or')
plt.colorbar()
plt.title("Qtot")

for i in range(3):
    plt.subplot(1,5,i+2)
    plt.imshow(Qsites[:,:,i], origin="lower", extent=extent)
    plt.plot(spos[:,0], spos[:,1], 'or')
    plt.colorbar()
    plt.title(f"Q site {i+1}")

plt.subplot(1,5,5)
plt.imshow(I_stm, origin="lower", extent=extent)
plt.plot(spos[:,0], spos[:,1], 'or')
plt.colorbar()
plt.title("STM")

plt.tight_layout()
plt.savefig("test_ChargeRings.png", bbox_inches='tight')
plt.show()
