import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt

# =================  Setup

#Q_tip     = 0.48
Q_tip     = 0.7
cCouling  = 0.03 * 0
E_Fermi   = 0.0
z_tip     = 8.0
L         = 20.0
npix      = 400
decay     = 0.7

# --- sites geometry
nsite  =  3
R      =  5.0  # radius of circle on which sites are placed
phiRot = -1.0

Q0  = 1.0
Qzz = 15.0

# =================  Main

# Energy of states on the sites
Esite = [-1.0, -1.0, -1.0]

# Setup system geometry
# distribute sites on a circle
phis = np.linspace(0,2*np.pi,nsite, endpoint=False)
spos = np.zeros((3,3))
spos[:,0] = np.cos(phis)*R
spos[:,1] = np.sin(phis)*R
# rotation of multipoles on sites
rots = chr.makeRotMats( phis + phiRot, nsite )
# Setup site multipoles
mpols = np.zeros((3,10))
mpols[:,4] = Qzz
mpols[:,0] = Q0

# Initialize global parameters
chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling )

# Setup scanning grid ( tip positions and charges )
extent = [-L,L,-L,L]
ps     = chr.makePosXY(n=npix, L=L, z0=z_tip )
Qtips  = np.ones(len(ps))*Q_tip

print( "ps ", ps)

# Calculate site occupancies
Qsites = chr.solveSiteOccupancies(ps, Qtips)
Qsites = Qsites.reshape((npix,npix,nsite))

# Calculate STM map
I_empty = chr.getSTM_map(ps, Qtips, Qsites.reshape(-1,nsite), decay=decay, bOccupied=False ); I_empty = I_empty.reshape((npix,npix))
I_occup = chr.getSTM_map(ps, Qtips, Qsites.reshape(-1,nsite), decay=decay, bOccupied=True  ); I_occup = I_occup.reshape((npix,npix))

# Plot results
plt.figure(figsize=(20,5))
nplot = 6
plt.subplot(1,nplot,1)
plt.imshow(np.sum(Qsites,axis=2), origin="lower", extent=extent)
plt.plot(spos[:,0], spos[:,1], 'og')
plt.colorbar()
plt.title("Qtot")

plt.subplot(1,nplot,2)
plt.imshow(I_empty, origin="lower", extent=extent)
plt.plot(spos[:,0], spos[:,1], 'or')
plt.colorbar()
plt.title("STM empty")

plt.subplot(1,nplot,3)
plt.imshow(I_occup, origin="lower", extent=extent)
plt.plot(spos[:,0], spos[:,1], 'or')
plt.colorbar()
plt.title("STM occupied")

for i in range(3):
    plt.subplot(1,nplot,i+4)
    plt.imshow(Qsites[:,:,i], origin="lower", extent=extent)
    plt.plot(spos[:,0], spos[:,1], 'og')
    plt.colorbar()
    plt.title(f"Q site {i+1}")

plt.tight_layout()
plt.savefig("test_ChargeRings.png", bbox_inches='tight')
plt.show()
