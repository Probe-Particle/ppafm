import numpy as np
import sys
import os
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt

# =================  Setup

# System parameters
z_tip     = 6.0
L         = 20.0
npix      = 400
decay     = 0.7
cCouling  = 0.02
E_Fermi   = 0.0
T         = 10.0

# Scan parameters
dQ = 0.02
Q_tips = np.linspace(0.0, 0.8, 40+1)  # Range of tip charges to scan
save_dir = "ChargeRings_Qscan_images"  # Directory to save images

# --- sites geometry
nsite  = 3
R      = 5.0  # radius of circle on which sites are placed
phiRot = -1.0

Q0  = 1.0
Qzz = 15.0 * 0.0

# =================  Main

# Energy of states on the sites
Esite = [-1.0, -1.0, -1.0]

# Setup system geometry
# distribute sites on a circle
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
spos = np.zeros((3,3))
spos[:,0] = np.cos(phis)*R
spos[:,1] = np.sin(phis)*R
# rotation of multipoles on sites
rots = chr.makeRotMats(phis + phiRot, nsite)
# Setup site multipoles
mpols = np.zeros((3,10))
mpols[:,4] = Qzz
mpols[:,0] = Q0

# Initialize global parameters
chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T, onSiteCoulomb=3.0)

# Setup scanning grid (tip positions)
extent = [-L,L,-L,L]
ps = chr.makePosXY(n=npix, L=L, z0=z_tip)

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Loop over different tip charges
for iq, Q_tip in enumerate(Q_tips):
    print(f"Processing Q_tip = {Q_tip:.3f}")
    
    Qtips = np.ones(len(ps)) * Q_tip
    
    # Calculate for Q_tip
    Q_1, Ec_1 = chr.solveSiteOccupancies(ps, Qtips)
    I_1 = chr.getSTM_map(ps, Qtips, Q_1.reshape(-1,nsite), decay=decay)
    
    # Calculate for Q_tip + dQ
    Q_2, Ec_2 = chr.solveSiteOccupancies(ps, Qtips+dQ)
    I_2 = chr.getSTM_map(ps, Qtips+dQ, Q_2.reshape(-1,nsite), decay=decay)
    
    # Calculate dI/dQ
    dIdQ = (I_2-I_1)/dQ
    
    # Reshape arrays for plotting
    Q_1 = Q_1.reshape((npix,npix,nsite))
    I_1 = I_1.reshape((npix,npix))
    dIdQ = dIdQ.reshape((npix,npix))
    
    # Plot results
    plt.figure(figsize=(6*5,5))
    
    plt.subplot(1,3,1)
    plt.imshow(np.sum(Q_1,axis=2), origin="lower", extent=extent)
    plt.plot(spos[:,0], spos[:,1], '+g')
    plt.colorbar()
    plt.title(f"Qtot (Q_tip={Q_tip:.3f})")
    
    plt.subplot(1,3,2)
    plt.imshow(I_1, origin="lower", extent=extent)
    plt.plot(spos[:,0], spos[:,1], '+g')
    plt.colorbar()
    plt.title(f"STM empty (Q_tip={Q_tip:.3f})")
    
    plt.subplot(1,3,3)
    plt.imshow(dIdQ, origin="lower", extent=extent)
    plt.plot(spos[:,0], spos[:,1], '+g')
    plt.colorbar()
    plt.title(f"dI/dQ (Q_tip={Q_tip:.3f})")
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'ChargeRings_Q{Q_tip:.3f}.png'))
    plt.close()

print(f"All images have been saved to {save_dir}/")
