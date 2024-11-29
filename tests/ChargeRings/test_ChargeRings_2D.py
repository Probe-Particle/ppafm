import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt

# =================  Setup

#Q_tip     = 0.48
Q_tip     = 0.6
cCouling  = 0.02 # * 0.0
E_Fermi   = 0.0
z_tip     = 6.0
L         = 20.0
npix      = 400
decay     = 0.7

dQ =0.02

T = 10.0

# --- sites geometry
nsite  =  3
R      =  5.0  # radius of circle on which sites are placed
phiRot = -1.0

Q0  = 1.0
Qzz = 15.0 * 0.0

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
chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T, onSiteCoulomb=3.0 )
#chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T, onSiteCoulomb=0.2 )

# Setup scanning grid ( tip positions and charges )
extent = [-L,L,-L,L]
ps    = chr.makePosXY(n=npix, L=L, z0=z_tip )
Qtips = np.ones(len(ps))*Q_tip

ps_line = chr.getLine(spos, [0.5,0.5,-3.0], [-3.0,-3.0,1.0], n=100 )
ps_line[:,2] = z_tip


# ================= 2D x,y

print( "ps ", ps)

# Calculate site occupancies
Q_1, Es_1, Ec_1 = chr.solveSiteOccupancies(ps, Qtips)
I_1             = chr.getSTM_map(ps, Qtips    , Q_1.reshape(-1,nsite), decay=decay );

Q_2, Es_2, Ec_2 = chr.solveSiteOccupancies(ps, Qtips+dQ)
I_2             = chr.getSTM_map(ps, Qtips+dQ , Q_2.reshape(-1,nsite), decay=decay );

dIdQ = (I_2-I_1)/dQ

Q_1  = Q_1.reshape((npix,npix,nsite))
I_1  = I_1.reshape((npix,npix))
dIdQ = dIdQ.reshape((npix,npix))


# Plot results
plt.figure(figsize=(6*5,5))
nplot = 6
plt.subplot(1,nplot,1)
plt.imshow(np.sum(Q_1,axis=2), origin="lower", extent=extent)
plt.plot(spos[:,0], spos[:,1], '+g')
plt.colorbar()
plt.plot(ps_line[:,0], ps_line[:,1], '.-r')
plt.title("Qtot")


plt.subplot(1,nplot,2)
plt.imshow(I_1, origin="lower", extent=extent)
plt.plot(spos[:,0], spos[:,1], '+g')
plt.colorbar()
plt.title("STM empty")

plt.subplot(1,nplot,3)
plt.imshow(dIdQ, origin="lower", extent=extent)
plt.plot(spos[:,0], spos[:,1], '+g')
plt.colorbar()
plt.title("dI/dQ")

for i in range(3):
    plt.subplot(1,nplot,i+4)
    plt.imshow(Q_1[:,:,i], origin="lower", extent=extent)
    plt.plot(spos[:,0], spos[:,1], '+g')
    plt.colorbar()
    plt.title(f"Q site {i+1}")

plt.tight_layout()
plt.savefig("STM_sim_%2.3f.png" %(cCouling), bbox_inches='tight')

plt.show()
