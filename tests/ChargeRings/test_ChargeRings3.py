import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt

# =================  Setup

#Q_tip     = 0.48
Q_tip     = 0.6
cCouling  = 0.02 
E_Fermi   = 0.0
z_tip     = 6.0
L         = 20.0
npix      = 400
decay     = 0.7

dQ =0.01

T = 100.0

# --- sites geometry
nsite  =  3
R      =  5.0  # radius of circle on which sites are placed
phiRot = -1.0

Q0  = 1.0
Qzz = 15.0

# =================  Functions


def makePosXY(n=100, L=10.0, z0=5.0):
    x = np.linspace(-L,L,n)
    y = np.linspace(-L,L,n)
    Xs,Ys = np.meshgrid(x,y)
    ps = np.zeros((n*n,3))
    ps[:,0] = Xs.flatten()
    ps[:,1] = Ys.flatten()
    ps[:,2] = z0
    return ps

def makeRotMats(phi, nsite=3 ):
    rot = np.zeros((nsite,3,3))
    ca = np.cos(phi)
    sa = np.sin(phi)
    rot[:,0,0] = ca
    rot[:,1,1] = ca
    rot[:,0,1] = -sa
    rot[:,1,0] = sa
    rot[:,2,2] = 1.0
    return rot

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
rots = makeRotMats( phis + phiRot, nsite )
# Setup site multipoles
mpols = np.zeros((3,10))
mpols[:,4] = Qzz
mpols[:,0] = Q0

# Initialize global parameters
chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T )

# Setup scanning grid ( tip positions and charges )
extent = [-L,L,-L,L]
ps = makePosXY(n=npix, L=L, z0=z_tip )
Qtips = np.ones(len(ps))*Q_tip

print( "ps ", ps)

# Calculate site occupancies
Q_1 = chr.solveSiteOccupancies(ps, Qtips)
I_1 = chr.getSTM_map(ps, Qtips    , Q_1.reshape(-1,nsite), decay=decay );

Q_2 = chr.solveSiteOccupancies(ps, Qtips+dQ)
I_2 = chr.getSTM_map(ps, Qtips+dQ , Q_2.reshape(-1,nsite), decay=decay );

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


# ----------------- Plot Eigen-States

evals, evecs, G = chr.solveHamiltonians( ps, Qtips )

# print( "evals ", evals.shape)
# print( "evecs ", evecs.shape)

Emin=evals.min()
Emax=evals.max()
Eabsmax = max(abs(Emin),abs(Emax))

evals = evals.reshape((npix,npix,3))
evecs = evecs.reshape((npix,npix,3,3))



# Plot results
plt.figure(figsize=(3*5,4*5))

for i in range(3):
    plt.subplot(4,3,i+1)
    plt.imshow(evals[:,:,i], origin="lower", extent=extent, cmap="bwr", vmin=-Eabsmax, vmax=Eabsmax )
    plt.plot(spos[:,0], spos[:,1], '+g')
    plt.colorbar()
    plt.title(f"E_{i+1}")

for i in range(3):
    for j in range(3):
        plt.subplot(4,3,(i+1)*3+j+1)
        plt.imshow(evecs[:,:,i,j], origin="lower", extent=extent)
        plt.plot(spos[:,0], spos[:,1], '+g')
        plt.colorbar()
        plt.title(f"<Psi_{i+1}| site {j}>")

plt.tight_layout()
plt.savefig("STM_sim_%2.3f.png" %(cCouling), bbox_inches='tight')






plt.show()
