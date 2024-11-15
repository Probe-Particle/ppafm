import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt

# =================  Setup

#Q_tip     = 0.48
Q_tip     = 0.6
cCouling  = 0.03 # * 0.0
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
chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T )

# Setup scanning grid ( tip positions and charges )
extent = [-L,L,-L,L]
ps    = chr.makePosXY(n=npix, L=L, z0=z_tip )
Qtips = np.ones(len(ps))*Q_tip

ps_line = chr.getLine(spos, [0.5,0.5,-3.0], [-3.0,-3.0,1.0], n=100 )
ps_line[:,2] = z_tip


# ================= 1D scan
"""
Qs                   = np.ones(len(ps_line))*Q_tip 
Q_1                  = chr.solveSiteOccupancies( ps_line, Qs )
evals, evecs, Hs, Gs = chr.solveHamiltonians( ps_line, Qs, bH=True  )

plt.figure( figsize=(10,5) )
plt.subplot(1,2,1)
plt.plot( Hs[:,0,0], label="H[1,1]" )
plt.plot( Hs[:,1,1], label="H[2,2]" )
plt.plot( Hs[:,2,2], label="H[3,3]" )
plt.title("On-site energies")
plt.legend()
plt.ylim(-1.0,3.0)
plt.subplot(1,2,2)
plt.plot( evals[:,0], label="E_1" )
plt.plot( evals[:,1], label="E_2" )
plt.plot( evals[:,2], label="E_3" )
plt.title("eigenvalues")
plt.legend()
plt.ylim(-1.0,3.0)
"""

# ================= 2D x,y

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



# ----------------- Plot Eigen-States

"""
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
"""

# ================= 2D scan   xy(t),Q 
"""

Qtips = np.linspace(0.0, 1.0, 50)
nQs     = len(Qtips)
npoints = len(ps_line)

ps, qs = chr.makePosQscan ( ps_line, Qtips )
#ps, qs = chr.makePosQscan_( ps_line, Qtips )

# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1); plt.imshow(qs);      plt.colorbar(); plt.title("Qs")
# plt.subplot(1,3,2); plt.imshow(ps[:,:,0]); plt.colorbar(); plt.title("ps.x")
# plt.subplot(1,3,3); plt.imshow(ps[:,:,1]); plt.colorbar(); plt.title("ps.y")
 
ps=ps.reshape((-1,3)).copy()
qs=qs.flatten().copy()
#print( "ps ", ps)
#print( "qs ", qs)

Q_1 = chr.solveSiteOccupancies( ps, qs )
#I_1 = chr.getSTM_map(ps, Qtips    , Q_1.reshape(-1,nsite), decay=decay );

# Q_2 = chr.solveSiteOccupancies(ps, Qtips+dQ)
# I_2 = chr.getSTM_map(ps, Qtips+dQ , Q_2.reshape(-1,nsite), decay=decay );

Q_1   = Q_1.reshape((nQs,npoints,nsite))
Q1tot = np.sum(Q_1,axis=2)

extent = [ 0.0, 1.0, Qtips.min(), Qtips.max()]

print( f"Scan along line {ps_line[0,:2]} {ps_line[-1,:2]}" );

plt.figure(figsize=(20,5))
plt.subplot(1,4,1);  plt.imshow( Q1tot     , origin="lower", extent=extent, ); plt.colorbar(); plt.title("Q tot   "); plt.xlabel(f"Scan along the line "); plt.ylabel("Q_tip [e])")
plt.subplot(1,4,2);  plt.imshow( Q_1[:,:,0], origin="lower", extent=extent, ); plt.colorbar(); plt.title("Q site 1"); plt.xlabel(f"Scan along the line "); plt.ylabel("Q_tip [e])")
plt.subplot(1,4,3);  plt.imshow( Q_1[:,:,1], origin="lower", extent=extent, ); plt.colorbar(); plt.title("Q site 2"); plt.xlabel(f"Scan along the line "); plt.ylabel("Q_tip [e])")
plt.subplot(1,4,4);  plt.imshow( Q_1[:,:,2], origin="lower", extent=extent, ); plt.colorbar(); plt.title("Q site 3"); plt.xlabel(f"Scan along the line "); plt.ylabel("Q_tip [e])")

evals, evecs, Hs, Gs = chr.solveHamiltonians( ps, qs , bH=True  )
evals                = evals.reshape((nQs,npoints,3)    )
Hs                   = Hs   .reshape((nQs,npoints,3,3)  )

# plt.figure(figsize=(15,5))
# plt.subplot(1,4,2);  plt.imshow( evals[:,:,0], origin="lower", extent=extent, ); plt.colorbar(); plt.title("Q site 1"); plt.xlabel(f"Scan along the line "); plt.ylabel("E_1 [e])")
# plt.subplot(1,4,3);  plt.imshow( evals[:,:,1], origin="lower", extent=extent, ); plt.colorbar(); plt.title("Q site 2"); plt.xlabel(f"Scan along the line "); plt.ylabel("E_2 [e])")
# plt.subplot(1,4,4);  plt.imshow( evals[:,:,2], origin="lower", extent=extent, ); plt.colorbar(); plt.title("Q site 3"); plt.xlabel(f"Scan along the line "); plt.ylabel("E_3 [e])")

iQ = nQs//2
plt.figure( figsize=(10,5) )
plt.subplot(1,2,1)
plt.plot( Hs[iQ,:,0,0], label="H[1,1]" )
plt.plot( Hs[iQ,:,1,1], label="H[2,2]" )
plt.plot( Hs[iQ,:,2,2], label="H[3,3]" )
plt.ylim(-1.0,3.0)
plt.subplot(1,2,2)
plt.plot( evals[iQ,:,0], label="E_1" )
plt.plot( evals[iQ,:,1], label="E_2" )
plt.plot( evals[iQ,:,2], label="E_3" )
plt.ylim(-1.0,3.0)
"""

plt.show()
