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

T = 10.0

# --- sites geometry
nsite  =  3
R      =  5.0  # radius of circle on which sites are placed
phiRot = -1.0

Q0  = 1.0
Qzz = 15.0 * 0.0

bDebugRun = False
#bDebugRun = True

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

confstrs = ["000","001", "010", "100", "110", "101", "011", "111"] 
confColors = chr.colorsFromStrings(confstrs, hi="A0", lo="00")
confs = chr.confsFromStrings( confstrs )
print( "confs: \n", confs)
nconfs = chr.setSiteConfBasis(confs)


# Setup scanning grid ( tip positions and charges )
extent = [-L,L,-L,L]
ps    = chr.makePosXY(n=npix, L=L, z0=z_tip )
Qtips = np.ones(len(ps))*Q_tip


if bDebugRun:
    ps_line = chr.getLine(spos, [0.5,0.5,-5.0], [-4.0,-4.0,1.0], n=5 )
    chr.setVerbosity(3)
else:
    ps_line = chr.getLine(spos, [0.5,0.5,-5.0], [-4.0,-4.0,1.0], n=300 )
ps_line[:,2] = z_tip

plt.figure(figsize=(5,5))
plt.plot(spos[:,0], spos[:,1], '+g')
plt.plot(ps_line[:,0], ps_line[:,1], '.-r', ms=0.5)
plt.title("Tip Trajectory")
plt.axis("equal")
plt.grid(True)


# ================= 2D scan   xy(t),Q   Ocupancy, STM

Qtips = np.linspace(0.0, 1.0, 50)
nQs     = len(Qtips)
npoints = len(ps_line)

ps, qs = chr.makePosQscan(ps_line, Qtips)

ps=ps.reshape((-1,3)).copy()
qs=qs.reshape(-1).copy()

Qsites,_,_ = chr.solveSiteOccupancies(ps, qs, bUserBasis=True)
I_stm    = chr.getSTM_map(ps, qs, Qsites)
I_stm    = I_stm.reshape((nQs,npoints))

Qsites   = Qsites.reshape((nQs,npoints,nsite))
plt.figure(figsize=(10,10))
plt.subplot(2,2,1); plt.imshow(Qsites[:,:,0], extent=[0,npoints,0,1], aspect='auto', origin='lower'); plt.colorbar(); plt.title("Q1")
plt.subplot(2,2,2); plt.imshow(Qsites[:,:,1], extent=[0,npoints,0,1], aspect='auto', origin='lower'); plt.colorbar(); plt.title("Q2")
plt.subplot(2,2,3); plt.imshow(Qsites[:,:,2], extent=[0,npoints,0,1], aspect='auto', origin='lower'); plt.colorbar(); plt.title("Q3")
plt.subplot(2,2,4); plt.imshow(I_stm,        extent=[0,npoints,0,1], aspect='auto', origin='lower'); plt.colorbar(); plt.title("STM")

plt.show()