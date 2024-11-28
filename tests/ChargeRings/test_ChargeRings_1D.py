import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt

# =================  Setup

"""
#Q_tip     = 0.48
#Q_tip     = 0.08
Q_tip     = 0.6
cCouling  = 0.03 # * 0.0
E_Fermi   = 0.0
z_tip     = 6.0
L         = 20.0
npix      = 400
decay     = 0.7
onSiteCoulomb = 3.0
dQ =0.01
T = 2.0
# --- sites geometry
nsite  =  3
R      =  5.0  # radius of circle on which sites are placed
phiRot = -1.0
Q0  = 1.0
Qzz = 15.0 * 0.0
Esite = [-1.0, -1.0, -1.0]
"""



Q_tip     = 0.6*0.2
cCouling  = 0.03*0.2 # * 0.0
E_Fermi   = 0.0
z_tip     = 6.0
L         = 20.0
npix      = 400
decay     = 0.7
onSiteCoulomb = 3.0
dQ = 0.01
T = 2.0
# --- sites geometry
nsite  =  3
R      =  5.0  # radius of circle on which sites are placed
phiRot = -1.0
Q0  = 1.0
Qzz = 15.0 * 0.0
Esite = [-0.2, -0.2, -0.2]

bDebugRun = False
#bDebugRun = True

# =================  Main

# Energy of states on the sites



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
chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling, onSiteCoulomb=onSiteCoulomb, temperature=T )

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

# ================= 1D scan

# ================= 1D scan  : Eigen-States, Hamiltonian, and Configuration Energies

Qs = np.ones(len(ps_line))*Q_tip 

print( "chr.nconfs ", chr.nconfs)

Qsites, Econf        = chr.solveSiteOccupancies( ps_line, Qs, bEconf=True, bUserBasis=True)
evals, evecs, Hs, Gs = chr.solveHamiltonians   ( ps_line, Qs, Qsites=Qsites, bH=True)

Qtot = np.sum(Qsites, axis=1)

# Create figure with 3 vertically stacked subplots
plt.figure(figsize=(10,12))

nplt=3
iplt=1

# Plot 1: Configuration Energies
plt.subplot(nplt,1,iplt); iplt+=1
for i in range(nconfs):
    plt.plot(Econf[:,i] - E_Fermi, linewidth=2, label=confstrs[i], c=confColors[i])
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Add Fermi level reference line
plt.title(f"Configuration Energies (relative to E_Fermi={E_Fermi})")
plt.ylabel("Energy (eV)")
plt.xlabel("Position along line")
plt.legend()
plt.grid(True)

# Plot 2: On-site energies
plt.subplot(nplt,1,iplt); iplt+=1
plt.plot(Qsites[:,0], label="Q 1")
plt.plot(Qsites[:,1], label="Q 2")
plt.plot(Qsites[:,2], label="Q 3")
plt.plot(Qtot,        label="Qtot")
plt.title("On-site energies")
plt.legend()
plt.ylim(-0.5,3.5)
plt.grid(True)

# Plot 3: STM
plt.subplot(nplt,1,iplt); iplt+=1
#I_occup = chr.getSTM_map(ps_line, Qtips, Qsites, decay=decay, bOccupied=True  ); #I_occup = I_occup.reshape((npix,npix))
I_empty = chr.getSTM_map(ps_line, Qtips, Qsites, decay=decay, bOccupied=False ); #I_empty = I_empty.reshape((npix,npix))
#plt.plot( I_occup, label="STM occupied" )
plt.plot( I_empty, label="STM empty" )
#plt.plot( Qtot,    label="Qtot" )
plt.grid(True)
plt.title("STM")


# plt.subplot(nplt,1,iplt); iplt+=1
# plt.plot(Hs[:,0,0], label="H[0,0]")
# plt.plot(Hs[:,1,1], label="H[1,1]")
# plt.plot(Hs[:,2,2], label="H[2,2]")
# plt.title("On-site energies")
# plt.legend()
# plt.ylim(-1.0,3.0)
# plt.grid(True)


# Plot 2: Eigenvalues
# plt.subplot(3,1,2)
# plt.plot(evals[:,0], label="E_1")
# plt.plot(evals[:,1], label="E_2")
# plt.plot(evals[:,2], label="E_3")
# plt.title("Eigenvalues")
# plt.legend()
# plt.ylim(-1.0,3.0)
# plt.grid(True)


plt.tight_layout()
plt.savefig("test_ChargeRings_1D.png", bbox_inches='tight')


plt.show()