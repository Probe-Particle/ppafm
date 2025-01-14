import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt

path.insert(0, '/home/prokop/bin/home/prokop/venvs/ML/lib/python3.12/site-packages/qmeq/')
#path.insert(0, '/home/pokorny/bin/qmeq-1.1/')
import qmeq

# =================  Setup

# ============  Scan Params

V_Bias    = 0.1
Q_tip     = 0.6*0.2
cCouling  = 0.03*0.2 #* 0.0
E_Fermi   = 0.0
z_tip     = 6.0
L         = 20.0
npix      = 400
decay     = 0.2
onSiteCoulomb = 3.0
dQ        = 0.01
T         = 2.0
# --- sites geometry
nsite     =  3
R         =  5.0  # radius of circle on which sites are placed
phiRot    = -1.0
Q0        = 1.0
Qzz       = 15.0 *0.0
Esite     = [-0.2, -0.2, -0.2]

# ============  QmeQ params

muS    = 0.0    ## substrate chemical potential
muT    = 0.0    ## scanning tip chemical potential
Temp   = 0.224  ## (2.6K) temperature in meV, 1meV = 11.6K, 1K = 0.0862meV
DBand  = 1000.0 ## lead bandwidth
GammaS = 0.20 ## coupling to substrate
GammaT = 0.05 ## coupling to scanning tip
## tunneling amplitudes weighted by lead DOS
VS = np.sqrt(GammaS/np.pi)
VT = np.sqrt(GammaT/np.pi)

siteColors = ["r", "g", "b"]

bDebugRun = False
#bDebugRun = True

# =================  Functions

def call_qmeq( 
    VBias,           ## bias between tip and substrate
    Ttips,           ## hopping between tip and each of the sites
    Eps,             ## energy levels of sites shifted by tip bias
    Uij  = 20.0,     ## Coulombinc coupling between different sites
    tij  =  0.0,     ## Direct hopping between different sites
    Tsub = GammaS,   ## hopping between substrate and each of the sites
    # ------- Constant /  not interesting parameters
    Temp  = 0.224,  ## (2.6K) temperature in meV, 1meV = 11.6K, 1K = 0.0862meV
    muS   = 0.0,     ## substrate chemical potential
    muT   = 0.0,     ## scanning tip chemical potential
    DBand = 1000.0,  ## lead bandwidth
    U      = 220.0,  ## on-site Coulomb interaction, useless for spinless case
    NLeads=2,
    kerntype = 'Pauli',
):
    NSingle = len(Eps)
    ## one-particle Hamiltonian
    H1p = {(0,0):  Eps[0], (0,1): t, (0,2): t,
            (1,1): Eps[1],           (1,2): t,
            (2,2): Eps[2]
    }
    ## coupling between leads (1st number) and impurities (2nd number)
    TLeads = {
        (0,0): Tsub,     # substrate  to site #1        
        (0,1): Tsub,     # substrate  to site #2    
        (0,2): Tsub,     # substrate  to site #3    
        (1,0): Ttips[0], # tip        to site #1
        (1,1): Ttips[1], # tip        to site #2
        (1,2): Ttips[2]  # tip        to site #3
    }
    ## leads: substrate (S) and scanning tip (T)
    mu_L   = {0: muS,  1: muT + VBias}
    Temp_L = {0: Temp, 1: Temp}
    ## two-particle Hamiltonian: inter-site coupling
    H2p = { (0,1,1,0): Uij,
            (0,2,2,0): Uij,
            (1,2,2,1): Uij
    }
    system = qmeq.Builder(NSingle, H1p, H2p, NLeads, TLeads, mu_L, Temp_L, DBand, kerntype=kerntype, indexing='Lin', itype=0, symq=True, solmethod='lsqr', mfreeq=0)
    system.solve()
    return system.current[1], system.Ea

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

#Qsites, Esite, Econf = chr.solveSiteOccupancies( ps_line, Qs, bEconf=True, bEsite=True, solver_type=1 )
Qsites, Esite, Econf = chr.solveSiteOccupancies( ps_line, Qs, bEconf=True, bEsite=True, solver_type=2 )
#evals, evecs, Hs, Gs = chr.solveHamiltonians   ( ps_line, Qs, Qsites=Qsites, bH=True)

Qtot = np.sum(Qsites, axis=1)

# Create figure with 3 vertically stacked subplots
plt.figure(figsize=(10,12))

nplt=3
iplt=1

# Plot 1: Configuration Energies
plt.subplot(nplt,1,iplt); iplt+=1
if Econf is not None:
    for i in range(nconfs):
        plt.plot(Econf[:,i], lw=1.5, label=confstrs[i], c=confColors[i])
# Add Esite plots with dashed lines
if Esite is not None:
    for i in range(Esite.shape[1]):
        plt.plot( Esite[:,i], '-', alpha=0.5, lw=3.0, label=f'Esite {i}', c=siteColors[i])    
plt.axhline(y=E_Fermi,       color='k', linestyle='--', alpha=0.5)  # Add Fermi level reference line
plt.axhline(y=E_Fermi+V_Bias, color='r', linestyle='--', alpha=0.5)
plt.title(f"Configuration Energies (relative to E_Fermi={E_Fermi})")
plt.ylabel("Energy (eV)")
plt.xlabel("Position along line")
plt.legend()
plt.grid(True)

# Plot 2: On-site energies
plt.subplot(nplt,1,iplt); iplt+=1
for i in range(Qsites.shape[1]):
    plt.plot( Qsites[:,i], '-', alpha=0.5, lw=2.0, label=f'Q {i}', c=siteColors[i])    
#plt.plot(Qsites[:,0], label="Q 1")
#plt.plot(Qsites[:,1], label="Q 2")
#plt.plot(Qsites[:,2], label="Q 3")
plt.plot(Qtot,     'k-', label="Qtot" )
plt.title("On-site energies")
plt.legend()
plt.ylim(-0.5,3.5)
plt.grid(True)

# Plot 3: STM
plt.subplot(nplt,1,iplt); iplt+=1

bSTMcpp = False
if bSTMcpp:
    #I_STM = chr.getSTM_map(ps_line, Qtips, Qsites, decay=decay, bOccupied=True  ); #I_occup = I_occup.reshape((npix,npix))
    I_STM = chr.getSTM_map(ps_line, Qtips, Qsites, decay=decay, bOccupied=False ); #I_empty = I_empty.reshape((npix,npix))
else:
    # Using new Python implementation
    I_STM = chr.calculate_tunneling_current(
        ps_line,          # tip positions
        Esite,            # site energies
        E_fermi_sub=E_Fermi,  # assuming substrate Fermi level at 0 
        E_fermi_tip=E_Fermi+V_Bias,  # tip Fermi level above tip
        decay=decay,      # using same decay as C++ version
        T=T,             # room temperature
    )

plt.plot( I_STM, label="STM empty" )
#plt.plot( Qtot,    label="Qtot" )
plt.grid(True)
plt.title("STM")

plt.tight_layout()
plt.savefig("test_ChargeRings_1D.png", bbox_inches='tight')


plt.show()