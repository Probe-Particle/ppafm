import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt


import TipMultipole as tmul
#sys.path.append('/home/prokop/bin/home/prokop/venvs/ML/lib/python3.12/site-packages/qmeq/')
sys.path.append('/home/prokop/git_SW/qmeq')
#path.insert(0, '/home/pokorny/bin/qmeq-1.1/')
import qmeq

# =================  Setup

# ============  Scan Params

V_Bias    = 0.1
Q_tip     = 0.6*0.1
cCouling  = 0.03*0.01 #* 0.0
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
Esite0    = [-0.1, -0.1, -0.1]

# ============  QmeQ params


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
    H1p = {(0,0):  Eps[0], (0,1): tij, (0,2): tij,
            (1,1): Eps[1],           (1,2): tij,
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

def get_current_Qmeq( VBias, Esites, Ttips ):
    ntip = len(Esites)
    Is = np.zeros(ntip)
    for itip in range(ntip):
       I,Ea = call_qmeq( VBias, Ttips[itip], Esites[itip]  )
       Is[itip] = I
       print( "itip,I,Ea", itip, I )
    return Is


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

# --------- 1D scan tip trajectory
# --------- Setup scanning grid ( tip positions and charges )
extent = [-L,L,-L,L]
ps    = chr.makePosXY(n=npix, L=L, z0=z_tip )
Qtips = np.ones(len(ps))*Q_tip
if bDebugRun:
    ps_line = chr.getLine(spos, [0.5,0.5,-5.0], [-4.0,-4.0,1.0], n=5 )
    chr.setVerbosity(3)
else:
    ps_line = chr.getLine(spos, [0.5,0.5,-5.0], [-4.0,-4.0,1.0], n=300 )
ps_line[:,2] = z_tip
# -------- plot 1D scan tip trajectory
plt.figure(figsize=(5,5))
plt.plot(spos[:,0], spos[:,1], '+g')
plt.plot(ps_line[:,0], ps_line[:,1], '.-r', ms=0.5)
plt.title("Tip Trajectory")
plt.axis("equal")
plt.grid(True)

# --------- Calculate site energy shifts, hopping, tunelling coefs and current
plt.figure(figsize=(10,5))
Esites, Ttips = tmul.compute_site_energies_and_hopping( ps_line, spos, rots, mpols, Esite0, Q_tip, beta=0. )
Is      = get_current_Qmeq( V_Bias, Esites, Ttips )

# --------- Plot site energies along the 1D tip trajectory
plt.subplot(2,1,1)
for i in range(3):
    plt.plot(Esites[:,i], lw=1.5, label=i ) 
plt.legend()
plt.grid()

# --------- Plot current along the 1D tip trajectory
plt.subplot(2,1,2)
plt.plot(Is, lw=1.5, label="I" )
plt.legend()
plt.grid()
plt.show()
