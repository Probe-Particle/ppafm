import numpy as np
import sys
sys.path.append("../../")
#from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt


import TipMultipole as tmul
#sys.path.append('/home/prokop/bin/home/prokop/venvs/ML/lib/python3.12/site-packages/qmeq/')
sys.path.append('/home/prokop/git_SW/qmeq')
#path.insert(0, '/home/pokorny/bin/qmeq-1.1/')
import qmeq

# =================  Setup

eV2meV = 1000.0

# ============  Scan Params

V_Bias    = 1.0
Uij       = 0.01 *0 #
z_tip     = 6.0
L         = 20.0
#npix      = 400
npix      = 100
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

E0        = -0.05
Esite0    = [E0,E0,E0]
Rtip      = 2.0
zV0       = -3.0

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
    H1p = {
        (0,0):  Eps[0], (0,1): tij, (0,2): tij,
        (1,1): Eps[1], (1,2): tij,
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
    H2p = { 
        (0,1,1,0): Uij,
        (0,2,2,0): Uij,
        (1,2,2,1): Uij
    }
    system = qmeq.Builder(NSingle, H1p, H2p, NLeads, TLeads, mu_L, Temp_L, DBand, kerntype=kerntype, indexing='Lin', itype=0, symq=True, solmethod='lsqr', mfreeq=0)
    system.solve()
    return system.current[1], system.Ea

def scan2D_QmeQ( a1_range, a2_range, Ttip=0.1, VBias=1.0, E0=-0.1, Uij=0.1 ):
    n1 = len(a1_range)
    n2 = len(a2_range)
    Is = np.zeros((n1,n2))
    Ttips = [ Ttip, Ttip, Ttip ]
    for i1 in range(n1):
        for i2 in range(n2):
            E1 = ( E0 + VBias * a1_range[i1] )*eV2meV
            E2 = ( E0 + VBias * a2_range[i2] )*eV2meV
            I,Ea = call_qmeq( VBias*eV2meV, Ttips, [ E1, E2, E2 ], Uij*eV2meV  )
            Is[i1,i2] = I
            print( "i1,i2,I", i1,i2, I, E1, E2 )
    return Is

def scanVBias_QmeQ( VBias_range, a1=0.0, a2=0.0, Ttip=0.001, E0=-0.1, Uij=0.1 ):
    n = len(VBias_range)
    Ttips = [ Ttip, Ttip, Ttip ]
    Is = np.zeros((n))
    for i in range(n):
        VBias = VBias_range[i]
        E1 = ( E0 + VBias * a1 )*eV2meV
        E2 = ( E0 + VBias * a2 )*eV2meV
        I,Ea = call_qmeq( VBias*eV2meV, Ttips, [ E1, E2, E2 ], Uij*eV2meV  )
        Is[i] = I
        print( f"i {i} V {VBias*eV2meV} I {I} E1 {E1} E2 {E2} VT {VT}" )
    return Is

# a1_range = np.linspace( -0.5, 1.0, 50 )
# a2_range = np.linspace( -1.0, 1.0, 50 )
# extent = [ a1_range[0], a1_range[-1], a2_range[0], a2_range[-1] ]
# Is = scan2D_QmeQ( a1_range, a2_range, VBias=1.0, E0=0.0, Uij=0.01 )
# plt.figure()
# plt.imshow(Is, origin='lower', extent=extent )
# plt.xlabel("a1")
# plt.ylabel("a2")
# plt.colorbar()


# a1_range = np.linspace(  0.8 , 1.0, 50 )
# a2_range = np.linspace(  0.85, 1.0, 50 )
# extent = [ a1_range[0], a1_range[-1], a2_range[0], a2_range[-1] ]
# Is = scan2D_QmeQ( a1_range, a2_range, VBias=1.0, E0=0.0, Uij=0.01 )
# plt.figure()
# plt.imshow(Is, origin='lower', extent=extent )
# plt.xlabel("a1")
# plt.ylabel("a2")
# plt.colorbar()

VBias_range  = np.arange(0.0,0.040,0.0005)
Is = scanVBias_QmeQ( VBias_range, a1=-0.4, Ttip=0.001, E0=-0.01, Uij=0.02 )
plt.plot( VBias_range, Is )


plt.show()



