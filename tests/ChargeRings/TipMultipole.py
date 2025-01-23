import numpy as np
import sys

# Constants
SQRT3           =   1.7320508
COULOMB_CONST   =   14.3996448915     # [eV A]
const_Boltzman  =   8.617333262145e-5 # [eV/K]

def multipole_energy(d, order, cs, ir=None ):
    """
    Computes multipole interaction energy between a point and a charge distribution for an array of positions.

    Parameters:
    d (np.ndarray): Array of vectors between interaction points (shape: (n, 3)).
    order (int): Maximum order of multipole expansion (0=monopole, 1=dipole, 2=quadrupole).
    cs (np.ndarray): Array of multipole coefficients (shape: (10,)).

    Returns:
    np.ndarray: Array of total interaction energies for all positions.
    """
    dx=d[:,0]
    dy=d[:,1]
    dz=d[:,2]
    if ir is None:
        ir2 = 1.0 / ( dx*dx + dy*dy + dz*dz )  # 1 / r^2 for all positions
        ir  = np.sqrt(ir2)                     # 1 / r for all positions
    else:
        ir2 = ir*ir
    print( "ir.shape, cs.shape ", ir.shape, cs.shape)
    E = cs[0] * ir                         # Monopole term
    if order > 0:
        ir3 = ir2 * ir
        E += ir3 * ( cs[1] * dx 
                   + cs[2] * dy 
                   + cs[3] * dz )  # Dipole term
    if order > 1:
        ir5 = ir3 * ir2
        E += ir5 * (
            (cs[4] * dx + cs[9] * dy) * dx +  # Quadrupole terms
            (cs[5] * dy + cs[7] * dz) * dy +
            (cs[6] * dz + cs[8] * dx) * dz )
    return E

def makePosXY(n=100, L=10.0, axs=(0,1,2), p0=(0.0,0.0,0.0) ):
    x = np.linspace(-L,L,n)
    y = np.linspace(-L,L,n)
    Xs,Ys = np.meshgrid(x,y)
    ps = np.zeros((n*n,3))
    ps[:,axs[0]] = p0[axs[0]] + Xs.flatten()
    ps[:,axs[1]] = p0[axs[1]] + Ys.flatten()
    ps[:,axs[2]] = p0[axs[2]] 
    return ps, Xs, Ys

def makeCircle( n=10, R=1.0, p0=(0.0,0.0,0.0), axs=(0,1,2), phi0=0.0 ):
    phis  = np.linspace(0,2*np.pi,n, endpoint=False) + phi0
    ps    = np.zeros((n,3))
    ps[:,axs[0]] = p0[axs[0]] + np.cos(phis)*R
    ps[:,axs[1]] = p0[axs[1]] + np.sin(phis)*R
    ps[:,axs[2]] = p0[axs[2]]
    return ps, phis
    
def makeRotMats(phis, nsite=3 ):
    rot = np.zeros((nsite,3,3))
    ca = np.cos(phis)
    sa = np.sin(phis)
    rot[:,0,0] = ca
    rot[:,1,1] = ca
    rot[:,0,1] = -sa
    rot[:,1,0] = sa
    rot[:,2,2] = 1.0
    return rot

def getLine(spos, comb1, comb2, n=10):
    " Get line between linear combination of the charge sites"
    ps = np.zeros((n, spos.shape[1]))
    ts = np.linspace(0.0, 1.0, n, endpoint=False)
    ms = 1 - ts
    for i in range(spos.shape[0]):
        ps += spos[i] * (comb1[i] * ms[:, None] + comb2[i] * ts[:, None])
    return ps

def makePosQscan( ps, qs ):
    npoint  = ps.shape[0]
    ncharge = qs.shape[0]
    Ps = np.expand_dims(ps, axis=0)                 
    Ps = np.broadcast_to(Ps, (ncharge, npoint, 3))  
    Qs = np.expand_dims(qs, axis=1)  
    Qs = np.broadcast_to(Qs, (ncharge, npoint))  
    print("npoint ", Ps.shape, " ncharge ", ncharge )
    print("Ps.shape ", Ps.shape, " Qs.shape", Qs.shape)
    return  Ps.copy(), Qs.copy()

def rotate( dpos, rot ):
    dpos_ = np.zeros( dpos.shape )
    dpos_[:,0] = rot[0,0]*dpos[:,0] + rot[0,1]*dpos[:,1] + rot[0,2]*dpos[:,2]
    dpos_[:,1] = rot[1,0]*dpos[:,0] + rot[1,1]*dpos[:,1] + rot[1,2]*dpos[:,2]
    dpos_[:,2] = rot[2,0]*dpos[:,0] + rot[2,1]*dpos[:,1] + rot[2,2]*dpos[:,2]
    return dpos_

def length(d):
    return np.sqrt( np.sum( d*d, axis=1) )

def compute_site_energies_and_hopping( pTip, pSites, siteRots, mpols, Esite0,  Qt, beta=1.0 ):
    """
    Constructs coupling matrix including site energies and interactions using vectorized operations.

    Parameters:
    nsite (int): Number of molecular sites.
    spos (np.ndarray): Array of site positions (shape: (nsite, 3)).
    rot (np.ndarray): Array of rotation matrices for multipole orientations (shape: (nsite, 3, 3)).
    multi_poles (np.ndarray): Array of multipole moments for each site (shape: (nsite, 10)).
    esite0 (np.ndarray): Array of bare site energies (shape: (nsite,)).
    pT (np.ndarray): Tip position (shape: (3,)).
    Qt (float): Tip charge.
    c_coupling (float): Coupling strength parameter.

    Returns:
    tuple: (Esite, Coupling) where Esite is the array of site energies including interactions,
           and Coupling is the matrix for inter-site couplings.
    """
    # Compute tip-site interactions
    nsite = len(pSites)
    ntip = len(pTip)
    Vtip  = np.zeros( ntip )
    dpos  = np.zeros( pTip.shape )
    #if multi_poles is not None:
    #    dpos_ = np.zeros( pTip.shape )
    Esites = np.zeros( (ntip,nsite) )
    Ttips  = np.zeros( (ntip,nsite) )
    for isite in range(nsite):
        dpos[:,:] = pSites[isite,:][np.newaxis,:] - pTip[:,:]
        print( "dpos.shape, pTip.shape, pSites.shape ", dpos.shape, pTip.shape, pSites.shape )
        r  = np.sqrt( np.sum( dpos*dpos, axis=1) )
        ir = 1.0 / r
        if mpols is not None:
            dpos_   = rotate( dpos, siteRots[isite] )
            Vtip[:] = multipole_energy( dpos_, 2, mpols[isite], ir=ir ) * COULOMB_CONST * Qt
        else:
            Vtip[:] = COULOMB_CONST * Qt / np.sqrt( np.sum( dpos*dpos, axis=1) )
        Esites[:,isite] = Vtip[:] + Esite0[isite]

        Ttips[:,isite] = np.exp( -beta * r )
    return Esites, Ttips



def compute_site_energies_and_hopping_mirror( pTip, pSites, siteRots, mpols, Esite0, VBias, Rtip=1.0, beta=1.0, zV0=1.0 ):
    """
    Constructs coupling matrix including site energies and interactions using vectorized operations.

    Parameters:
    nsite (int): Number of molecular sites.
    spos (np.ndarray): Array of site positions (shape: (nsite, 3)).
    rot (np.ndarray): Array of rotation matrices for multipole orientations (shape: (nsite, 3, 3)).
    multi_poles (np.ndarray): Array of multipole moments for each site (shape: (nsite, 10)).
    esite0 (np.ndarray): Array of bare site energies (shape: (nsite,)).
    pT (np.ndarray): Tip position (shape: (3,)).
    Qt (float): Tip charge.
    c_coupling (float): Coupling strength parameter.

    Returns:
    tuple: (Esite, Coupling) where Esite is the array of site energies including interactions,
           and Coupling is the matrix for inter-site couplings.
    """
    # Compute tip-site interactions
    nsite = len(pSites)
    ntip  = len(pTip)
    Vtip  = np.zeros( ntip )
    #dpos  = np.zeros( pTip.shape )
    #if multi_poles is not None:
    #    dpos_ = np.zeros( pTip.shape )
    Esites = np.zeros( (ntip,nsite) )
    Ttips  = np.zeros( (ntip,nsite) )

    VRtip = VBias*Rtip

    pTip_      =         pTip.copy()
    pTip_[:,2] = 2*zV0 - pTip_[:,2]
    #pTip_[:,2] *= -1 

    for isite in range(nsite):
        psite  = pSites[isite,:]
        psite_ = psite*1.0
        psite_[2] = 2*zV0 - psite[2]
        dpos  = psite [np.newaxis,:]   - pTip [:,:]
        dpos_ = psite_[np.newaxis,:]   - pTip_[:,:]
        r = length( dpos )
        if mpols is not None:
            dpos_   = rotate( dpos, siteRots[isite] )
            Vtip[:]  = multipole_energy( dpos_, 2, mpols[isite] ) * VRtip
            Vtip[:] -= multipole_energy( dpos_, 2, mpols[isite] ) * VRtip    # mirror image of the tip
        else:
            Vtip[:]    = VRtip / length( dpos )
            Vtip[:]   -= VRtip / length( dpos_ )      # mirror image of the tip
        Esites[:,isite] = Vtip[:] + Esite0[isite]

        Ttips[:,isite] = np.exp( -beta * r )
    return Esites, Ttips


def compute_site_energies_and_hopping_mirror( pTips, pSites, VBias, Rtip=1.0, zV0=1.0 ):
    # ...
    for isite in range(nsite):
        psite  = pSites[isite,:]
        # ...
    return Esites, Ttips

def compute_V_mirror_2( pTip, pSites, VBias, Rtip=1.0, zV0=1.0 ):
    VR = VBias*Rtip   # VR = COULOMB_CONST by because V = K_coul * Q/r = V0 * (R0/r)
    pTip_    = pTip.copy() 
    pTip_[2] = 2*zV0 - pTip_[2]
    Vtip     =  VR / length( pTip [np.newaxis,:] - pSites[:,:] )
    Vtip_    = -VR / length( pTip_[np.newaxis,:] - pSites[:,:] )      # mirror image of the tip
    return Vtip, Vtip_

def compute_V_mirror( pTip, pSites, VBias, Rtip=1.0, zV0=1.0 ):
    VR       = VBias*Rtip   # VR = COULOMB_CONST by because V = K_coul * Q/r = V0 * (R0/r)
    pTip_    =  pTip.copy() 
    pTip_[2] =  2*zV0 - pTip_[2]
    Vtip     =  VR / length( pTip [np.newaxis,:] - pSites[:,:] )
    Vtip    -=  VR / length( pTip_[np.newaxis,:] - pSites[:,:] )      # mirror image of the tip
    return Vtip

# def compute_site_energies( pTips, pSites, VBias, Rtip=1.0, zV0=1.0):
#     ntip   = pTips.shape[0]  
#     nsite  = pSites.shape[0]  
#     Esites = np.zeros((ntip, nsite)) 
#     print( "pTips.shape ", pTips.shape )
#     print( "pSites.shape ", pSites.shape )
#     for isite in range(nsite):
#         pSite = pSites[isite,:]
#         Esites[:,isite] = compute_V_mirror( np.array((0.0,0.0,0.0)), pSite[np.newaxis,:]-pTips, VBias, Rtip=1.0, zV0=1.0 )
#     return Esites

def compute_site_energies(pTips, pSites, VBias, Rtip=1.0, zV0=1.0, E0s=None ):
    #print( "pTips.shape pTips.shape, pTips, Rtip=1.0, zV0=1.0", pTips.shape, pSites, Rtip, zV0 )
    ntip  = len(pTips)  
    nsite = len(pSites)  
    VR = VBias * Rtip  
    pTips_       = pTips.copy()
    pTips_[:, 2] = 2*zV0 - pTips_[:, 2] 
    Esites = np.zeros((ntip, nsite)) 
    if E0s is None:
        E0s = np.zeros(nsite)
    for isite in range(nsite):
        psite = pSites[isite, :] 
        Vtip  = VR / np.linalg.norm( pTips  - psite, axis=1) 
        Vtip -= VR / np.linalg.norm( pTips_ - psite, axis=1) 
        Esites[:, isite] = Vtip
        #Esites[:, isite] = 1 /( 1 + np.linalg.norm( pTips  - psite, axis=1) )
        if E0s is not None:
            Esites[:, isite] += E0s[isite]
    return Esites

def compute_site_tunelling(pTips, pSites, beta, Amp=1.0 ):
    ntip  = len(pTips)  
    nsite = len(pSites)  
    Tsites = np.zeros((ntip, nsite)) 
    for isite in range(nsite):
        psite = pSites[isite, :] 
        Tsites[:, isite] = np.exp( -beta * np.linalg.norm( pTips - psite, axis=1) )
    return Tsites

def occupancy_FermiDirac( Es, T, mu=0.0 ):
    beta = 1.0 / (const_Boltzman * T )
    return 1.0 / ( 1.0 + np.exp( (Es - mu) * beta ) )


# def compute_site_energies_and_hopping_mirror_2(pTip, pSites, Esite0, VBias, Rtip=1.0, beta=1.0, zV0=1.0):
#     # Compute tip-site interactions
#     nsite = len(pSites)
#     ntip  = len(pTip)
#     Vtip  = np.zeros(ntip)
#     Vtip_ = np.zeros(ntip)
#     Esites = np.zeros((ntip, nsite))
#     Ttips  = np.zeros((ntip, nsite))

#     VRtip = VBias * Rtip

#     # Create the mirror image of the tip positions
#     pTip_ = pTip.copy()
#     pTip_[:, 2] = 2 * zV0 - pTip_[:, 2]  # Mirror the z-coordinate

#     print( "pTip  ", pTip [:, 2].reshape(100,100)[:,0] )
#     print( "pTip_ ", pTip_[:, 2].reshape(100,100)[:,0] )
        
#     # Calculate the potential due to the original tip and its mirror image
#     #Vtip [:] =  VRtip / np.linalg.norm(pTip,  axis=1)
#     #Vtip_[:] = -VRtip / np.linalg.norm(pTip_, axis=1)  # Subtract the mirror image potential

#     Vtip [:] =  VRtip / pTip [:,2]**2
#     Vtip_[:] = -VRtip / pTip_[:,2]**2

#     print( "compute_site_energies_and_hopping_mirror_2() np.sum(Vtip - Vtip_)",  np.sum(Vtip + Vtip_) )    
#     return Vtip, Vtip_


# # Example usage
# nsite = 3
# spos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# rot = np.array([np.eye(3) for _ in range(nsite)])  # Identity matrices for simplicity
# multi_poles = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(nsite)])  # Example multipole coefficients
# esite0 = np.array([0.0, 0.0, 0.0])  # Bare site energies
# pT = np.array([0.0, 0.0, 0.0])  # Tip position
# Qt = 1.0  # Tip charge
# c_coupling = 1.0  # Coupling strength parameter

# Esite, Coupling = make_coupling_matrix_vectorized(nsite, spos, rot, multi_poles, esite0, pT, Qt, c_coupling)
# print("Esite:", Esite)
# print("Coupling Matrix:\n", Coupling)