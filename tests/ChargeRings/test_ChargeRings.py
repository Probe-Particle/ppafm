import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
#from pyProbeParticle import atomicUtils as au
import matplotlib.pyplot as plt


# ==================== Functions


def makePosXY( n=100, L=10.0, z0=5.0 ):
    x = np.linspace(-L,L,n)
    y = np.linspace(-L,L,n)
    Xs,Ys = np.meshgrid(x,y)
    ps = np.zeros( (n*n,3) )
    ps[:,0] = Xs.flatten()
    ps[:,1] = Ys.flatten()
    ps[:,2] = z0
    return ps


# ==================== Main

chr.setVerbosity( 0 );

Qtip = 0.48
ps    = [[0.0,0.0,0.0],]


R = 5.0
nsite = 3
phis = np.linspace(0,2*np.pi,nsite, endpoint=False); print(phis)
spos = np.zeros((3,3));
spos[:,0] = np.cos(phis)*R
spos[:,1] = np.sin(phis)*R
#print( "spos ", spos)
#plt.plot( spos[:,0], spos[:,1], 'o'); plt.axis('equal'); 

Esite = [ -1.0, -1.0, -1.0 ]

# ----- Multi Poles
rot   = np.zeros( (3,3,3) )
mpols = np.zeros( (3,10)  )
phi2 = phis + 0.3
ca = np.cos(phi2)
sa = np.sin(phi2) 
rot[:,0,0] =  ca
rot[:,1,1] =  ca
rot[:,0,1] = -sa
rot[:,1,0] =  sa
rot[:,2,2] =  1.0
mpols[:,4] = 10.0
mpols[:,5] = 1.0*0
mpols[:,0] = 1.0

# ------ Make tip positions over site1
#ps = spos[0:1,:].copy();  
#ps[:,2] = 5.0 
#Qtips = [-Qtip]
#print( "ps ", ps)

L=20.0
npix=400
extent=[-L,L,-L,L]
ps    = chr.makePosXY( n=npix, L=L, z0=5.0 )
Qtips = np.ones( len(ps) )*Qtip
print( "ps ", ps)

print("==== to C++ ===")

#Qsites = chr.solveSiteOccupancies( ps, Qtips, spos, Esite, E_mu=0.0, cCouling=-0.01, niter=1000, tol=1e-6, dt=0.1 ).reshape( (npix,npix,nsite) )
#Qsites, niters = chr.solveSiteOccupancies( ps, Qtips, spos, Esite, MultiPoles=mpols, rot=rot, E_fermi=0.0, cCouling=0.03, niter=1000, tol=1e-6, dt=0.5 )
#Qsites, niters = chr.solveSiteOccupancies_old( ps, Qtips, spos, Esite, MultiPoles=mpols, rot=rot, E_fermi=0.0, cCoupling=0.03 )
Qsites,_,_ = chr.solveSiteOccupancies_old( ps, Qtips, spos, Esite, MultiPoles=mpols, rot=rot, E_fermi=0.0, cCoupling=0.03, temperature=100.0 )

Qsites = Qsites.reshape( (npix,npix,nsite) )
#niters = niters.reshape( (npix,npix) )


Qtot = np.sum( Qsites, axis=2 )

#print( "Qsites ", Qsites)
#print( " Qsites ", Qsites )

# plt.figure(figsize=(25,5));
# plt.subplot(1,5,1); plt.imshow( Qtot         , origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'or'); plt.colorbar(); plt.title("Qtot")
# plt.subplot(1,5,2); plt.imshow( Qsites[:,:,0], origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'or'); plt.colorbar(); plt.title("Q site 1")
# plt.subplot(1,5,3); plt.imshow( Qsites[:,:,1], origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'or'); plt.colorbar(); plt.title("Q site 2")
# plt.subplot(1,5,4); plt.imshow( Qsites[:,:,2], origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'or'); plt.colorbar(); plt.title("Q site 3")
# plt.subplot(1,5,5); plt.imshow( niters       , origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'or'); plt.colorbar(); plt.title("niters")

plt.figure(figsize=(20,5));
plt.subplot(1,4,1); plt.imshow( Qtot         , origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'og'); plt.colorbar(); plt.title("Qtot")
plt.subplot(1,4,2); plt.imshow( Qsites[:,:,0], origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'og'); plt.colorbar(); plt.title("Q site 1")
plt.subplot(1,4,3); plt.imshow( Qsites[:,:,1], origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'og'); plt.colorbar(); plt.title("Q site 2")
plt.subplot(1,4,4); plt.imshow( Qsites[:,:,2], origin="lower", extent=extent ); plt.plot( spos[:,0], spos[:,1], 'og'); plt.colorbar(); plt.title("Q site 3")
plt.tight_layout()
plt.savefig("test_ChargeRings.png", bbox_inches='tight')



# After calculating Qsites, compute STM map
# I_stm = chr.getSTM_map(ps, Qsites.reshape(-1,nsite), spos, Esite,  rot=rot, MultiPoles=mpols, Q_tip=Qtip, E_Fermi=0.0, cCoupling=0.03, beta=1.0)

# # Plot STM map
# plt.figure(figsize=(25,5))
# plt.subplot(1,5,5)
# plt.imshow(I_stm.reshape(npix,npix), origin="lower", extent=extent)
# plt.plot(spos[:,0], spos[:,1], 'or')
# plt.colorbar()
# plt.title("STM")



plt.show()