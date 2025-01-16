import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
import matplotlib.pyplot as plt


import TipMultipole as tmul

def makeCircle( n=10, R=1.0, p0=(0.0,0.0,0.0), axs=(0,1,2) ):
    phis  = np.linspace(0,2*np.pi,n, endpoint=False)
    ps    = np.zeros((n,3))
    ps[:,axs[0]] = p0[axs[0]] + np.cos(phis)*R
    ps[:,axs[1]] = p0[axs[1]] + np.sin(phis)*R
    ps[:,axs[2]] = p0[axs[2]]
    return ps, phis

# Energy of states on the sites
# Rtip   = 1.0
# VBias  = 0.1
# Rcirc  = 0.0
# phiRot = 0.0
# Qzz    = 0.0
# Q0     = 1.0
# L      = 10.0
# npix   = 100
# z_tip  = 6.0
# #zV0    = -2.5
zV0    = -2.5
# zQd    = 0.0

def plot1DpoentialX( VBias=1.0, Rtip=1.0, z_tip=6.0, zVO=-2.5, zQd=0.0, npix=100, L=10.0, bPlot = True, c=None, label=None ):
    ps      = np.zeros((npix,3))
    ps[:,0] = np.linspace(-L,L,npix, endpoint=False)
    ps[:,2] = z_tip+Rtip
    Vtip = tmul.compute_site_energies( ps, np.array([[0.0,0.0,zQd],]), VBias, Rtip, zV0=zV0 )
    if bPlot:
        plt.plot( ps[:,0], Vtip, c=c, label=label )
    return Vtip, ps

def plotTipPotXZ( VBias=1.0, Rtip=1.0, z_tip=3.0, zVO=-2.5, zQd=0.0, npix=100, L=10.0, bPlot=True ):
    zT = z_tip+Rtip
    ps,Xs,Ys = chr.makePosXY(n=npix, L=L, axs=(0,2,1) )
    Vtip = tmul.compute_V_mirror( np.array([0.0,0.0,zT]), ps, VBias=VBias, Rtip=Rtip, zV0=zV0 )
    Vtip = Vtip .reshape(npix,npix)
    if bPlot:
        plt.title(f'Tip potential R_tip: {Rtip} V_Bias: {VBias}')
        extent     = [-L,L,-L,L]
        circ1,_ = makeCircle( 16, R=Rtip, axs=(0,2,1), p0=(0.0,0.0,      zT) )
        circ2,_ = makeCircle( 16, R=Rtip, axs=(0,2,1), p0=(0.0,0.0,2*zV0-zT) )
        #plt.title('Tip Potnetial')
        im = plt.imshow( Vtip , extent=extent, cmap='bwr', origin='lower', vmin=-VBias, vmax=VBias) 
        #plt.colorbar()
        contour_levels = np.linspace( -VBias, VBias, 25)  # Define levels for the contours
        contour_plot   = plt.contour( Xs, Ys, Vtip, levels=contour_levels, colors='k', linewidths=0.5)
        cbar=plt.colorbar(im)
        cbar.add_lines(contour_plot)
        plt.plot(circ1[:,0], circ1[:,2], ':k')
        plt.plot(circ2[:,0], circ2[:,2], ':k')
        plt.xlabel('x [A]')
        plt.ylabel('y [A]')
        plt.grid()
    return Vtip, ps

def plot2DpoentialXZ( VBias=1.0, Rtip=1.0, z_tip=3.0, zVO=-2.5, zQd=0.0, npix=100, L=10.0, bPlot=True ):
    pSites = np.array([[0.0,0.0,zQd],])
    ps,Xs,Ys = chr.makePosXY(n=npix, L=L, axs=(0,2,1) )
    Esites   = tmul.compute_site_energies( ps, pSites, VBias, Rtip, zV0=zV0 )
    Esites   = Esites.reshape(npix,npix, len(pSites))
    if bPlot:
        plt.title(f'Site Energy shift R_tip: {Rtip} V_Bias: {VBias}')
        extent   = [-L,L,-L,L]
        im = plt.imshow( Esites[:,:,0] , extent=extent, cmap='bwr', origin='lower', vmin=-VBias, vmax=VBias)
        #plt.colorbar()
        #contour_levels = np.linspace( -VBias, VBias, 25)  # Define levels for the contours
        #contour_plot   = plt.contour( Xs, Ys, Vtip, levels=contour_levels, colors='k', linewidths=0.5)
        plt.axhline(zV0, ls='--',c='k', label='mirror surface')
        plt.axhline(zQd, ls='--',c='g', label='Qdot height')
        plt.xlabel('x_Tip [A]')
        plt.ylabel('z_Tip [A]')
        plt.legend()
        plt.grid()
        cbar=plt.colorbar(im)
    return Esites, ps

plt.figure()
plotTipPotXZ( VBias=1.0, Rtip=1.0, z_tip=3.0, zVO=-2.5, zQd=0.0, npix=100, L=10.0, bPlot=True )
#plt.savefig("test_TipPotential_mirror_1.png")

plt.figure()
plot2DpoentialXZ( VBias=1.0, Rtip=1.0, z_tip=3.0, zVO=-2.5, zQd=0.0, npix=100, L=10.0, bPlot=True )



zV0s  = [-2.0, -2.5, -3.0   ]
zTips = [1.0, 2.0, 3.0, 4.0 ]
Rtips = [0.5, 1.0, 1.5, 2.0]

npx = len(zTips)
npy = len(zV0s)
plt.figure( figsize=(npx*5,npy*5)  )
for iz0, zV0 in enumerate( zV0s ):
    for iz, z_tip in enumerate( zTips ):
        plt.subplot(npy,npx, iz0*npx + iz + 1 )
        plt.title( f'z_tip: {z_tip}  z_V0: {zV0} ' )
        for ir,Rtip in enumerate( Rtips ):
            #plt.plot( ps[:,0], Esites[:,:,i].reshape(npix,npix)[0], label='i='+str(i) )
            plot1DpoentialX( Rtip=Rtip, z_tip=z_tip, zVO=zV0, label=f'Rtip: {Rtip}' )
            plt.legend()
        plt.xlabel('x_Tip [A]')
        plt.ylabel('V_tip [V]')
        plt.grid()
        plt.ylim(0.0,1.0)

plt.savefig("test_TipPotential_mirror_1D.png")

plt.show()

