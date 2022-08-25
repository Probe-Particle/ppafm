#!/usr/bin/python3

import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import Ramman as rm
from pyProbeParticle import atomicUtils as au
import matplotlib.pyplot as plt

# ============ SETUP
mode_selection=[0,1,2,3,10,20,30,40]
wdir='./'
fname_geom='input-yz.xyz'; 
fname_modes='data_NModes.dat'; 
fname_alphas='data_Dalpha-wrt-X.dat'
nx=150; ny=100     # Image Resolution

# --- Efield imagind setup
nx=100; ny=150     # Image Resolution
cutPlaneHeight = 10.0  
bPlotFieldOnMod = bool
choosePlane = 0  #   0=(x,yz), 1=(y,xz), 2=(z,xy)
Emax = 1.0

# ----- Load Data
alphas = rm.reOrder( np.genfromtxt(wdir+fname_alphas) )  
modes  =             np.genfromtxt(wdir+fname_modes).transpose().copy() 
apos,Zs,enames,qs = au.loadAtomsNP(wdir+fname_geom)

#                          s   px  py   pz
#rm.setEfieldMultipole( [1.0              ] )   # monopole tip field
#rm.setEfieldMultipole(  [0.0, 0.0, 0.0,1.0] )   # field of tip with dipole along z-axis
#rm.setEfieldMultipole(  [0.0, 1.0, 0.0,0.0] )    # field of tip with dipole along x-axis
#rm.setEfieldMultipole(  [0.0, 0.0, 1.0,0.0] )    # field of tip with dipole along y-axis
rm.setEfieldMultipole(  [0.9, 0.5,-0.5,0.8] )   # field of general asymmetric tip with dipole/monopole mix

# ----- Generate Tip Position ( 2D for Image Scan)

ax0=2;ax1=0;ax2=1  #  2=(z,xy)
if   choosePlane==0:
    ax0=0;ax1=1;ax2=2  # 0=(x,yz)
elif choosePlane==1:  
    ax0=1;ax1=0;ax2=2  # 1=(y,xz) 
extent=(-nx*0.1,nx*0.1, -ny*0.1,ny*0.1)
xs    = np.linspace(-nx*0.1,nx*0.1,nx) 
ys    = np.linspace(-ny*0.1,ny*0.1,ny) 
poss  = np.zeros( (ny,nx,3) )
poss[:,:,ax1],poss[:,:,ax2] = np.meshgrid( xs, ys ); poss[:,:,ax0] = cutPlaneHeight # z above surface
tippos = poss.reshape((nx*ny,3))

#xs    = np.linspace(-15.0,15.0,nx) 
#ys    = np.linspace(-10.0,10.0,ny) 
#tpos  = np.zeros( (ny,nx,3) )
#tpos[:,:,0],tpos[:,:,1] = np.meshgrid( xs, ys ); tpos[:,:,2] = 2.0 # z above surface
#tpos[:,:,2],tpos[:,:,1] = np.meshgrid( xs, ys ); tpos[:,:,0] = 3.0 # z above surface
#tippos = tpos.reshape((nx*ny,3))
#extent=(-15.0,15.0, -10.0,10.0)

# ----- Scan Selected modes
#mode_selection=[15]
nmod=len(mode_selection)
ncol=4
nrow=np.ceil(nmod/ncol)
plt.figure( figsize=(5*ncol,5*nrow) )
for i,imode in enumerate(mode_selection):
    # DEBUG
    Amps = rm.RammanAmplitudes( tippos, apos, alphas, modes, imode=imode ) 
    displ = modes[imode].reshape((len(apos),3))
    #print( modes[imode] )
    #print( "displ.shape ", displ.shape, modes.shape )
    plt.subplot(nrow,ncol,i+1)
    plt.imshow( Amps.reshape((ny,nx))**2, extent=extent )
    #plt.plot  ( apos[:,0],apos[:,1], '.k' )
    #plt.quiver( apos[:,0],apos[:,1], displ[:,0],displ[:,1], width=0.005, headwidth=0.005 )
    plt.plot  ( apos[:,ax1],apos[:,ax2], '.k' )
    plt.quiver( apos[:,ax1],apos[:,ax2], displ[:,ax1],displ[:,ax2], width=0.005, headwidth=0.005 )
    plt.title ( 'Mode#%i' %imode )

#print( Amp )

#plt.plot( tpos[:,0], Amp )
plt.show()