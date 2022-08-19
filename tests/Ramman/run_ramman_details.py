#!/usr/bin/python3

import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import Ramman as rm
from pyProbeParticle import atomicUtils as au
import matplotlib.pyplot as plt

# ============ SETUP
wdir='./'
fname_geom='input-yz.xyz'; 
fname_modes='data_NModes.dat'; 
fname_alphas='data_Dalpha-wrt-X.dat'

# --- Efield imagind setup
nx=100; ny=150     # Image Resolution
cutPlaneHeight = 0.0  
bPlotFieldOnMod = bool
choosePlane = 1  #   0=(x,yz), 1=(y,xz), 2=(z,xy)
Emax = 1.0

# --- Detailed raman calculation
bDefuLog = False
imode=20
tip_pos= np.array([0.0,0.0,10.0])

# ============ SETUP

# ----- Load Data
alphas = rm.reOrder( np.genfromtxt(wdir+fname_alphas) )  
modes  =             np.genfromtxt(wdir+fname_modes).transpose().copy() 
apos,Zs,enames,qs = au.loadAtomsNP(wdir+fname_geom)

#                          s   px  py   pz
#rm.setEfieldMultipole( [0.0              ], Ehomo=[1.0,0.0,0.0]  ) 
#rm.setEfieldMultipole( [1.0, 0.0,0.0,0.0    ], Ehomo=[0.0,0.0,0.2]  ) 
#rm.setEfieldMultipole( [1.0              ] )   # monopole tip field
#rm.setEfieldMultipole(  [0.0, 0.0, 0.0,1.0] )   # field of tip with dipole along z-axis
#rm.setEfieldMultipole(  [0.0, 1.0, 0.0,0.0] )    # field of tip with dipole along x-axis
#rm.setEfieldMultipole(  [0.0, 0.0, 1.0,0.0] )    # field of tip with dipole along y-axis
#rm.setEfieldMultipole(  [0.9, 0.5,-0.5,0.8] )   # field of general asymmetric tip with dipole/monopole mix

rm.setEfieldMultipole( [3.5              ], Ehomo=[0.0,0.0,1.0], gaussR=[10.,10.,10.]  ) 


'''
# ----- Plot Electri Field Components 1D
nz    = 200 
zs    = np.linspace(-20.0,20.0,nz)
tpos  = np.zeros( (len(zs),3) )
tpos[:,2]=zs
tpos[:,0]=0.1 # x
Es    = rm.EfieldAtPoints( tpos )
plt.plot(zs,Es[:,0],label="E_x")
plt.plot(zs,Es[:,1],label="E_y")
plt.plot(zs,Es[:,2],label="E_z")
ymax=10.0
plt.ylim(-ymax,ymax)
plt.legend()
'''

# ----- Plot Electri Field Components 2D
ax0=2;ax1=0;ax2=1  #  2=(z,xy)
if   choosePlane==0:
    ax0=1;ax1=0;ax2=2  # 1=(y,xz)
elif choosePlane==1:   
    ax0=0;ax1=1;ax2=2      # 0=(x,yz)

extent=(-nx*0.1,nx*0.1, -ny*0.1,ny*0.1)
xs    = np.linspace(-nx*0.1,nx*0.1,nx) 
ys    = np.linspace(-ny*0.1,ny*0.1,ny) 
poss  = np.zeros( (ny,nx,3) )
poss[:,:,ax1],poss[:,:,ax2] = np.meshgrid( xs, ys ); poss[:,:,ax0] = cutPlaneHeight # z above surface
poss_mol = poss.copy()
if bPlotFieldOnMod:
    poss[:,:,0]-=tip_pos[0]
    poss[:,:,1]-=tip_pos[1]
    poss[:,:,2]-=tip_pos[2]
poss = poss.reshape((nx*ny,3))
Es = rm.EfieldAtPoints( poss )
Es = Es.reshape((ny,nx,3))

plt.figure(figsize=(20,5))
plt.subplot(1,4,1); plt.imshow( Es[:,:,0], extent=extent, vmin=-Emax, vmax=Emax,cmap='seismic', origin='lower' ); plt.title("E_x"); plt.colorbar();
plt.subplot(1,4,2); plt.imshow( Es[:,:,1], extent=extent, vmin=-Emax, vmax=Emax,cmap='seismic', origin='lower' ); plt.title("E_y"); plt.colorbar();
plt.subplot(1,4,3); plt.imshow( Es[:,:,2], extent=extent, vmin=-Emax, vmax=Emax,cmap='seismic', origin='lower' ); plt.title("E_z"); plt.colorbar();
plt.subplot(1,4,4); plt.imshow( np.sqrt(Es[:,:,0]**2+Es[:,:,1]**2+Es[:,:,2]**2), extent=extent, vmin=0, vmax=Emax, origin='lower' ); plt.title("|E|"); plt.colorbar();
if bPlotFieldOnMod:
    plt.plot  ( apos[:,ax1], apos[:,ax2],'ok')
    vsc=100.0;
    #plt.quiver( apos[:,ax1], apos[:,ax2], Es[:,ax1]*vsc,Es[:,ax2]*vsc, width=0.005, headwidth=0.05 ); 

# ----- Electric field and polarization at each atom
if bDefuLog:
    rm.setVerbosity(3) # Uncomment this to get verbous debug log during calculation (this should be off when doing many calculations, very much slower)

Amp, Einc, Eind, modeOut, Ampis = rm.RammanDetails( tip_pos, apos, alphas, modes, imode ) 
Einc /= np.max(  np.sqrt((Einc**2).sum(axis=1)) ) # Normalize
Eind /= np.max(  np.sqrt((Eind**2).sum(axis=1)) ) # Normalize

def plotVecsAtAtoms( apos, vecs, ax1=2, ax2=1, width=0.005, headwidth=0.05, vsc=1 ):
    plt.plot  ( apos[:,ax1],apos[:,ax2], '.b' ); 
    plt.quiver( apos[:,ax1],apos[:,ax2], vecs[:,ax1]*vsc,vecs[:,ax2]*vsc, width=width, headwidth=headwidth ); 
    plt.axis('equal');

def setPlotExtent(extent):
    plt.xlim(extent[0],extent[1]) 
    plt.ylim(extent[2],extent[3])

plt.figure(figsize=(15,5))
vsc=100.0;
plt.subplot(1,5,1); plotVecsAtAtoms( apos, Einc   , ax1=ax1, ax2=ax2, vsc=vsc ); setPlotExtent(extent); plt.title('Efield')
plt.subplot(1,5,2); plotVecsAtAtoms( apos, modeOut, ax1=ax1, ax2=ax2, vsc=vsc ); setPlotExtent(extent); plt.title('vib.mode')
plt.subplot(1,5,3); plotVecsAtAtoms( apos, Eind   , ax1=ax1, ax2=ax2, vsc=vsc ); setPlotExtent(extent); plt.title('polarization')
vmax=np.max(np.abs(Ampis))
plt.subplot(1,5,4); plt.scatter    ( apos[:,ax1],apos[:,ax2], c=Ampis,    cmap='seismic', vmin=-vmax, vmax=vmax );  plt.title('Ampi');     plt.axis('equal'); setPlotExtent(extent);
plt.subplot(1,5,5); plt.scatter    ( apos[:,ax1],apos[:,ax2], c=np.abs(Ampis), cmap='binary'); plt.title('|Ampi|'); setPlotExtent(extent); plt.axis('equal');  setPlotExtent(extent);


rm.write_xyz_vecs(  "Einc.xyz", enames, apos, Einc, tpos=tip_pos )
rm.write_xyz_vecs(  "Eind.xyz", enames, apos, Eind, tpos=tip_pos )
rm.write_xyz_vecs(  "modeOut.xyz", enames, apos, modeOut, tpos=tip_pos )

#f=open("Einc.xyz",'w'); f.write("%i\n\n", len(apos)+1 ); rm.write_to_xyz_vecs( f, enames, apos, Einc ); rm.write_to_xyz_vecs( f, enames, apos, Einc )
#f=open("Eind.xyz",'w'); f.write("%i\n\n", len(apos)+1 ); rm.write_to_xyz_vecs( f, enames, apos, Einc ); rm.write_to_xyz_vecs( f, enames, apos, Eind )



#print( Amp )

#plt.plot( tpos[:,0], Amp )
plt.show()