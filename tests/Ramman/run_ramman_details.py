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
nx=150; ny=100     # Image Resolution

# ============ SETUP

# ----- Load Data
alphas = rm.reOrder( np.genfromtxt(wdir+fname_alphas) )  
modes  =             np.genfromtxt(wdir+fname_modes).transpose().copy() 
apos,Zs,enames,qs = au.loadAtomsNP(wdir+fname_geom)

#                          s   px  py   pz
#rm.setEfieldMultipole( [0.0              ], Ehomo=[1.0,0.0,0.0]  ) 
rm.setEfieldMultipole( [1.0, 0.0,0.0,0.0    ], Ehomo=[0.0,0.0,0.2]  ) 
#rm.setEfieldMultipole( [1.0              ] )   # monopole tip field
#rm.setEfieldMultipole(  [0.0, 0.0, 0.0,1.0] )   # field of tip with dipole along z-axis
#rm.setEfieldMultipole(  [0.0, 1.0, 0.0,0.0] )    # field of tip with dipole along x-axis
#rm.setEfieldMultipole(  [0.0, 0.0, 1.0,0.0] )    # field of tip with dipole along y-axis
#rm.setEfieldMultipole(  [0.9, 0.5,-0.5,0.8] )   # field of general asymmetric tip with dipole/monopole mix

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
xs    = np.linspace(-15.0,15.0,nx) 
ys    = np.linspace(-10.0,10.0,ny) 
tpos  = np.zeros( (ny,nx,3) )
#tpos[:,:,0],tpos[:,:,1] = np.meshgrid( xs, ys ); tpos[:,:,2] = 2.0 # z above surface
tpos[:,:,2],tpos[:,:,1] = np.meshgrid( xs, ys ); tpos[:,:,0] = 3.0 # z above surface
tippos = tpos.reshape((nx*ny,3))
extent=(-15.0,15.0, -10.0,10.0)
Es = rm.EfieldAtPoints( tippos )
Es = Es.reshape((ny,nx,3))

plt.figure(figsize=(20,5))
plt.subplot(1,4,1); plt.imshow( Es[:,:,0], extent=extent ); plt.title("E_x"); plt.colorbar();
plt.subplot(1,4,2); plt.imshow( Es[:,:,1], extent=extent ); plt.title("E_y"); plt.colorbar();
plt.subplot(1,4,3); plt.imshow( Es[:,:,2], extent=extent ); plt.title("E_z"); plt.colorbar();
plt.subplot(1,4,4); plt.imshow( np.sqrt(Es[:,:,0]**2+Es[:,:,1]**2+Es[:,:,2]**2), extent=extent ); plt.title("|E|"); plt.colorbar();


# ----- Electric field and polarization at each atom
#rm.setVerbosity(3)
imode=20
tpos= [0.0,0.0,10.0]
Amp, Einc, Eind, modeOut = rm.RammanDetails( tpos, apos, alphas, modes, imode ) 
Einc /= np.max(  np.sqrt((Einc**2).sum(axis=1)) ) # Normalize
Eind /= np.max(  np.sqrt((Eind**2).sum(axis=1)) ) # Normalize

def plotVecsAtAtoms( apos, vecs, axis1=2, axis2=1, width=0.005, headwidth=0.05, vsc=1 ):
    plt.plot( apos[:,axis1],apos[:,axis2], '.b' ); 
    plt.quiver( apos[:,axis1],apos[:,axis2], vecs[:,axis1]*vsc,vecs[:,axis2]*vsc, width=width, headwidth=headwidth ); 
    plt.axis('equal');

plt.figure(figsize=(15,5))
vsc=100.0; axis1=1; axis2=2;
plt.subplot(1,3,1); plotVecsAtAtoms( apos, modeOut, axis1=axis1, axis2=axis2, vsc=vsc ); plt.title('vib.mode')
plt.subplot(1,3,2); plotVecsAtAtoms( apos, Eind   , axis1=axis1, axis2=axis2, vsc=vsc ); plt.title('polarization')
plt.subplot(1,3,3); plotVecsAtAtoms( apos, Einc   , axis1=axis1, axis2=axis2, vsc=vsc ); plt.title('Efield')

rm.write_xyz_vecs(  "Einc.xyz", enames, apos, Einc, tpos=tpos )
rm.write_xyz_vecs(  "Eind.xyz", enames, apos, Eind, tpos=tpos )
rm.write_xyz_vecs(  "modeOut.xyz", enames, apos, modeOut, tpos=tpos )

#f=open("Einc.xyz",'w'); f.write("%i\n\n", len(apos)+1 ); rm.write_to_xyz_vecs( f, enames, apos, Einc ); rm.write_to_xyz_vecs( f, enames, apos, Einc )
#f=open("Eind.xyz",'w'); f.write("%i\n\n", len(apos)+1 ); rm.write_to_xyz_vecs( f, enames, apos, Einc ); rm.write_to_xyz_vecs( f, enames, apos, Eind )



#print( Amp )

#plt.plot( tpos[:,0], Amp )
plt.show()