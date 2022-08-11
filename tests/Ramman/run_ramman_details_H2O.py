#!/usr/bin/python3

import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import Ramman as rm
from pyProbeParticle import atomicUtils as au
import matplotlib.pyplot as plt

# ============ SETUP
wdir='./'
fname_geom='H2O.xyz'; 
fname_modes='H2O_modes.txt'; 
fname_alphas='H2O_DaX.txt'

# ============ SETUP

# ----- Load Data
alphas = rm.reOrder( np.genfromtxt(wdir+fname_alphas) )  
modes  =             np.genfromtxt(wdir+fname_modes).transpose().copy() ; print("modes.shape ", modes.shape)
#modes  =             np.genfromtxt(wdir+fname_modes); print("modes.shape ", modes.shape)
apos,Zs,enames,qs = au.loadAtomsNP(wdir+fname_geom)

#                          s   px  py   pz
#rm.setEfieldMultipole( [1.0              ] )   # monopole tip field
#rm.setEfieldMultipole(  [0.0, 0.0, 0.0,1.0] )   # field of tip with dipole along z-axis
rm.setEfieldMultipole(  [0.0, 1.0, 0.0,0.0] )    # field of tip with dipole along x-axis
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

# ----- Plot Electri Field Components 2D
nx=150; ny=100     # Image Resolution
xs    = np.linspace(-15.0,15.0,nx) 
ys    = np.linspace(-10.0,10.0,ny) 
tpos  = np.zeros( (ny,nx,3) )
#tpos[:,:,0],tpos[:,:,1] = np.meshgrid( xs, ys ); tpos[:,:,2] = 2.0 # z above surface
tpos[:,:,2],tpos[:,:,1] = np.meshgrid( xs, ys ); tpos[:,:,0] = 3.0 # z above surface
tippos = tpos.reshape((nx*ny,3))
extent=(-15.0,15.0, -10.0,10.0)
Es = rm.EfieldAtPoints( tippos )
Es = Es.reshape((ny,nx,3))

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow( Es[:,:,0], extent=extent ); plt.title("E_x")
plt.subplot(1,3,2); plt.imshow( Es[:,:,1], extent=extent ); plt.title("E_y")
plt.subplot(1,3,3); plt.imshow( Es[:,:,2], extent=extent ); plt.title("E_z")
'''

# ----- Electric field and polarization at each atom
rm.setVerbosity(3)
imode=2
tpos= [0.0,0.0,5.0]
Amp, Einc, Eind, modeOut = rm.RammanDetails( tpos, apos, alphas, modes, imode ) 
Einc /= np.max(  np.sqrt((Einc**2).sum(axis=1)) ) # Normalize
Eind /= np.max(  np.sqrt((Eind**2).sum(axis=1)) ) # Normalize

#displ = modes[imode].reshape((len(apos),3))
rm.write_xyz_vecs(  "Einc.xyz", enames, apos, Einc, tpos=tpos )
rm.write_xyz_vecs(  "Eind.xyz", enames, apos, Eind, tpos=tpos )
rm.write_xyz_vecs(  "modeOut.xyz", enames, apos, modeOut, tpos=tpos )

#f=open("Einc.xyz",'w'); f.write("%i\n\n", len(apos)+1 ); rm.write_to_xyz_vecs( f, enames, apos, Einc ); rm.write_to_xyz_vecs( f, enames, apos, Einc )
#f=open("Eind.xyz",'w'); f.write("%i\n\n", len(apos)+1 ); rm.write_to_xyz_vecs( f, enames, apos, Einc ); rm.write_to_xyz_vecs( f, enames, apos, Eind )



#print( Amp )

#plt.plot( tpos[:,0], Amp )
plt.show()