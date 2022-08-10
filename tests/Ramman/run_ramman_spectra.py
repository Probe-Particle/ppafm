#!/usr/bin/python3


## --  history:
# 2022-06-23 _ prokop: code
# 2022-06-28 _ sofia: do simulation for single tip position
# 2022-07-01 _ sofia: read redmass/freq; multiply each mode by this 
# 2022-07-01 _ sofia: read single tip coordinates from file (inp_tippos)
# 2022-08-02 _ prokop: introduce new electric fields
#
## --  TO do:
# feed external Efield from input
# ........................................................... #
#

import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import Ramman as rm
from pyProbeParticle import atomicUtils as au
import matplotlib.pyplot as plt


# ============ SETUP
wdir='./'
# ----- NModes: all or a selection
nmod = 108  # sofia
 #mode_selection=[0,1,2,3,10,20,30,40]
# ----- Read input files
fname_geom='input-xy.xyz'; 
fname_modes='data_NModes.dat'; 
fname_alphas='data_Dalpha-wrt-X.dat'
fname_Freq12='data_Freqau-m12.dat' # sofia 
fname_RedM12='data_RedMass-m12.dat' # sofia
fname_tippos='inp_tippos' # sofia
#fname_Efield='inp_Efield' # sofia
fname_Freq='data_Freq.dat' # sofia
# ----- Image Resolution 
nx=150; ny=100     


# ============ Load Data
alphas = rm.reOrder( np.genfromtxt(wdir+fname_alphas) )  
modes  =             np.genfromtxt(wdir+fname_modes).transpose().copy() 
RedM12  =            np.genfromtxt(wdir+fname_RedM12).transpose().copy()    # sofia
Freq12  =            np.genfromtxt(wdir+fname_Freq12).transpose().copy()    # sofia
Freq  =              np.genfromtxt(wdir+fname_Freq).copy()    # sofia
apos,Zs,enames,qs = au.loadAtomsNP(wdir+fname_geom)
 # tippos = np.array( [np.genfromtxt(wdir+fname_tippos).transpose().copy(),] ) # sofia , read after to not rewrite
#Efield = np.array( [np.genfromtxt(wdir+fname_Efield).transpose().copy(),] ) # sofia
#Efield = np.array( [np.genfromtxt(wdir+fname_Efield).copy(),] ) # sofia

# ============ Electric field definition
#                          s   px  py   pz
 #rm.setEfieldMultipole( [1.0              ] )     # monopole tip field
 #rm.setEfieldMultipole(  [0.0, 0.0, 0.0,0.8] )   # field of tip with dipole along z-axis
 #rm.setEfieldMultipole(  [0.9, 0.5,-0.5,0.8] )   # field of general asymmetric tip with dipole/monopole mix
rm.setEfieldMultipole( [5.0, 0.0, 0.0, 0.0] )
 #print ( " " ) 
 #print ( " Electric field : ", Efield)  # sofia
 #rm.setEfieldMultipole( [Efield[0], Efield[1], Efield[2], Efield[3]] )     # sofia 


# ============ Generate Tip Position . NOTA: somehow implicit assumed the molecule centered in 0,0,0
# -- 1) Tip positions as 2D array, for Image Scan: scan on the "molecule" area (Angstrom)
xs    = np.linspace(-15.0,15.0,nx) 
ys    = np.linspace(-10.0,10.0,ny) 
tpos  = np.zeros( (ny,nx,3) )
 #tpos[:,:,0],tpos[:,:,1] = np.meshgrid( xs, ys ); tpos[:,:,2] = 2.0 # z above surface
tpos[:,:,2],tpos[:,:,1] = np.meshgrid( xs, ys ); tpos[:,:,0] = 3.0 # z above surface
tippos = tpos.reshape((nx*ny,3))
extent=(-15.0,15.0, -10.0,10.0)
# -- 2) Tip position as a single point, for simulate single spectra - - tpos xyz vectr (Angstrom)
tippoint = np.array( [np.genfromtxt(wdir+fname_tippos).transpose().copy(),] ) # sofia , read after to not rewrite
print ( "  " )  # sofia
print ( " tip placed in [A] : " , tippoint)  # sofia


# ============ SIMULATION 
'''
# ----- 1. Image 2D, Scan Selected modes
nmod=len(mode_selection)
ncol=4
nrow=np.ceil(nmod/ncol)
plt.figure( figsize=(5*ncol,5*nrow) )
for i,imode in enumerate(mode_selection):
    # DEBUG
    prefact = Freq12[imode]*RedM12[imode]    # sofia
    modesFRM = np.zeros(modes.shape)          # sofia
    modesFRM[imode,:] = prefact * modes[imode,:] # sofia
     #Amps = rm.RammanAmplitudes( tippos, apos, alphas, modes, imode=imode ) 
    Amps = rm.RammanAmplitudes( tippos, apos, alphas, modesFRM, imode=imode ) 
    displ = modes[imode].reshape((len(apos),3))
    #print( modes[imode] )
    #print( "displ.shape ", displ.shape, modes.shape )
    plt.subplot(nrow,ncol,i+1)
    plt.imshow( Amps.reshape((ny,nx))**2, extent=extent )
    #plt.plot  ( apos[:,0],apos[:,1], '.k' )
    #plt.quiver( apos[:,0],apos[:,1], displ[:,0],displ[:,1], width=0.005, headwidth=0.005 )
    plt.plot  ( apos[:,2],apos[:,1], '.k' )
    plt.quiver( apos[:,2],apos[:,1], displ[:,2],displ[:,1], width=0.005, headwidth=0.005 )
    plt.title ( 'Mode#%i' %imode )
'''
# ----- 2. Spectra simulation with given tip position
with open("out_Amps2.dat",'wb') as f_out1:           # sofia
    for imode in range(0,nmod):                  # sofia 
       prefact = Freq12[imode]*RedM12[imode]    # sofia
       modesFRM = np.zeros(modes.shape)          # sofia
       modesFRM[imode,:] = prefact * modes[imode,:] # sofia
       Amps = rm.RammanAmplitudes( tippoint, apos, alphas, modesFRM, imode=imode ) #sofia
       np.savetxt(f_out1, Amps**2, fmt='%.6f') # sofia


# ----- 3. Image 2D, Scan ALL modes
'''
 #nmod=len(mode_selection)
 #ncol=4
ncol=6
nrow=np.ceil(nmod/ncol)
 #plt.figure( figsize=(5*ncol,5*nrow) )
plt.figure( figsize=(1*ncol,1*nrow) )
for imode in range(0,nmod): # sofia
    prefact = Freq12[imode]*RedM12[imode]    # sofia
    modesFRM = np.zeros(modes.shape)          # sofia
    modesFRM[imode,:] = prefact * modes[imode,:] # sofia
     #Amps = rm.RammanAmplitudes( tippos, apos, alphas, modes, imode=imode ) 
    Amps = rm.RammanAmplitudes( tippos, apos, alphas, modesFRM, imode=imode ) 
    displ = modes[imode].reshape((len(apos),3))
    #print( modes[imode] )
    #print( "displ.shape ", displ.shape, modes.shape )
    #plt.subplot(nrow,ncol,i+1)
    plt.subplot(nrow,ncol,imode+1) # sofia
    plt.imshow( Amps.reshape((ny,nx))**2, extent=extent )
    #plt.plot  ( apos[:,0],apos[:,1], '.k' )
    #plt.quiver( apos[:,0],apos[:,1], displ[:,0],displ[:,1], width=0.005, headwidth=0.005 )
    plt.plot  ( apos[:,2],apos[:,1], '.k' )
    plt.quiver( apos[:,2],apos[:,1], displ[:,2],displ[:,1], width=0.005, headwidth=0.005 )
    plt.title ( 'Mode#%i' %imode )
'''

'''

# =============== DEBUG - Plot just Polarizability (i.e. Ignore normal modes (vibration states)) 
dirnames =['x','y','z']
compnames=['xx','yy','zz',    'yz','xz','xy']
plt.figure( figsize=(3*9,3*3) )
for idir in range(3):
    for icomp in range(6):
        #Amps = rm.ProjectPolarizability( tpos1d, apos, alphas, idir=idir, icomp=icomp )
        Amps = rm.ProjectPolarizability( tippos, apos, alphas, idir=idir, icomp=icomp )
        plt.subplot(3,6,1+icomp+idir*6)
        #plt.imshow( Amps.reshape((ny,nx))**2, extent=extent )
        plt.imshow( Amps.reshape((ny,nx)), extent=extent )
        plt.title ( 'coord %i|%s  comp %i|%s' %(idir,dirnames[idir], icomp,compnames[icomp]) )
'''

#print( Amp )

#plt.plot( tpos[:,0], Amp )
plt.show()
