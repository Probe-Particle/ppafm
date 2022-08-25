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

# -- 2) Tip position as a single point, for simulate single spectra - - tpos xyz vectr (Angstrom)
tippoint = np.array( [np.genfromtxt(wdir+fname_tippos).transpose().copy(),] ) # sofia , read after to not rewrite
print ( "  " )  # sofia
print ( " tip placed in [A] : " , tippoint)  # sofia

# ----- 2. Spectra simulation with given tip position
with open("out_Amps2.dat",'wb') as f_out1:           # sofia
    for imode in range(0,nmod):                  # sofia 
       prefact = Freq12[imode]*RedM12[imode]    # sofia
       modesFRM = np.zeros(modes.shape)          # sofia
       modesFRM[imode,:] = prefact * modes[imode,:] # sofia
       Amps = rm.RammanAmplitudes( tippoint, apos, alphas, modesFRM, imode=imode ) #sofia
       np.savetxt(f_out1, Amps**2, fmt='%.6f') # sofia

#print( Amp )

#plt.plot( tpos[:,0], Amp )
plt.show()
