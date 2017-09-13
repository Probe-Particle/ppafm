#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys

import numpy as np
import matplotlib.pyplot as plt   
import pyProbeParticle               as PPU
#import pyProbeParticle.GridUtils     as GU
import pyProbeParticle.basUtils      as BU
import matplotlib.pyplot as plt
from   optparse import OptionParser

parser = OptionParser()
parser.add_option( "-i",   action="store", type="string", help="input file",                   default='OutFz'        )
(options, args) = parser.parse_args()

# ====== functions

def fitExp( zs, fs, zmin, zmax ):
    fs = np.log( fs )
    f0,f1 = np.interp( [zmin,zmax], zs, fs )
    alpha = (f1-f0)/(zmax-zmin)
    A     = np.exp( f0 - alpha*zmin )
    return alpha, A

def getIsoArg( zs, fs, iso=0.01 ):
    i = np.searchsorted( -fs, -iso )
    x0 = zs[i-1]
    f0 = fs[i-1]
    dx = (zs[i] - x0 )
    df = (fs[i] - f0)
    return x0 +  dx*( iso - f0 )/df

def getMorse( r, R0=3.5, eps=0.03, alpha=-1.8,  cpull=1.0, cpush=1.0 ):
    expar = np.exp( alpha*(r - R0) )
    return eps*( expar*expar*cpush - 2*expar*cpull )

def getLJ( r, R0=3.5, eps=0.03, cpull=1.0, cpush=1.0 ):
    rmr6 = (R0/r)**6
    return eps*(rmr6*rmr6*cpush - 2*rmr6*cpull)

# ====== Main

fname, fext     = os.path.splitext( options.i ); fext = fext[1:]
atoms,nDim,lvec = BU.loadGeometry( options.i, params=PPU.params )
atoms_z = atoms[3]
atoms_e = atoms[0]
data            = np.transpose( np.genfromtxt("atom_density_zlines.dat") )

zs_bare = data[0]



zmin = 1.2
zmax = 2.2

#REAs         = PPU.getAtomsREA( iZPP, Zs, self.TypeParams, alphaFac=-self.bxMorse.value() )

FFparams = PPU.loadSpecies( fname=None )

iZs = atoms[0]

REAs     = PPU.getSampleAtomsREA( iZs, FFparams )

print REAs

ilist =  range( len(atoms[0]) )
#ilist = [0]
#ilist = [0,18,26]
for i in ilist:
    fs = data[1+i]
    zs = zs_bare - atoms_z[i]
    alpha, A = fitExp( zs, fs, zmin, zmax )

    Riso = getIsoArg( zs, fs, iso=0.017 )
    plt.axvline(Riso)
    print " elem %i a_z %f Riso %f alpha %f alpha/2 %f" %( atoms_e[i], atoms_z[i], Riso, alpha, alpha/2.0 ), REAs[i]

    REAs[i][0] = Riso
    REAs[i][2] = alpha/2.0

    '''
    zmid   = (zmin+zmax)*0.5
    fmid   = np.interp( zmid, zs, fs )
    fmorse = getMorse( zmid, R0=1.8, eps=0.03, alpha=-1.8, cpull=0.0)

    dens2Pauli = fmorse/fmid
    #A *= dens2Pauli
    print " elem %i a_z %f alpha %f alpha/2 %f A  %f d2p %f " %( atoms_e[i], atoms_z[i], alpha, alpha*0.5, A, dens2Pauli )
    '''
    
    plt.plot( zs, fs, label=("%i" %atoms_e[i])  )
    #plt.plot( zs, A*np.exp( alpha*zs ) )
    #plt.plot( zs, dens2Pauli*A*np.exp( alpha*zs ) )
    #plt.plot( zs, getMorse(zs, R0=1.8, eps=0.03, alpha=-1.8, cpull=0.0) )

plt.legend()    
plt.xlim(0.0, 5.0)
plt.ylim(1e-8, 1e+5)
plt.yscale("log")
plt.axvline( zmin, ls="--", c="k" ); plt.axvline( zmax, ls="--", c="k" )
plt.grid()


atoms = np.transpose( np.array(atoms) )
print atoms.shape, REAs.shape
data =  np.concatenate( ( atoms[:,:4], REAs ), axis=1 )
np.savetxt( "atom_REAs.xyz", data, header=("%i \n # e,xyz,REA" %len(data) ) )



'''
plt.figure()

eps   =  0.030
R0    =  3.6
alpha = -1.8

Vmorse = getMorse(zs, R0=R0, eps=eps, alpha=alpha)
VLJ    = getLJ   (zs, R0=R0, eps=eps )

plt.plot( zs, Vmorse )
plt.plot( zs, VLJ )

plt.xlim(0.0, 5.0)
plt.ylim(-eps*1.2, eps)
plt.axvline( eps, ls="--", c="k" ); plt.axhline( R0, ls="--", c="k" )
plt.grid()
'''


plt.show()