#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys

import numpy as np
import matplotlib.pyplot as plt   
import pyProbeParticle               as PPU
#import pyProbeParticle.GridUtils     as GU
import pyProbeParticle.basUtils      as BU
from   optparse import OptionParser

parser = OptionParser()
parser.add_option( "-i",           action="store", type="string", help="input file",                              default= 'CHGCAR'        )
parser.add_option( "-o",           action="store", type="string", help="output xyz file name",                    default= 'new_xyz.xyz'   )
parser.add_option( "-z", "--zcut", action="store", type="float",  help="cut-out atoms bellow this height",        default= -100.           )
parser.add_option( "--height",     action="store", type="float",  help="how far above atom is isosurface fitted", default= +5.0            )
parser.add_option( "--old",        action="store_true",           help="use old version of params.ini",           default= False	   )
parser.add_option( "--plot",       action="store_false"        ,  help="not-plot ?",                              default= True            )
parser.add_option( "--debug",      action="store_true"         ,  help="plot and pr. all lines and data from fit",default= False           )

(options, args) = parser.parse_args()

# ====== functions

def fitExp( zs, fs, zmin, zmax ):
    fs = np.log( fs )
    f0,f1 = np.interp( [zmin,zmax], zs, fs )
    alpha = (f1-f0)/(zmax-zmin)
    A     = np.exp( f0 - alpha*zmin )
    return alpha, A

def getIsoArg( zs_, fs, iso=0.01, atom_z=0.0 ):
    zi = int((atom_z -zs_[0]) // (zs_[1]-zs_[0]))
    ai = int((atom_z+options.height +0.1 - zs_[0]) // (zs_[1]-zs_[0]))
    zs = zs_ - atom_z
    i = np.searchsorted( -fs[zi:ai], -iso )
    #i = np.searchsorted( fs[:160], iso , side='right') # Not working either way
    x0 = zs[zi+i-1]
    f0 = fs[zi+i-1]
    dx = (zs[zi+i] - x0 )
    df = (fs[zi+i] - f0)
    if options.debug or x0 +  dx*( iso - f0 )/df >= 5.0 or x0 +  dx*( iso - f0 )/df <= 1.0:
        print "atom_z", atom_z
        print "zs_[0]", zs_[0]
        print "(atom_z -zs_[0])", (zs_[1]-zs_[0])
        print "(zs_[1]-zs_[0])", (zs_[1]-zs_[0])
        print(x0, f0, dx, df, i, zi, ai)
        plt.plot(zs_[zi:ai],fs[zi:ai],[zs_[zi],zs_[ai]],[iso,iso])
        plt.show()
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

data            = np.transpose( np.genfromtxt("atom_density_zlines.dat") )

zs_bare = data[0]

zmin = 1.2
zmax = 2.2

#REAs         = PPU.getAtomsREA( iZPP, Zs, self.TypeParams, alphaFac=-self.bxMorse.value() )

FFparams = PPU.loadSpecies( fname=None )

# remove atoms lower than zcut:
mask = np.array(atoms[3]) >= float(options.zcut) ;

for i in range(len(atoms)):
    atoms[i] = np.array(atoms[i])[mask];

atoms_z = atoms[3]
atoms_e = atoms[0]
iZs = atoms[0]

REAs     = PPU.getSampleAtomsREA( iZs, FFparams )

print "REAs:"
print REAs
print

mask = np.append([True], mask)

data = data[mask]

del mask;

if options.plot:
    import matplotlib.pyplot as plt

# f1 . pseudo-xyz file with all atoms about z-cut
ilist =  range( len(atoms[0]) )
f1 = open(options.o,"w")
f1.write(str(len(atoms[0])) + '\n')
f1.write('\n')
# f2 - atomtypes.ini file with Riso (and Alpha) coresponding to each atom
f2 = open("atomtypes.ini","w")
for i in ilist:
    fs = data[1+i]
    zs = zs_bare #- atoms_z[i]
    alpha, A = fitExp( zs - atoms_z[i] , fs, zmin, zmax )

    Riso = getIsoArg( zs, fs, iso=0.017, atom_z=atoms_z[i] )
    if not (0.5 < Riso < 5.0) :
        print "!!! Problem with Riso for atom no. %i : Riso %f, we will use tabled number." %(i,Riso)
        Riso = REAs[i][0]
    f1.write(str(i+1)+' '+str(atoms[1][i])+' '+str(atoms[2][i])+' '+str(atoms[3][i])+'\n')
    if options.old: # old verison of atomtypes.inis
        f2.write(str(Riso)+' '+str(REAs[i][1])+' '+str(i+1)+' '+FFparams[iZs[i]-1][4]+str(i)+'\n')
    else: # ocl version of params.ini
        f2.write(str(Riso)+' '+str(REAs[i][1])+' '+str(alpha/2)+' '+str(i+1)+' '+FFparams[iZs[i]-1][4]+str(i)+'\n')
    #plt.axvline(Riso)
    #print " elem %i a_z %f Riso %f alpha %f alpha/2 %f" %( atoms_e[i], atoms_z[i], Riso, alpha, alpha/2.0 ), REAs[i]
    print " elem %i a_z %f Riso %f " %( atoms_e[i], atoms_z[i], Riso ), REAs[i]

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
    
    if options.plot:
         plt.plot( zs - atoms_z[i], fs, label=("%i" %atoms_e[i])  )
         #plt.plot( zs, A*np.exp( alpha*zs ) )
         #plt.plot( zs, dens2Pauli*A*np.exp( alpha*zs ) )
         #plt.plot( zs, getMorse(zs, R0=1.8, eps=0.03, alpha=-1.8, cpull=0.0) )

f1.close()
f2.close()

if options.plot:
    plt.plot( [0.0,5.0],[0.017,0.017], label=("threshold") )
    plt.legend()    
    plt.xlim(0.0, options.height)
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

if options.plot:
    plt.show()
