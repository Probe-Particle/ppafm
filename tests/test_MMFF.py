#!/usr/bin/python

import sys
import os
import numpy as np
import time

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import common    as PPU
#from optparse import OptionParser


sys.path.append("../")

#import pyProbeParticle.PolyCycles  as pcff
import pyProbeParticle.atomicUtils as au
#import pyProbeParticle.chemistry   as ch

import pyProbeParticle.MMFF   as mmff
#include "DynamicOpt.h"

import matplotlib.pyplot as plt


if __name__ == "__main__":

    apos = np.array([
        [-2.0,0.0,0.0],  # 0
        [-1.0,2.0,0.0],  # 1
        [+1.0,2.0,0.0],  # 2
        [+2.0,0.0,1.0],  # 3
    ])
    aconf = np.array([
        [0,1],  # 0
        [0,1],  # 1
        [1,0],  # 2
        [0,0],  # 3
    ],dtype=np.int32)
    bonds2atom = np.array([
        [0,1],  # 0
        [1,2],  # 1
        [2,3],  # 2
    ],dtype=np.int32)
    '''
    angles2bonds = np.array([
        {0,1},  # 0
        {1,2},  # 1
    ])
    dihedrals2bond = np.array([
        {0,1,2},  # 0
    ])
    a2b = np.array([
        2.0, # 0
        2.0, # 1
        2.0, # 2
    ])
    '''

    t1=time.clock()
    mmff.addAtoms(apos, aconf )                    #;print "DEBUG 1"
    mmff.addBonds(bonds2atom, l0s=None, ks=None)   #;print "DEBUG 2"
    natom = mmff.buildFF(True,True,True,True)           #;print "DEBUG 4"
    #mmff.setNonBonded(None)                        #;print "DEBUG 3"   # this activate NBFF with default atom params
    mmff.setupOpt()                                #;print "DEBUG 5"
    pos = mmff.getPos(natom)
    print(" Time to build molecule [s] ", time.clock()-t1)
    
    natom0 =  len(apos)
    
    #            0    1    2
    ne2elem = [ 'C', 'N', 'O' ]
    elems   = [   ne2elem[ne] for ne in aconf[:,1] ]
    
    types = mmff.getAtomTypes(natom)            ;print(types)
    #              0!  -1!   -2=pi   -3=e   -4=H
    type2elem  = [ 'U', 'U'  , 'He',  'Ne', 'H'  ]
    elems     +=  [  type2elem[-t] for t in types[natom0:] ]  ;print(elems) 
    
    au.saveXYZ( elems, pos, "test_MMFF_start.xyz", qs=None )
    
    #exit()
    
    errs = []
    fout = open("test_MMFF_movie.xyz", "w")
    for i in range(1000):
        print(i," ", end=' ')
        f = mmff.relaxNsteps(1)
        errs.append( f )
        au.writeToXYZ( fout, elems, pos )
        if(f<1e-6): break
    fout.close()
    
    plt.plot(errs); plt.yscale('log'); plt.xlim(0,1000); plt.ylim(1e-6,1e+2);
    plt.show()
    

    '''
    t1=time.clock()
    mmff.relaxNsteps(1000)
    print " Time to relax molecule [s] ", time.clock()-t1
    au.saveXYZ( elems, pos, "test_MMFF.xyz", qs=None )
    '''
    
    print("pos ", pos)



