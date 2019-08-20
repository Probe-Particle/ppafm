#!/usr/bin/python

import sys
import os
import numpy as np
import time

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import common    as PPU
#from optparse import OptionParser

#import pyProbeParticle.PolyCycles  as pcff
#import pyProbeParticle.atomicUtils as au
#import pyProbeParticle.chemistry   as ch

import pyProbeParticle.MMFF   as mmff


import matplotlib.pyplot as plt


if __name__ == "__main__":

    apos = np.array([
        [-2.0,0.0,0.0],  # 0
        [-1.0,2.0,0.0],  # 1
        [+1.0,2.0,0.0],  # 2
        [+2.0,0.0,1.0],  # 3
    ])
    aconf = np.array([
        [0,0],  # 0
        [0,0],  # 1
        [0,0],  # 2
        [0,0],  # 3
    ])
    bonds2atom = np.array([
        [0,1],  # 0
        [1,2],  # 1
        [2,3],  # 2
    ])
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

    mmff.addAtoms(apos, aconf )                    ;print "DEBUG 1"
    mmff.addBonds(bonds2atom, l0s=None, ks=None)   ;print "DEBUG 2"
    mmff.setNonBonded(None)                        ;print "DEBUG 3"   # this activate NBFF with default atom params
    mmff.buildSystem(True,True,True)               ;print "DEBUG 4"
    mmff.setupOpt()                                ;print "DEBUG 5"
    mmff.relaxNsteps(100)                          ;print "DEBUG 6"







