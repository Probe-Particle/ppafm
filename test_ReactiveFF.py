#!/usr/bin/python

import sys
import os
import numpy as np
import time

import pyProbeParticle.ReactiveFF  as rff
import pyProbeParticle.atomicUtils as au

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import common    as PPU
#from optparse import OptionParser

N1 = int(sys.argv[1])
N2 = int(sys.argv[2])
predir = "gen_"


if __name__ == "__main__":
    print " ================ START "
    print " ================ START "

    # --- prepare atom-types 
    c6    = -15.0
    R2vdW = 8.0
    rff.insertAtomType( 3, 1, 0.65,  1.0, -0.7, c6, R2vdW, 0.2 )
    rff.insertAtomType( 4, 2, 0.8 ,  1.0, -0.7, c6, R2vdW, 0.0 )

    # --- prepare system (resize buffers)
    natom = 20
    rff.ralloc(natom)
    types  = rff.getTypes(natom)
    poss   = rff.getPoss(natom)
    qrots  = rff.getQrots(natom)
    hbonds = rff.getHbonds(natom)
    ebonds = rff.getEbonds(natom)
    caps   = rff.getBondCaps(natom)

    # --- generate set of molecules
    for i in range(N1,N2):
        dname = predir+("%03i" %i )
        print dname
        os.makedirs( dname )

        # --- distribute carbond atoms randomly
        rff.clean()
        # -- set random atom types
        itypes  = (np.random.rand( natom )*1.3 ).astype(np.int32); # print "itypes", self.rff_itypes
        rff.setTypes( natom, itypes )
        # -- set random atom  positions
        poss [:,:]  = ( np.random.rand(natom,3)-0.5 ) * 10.0
        poss [:,2] *= 0.15
        # -- set random atom rotation
        qrots[:,:]  = np.random.rand( natom,4)-0.5
        rs          = np.sum( qrots**2, axis=1 )
        qrots      /= rs[:,None]
        # --- setup environment boundary ( box, surface )
        rff.setBox( p0=np.array([-5.0,-5.0,-1.0]), p1=np.array([5.0,5.0,1.0]), K=-1.0, fmax=1.0  )
        rff.setSurf(K=-0.2, x0=0.0, h=np.array([0.0,0.0,1.0]) )

        # --- relax molecule
        t1 = time.clock();
        #if (rff_debug_xyz): fout = open( "rff_movie.xyz",'w')
        for itr in range(10):
            F2 = rff.relaxNsteps( nsteps=50, F2conf=0.0, dt=0.15, damp=0.9 )
            #print ">> itr ", itr," F2 ", F2 #, self.rff_caps
            #if (rff_debug_xyz):
            #    xyzs, itypes_ = rff.h2bonds( self.rff_itypes, self.rff_poss, self.rff_hbonds, bsc=1.1 )
            #    xyzs, itypes_ = rff.removeSaturatedBonds(self.rff_caps, self.rff_itypes_, xyzs )
            #    au.writeToXYZ( fout, itypes_, xyzs  )
        rff.passivateBonds( -0.1 );
        #print "passivation ", self.rff_caps
        for itr in range(30):
            F2 = rff.relaxNsteps( nsteps=50, F2conf=0.0, dt=0.05, damp=0.9 )
            #print ">> itr ", itr," F2 ", F2 #, self.rff_caps
            #if (rff_debug_xyz):
            #    xyzs, itypes_ = rff.h2bonds( self.rff_itypes, self.rff_poss, self.rff_hbonds, bsc=1.1 )
            #    xyzs, itypes_ = rff.removeSaturatedBonds(self.rff_caps, itypes_, xyzs )
            #    au.writeToXYZ( fout, itypes_, xyzs  )
        #if (rff_debug_xyz): fout.close()
        t2 = time.clock();
        print " molecule gen time ", t2-t1 

        # ---- store result
        xyzs, itypes_ = rff.h2bonds( itypes, poss, hbonds, bsc=1.1 ) 
        xyzs, itypes_ = rff.removeSaturatedBonds(caps, itypes_, xyzs )
        au.saveXYZ( itypes_, xyzs, dname+"/pos.xyz" )







