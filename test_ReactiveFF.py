#!/usr/bin/python

import sys
import os
import numpy as np
import time

#import pyMolecular.ReactiveFF  as rff
#import pyMolecular.atomicUtils as au

import pyProbeParticle.ReactiveFF  as rff
import pyProbeParticle.atomicUtils as au

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import common    as PPU

if __name__ == "__main__":
    print " ================ START "
    print " ================ START "

    #os.chdir( "/u/25/prokoph1/unix/git/SimpleSimulationEngine/cpp/Build/apps/MolecularEditor2" )

    c6    = -15.0
    R2vdW = 8.0
    rff.insertAtomType( 3, 1, 0.65,  1.0, -0.7, c6, R2vdW, 0.2 )
    rff.insertAtomType( 4, 2, 0.8 , 1.0, -0.7, c6, R2vdW, 0.0 )


    '''
    natom = 2
    rff.ralloc(natom)
    types  = rff.getTypes(natom)
    poss   = rff.getPoss(natom)
    qrots  = rff.getQrots(natom)
    hbonds = rff.getHbonds(natom)
    ebonds = rff.getEbonds(natom)
    itypes  = np.zeros(natom).astype(np.int32); print "itypes", itypes
    rff.setTypes( natom, itypes )
    rff.setSurf(K=1.0, x0=0.0, h=np.array([0.0,0.0,1.0]) )

    poss [:,:]  = np.array( [[-1.0,0.0,0.0],[1.0,0.0,0.0]] )
    qrots[:,:]  = np.random.rand(natom,4)-0.5
    rs          = np.sum(qrots**2, axis=1 )
    qrots      /= rs[:,None]
    '''

    natom = 20
    rff.ralloc(natom)
    types  = rff.getTypes(natom)
    poss   = rff.getPoss(natom)
    qrots  = rff.getQrots(natom)
    hbonds = rff.getHbonds(natom)
    ebonds = rff.getEbonds(natom)
    caps   = rff.getBondCaps(natom)
    #itypes  = np.random.randint( 2, size=natom, dtype=np.int32 ); print "itypes", itypes
    itypes  = (np.random.rand( natom )*1.3 ).astype(np.int32); print "itypes", itypes
    rff.setTypes( natom, itypes )
    poss [:,:]  = ( np.random.rand(natom,3)-0.5 ) * 10.0
    poss [:,2]  = 0.15
    qrots[:,:]  = np.random.rand(natom,4)-0.5
    rs          = np.sum(qrots**2, axis=1 )
    qrots      /= rs[:,None]

    rff.setBox( p0=np.array([-5.0,-5.0,-1.0]), p1=np.array([5.0,5.0,1.0]), K=-1.0, fmax=1.0  )
    rff.setSurf(K=-0.2, x0=0.0, h=np.array([0.0,0.0,1.0]) )

    #rff.relaxNsteps( nsteps=2000, F2conf=0.0, dt=0.05, damp=0.9 )

    '''
    fout = open( "rff_movie.xyz",'w')
    for itr in range(50):
        F2 = rff.relaxNsteps( nsteps=50, F2conf=0.0, dt=0.15, damp=0.9 )
        print ">> itr ", itr," F2 ", F2
        #au.writeToXYZ( fout, itypes, poss  )
        xyzs, itypes_ = rff.h2bonds( itypes, poss, hbonds, bsc=1.1 )
        #print "itypes_,xyzs shapes : ", itypes_.shape,xyzs.shape
        xyzs, itypes_ = rff.removeSaturatedBonds(ebonds, itypes_, xyzs, Ecut=-0.1 )
        #print ebonds
        au.writeToXYZ( fout, itypes_, xyzs  )
    fout.close()
    t2 = time.clock();
    '''
    
    t1 = time.clock();
    fout = open( "rff_movie.xyz",'w')
    for itr in range(10):
        F2 = rff.relaxNsteps( nsteps=50, F2conf=0.0, dt=0.15, damp=0.9 )
        print ">> itr ", itr," F2 ", F2 #, caps
        xyzs, itypes_ = rff.h2bonds( itypes, poss, hbonds, bsc=1.1 )
        xyzs, itypes_ = rff.removeSaturatedBonds(caps, itypes_, xyzs )
        au.writeToXYZ( fout, itypes_, xyzs  )
    rff.passivateBonds( -0.1 );
    print "passivation ", caps
    for itr in range(30):
        F2 = rff.relaxNsteps( nsteps=50, F2conf=0.0, dt=0.05, damp=0.9 )
        print ">> itr ", itr," F2 ", F2 #, caps
        xyzs, itypes_ = rff.h2bonds( itypes, poss, hbonds, bsc=1.1 )
        xyzs, itypes_ = rff.removeSaturatedBonds(caps, itypes_, xyzs )
        au.writeToXYZ( fout, itypes_, xyzs  )
    fout.close()
    t2 = time.clock();
    print "Relaxation time ", t2-t1

    #print ebonds
    au.saveXYZ( itypes+5, poss, "rff_skelet.xyz" )

    #print "rots_", rots_
    print ">>>> ALL DONE <<<<"
