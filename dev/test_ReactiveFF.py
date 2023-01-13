#!/usr/bin/python

import os
import sys
import time

import numpy as np

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
import ppafm.atomicUtils as au
import ppafm.ReactiveFF as rff
from ppafm import basUtils

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import common    as PPU
#from optparse import OptionParser

N1 = int(sys.argv[1])
N2 = int(sys.argv[2])
predir = "gen_"


#                   H        He       Li      Be     B       C       N         O       F
taffins = np.array([-4.5280,-4.5280, -3.006,-3.006, -5.343, -5.343, -6.899, -8.741, -10.874] )
thards  = np.array([13.8904, 13.8904, 4.772,  4.772, 10.126, 10.126, 11.760, 13.364,  14.948] )


if __name__ == "__main__":
    print(" ================ START ")
    print(" ================ START ")

    # --- prepare atom-types
    c6    = -15.0
    R2vdW = 8.0
    rff.insertAtomType( 3, 1, 0.65,  1.0, -0.7, c6, R2vdW, 0.2 )
    rff.insertAtomType( 4, 2, 0.8 ,  1.0, -0.7, c6, R2vdW, 0.0 )


    # --- prepare system (resize buffers)
    natom = 20
    rff.reallocFF(natom)
    types  = rff.getTypes(natom)
    poss   = rff.getPoss(natom)
    qrots  = rff.getQrots(natom)
    hbonds = rff.getHbonds(natom)
    ebonds = rff.getEbonds(natom)
    caps   = rff.getBondCaps(natom)

    # --- generate set of molecules
    for i in range(N1,N2):
        dname = predir+("%03i" %i )
        print(dname)
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
        print(" molecule gen time ", t2-t1)


        # ---- store result
        xyzs, itypes_ = rff.h2bonds( itypes, poss, hbonds, bsc=1.1 )
        xyzs, itypes_ = rff.removeSaturatedBonds(caps, itypes_, xyzs )

        # --- charge equlibraion
        #print itypes_
        #print xyzs
        #itypes_[0] = 7  # test other elements
        #itypes_[1] = 8
        rff.setupChargePos( xyzs, itypes_, taffins, thards )
        n = len(itypes_)
        #print "Affinitis ", rff.getChargeAffinitis(n)
        #print "Hardness  ", rff.getChargeHardness(n)
        #print "J ",         rff.getChargeJ(n)
        rff.setTotalCharge(0.0);
        rff.relaxCharge( nsteps=100, F2conf=0.0, dt=0.05, damp=0.9 )
        qs = rff.getChargeQs(n) * -1
        print("Qs        ", qs)

        #itypes_[itypes_==5] = 6 # replace borons by carbons
        basUtils.saveXYZ(dname+"/pos.xyz", xyzs, itypes_, qs=qs )

        import matplotlib.pyplot as plt
        plt.scatter(xyzs[:,0], xyzs[:,1], c=qs, alpha=1.0, vmin=-0.5,vmax=0.5, cmap='bwr')
        plt.colorbar()
        plt.show()
