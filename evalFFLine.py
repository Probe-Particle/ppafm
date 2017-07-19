#!/usr/bin/python -u

import os
import numpy as np
#import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt

import pyProbeParticle                as PPU     
#import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.core           as PPC
#import pyProbeParticle.HighLevel     as PPH
import pyProbeParticle.cpp_utils as cpp_utils
from   pyProbeParticle            import basUtils

# ======== setup

lines= [
( [0.0,0.0,0.0], [1.0,2.0,3.0], 100 ),
]

# ======== main

#FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 

FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )

atoms,nDim,lvec = basUtils.loadGeometry("input.xyz", params=PPU.params)
iZs,Rs,Qs       = PPU.parseAtoms(atoms, autogeom = False, PBC = PPU.params['PBC'], FFparams=FFparams )

print "iZs", iZs; print "Rs", Rs; print "Qs", Qs

cLJs            = PPU.getAtomsLJ( PPU.params['probeType'], iZs, FFparams )
PPU.loadParams( 'params.ini',FFparams=FFparams )

plt.figure(figsize=(10,5))

for (p1,p2,nps) in lines:
    ts      = np.linspace(0.0,1.0,nps)
    ps      = np.zeros((nps,3))
    ps[:,1] = p1[0] + (p2[0]-p1[0])*ts
    ps[:,1] = p1[1] + (p2[1]-p1[1])*ts
    ps[:,1] = p1[2] + (p2[2]-p1[2])*ts
    FEs = PPC.getInPoints_LJ( ps, Rs, cLJs )
    plt.subplot(2,1,1); plt.plot( ps[:,2], FEs[:,2] )
    plt.subplot(2,1,2); plt.plot( ps[:,2], FEs[:,3] )

plt.show()
