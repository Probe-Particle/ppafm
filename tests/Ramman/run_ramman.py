#!/usr/bin/python3

import numpy as np
import sys
sys.path.append("../../")
from pyProbeParticle import Ramman as rm
import matplotlib.pyplot as plt

nps       = 100
tpos      = np.zeros( (nps,3) )
tpos[:,0] = np.linspace(-10.0,10.0,nps)   #  ;print(tpos)

Amp = rm.RunRaman( tpos, wdir='./', imode=2 )

#print( Amp )

plt.plot( tpos[:,0], Amp )
plt.show()