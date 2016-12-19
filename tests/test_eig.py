#!/usr/bin/python

import sys

sys.path.append('../')
#sys.path.append( "/home/prokop/git/ProbeParticleModel" )

import numpy as np
import pyProbeParticle          as PPU     
import pyProbeParticle.core     as PPC

mat = np.array([  
[1.0,0.6, 0.9],
[0.6,0.5, -0.2],
[0.9,-0.2,-0.1],
])


evas, evcs = np.linalg.eig( mat )

print evas
print evcs

evas = PPC.test_eigen3x3(mat)

print evas
