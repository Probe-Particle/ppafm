#!/usr/bin/python -u

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
#import pyProbeParticle               as PPU 
import pyProbeParticle.core           as PPC

S = np.genfromtxt('TipRSpline.ini')
print("TipRSpline.ini overrides harmonic tip")
xs   = S[:,0].copy();  print("xs: ",   xs)
ydys = S[:,1:].copy(); print("ydys: ", ydys)
plt.plot(xs, ydys[:,0]          , 'o' )
plt.plot(xs, ydys[:,0]+ydys[:,1], '.' )
PPC.setTipSpline( xs, ydys )

PPC.setTip( )

fs = np.zeros((60,3))
r0 = np.array([ 0.0,0.0,0.5])
dr = np.array([ 0.0,0.0,0.1])
R  = np.array([ 0.0,0.0,0.0])
xs = np.array( list(range(len(fs))) )*dr[2] + r0[2]
#print "xs=",xs

print(">>>  PPC.test_force( 1, r0, dr, R, fs )")
PPC.test_force( 1, r0, dr, R, fs )
plt.plot(xs, fs[:,2] )

print(">>>  PPC.test_force( 2, r0, dr, R, fs )")
PPC.test_force( 2, r0, dr, R, fs )
plt.plot(xs, fs[:,2] )

#print "fs:", fs

plt.grid()
plt.show()



