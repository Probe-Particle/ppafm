#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases


# https://stackoverflow.com/questions/22508593/numpy-polyfit-or-any-fitting-to-x-and-y-multidimensional-arrays


import sys
import os
import time
import random
import matplotlib;
import numpy as np

#import matplotlib as mpl;  mpl.use('Agg'); print "plot WITHOUT Xserver";
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm


# ========== setup

npoly = 5

# ========== function

def evalPoly(xs, coefs):
    ys  = np.zeros( (coefs.shape[1], len(xs)) )
    xns = ys + 1 
    for c in coefs:
        ys += xns*c[:,None]
        xns *= xs[None,:]
    return ys

# ========== main

#Fdata = np.load("pos.xyz_Fout_z.npy")
Fdata = np.load("Fout_z.npy")
sh = Fdata.shape
print("Fdata.shape ", Fdata.shape)

iz0 = 5

ps = np.array([
#[20,20],
#[40,40],
#[50,50],
#[55,50],
#[50,65],
#[55,40],
#[60,60],

[10,10],
[55,50],
[55,55],
[47,65],
[46,36],
[64,46],


])

xs = np.linspace(0.0,1.0,sh[2])
ys = Fdata.reshape( -1,sh[2])
coefs = np.polynomial.polynomial.polyfit(xs, ys.T, npoly )
print("coefs.shape", coefs.shape)
ys_ = evalPoly(xs, coefs)

print("ys .shape ", ys .shape)
print("ys_.shape ", ys_.shape)

Fdata_ = ys_.reshape(sh[0],sh[1],-1)


plt.figure()
for i,p in enumerate(ps):
    c = cm.rainbow( i/float(len(ps)) )
    plt.plot( xs, Fdata [p[1],p[0],:], c=c, ls='-' )
    plt.plot( xs, Fdata_[p[1],p[0],:], c=c, ls='--' )
    Fdata [p[1],p[0],:] = np.NaN
    Fdata_[p[1],p[0],:] = np.NaN
plt.savefig( "polyfit_curves.png", bbox_inchjes = 'tight' )

#plt.show()

#for iz in [0,4,8,12,16]:
for iz in range(sh[2]):
    print("plot slice", iz)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow( Fdata [:,:,iz] ); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow( Fdata_[:,:,iz] ); plt.colorbar()
    plt.savefig( "polyfit_%03i.png" %iz, bbox_inchjes = 'tight' )


'''
plt.imshow( Fdata[:,:,iz0] )
plt.figure()

#colors = cm.rainbow(np.linspace(0,1,n))
for i,p in enumerate(ps):
    ys = Fdata[p[0],p[1],:]
    coefs = np.polynomial.polynomial.polyfit(xs, ys, 6)
    c = cm.rainbow( i/float(len(ps)) )
    plt.plot( xs, ys, c=c, ls='-' )
    plt.plot( xs, evalPoly(xs, coefs), c=c, ls='--' )
'''

#plt.show()



