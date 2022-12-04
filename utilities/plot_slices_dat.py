#!/usr/bin/python

import sys
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
import ppafm.file_dat    as file_dat

path = "./"

# ======== load from .dat

fnames   = glob.glob( path+'*.dat' )
fnames.sort()
print(fnames)
data = []
for fname in fnames:
    fname_ = os.path.basename(fname);
    #fnames.append( fname_ )
    print(fname)
    imgs = file_dat.readDat(fname)
    data.append( imgs[1] )

np.save("data.npy", data)
slices = data

# ======== load from NPY and plot

#print slices.shape

n=len(slices)
plt.figure( figsize=(5*n,5) )

#slices = np.load("data.npy")

for isl,sl in enumerate(slices):
    print(isl+1)
    #plt.figure()
    plt.subplot(1,n,isl+1)
    plt.imshow(sl)

plt.savefig( "data.png", bbox_inches='tight' )
#plt.show()
