#!/usr/bin/python3

# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#import GridUtils as GU
import pyProbeParticle.GridUtils      as GU

from optparse import OptionParser


parser = OptionParser()
parser.add_option(      "--dfrange", action="store", type="float", help="Range of plotted frequency shift (df)", nargs=2)
parser.add_option( "--npy" , action="store_true" ,  help="load and save fields in npy instead of xsf"     , default=False)
(options, args) = parser.parse_args()

if options.npy:
    data_format ="npy"
else:
    data_format ="xsf"

dfs,lvec,nDim=GU.load_scal_field('df',data_format=data_format)
#print lvec
#print nDim

print(" # ============  Plot Relaxed Scan 3D ")
slices = list(range( 0, len(dfs)))
#print slices
extent=( 0.0, lvec[1][0], 0.0, lvec[2][1])

for ii,i in enumerate(slices):
    print(" plotting ", i)
    plt.figure( figsize=( 10,10 ) )
    if(options.dfrange != None):
        fmin = options.dfrange[0]
        fmax = options.dfrange[1]
        plt.imshow( dfs[i], origin='lower', interpolation='bicubic', vmin=fmin, vmax=fmax,  cmap='gray', extent=extent)
    else:
        plt.imshow( dfs[i], origin='lower', interpolation='bicubic', cmap='gray', extent=extent)

    z=float(i)*(lvec[3][2]/nDim[0])
    plt.colorbar();
    plt.xlabel(r' Tip_x $\AA$')
    plt.ylabel(r' Tip_y $\AA$')
    plt.title( r"df Tip_z = %2.2f $\AA$" %z  )
    plt.savefig( 'df_%04i.png' %i, bbox_inches='tight' )


print(" ***** ALL DONE ***** ")
