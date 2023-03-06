#!/usr/bin/python

# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl;  mpl.use('Agg'); print("plot WITHOUT Xserver"); # this makes it run without Xserver (e.g. on supercomputer) # see http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server

#--- added later just to plot atoms
sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
from optparse import OptionParser

import ppafm as PPU
import ppafm.cpp_utils as cpp_utils
import ppafm.HighLevel as PPH
import ppafm.PPPlot as PPPlot
from ppafm import elements, io
from ppafm.atomicUtils import findBonds, getAtomColors

parser = OptionParser()
parser.add_option( "-i", action="store", type="string", help="input file name", default='df' )
parser.add_option( "--cmap", action="store", type="string", help="input file name", default='gray' )
parser.add_option( "--izs", action="store", type="int", help="Range of plotted z slice indexes", nargs=3)
parser.add_option( "--vrange", action="store", type="float", help="Range of values", nargs=2)
parser.add_option( "--npy" , action="store_true" ,  help="load and save fields in npy instead of xsf"     , default=False)

parser.add_option( "--atoms",    action="store", type="string", help="xyz geometry file", default='input_plot.xyz' )
parser.add_option( "--bonds",    action="store_true", default=False, help="plot bonds to images" )

(options, args) = parser.parse_args()


atoms = None
bonds = None
if options.atoms:
    xyzs, Zs, qs, _ = io.loadXYZ(options.atoms)
    atoms = [list(Zs), list(xyzs[:, 0]), list(xyzs[:, 1]), list(xyzs[:, 2]), list(qs)]
    if os.path.isfile( 'atomtypes.ini' ):
        print(">> LOADING LOCAL atomtypes.ini")
        FFparams=PPU.loadSpecies( 'atomtypes.ini' )
    else:
        FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
    iZs,Rs,Qstmp=PPH.parseAtoms(atoms, autogeom = False, PBC = True, FFparams=FFparams )
    atom_colors = getAtomColors(iZs,FFparams=FFparams)
    #print atom_colors
    Rs=Rs.transpose().copy()
    atoms= [iZs,Rs[0],Rs[1],Rs[2],atom_colors]
    #print "atom_colors: ", atom_colors
    if options.bonds:
        bonds = findBonds(atoms,iZs,1.0,FFparams=FFparams)
        #print "bonds ", bonds
atomSize = 0.15


if options.npy:
    data_format = "npy"
else:
    data_format = "xsf"

data,lvec,nDim=io.load_scal_field( options.i ,data_format=data_format)
#print lvec
#print nDim

print(" # ============  Plot Relaxed Scan 3D ")
if options.izs:
    slices = list(range( options.izs[0], options.izs[1], options.izs[2]))
else:
    slices = list(range( 0, len(data)))
#print slices
extent=( 0.0, lvec[1][0], 0.0, lvec[2][1])

for ii,i in enumerate(slices):
    print(" plotting ", i)
    plt.figure( figsize=( 10,10 ) )
    if options.vrange:
        plt.imshow( data[i], origin='upper', interpolation='bicubic', cmap=options.cmap, extent=extent, vmin=options.vrange[0], vmax=options.vrange[1])
    else:
        plt.imshow( data[i], origin='upper', interpolation='bicubic', cmap=options.cmap, extent=extent)
    PPPlot.plotGeom( atoms, bonds, atomSize=atomSize )
    z=float(i)*(lvec[3][2]/nDim[0])
    plt.colorbar();
    plt.xlabel(r' Tip_x $\AA$')
    plt.ylabel(r' Tip_y $\AA$')
    plt.title( r" Tip_z = %2.2f $\AA$" %z  )
    plt.savefig( options.i+'_%04i.png' %i, bbox_inches='tight' )


print(" ***** ALL DONE ***** ")
