#!/usr/bin/python
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
from optparse import OptionParser

import ppafm.GridUtils as GU

parser = OptionParser()
parser.add_option( "-p",   action="store", type="string", help="pixels (ix,iy) to take curve", default='quad_points.ini' )
parser.add_option( "-i",   action="store", type="string", help="input file",                   default='OutFz'        )
parser.add_option("-f","--data_format" , action="store" , type="string", help="Specify the output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
(options, args) = parser.parse_args()

try:
	points = np.genfromtxt( options.p )
	print("quad in points", points)
except:
	print(options.p+" not found => exiting ...")
	sys.exit()

#izs = range(100,300,10)
izs  = list(range(75,125,5))
print(izs)
dc = 1.0/len(izs)
cmap  = plt.get_cmap('jet_r')

isz = (200,130)

#print "DEBUG 1 "
F,lvec,nDim=GU.load_scal_field(options.i,data_format=options.data_format)
GU.setGridN( np.array( nDim, dtype='int32' ) )
print("nDim ", nDim)
Fquad = GU.interpolateQuad( F, points[0], points[1], points[2], points[3], sz=isz )  # grid coord
#vmax = 0.01
#plt.imshow(Fquad, origin='upper', vmin=-vmax, vmax=vmax, extent=extent)
plt.imshow(Fquad, origin='upper')
plt.axis('equal')

for i,iz in enumerate(izs):
    color = cmap(i*dc)
    #z = iz*extent[3]/200.0
    plt.axhline( iz, c=color )
    #plt.plot( [extent[0],extent[1]], [iz,iz], c=color)

plt.savefig("legend.png", bbox_inches='tight')

plt.figure()

xs = np.linspace(0,1,isz[1] )
#xs = np.linspace(extent[0],extent[1],130)
for i,iz in enumerate(izs):
    color = cmap(i*dc)
    #z = iz*extent[3]/2
    plt.plot( xs,Fquad[iz,:], c=color, label=("iz=%i" %iz) )
plt.legend()

plt.savefig("Linescans.png", bbox_inches='tight')

np.savetxt( "Fquad.dat", Fquad )




plt.show()
