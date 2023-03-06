#!/usr/bin/python
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import sys

import __main__ as main
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")

from optparse import OptionParser

import ppafm as PPU
import ppafm.core as core
from ppafm import io

parser = OptionParser()
parser.add_option( "-s", "--sample", action="store", type="string", default="CHGCAR.xsf", help="sample 3D data-file (.xsf)")
parser.add_option( "-R", "--Rcore", action="store", type="float", default="0.7", help="width of the core density radial function")
(options, args) = parser.parse_args()

valElDict = { 6:4.0, 8:6.0}   # number of valence electrons for each atomic number,   TODO: should read form external dictionary for every atoms
Rcut = options.Rcore

atoms,nDim,lvec = io.loadGeometry( options.sample, params=PPU.params )
Rs = np.array(atoms[1:4])                     # get just positions x,y,z
#Rs += np.array( [1.0, 1.0, 1.0] )[:,None]    # Arbitrary shift of atoms - for debugging

#print "Rs\n", Rs

corners   = []   # corners of the unit cell with margins - just for debugging
inds, Rs_ = PPU.findPBCAtoms3D_cutoff( Rs, np.array(lvec[1:]), Rcut=Rcut, corners=corners )  # find periodic images of PBC images of atom of radius Rcut which touch our cell
corners=corners[0]
elems = [ atoms[0][i] for i in inds ]   # atomic number of all relevant peridic images of atoms

#print "inds: \n", inds
#print "Rs_ \n", Rs_
#print "Rs_.shape ", Rs_.shape

# --- Add Corners
#elems +=  [2]*8
#Rs_ = np.hstack( [Rs_,corners] )

io.saveGeomXSF( "imaged_CO.xsf",elems,Rs_, lvec[1:], convvec=lvec[1:], bTransposed=True )    # for debugging - mapping PBC images of atoms to the cell

#exit()

cRAs = np.array([ (-valElDict[elem],Rcut) for elem in elems ])     #   parameters of radial functions (amplitude,radius)  WARRNING - renormalized by integral inside getDensityR4spline()
#print "cRAs ",cRAs.shape, "\n",  cRAs

print(">>> Loading ... ")
rho1, lvec1, nDim1, head1 = io.loadXSF( options.sample )
#V = lvec1[1,0]*lvec1[2,1]*lvec1[3,2]
V  = np.linalg.det( lvec )
N = nDim1[0]*nDim1[1]*nDim1[2]
dV = (V/N)  # volume of one voxel
#cRAs[:,0] *= dV    # Debugging
print(" dV ", dV)
print("sum(RHO), Nelec: ",  rho1.sum(),  rho1.sum()*dV)   # check sum

#rho1[:,:,:] *= 0   # Debugging

Rs_ = Rs_.transpose().copy()
core.setFF_shape   ( rho1.shape, lvec1 )     # set grid sampling dimension and shape
core.setFF_Epointer( rho1 )                  # set pointer to array with density data (to write into)
print(">>> Projecting Core Densities ... ")
core.getDensityR4spline( Rs_, cRAs.copy() )  # Do the job ( the Projection of atoms onto grid )
print("sum(RHO), Nelec: ",  rho1.sum(),  rho1.sum()*dV)   # check sum

print(">>> Saving ... ")
io.saveXSF( "rho_subCoreChg.xsf", rho1, lvec1, head=head1 )
