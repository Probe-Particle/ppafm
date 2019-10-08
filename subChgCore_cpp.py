#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys
import __main__ as main
import numpy as np
import matplotlib.pyplot as plt
#import GridUtils as GU
sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")

import pyProbeParticle                as PPU
import pyProbeParticle.core           as core
import pyProbeParticle.GridUtils      as GU
#import pyProbeParticle.fieldFFT       as fFFT
import pyProbeParticle.basUtils  as BU
from optparse import OptionParser

parser = OptionParser()
parser.add_option( "-s", "--sample", action="store", type="string", default="CHGCAR.xsf", help="sample 3D data-file (.xsf)")
(options, args) = parser.parse_args()

#pixOff    = np.array([-0.5,-0.5,-0.5])
pixOff    = np.array([0.0,0.0,0.0])
valElDict = { 6:4.0, 8:6.0}

atoms,nDim,lvec     = BU.loadGeometry( options.sample, params=PPU.params )
xyzs = np.array(atoms[1:4]).transpose().copy()
#xyzs = xyzs_[1:3].transpose()

#def PBCAtoms( Zs, Rs, Qs, avec, bvec, na=None, nb=None )

Zs=np.array(atoms[0])
Rs=np.array(atoms[1:4]).transpose()
Qs=np.array(atoms[4])

Zs_, xyzqs_, cLJs_ = PPU.PBCAtoms3D_np( Zs, Rs, Qs, None, np.array(lvec), npbc=[1,1,1] )

xyzs = xyzqs_[:,0:3].astype(np.float64)
#xyzs = xyzqs_[:,0:3].copy()

print "Zs_: \n", Zs_
sigma  = 1.0
renorm = -1./( sigma * np.sqrt(2*np.pi) )**3

#renorm *= 1.14780494243
#renorm *= 4*np.pi
#renorm = -0.80375499268 #/(sigma**3)
cRAs = np.array([ (valElDict[elem]*renorm,sigma) for elem in Zs_ ])
print "cRAs ",cRAs.shape, "\n",  cRAs


#print atoms
print "xyzs.shape ", xyzs.shape
print "xyzs.dtype", xyzs.dtype
print "xyzs: \n", xyzs

rho1, lvec1, nDim1, head1 = GU.loadXSF( options.sample )
V = lvec1[1,0]*lvec1[2,1]*lvec1[3,2]
N = nDim1[0]*nDim1[1]*nDim1[2]
dV = (V/N)

print " dV ", dV



print "sum(RHO), Nelec",  rho1.sum(),  rho1.sum()*dV

core.setFF_shape   ( rho1.shape, lvec1 )
core.setFF_Epointer( rho1 )
core.getGaussDensity( xyzs, cRAs )

print "sum(RHO), Nelec",  rho1.sum(),  rho1.sum()*dV

#( Rs, Qs*PPU.CoulombConst, kind=tipKind ) # THE MAIN STUFF HERE
#PPC.getGaussDensity( Rs, cRAs )

GU.saveXSF( "rho_core.xsf", rho1,       lvec1, head=head1 )





