#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys
import __main__ as main
import numpy as np
import matplotlib.pyplot as plt
#import GridUtils as GU
#sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")

import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT
from optparse import OptionParser

parser = OptionParser()
parser.add_option( "-s", "--sample", action="store", type="string", default="CHGCAR.xsf", help="sample 3D data-file (.xsf)")
parser.add_option( "-t", "--tip",    action="store", type="string", default="./tip/CHGCAR.xsf", help="tip 3D data-file (.xsf)")
parser.add_option( "-o", "--output", action="store", type="string", default="pauli", help="output 3D data-file (.xsf)")
(options, args) = parser.parse_args()

#rho1, lvec1, nDim1, head1 = GU.loadXSF("./pyridine/CHGCAR.xsf")
#rho2, lvec2, nDim2, head2 = GU.loadXSF("./CO_/CHGCAR.xsf")

rho1, lvec1, nDim1, head1 = GU.loadXSF( options.sample )
rho2, lvec2, nDim2, head2 = GU.loadXSF( options.tip    )

if "AECCAR" in options.sample:
    V1 = np.abs( np.linalg.det(lvec1[1:]) )
    rho1 /= V1

if "AECCAR" in options.sample:
    V2 = np.abs( np.linalg.det(lvec2[1:]) )
    rho2 /= V2

#fFFT.conv3DFFT( F2, F1 )
#GU.saveXSF( "Fpauli_x.xsf", Fx*PQ, lvec1, head=head1 )

Fx,Fy,Fz,E = fFFT.potential2forces_mem( rho1, lvec1, nDim1, rho=rho2, doForce=True, doPot=True, deleteV=True )
#Fx,Fy,Fz = fFFT.potential2forces( rho1, lvec1, nDim1, rho=rho2 )
#Fx,Fy,Fz =fFFT.rhos2forces( rho1, rho2, lvec, nDim )
#FFel = GU.packVecGrid(Fel_x,Fel_y,Fel_z); 

PQ = -1.0

namestr = options.output
print "save to ", namestr

# Density Overlap Model
GU.saveXSF( "E"+namestr+".xsf",    E*(PQ*-1.0), lvec1, head=head1 )
GU.saveXSF( "FF"+namestr+"_x.xsf", Fx*PQ,       lvec1, head=head1 )
GU.saveXSF( "FF"+namestr+"_y.xsf", Fy*PQ,       lvec1, head=head1 )
GU.saveXSF( "FF"+namestr+"_z.xsf", Fz*PQ,       lvec1, head=head1 )

#Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)






