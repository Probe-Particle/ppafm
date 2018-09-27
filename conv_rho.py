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

rho1, lvec1, nDim1, head1 = GU.loadXSF("./pyridine/CHGCAR.xsf")
rho2, lvec2, nDim2, head2 = GU.loadXSF("./CO/CHGCAR.xsf")

#fFFT.conv3DFFT( F2, F1 )
#GU.saveXSF( "Fpauli_x.xsf", Fx*PQ, lvec1, head=head1 )

Fx,Fy,Fz,E = fFFT.potential2forces_mem( rho1, lvec1, nDim1, rho=rho2, doForce=True, doPot=True, deleteV=True )
#Fx,Fy,Fz = fFFT.potential2forces( rho1, lvec1, nDim1, rho=rho2 )
#Fx,Fy,Fz =fFFT.rhos2forces( rho1, rho2, lvec, nDim )
#FFel = GU.packVecGrid(Fel_x,Fel_y,Fel_z); 

PQ = -1.0

# Density Overlap Model
GU.saveXSF( "Epauli.xsf",    E*(PQ*-1.0), lvec1, head=head1 )
GU.saveXSF( "FFpauli_x.xsf", Fx*PQ,       lvec1, head=head1 )
GU.saveXSF( "FFpauli_y.xsf", Fy*PQ,       lvec1, head=head1 )
GU.saveXSF( "FFpauli_z.xsf", Fz*PQ,       lvec1, head=head1 )

#Fx, Fy, Fz = getForces( V, rho, sampleSize, dims, dd, X, Y, Z)






