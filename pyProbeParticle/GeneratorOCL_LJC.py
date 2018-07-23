#!/usr/bin/python

# Refrences:
# - Keras Data Generator   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

from __future__ import unicode_literals
import sys
import os
import shutil
import time
import random
import matplotlib;
import numpy as np
from enum import Enum

#import matplotlib.pyplot as plt

import basUtils
#from   import PPPlot 
#import GridUtils as GU
import common    as PPU
#import cpp_utils as cpp_utils

import pyopencl    as cl
import oclUtils    as oclu 
import fieldOCL    as FFcl 
import RelaxOpenCL as oclr

import numpy as np
import keras


class Generator(keras.utils.Sequence):

    n_channels = 1
    n_classes  = 10

    # --- ForceField
    pixPerAngstrome = 10
    iZPP = 8
    Q    = -0.1;
    bPBC = True
    lvec = np.array([
        [ 0.0,  0.0,  0.0],
        [19.0,  0.0,  0.0],
        [ 0.0, 20.0,  0.0],
        [ 0.0,  0.0, 21.0]
    ])

    # --- Relaxation
    npbc      = (1,1,1)
    scan_dim  = ( 100, 100, 20)
    distAbove =  7.5

    'Generates data for Keras'
    def __init__(self, molecules, rotations, batch_size=32 ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.molecules = molecules

        self.n_channels = n_channels
        self.n_classes  = n_classes
        #self.shuffle    = shuffle
        #self.on_epoch_end()

        #rotations = hl.PPU.genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, 8) )
        self.irot = len(self.rotations) + 1

        self.typeParams = hl.loadSpecies('atomtypes.ini')
        self.ff_dim     = hl.genFFSampling(lvec, pixPerAngstrome );  print "ff_dim ", ff_dim
        self.poss       = FFcl.getposs( lvec, ff_dim );              print "poss.shape ", poss.shape

        self.forcefield = FFcl.ForceField_LJC()

        self.scanner = oclr.RelaxedScanner()
        self.scanner.relax_params = np.array( [ 0.1,0.9,0.1*0.2,0.1*5.0], dtype=np.float32 );
        self.scanner.stiffness    = np.array( [0.24,0.24,0.0, 30.0     ], dtype=np.float32 )/ -16.0217662

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):  
        'Generate one batch of data'
        print "index ", index
        n = self.batch_size
        Xs = np.empty((n, *self.xdim) )
        Ys = np.empty((n, *self.ydim) )
        for i in range(n):
            ioff = index*n
            if( self.irot >= len(self.rotations) ):
                self.nextMolecule(fname)
                self.irot = 0
            nextRotation( rotations[self.irot], Xs[i], Ys[i] )
        return Xs, Ys

    def nextMolecule(self, fname, X,Y ):
        fullname = self.preName+fname+self.postName
        print " ===== nextMolecule: ", fullname
        t1ff = time.clock();
        atom_lines = open( fullname ).readlines()
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( atom_lines )
        self.natoms0 = len(Zs)
        if( npbc is not None ):
            Zs, xyzs, qs = PPU.PBCAtoms3D( Zs, xyzs, qs, lvec[1:], npbc=npbc )
        cLJs  = PPU.getAtomsLJ( iZPP, Zs, typeParams ).astype(np.float32)
        FF,self.atoms = forcefield.makeFF( xyzs, qs, cLJs, poss=poss )

        self.FEin  = FF[:,:,:,:4] + self.Q*FF[:,:,:,4:];   
        del FF
        Tff = time.clock()-t1ff;   print "Tff %f [s]" %Tff

    def nextRotation(self, rot, X,Y ):
        t1scan = time.clock();
        pos0  = hl.posAboveTopAtom( self.atoms[:self.natoms0], rot[2], distAbove=self.distAbove )
        poss0 = scanner.setScanRot( pos0, rot=rot, start=(-10.0,-10.0), end=(10.0,10.0) )
        FEout = scanner.run()
        Tscan = time.clock()-t1scan;  print "Tscan %f [s]" %Tscan
        X[:,:,:] = FEout[:,:,:,2]

        poss[:,:,:] = poss[:,:,:] + rot[2][:,:,:,None]*self.planeShift   # shift toward surface
        self.getDistDens( atoms, poss, Y )

    def getDistDens(self, atoms, poss, F):
        for atom in atoms:
            dposs =  ( poss - atom[None,None,:] )**2
            F += rFunc( dposs[:,:,0] + dposs[:,:,0] ) + zFunc



