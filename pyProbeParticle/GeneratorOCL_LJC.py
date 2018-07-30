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

import pyopencl     as cl
import oclUtils     as oclu 
import fieldOCL     as FFcl 
import RelaxOpenCL  as oclr
import HighLevelOCL as hl

import numpy as np
from keras.utils import Sequence

verbose=1

def applyZWeith( F, zWeight ):
    #F_ = np.apply_along_axis( lambda m: np.convolve(m, zWeight, mode='valid'), axis=0, arr=F )
    #print "F.shape, zWeight.shape ", F.shape, zWeight.shape
    F_ = np.average( F, axis=2, weights=zWeight )
    return F_

class Generator(Sequence):
#class Generator():
    preName  = ""
    postName = ""

    n_channels = 1
    n_classes  = 10

    # --- ForceField
    pixPerAngstrome = 10
    iZPP = 8
    Q    = 0.0
    bPBC = True
    lvec = np.array([
        [ 0.0,  0.0,  0.0],
        [19.0,  0.0,  0.0],
        [ 0.0, 20.0,  0.0],
        [ 0.0,  0.0, 21.0]
    ])

    # --- Relaxation
    npbc       = (1,1,1)
    scan_dim   = ( 100, 100, 30)
    distAbove  =  7.5
    planeShift =  -4.0
    
    # ---- Atom Distance Density
    wr = 1.0
    wz = 1.0
    r2Func = staticmethod( lambda r2 : 1/(1.0+r2) )
    zFunc  = staticmethod( lambda x  : np.exp(-x)  )

    isliceY = -1

    minEntropy = 4.5

    debugPlots = False
    #debugPlotSlices   = [0,+2,+4,+6,+8,+10,+12,+14,+16]
    #debugPlotSlices   = [-1]
    debugPlotSlices    = [-5,5]
    #debugPlotSlices   = [-10]
    #debugPlotSlices   = [-15]

    'Generates data for Keras'
    def __init__(self, molecules, rotations, batch_size=32 ):
        'Initialization'

        self.molecules = molecules
        self.rotations = rotations
        self.batch_size = batch_size

        #self.dim = dim
        #self.labels = labels
        #self.n_channels = n_channels
        #self.n_classes  = n_classes
        #self.shuffle    = shuffle
        #self.on_epoch_end()

        #rotations = hl.PPU.genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, 8) )
        self.counter = 0

        self.typeParams = hl.loadSpecies('atomtypes.ini')
        self.ff_dim     = hl.genFFSampling( self.lvec, self.pixPerAngstrome );  print "ff_dim ",     self.ff_dim
        #self.ff_poss    = FFcl.getposs    ( self.lvec, self.ff_dim );           print "poss.shape ", self.ff_poss.shape  # should we store this?

        self.forcefield = FFcl.ForceField_LJC()

        self.scanner = oclr.RelaxedScanner()
        self.scanner.relax_params = np.array( [ 0.1,0.9,0.1*0.2,0.1*5.0], dtype=np.float32 );
        self.scanner.stiffness    = np.array( [0.24,0.24,0.0, 30.0     ], dtype=np.float32 )/ -16.0217662
        #self.zWeight =  np.ones( self.scan_dim[2] )
        self.zWeight =  self.getZWeights();

    def getMolRotIndex(self, i):
        nrot = len(self.rotations)
        nmol = len(self.molecules)
        return i/(nrot*nmol), (i/nrot)%nmol, i%nrot

    def __iter__(self):
        return self

    def evalRotation(self, rot ):
        zDir = rot[2].flat.copy()
        imax,xdirmax,entropy = PPU.maxAlongDirEntropy( self.atomsNonPBC, zDir )
        pos0 = self.atomsNonPBC[imax,:3] + zDir*self.distAbove
        return pos0, entropy

    def next(self):
        'Generate one batch of data'
        n  = self.batch_size
        Xs = np.empty( (n,)+ self.scan_dim     )
        Ys = np.empty( (n,)+ self.scan_dim[:2] )
        for i in range(n):
            while(True):
                iepoch, imol, irot = self.getMolRotIndex( self.counter )
                if(verbose>0): print " imol, irot ", imol, irot
                if( irot == 0 ):# recalc FF
                    self.nextMolecule( self.molecules[imol] ) 
                    if(self.counter>0): # not first step
                        if(verbose>1): print "scanner.releaseBuffers()"
                        self.scanner.releaseBuffers()
                    self.scanner.prepareBuffers( self.FEin, self.lvec, scan_dim=self.scan_dim, nDimConv=len(self.zWeight), nDimConvOut=20  )
                rot = self.rotations[irot]
                pos0, entropy = self.evalRotation( rot )
                print "rot entropy:", entropy
                if( entropy > self.minEntropy ): break
                print "skiped"
                self.counter +=1
            self.nextRotation( rot, pos0, entropy, Xs[i], Ys[i] )
            #self.nextRotation( self.rotations[irot], Xs[i], Ys[i] )
            self.counter +=1
        return Xs, Ys

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 10
        #return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        if(verbose>0): print "index ", index
        return self.next()

    def nextMolecule(self, fname ):
        fullname = self.preName+fname+self.postName
        if(verbose>0): print " ===== nextMolecule: ", fullname
        t1ff = time.clock();
        self.atom_lines = open( fullname ).readlines()
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( self.atom_lines )
        self.natoms0 = len(Zs)
        if( self.npbc is not None ):
            Zs, xyzs, qs = PPU.PBCAtoms3D( Zs, xyzs, qs, self.lvec[1:], npbc=self.npbc )
        cLJs  = PPU.getAtomsLJ( self.iZPP, Zs, self.typeParams ).astype(np.float32)
        #FF,self.atoms = self.forcefield.makeFF( xyzs, qs, cLJs, poss=self.ff_poss )
        FF,self.atoms = self.forcefield.makeFF( xyzs, qs, cLJs, lvec=self.lvec, pixPerAngstrome=self.pixPerAngstrome )
        self.atomsNonPBC = self.atoms[:self.natoms0].copy()
        self.FEin  = FF[:,:,:,:4] + self.Q*FF[:,:,:,4:];
        Tff = time.clock()-t1ff;   
        if(verbose>1): print "Tff %f [s]" %Tff

    #def nextRotation(self, rot, X,Y ):
    def nextRotation(self, rot, pos0, entropy, X,Y ):
        t1scan = time.clock();
        zDir = rot[2].flat.copy()

        self.scan_pos0s  = self.scanner.setScanRot( pos0, rot=rot, start=(-10.0,-10.0), end=(10.0,10.0) )

        FEout  = self.scanner.runTilted()

        self.scanner.updateBuffers( WZconv=self.dfWeight )
        FEconv = self.scanner.runZConv()
        XX = FEconv[:,:,:,0]*zDir[0] + FEconv[:,:,:,1]*zDir[1] + FEconv[:,:,:,2]*zDir[2]
        #self.saveDebugXSF( "df.xsf", XX )

        # Perhaps we should convert it to df using PPU.Fz2df(), but that is rather slow - maybe to it on GPU?
        #X[:,:,:] = FEout[:,:,:,2]
        #print "rot.shape, zDir.shape", rot.shape, zDir
        #print "FEout.shape ", FEout.shape

        X[:,:,:] = FEout[:,:,:,0]*zDir[0] + FEout[:,:,:,1]*zDir[1] + FEout[:,:,:,2]*zDir[2]
        Tscan = time.clock()-t1scan;  
        if(verbose>1): print "Tscan %f [s]" %Tscan
        
        t1y = time.clock();
        self.scanner.runFixed( FEout=FEout )
        #Y[:,:] = FEout[:,:,-1,2]
        #Y[:,:] =  FEout[:,:,self.isliceY,0]*zDir[0] + FEout[:,:,self.isliceY,1]*zDir[1] + FEout[:,:,self.isliceY,2]*zDir[2]
        #self.zWeight =  self.getZWeights(); print self.zWeight
        #Y_ = FEout[:,:,:,0]*zDir[0] + FEout[:,:,:,1]*zDir[1] + FEout[:,:,:,2]*zDir[2]

        # strategy 1  CPU saturated Weighted average
        #Y_ = np.tanh( Y_ )
        #Y[:,:] = applyZWeith( Y_, self.zWeight )
        # strategy 1  CPU zMin
        #Y[:,:] = -np.nanargmin( ( Y_+1.0 )**2, axis=2 )

        self.scanner.updateBuffers( WZconv=self.zWeight )
        FEconv = self.scanner.runZConv()
        YY = FEconv[:,:,:,0]*zDir[0] + FEconv[:,:,:,1]*zDir[1] + FEconv[:,:,:,2]*zDir[2]
        #self.saveDebugXSF( "FixedConv.xsf", YY )

        Ty =  time.clock()-t1scan;  
        if(verbose>1): print "Ty %f [s]" %Ty

        if(self.debugPlots):
            #self.plot(X,Y, Y_, entropy )
            self.plot( X=XX, Y_=YY, entropy=entropy )
            #self.plot(Y=Y, entropy=entropy )

    def saveDebugXSF(self, fname, F, d=(0.1,0.1,0.1) ):
        if hasattr(self, 'GridUtils'):
            GU = self.GridUtils
        else:
            import GridUtils as GU
            self.GridUtils = GU
        sh = F.shape
        #self.lvec_scan = np.array( [ [0.0,0.0,0.0],[self.scan_dim[0],0.0,0.0],[0.0,self.scan_dim[1],0.0],0.0,0.0, ] ] )
        lvec = np.array( [ [0.0,0.0,0.0],[sh[0]*d[0],0.0,0.0],[0.0,sh[1]*d[1],0.0], [ 0.0,0.0,sh[2]*d[2] ] ] )
        print "saveDebugXSF : ", fname
        GU.saveXSF( fname, F.transpose((2,1,0)), lvec )

    def plot(self,X=None,Y=None,Y_=None, entropy=None):
        import matplotlib as mpl;  mpl.use('Agg');
        import matplotlib.pyplot as plt
        iepoch, imol, irot = self.getMolRotIndex( self.counter )
        fname    = self.preName + self.molecules[imol] + ("/rot%03i_" % irot)
        print " plot to file : ", fname

        basUtils.writeDebugXYZ( self.preName + self.molecules[imol] + ("/rot%03i.xyz" %irot), self.atom_lines, self.scan_pos0s[::10,::10,:].reshape(-1,4) )

        title = "entropy %f" %entropy
        if Y is not None:
            plt.imshow( Y );             plt.colorbar()
            plt.title(entropy)
            plt.savefig(  fname+"Dens.png", bbox_inches="tight"  ); 
            plt.close()

        for isl in self.debugPlotSlices:
            #plt.imshow( FEout[:,:,isl,2] )
            if (X is not None) and (Y_ is not None):
                plt.figure(figsize=(10,5))
                plt.subplot(1,2,2); plt.imshow (X [:,:,isl] );            plt.colorbar()
                #plt.subplot(1,2,1); plt.imshow (Y_[:,:,isl] );
                plt.subplot(1,2,1); plt.imshow (np.tanh(Y_[:,:,isl]) );   plt.colorbar()
                plt.title(entropy)
                plt.savefig( fname+( "FzFixRelax_iz%03i.png" %isl ), bbox_inches="tight"  ); 
                plt.close()
            else:
                if X is not None:
                    plt.imshow(  X[:,:,isl] );    plt.colorbar()
                    plt.savefig(  fname+( "Fz_iz%03i.png" %isl ), bbox_inches="tight"  ); 
                    plt.close()
                if Y_ is not None:
                    plt.imshow ( Y_[:,:,isl] );   plt.colorbar()
                    plt.savefig( fname+( "FzFix_iz%03i.png" %isl ), bbox_inches="tight"  ); 
                    plt.close()

    def getZWeights(self):
        zs = np.mgrid[0:self.scan_dim[2]] * self.scanner.zstep * self.wz
        zWeights = self.zFunc( zs )
        zWeights = zWeights.astype(np.float32)
        return zWeights

    def getDistDens(self, atoms, poss, F):
        F[:,:] = 0
        for atom in atoms[:1]:
            ## TODO - we should project it with respect to rotation
            dposs =  poss - atom[None,None,:]
            #F[:,:] += self.r2Func( (dposs[:,:,0]**2 + dposs[:,:,1]**2)/self.wr ) # * self.zFunc( dposs[:,:,2]/self.wz )
            F[:,:] += self.r2Func( (dposs[:,:,0]**2 + dposs[:,:,1]**2 + dposs[:,:,2]**2 )/self.wr )



