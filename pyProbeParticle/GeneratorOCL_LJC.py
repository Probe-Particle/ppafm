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

import elements
import pyopencl     as cl
import oclUtils     as oclu 
import fieldOCL     as FFcl 
import RelaxOpenCL  as oclr
import HighLevelOCL as hl

import numpy as np
from keras.utils import Sequence

verbose=0

def applyZWeith( F, zWeight ):
    #F_ = np.apply_along_axis( lambda m: np.convolve(m, zWeight, mode='valid'), axis=0, arr=F )
    #print "F.shape, zWeight.shape ", F.shape, zWeight.shape
    F_ = np.average( F, axis=2, weights=zWeight )
    return F_

def modMolParams_def( Zs, qs, xyzs, REAs, rndQmax, rndRmax, rndEmax, rndAlphaMax ):
    if rndQmax > 0:
        qs[:]     += rndQmax * ( np.random.rand( len(qs) ) - 0.5 )
    if rndRmax > 0:
        REAs[:,0] += rndRmax * ( np.random.rand( len(qs) ) - 0.5 )
    if rndEmax > 0:
        REAs[:,1] *= ( 1 + rndEmax * ( np.random.rand( len(qs) ) - 0.5 ) )
    if rndAlphaMax > 0:
        REAs[:,2] *= ( 1 + rndAlphaMax * ( np.random.rand( len(qs) ) - 0.5 ) )
    return Zs, qs, xyzs, REAs

def setBBoxCenter( xyzs, cog ):
#    print "xyzs.shape ", xyzs.shape
#    print "cog ", cog
    xmin = np.min( xyzs[:,0] )
    xmax = np.max( xyzs[:,0] )
    ymin = np.min( xyzs[:,1] )
    ymax = np.max( xyzs[:,1] )
    zmin = np.min( xyzs[:,2] )
    zmax = np.max( xyzs[:,2] )
#    print "xmin ", xmin
    xc = 0.5*(xmin+xmax); 
    yc = 0.5*(ymin+ymax); 
    zc = 0.5*(zmin+zmax)
    #print xyzs[:,0].shape, len(cog[0]), xc, (cog[0]-xc)
    xyzs[:,0] += (cog[0]-xc)
    xyzs[:,1] += (cog[1]-yc)
    xyzs[:,2] += (cog[2]-zc)
#    print "min", (xmin,ymin,zmin), "max", (xmax,ymax,zmax), "cog ", (xc,yc,zc)
    return (xmin,ymin,zmin), (xmax,ymax,zmax), (xc,yc,zc)

class Generator(Sequence,):
    preName  = ""
    postName = ""

    n_channels = 1
    n_classes  = 10

    #Ymode = 'HeightMap'

    # --- ForceField
    #pixPerAngstrome = 10
    iZPP = 8
    Q    = 0.0

    # --- Relaxation
    scan_start = (-8.0,-8.0) 
    scan_end   = ( 8.0, 8.0)
    scan_dim   = ( 100, 100, 30)
    distAbove  =  7.5
    planeShift = -4.0
    
    #maxTilt0 = 0.5
    #maxTilt0 = 1.5
    maxTilt0 = 0.0
    tipR0    = 4.0
    
    # ---- Atom Distance Density
    wr = 1.0
    wz = 1.0
    r2Func = staticmethod( lambda r2 : 1/(1.0+r2) )
    zFunc  = staticmethod( lambda x  : np.exp(-x)  )

    isliceY        = -1
    minEntropy     = 4.5
    nBestRotations = 30
    shuffle_rotations = True
    shuffle_molecules = True
    Yrange = 2

    #npbc = None
    npbc = (1,1,1)

    debugPlots = False
    #debugPlotSlices   = [0,+2,+4,+6,+8,+10,+12,+14,+16]
    #debugPlotSlices   = [-1]
    #debugPlotSlices    = [-5,5]
    #debugPlotSlices    = [-5,5,10,15]
    debugPlotSlices    = [5,10,15]
    #debugPlotSlices   = [-10]
    #debugPlotSlices   = [-15]

    def __init__(self, molecules, rotations, batch_size=32, pixPerAngstrome=10, lvec=None, Ymode='HeightMap' ):
        'Initialization'

        # --- params randomization
        self.rndQmax  = 0.1 
        self.rndRmax  = 0.2
        self.rndEmax  = 0.5
        self.rndAlphaMax = -0.1
        #self.modMolParams = staticmethod(modMolParams_def)
        self.modMolParams = modMolParams_def

        if lvec is None:
            self.lvec = np.array([
                [ 0.0,  0.0,  0.0],
                [30.0,  0.0,  0.0],
                [ 0.0, 30.0,  0.0],
                [ 0.0,  0.0, 30.0]
            ])
        else:
            self.lvec = lvec
        self.pixPerAngstrome=pixPerAngstrome

        self.molecules = molecules
        self.rotations = rotations
        self.batch_size = batch_size

        #self.labels = labels
        #self.n_channels = n_channels
        #self.n_classes  = n_classes
        #self.shuffle    = shuffle
        #self.on_epoch_end()

        #rotations = hl.PPU.genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, 8) )
        self.counter = 0

        self.typeParams = hl.loadSpecies('atomtypes.ini')
        self.ff_dim     = hl.genFFSampling( self.lvec, self.pixPerAngstrome );  #print "ff_dim ",     self.ff_dim
        #self.ff_poss    = FFcl.getposs    ( self.lvec, self.ff_dim );           print "poss.shape ", self.ff_poss.shape  # should we store this?

        self.forcefield = FFcl.ForceField_LJC()

        self.Ymode     = Ymode
        self.projector = None; self.FE2in=None
        self.bZMap = False; self.bFEmap = False;
        print "Ymode", self.Ymode
        if self.Ymode == 'Lorenzian' or self.Ymode == 'Spheres' or self.Ymode == 'Disks' or self.Ymode == 'DisksOcclusion' or self.Ymode == 'QDisks':
            self.projector  = FFcl.AtomProcjetion()
        if self.Ymode == 'HeightMap': 
            self.bZMap = True
        if self.Ymode == 'ElectrostaticMap':
            self.bZMap  = True
            self.bFEmap = True

        self.scanner = oclr.RelaxedScanner()
        self.scanner.relax_params = np.array( [ 0.1,0.9,0.1*0.2,0.1*5.0], dtype=np.float32 );
        self.scanner.stiffness    = np.array( [0.24,0.24,0.0, 30.0     ], dtype=np.float32 )/ -16.0217662
        #self.zWeight =  np.ones( self.scan_dim[2] )
        self.zWeight =  self.getZWeights();

    def __iter__(self):
        return self

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int( np.ceil( len(self.molecules) * self.nBestRotations / float(self.batch_size) ) )

    def __getitem__(self, index):
        if(verbose>0): print "index ", index
        return self.next()

    def on_epoch_end(self):
        if self.shuffle_molecules:
            permut = np.array( range(len(self.molecules)) )
            np.random.shuffle( permut )
            self.molecules = [ self.molecules[i] for i in permut ]

    def getMolRotIndex(self, i):
        #nrot = len(self.rotations)
        nrot = self.nBestRotations
        nmol = len(self.molecules)
        return i/(nrot*nmol), (i/nrot)%nmol, i%nrot

    def evalRotation(self, rot ):
        zDir = rot[2].flat.copy()
        imax,xdirmax,entropy = PPU.maxAlongDirEntropy( self.atomsNonPBC, zDir )
        pos0 = self.atomsNonPBC[imax,:3] 
        return pos0, entropy

    def sortRotationsByEntropy(self):
        rots = []
        for rot in self.rotations:
            pos0, entropy = self.evalRotation( rot )
            rots.append( (entropy, pos0, rot) )
        rots.sort(  key=lambda item: -item[0] )
        return rots

    def next(self):
        'Generate one batch of data'
        n  = self.batch_size
        Xs = np.empty( (n,)+ self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight),) )
        #Xs = np.empty( (n,)+ self.scan_dim     )
        #Xs = np.empty( (n,)+ self.scan_dim[:2]+(20,)  )
        Ys = np.empty( (n,)+ self.scan_dim[:2] )
        for ibatch in range(n):
            self.iepoch, self.imol, self.irot = self.getMolRotIndex( self.counter )
            if( self.irot == 0 ):# recalc FF
                if self.projector is not None:
                    self.projector.tryReleaseBuffers()
                self.molName =  self.molecules[self.imol]
                self.nextMolecule( self.molName ) 
                if(self.counter>0): # not first step
                    if(verbose>1): print "scanner.releaseBuffers()"
                    self.scanner.releaseBuffers()

                self.scanner.prepareBuffers( self.FEin, self.lvec, scan_dim=self.scan_dim, nDimConv=len(self.zWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in )
                self.rotations_sorted = self.sortRotationsByEntropy()
                self.rotations_sorted = self.rotations_sorted[:self.nBestRotations]
                if self.shuffle_rotations:
                    permut = np.array( range(len(self.rotations_sorted)) )
                    np.random.shuffle( permut )
                    #print permut
                    self.rotations_sorted = [ self.rotations_sorted[i] for i in permut ]
                    #self.rotations_sorted = self.rotations_sorted[permut]
                    #print self.rotations_sorted
            ##rot = self.rotations[self.irot]
            ##pos0, entropy = self.evalRotation( rot )
            ##print "rot entropy:", entropy
            ##if( entropy > self.minEntropy ): break
            ##print "skiped"
            #print "batch i : ", ibatch
            rot = self.rotations_sorted[self.irot]
            self.nextRotation( Xs[ibatch], Ys[ibatch] )
            #self.nextRotation( self.rotations[self.irot], Xs[ibatch], Ys[ibatch] )
            self.counter +=1
        return Xs, Ys

    def nextMolecule(self, fname ):
        fullname = self.preName+fname+self.postName
        if(verbose>0): print " ===== nextMolecule: ", fullname
        t1ff = time.clock();
        self.atom_lines = open( fullname ).readlines()
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( self.atom_lines )
        setBBoxCenter( xyzs, (self.lvec[1,0]*0.5,self.lvec[2,1]*0.5,self.lvec[3,2]*0.5) )
        self.natoms0 = len(Zs)
        #cLJs  = PPU.getAtomsLJ( self.iZPP, Zs, self.typeParams )
        #print "cLJs ref : ", cLJs
        
        REAs = PPU.getAtomsREA(  self.iZPP, Zs, self.typeParams, alphaFac=-1.0 )
        self.modMolParams( Zs, qs, xyzs, REAs, self.rndQmax, self.rndRmax, self.rndEmax, self.rndAlphaMax )
        cLJs = PPU.REA2LJ( REAs )

        #print "Qs   : ", qs
        #print "REAs : ", REAs
        #print "cLJs : ", cLJs

        if( self.npbc is not None ):
            Zs, xyzs, qs, cLJs = PPU.PBCAtoms3D( Zs, xyzs, qs, cLJs, self.lvec[1:], npbc=self.npbc )
        self.Zs = Zs

        #print "cLJs : ", cLJs

        #FF,self.atoms = self.forcefield.makeFF( xyzs, qs, cLJs, poss=self.ff_poss )
        FF,self.atoms = self.forcefield.makeFF( xyzs, qs, cLJs, lvec=self.lvec, pixPerAngstrome=self.pixPerAngstrome )
        self.atomsNonPBC = self.atoms[:self.natoms0].copy()
        self.FEin  = FF[:,:,:,:4] + self.Q*FF[:,:,:,4:];
        
        if self.Ymode == 'ElectrostaticMap':
            self.FE2in = FF[:,:,:,4:].copy();

        if self.projector is not None:
            na = len(self.atomsNonPBC)
            #print "self.atomsNonPBC", self.atomsNonPBC
            #print "Zs[:self.atomsNonPBC]", Zs[:self.atomsNonPBC]
            coefs=self.projector.makeCoefsZR( Zs[:na], elements.ELEMENTS )
            self.projector.prepareBuffers( self.atomsNonPBC, self.scan_dim[:2]+(1,), coefs=coefs )

        #self.saveDebugXSF( self.preName+fname+"/FF_z.xsf", self.FEin[:,:,:,2], d=(0.1,0.1,0.1) )

        Tff = time.clock()-t1ff;   
        if(verbose>1): print "Tff %f [s]" %Tff

    #def nextRotation(self, rot, X,Y ):
    def nextRotation(self, X,Y ):
        t1scan = time.clock();
        (entropy, self.pos0, rot) = self.rotations_sorted[self.irot]

        if(verbose>0): print " imol, irot, entropy ", self.imol, self.irot, entropy
        zDir = rot[2].flat.copy()

        '''
        tipR0      = self.maxTilt0 * (np.random.rand(3) - 0.5); 
        #tipR0[2] += self.tipR0 # augumentation of data by varying tip
        tipR0[2]   = self.tipR0 # augumentation of data by varying tip
        '''
        tipR0 = np.array( [0.5,0.0,self.tipR0] )


        self.scan_pos0s  = self.scanner.setScanRot( self.pos0+rot[2]*self.distAbove, rot=rot, start=self.scan_start, end=self.scan_end, tipR0=tipR0  )

        FEout  = self.scanner.run_relaxStrokesTilted()

        if( len(self.dfWeight) != self.scanner.scan_dim[2] - self.scanner.nDimConvOut   ):
            print "len(dfWeight) must be scan_dim[2] - nDimConvOut ", len(self.dfWeight),  self.scanner.scan_dim[2], self.scanner.nDimConvOut
            exit()
        #print "self.dfWeight ", self.dfWeight
        self.scanner.updateBuffers( WZconv=self.dfWeight )
        #FEconv = self.scanner.runZConv()
        FEout = self.scanner.run_convolveZ()

        #exit(0);

        #XX = FEconv[:,:,:,0]*zDir[0] + FEconv[:,:,:,1]*zDir[1] + FEconv[:,:,:,2]*zDir[2]
        #self.saveDebugXSF( "df.xsf", XX )

        # Perhaps we should convert it to df using PPU.Fz2df(), but that is rather slow - maybe to it on GPU?
        #X[:,:,:] = FEout[:,:,:,2]
        #print "rot.shape, zDir.shape", rot.shape, zDir
        #print "FEout.shape ", FEout.shape

        #X[:,:,:] = FEout[:,:,:,0]*zDir[0] + FEout[:,:,:,1]*zDir[1] + FEout[:,:,:,2]*zDir[2]
        #X[:,:,:] = FEout[:,:,:,2].copy()  # Rotation is now done in kernel
        X[:,:,:FEout.shape[2]] = FEout[:,:,:,2].copy()
        #print "FEout.z min max ", np.min(FEout[:,:,:,2]), np.max(FEout[:,:,:,2])
        #print "X shape min max ", X[:,:,:FEout.shape[2]].shape,   np.min(X[:,:,:FEout.shape[2]]), np.max(X[:,:,:FEout.shape[2]])

        Tscan = time.clock()-t1scan;  
        if(verbose>1): print "Tscan %f [s]" %Tscan
        
        t1y = time.clock();
        #self.scanner.runFixed( FEout=FEout )
        #self.scanner.run_getFEinStrokesTilted( FEout=FEout )
        #Y[:,:] = FEout[:,:,-1,2]
        #Y[:,:] =  FEout[:,:,self.isliceY,0]*zDir[0] + FEout[:,:,self.isliceY,1]*zDir[1] + FEout[:,:,self.isliceY,2]*zDir[2]
        #self.zWeight =  self.getZWeights(); print self.zWeight
        #Y_ = FEout[:,:,:,0]*zDir[0] + FEout[:,:,:,1]*zDir[1] + FEout[:,:,:,2]*zDir[2]
        #Y_ = FEout[:,:,:,2].copy()

        # -- strategy 1  CPU saturated Weighted average
        #Y_ = np.tanh( Y_ )
        #Y[:,:] = applyZWeith( Y_, self.zWeight )
        # -- strategy 1  CPU zMin
        #Y[:,:] = np.nanargmin( ( Y_-1.0 )**2, axis=2 )
        #Yf = Y.flat; Yf[Yf<5] = Y_.shape[2]
        #Yf = Y.flat; Yf[Yf<5] = np.NaN
        #Yf = Y.flat; Yf[Y.fla_] = np.NaN
        # -- strategy 3  GPU convolve
        #self.scanner.updateBuffers( WZconv=self.zWeight )
        #FEconv = self.scanner.runZConv()
        #YY = FEconv[:,:,:,0]*zDir[0] + FEconv[:,:,:,1]*zDir[1] + FEconv[:,:,:,2]*zDir[2]
        #self.saveDebugXSF( "FixedConv.xsf", YY )
        # -- strategy 4  GPU izoZ
        #Y[:,:] = self.scanner.runIzoZ( iso=0.1 )
        #Y[:,:] = self.scanner.runIzoZ( iso=0.1, nz=40 )
        #Y[:,:] = ( self.scanner.run_getZisoTilted( iso=0.1, nz=100 ) *-1 ) . copy()
        

        if self.Ymode == 'HeightMap':
            '''
            Y[:,:] = ( self.scanner.run_getZisoTilted( iso=0.1, nz=100 ) ) . copy()
            Yf=Y.flat; Yf[Yf<0]=+39+5; Yf[:]-=39
            Y *= (-self.scanner.zstep)
            '''
            Y[:,:] = ( self.scanner.run_getZisoTilted( iso=0.1, nz=100 ) *-1 ) . copy()
            Y *= (self.scanner.zstep)
            Ymin = max(Y[Y<=0].flatten().max() - self.Yrange, Y.flatten().min())
            Y[Y>0] = Ymin
            Y[Y<Ymin] = Ymin
            Y -= Ymin
        elif self.Ymode == 'ElectrostaticMap':
            zMap, feMap = self.scanner.run_getZisoFETilted( iso=0.1, nz=100 )
            #Y[:,:] = ( feMap[:,:,0] ).copy() # Fel_x
            #Y[:,:] = ( feMap[:,:,1] ).copy() # Fel_y
            Y[:,:] = ( feMap[:,:,2] ).copy() # Fel_z
            #Y[:,:]  = ( feMap[:,:,3] ).copy() # Vel
            '''
            Y *= (self.scanner.zstep)
            Ymin = max(Y[Y<=0].flatten().max() - self.Yrange, Y.flatten().min())
            Y[Y>0] = Ymin
            Y[Y<Ymin] = Ymin
            Y -= Ymin
            '''
        elif self.Ymode == 'Lorenzian':
            dirFw = np.append( rot[2], [0] ); print "dirFw ", dirFw
            poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAbove-1.0))[None,None,:] )
            Y[:,:] =  self.projector.run_evalLorenz( poss = poss_ )[:,:,0]
        elif self.Ymode == 'Spheres':
            dirFw = np.append( rot[2], [0] ); print "dirFw ", dirFw
            poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAbove-1.0))[None,None,:] )
            Y[:,:] = self.projector.run_evalSpheres( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'Disks':
            dirFw = np.append( rot[2], [0] ); print "dirFw ", dirFw
            poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAbove-1.0))[None,None,:] )
            Y[:,:] = self.projector.run_evaldisks( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'DisksOcclusion':
            dirFw = np.append( rot[2], [0] ); print "dirFw ", dirFw
            poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAbove-1.0))[None,None,:] )
            Y[:,:] = self.projector.run_evaldisks_occlusion( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'QDisks':
            dirFw = np.append( rot[2], [0] ); print "dirFw ", dirFw
            poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAbove-1.0))[None,None,:] )
            Y[:,:] = self.projector.run_evalQdisks( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]

        Ty =  time.clock()-t1scan;  
        if(verbose>1): print "Ty %f [s]" %Ty

        #print "Y.shape", Y.shape
        #print "X.shape", X.shape

        if(self.debugPlots):
            #self.plot(X,Y, Y_, entropy )
            #self.plot( X=XX, Y_=YY, entropy=entropy )
            #self.plot(Y=Y, entropy=entropy )
            self.plot( ("/rot%03i_" % self.irot), self.molName, X=X, Y=Y, entropy=entropy )

    def getAFMinRot(self, rot, X ):
        t1scan = time.clock();

        tipR0 = self.maxTilt0 * (np.random.rand(3) - 0.5); 
        tipR0[2]   = self.tipR0 # augumentation of data by varying tip

        self.scan_pos0s  = self.scanner.setScanRot( self.pos0+rot[2]*self.distAbove, rot=rot, start=self.scan_start, end=self.scan_end, tipR0=tipR0  )

        FEout  = self.scanner.run_relaxStrokesTilted()

        if( len(self.dfWeight) != self.scanner.scan_dim[2] - self.scanner.nDimConvOut   ):
            print "len(dfWeight) must be scan_dim[2] - nDimConvOut ", len(self.dfWeight),  self.scanner.scan_dim[2], self.scanner.nDimConvOut
            exit()
        self.scanner.updateBuffers( WZconv=self.dfWeight )
        FEout = self.scanner.run_convolveZ()
        X[:,:,:FEout.shape[2]] = FEout[:,:,:,2].copy()

    def match(self, Xref ):
        return

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

    def plot(self, rotName, molName, X=None,Y=None,Y_=None, entropy=None ):
        import matplotlib as mpl;  mpl.use('Agg');
        import matplotlib.pyplot as plt

        fname    = self.preName + molName + rotName
        print " plot to file : ", fname

        #self.saveDebugXSF( self.preName + self.molecules[imol] + ("/rot%03i_Y.xsf" %irot), Y_ )
        basUtils.writeDebugXYZ_2( self.preName + molName + rotName+".xyz", self.atoms, self.Zs, self.scan_pos0s[::10,::10,:].reshape(-1,4), pos0=self.pos0 )
        #basUtils.writeDebugXYZ( self.preName + molName + rotName, self.atom_lines, self.scan_pos0s[::10,::10,:].reshape(-1,4), pos0=self.pos0 )

        #self.saveDebugXSF(  self.preName + molName + rotName+"_Fz.xsf", X, d=(0.1,0.1,0.1) )

        title = "entropy %f" %entropy
        if Y is not None:
            if self.Ymode == 'ElectrostaticMap':
                vmax = max( Y.max(), -Y.min() )
                #plt.imshow( Y, vmin=-vmax, vmax=vmax, cmap='seismic' );
                plt.imshow( Y, vmin=-vmax, vmax=vmax, cmap='bwr' );
            else:
                plt.imshow( Y );
            plt.colorbar()
            #print "Y = ", Y
            #plt.imshow( Y, vmin=-5, vmax=5 );  
            plt.title(entropy)
            plt.savefig(  fname+"Dens.png", bbox_inches="tight"  );
            #plt.savefig(  fname+"Dens.png", bbox_inches="tight"  ); 
            plt.close()

        for isl in self.debugPlotSlices:
            #plt.imshow( FEout[:,:,isl,2] )
            if (X is not None) and (Y_ is not None):
                plt.figure(figsize=(10,5))
                #print isl, np.min(X[:,:,isl]), np.max(X[:,:,isl])
                plt.subplot(1,2,2); plt.imshow (X [:,:,isl] );            plt.colorbar()
                #plt.subplot(1,2,1); plt.imshow (Y_[:,:,isl] );
                plt.subplot(1,2,1); plt.imshow (np.tanh(Y_[:,:,isl]) );   plt.colorbar()
                plt.title(entropy)
                plt.savefig( fname+( "FzFixRelax_iz%03i.png" %isl ), bbox_inches="tight"  ); 
                plt.close()
            else:
                if X is not None:
                    print isl, np.min(X[:,:,isl]), np.max(X[:,:,isl])
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



