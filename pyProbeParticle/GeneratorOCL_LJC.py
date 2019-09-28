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
#from keras.utils import Sequence

verbose  = 0
bRunTime = False

class Sequence:
    pass

def getRandomUniformDisk():
    '''
    generate points unifromly distributed over disk
    # see: http://mathworld.wolfram.com/DiskPointPicking.html
    '''
    rnd = np.random.rand(2)
    rnd[0]    = np.sqrt( rnd[0] ) 
    rnd[1]   *= 2.0*np.pi
    return  rnd[0]*np.cos(rnd[1]), rnd[0]*np.sin(rnd[1])


def rotAtoms(rot, atoms):
    '''
    rotate atoms by matrix "rot"
    '''
    #print "atoms.shape ", atoms.shape
    atoms_ = np.zeros(atoms.shape)
    atoms_[:,0] =  rot[0,0] * atoms[:,0]   +  rot[0,1] * atoms[:,1]   +    rot[0,2] * atoms[:,2]
    atoms_[:,1] =  rot[1,0] * atoms[:,0]   +  rot[1,1] * atoms[:,1]   +    rot[1,2] * atoms[:,2]
    atoms_[:,2] =  rot[2,0] * atoms[:,0]   +  rot[2,1] * atoms[:,1]   +    rot[2,2] * atoms[:,2]
    return atoms_

def applyZWeith( F, zWeight ):
    '''
    Weighted average of 1D array
    '''
    #F_ = np.apply_along_axis( lambda m: np.convolve(m, zWeight, mode='valid'), axis=0, arr=F )
    #print "F.shape, zWeight.shape ", F.shape, zWeight.shape
    F_ = np.average( F, axis=2, weights=zWeight )
    return F_

def modMolParams_def( Zs, qs, xyzs, REAs, rndQmax, rndRmax, rndEmax, rndAlphaMax ):
    '''
    random variation of molecular parameters
    '''
    if rndQmax > 0:
        qs[:]     += rndQmax * ( np.random.rand( len(qs) ) - 0.5 )
    if rndRmax > 0:
        REAs[:,0] += rndRmax * ( np.random.rand( len(qs) ) - 0.5 )
    if rndEmax > 0:
        REAs[:,1] *= ( 1 + rndEmax * ( np.random.rand( len(qs) ) - 0.5 ) )
    if rndAlphaMax > 0:
        REAs[:,2] *= ( 1 + rndAlphaMax * ( np.random.rand( len(qs) ) - 0.5 ) )
    return Zs, qs, xyzs, REAs

def getBBox( xyzs ):
    xmin = np.min( xyzs[:,0] )
    xmax = np.max( xyzs[:,0] )
    ymin = np.min( xyzs[:,1] )
    ymax = np.max( xyzs[:,1] )
    zmin = np.min( xyzs[:,2] )
    zmax = np.max( xyzs[:,2] )
    return np.array([xmin,ymin,zmin]), np.array([xmax,ymax,zmax])

def setBBoxCenter( xyzs, cog ):
    '''
    find bounding box for set of xyz points
    '''
    pmin,pmax = getBBox( xyzs )
    pc = (pmin+pmax)*0.5
    #print xyzs[:,0].shape, len(cog[0]), xc, (cog[0]-xc)
    xyzs[:,0] += (cog[0]-pc[0])
    xyzs[:,1] += (cog[1]-pc[1])
    xyzs[:,2] += (cog[2]-pc[2])
#    print "min", (xmin,ymin,zmin), "max", (xmax,ymax,zmax), "cog ", (xc,yc,zc)
    return pmin, pmax, pc

def getAtomsRotZmin( rot, xyzs, zmin, Zs=None ):
    '''
    get all atoms closer to camera than "zmin"
    '''
    #xdir = np.dot( atoms[:,:3], hdir[:,None] )
    #print xyzs.shape
    #xyzs_ = np.empty(xyzs.shape)
    #xyzs_[:,0]  = rot[0,0]*xyzs[:,0] + rot[0,1]*xyzs[:,1] + rot[0,2]*xyzs[:,2]
    #xyzs_[:,1]  = rot[1,0]*xyzs[:,0] + rot[1,1]*xyzs[:,1] + rot[1,2]*xyzs[:,2]
    #xyzs_[:,2]  = rot[2,0]*xyzs[:,0] + rot[2,1]*xyzs[:,1] + rot[2,2]*xyzs[:,2]
    xyzs_ = rotAtoms(rot, xyzs )
    mask  =  xyzs_[:,2] > zmin
    #print xyzs_.shape, mask.shape
    #print xyzs_
    #print mask
    if Zs is not None:
        Zs = Zs[mask]
    return xyzs_[mask,:], Zs

def getAtomsRotZminNsort( rot, xyzs, zmin, RvdWs=None, Zs=None, Nmax=30, RvdW_H = 1.4870 ):
    '''
    get <=Nmax atoms closer to camera than "zmin" and sort by distance from camera
    '''
    xyzs_ = rotAtoms(rot, xyzs )
    #print " np.dot( rot, rot.transpse() ) ", np.dot( rot, rot.transpose() )
    #zs = xyzs_[:,2]
    zs = xyzs_[:,2].copy()
    if RvdWs is not None:
        #print " :::: xyzs_.shape, RvdWs.shape :: ", xyzs_.shape, RvdWs.shape
        #print " :::: RvdW      ", RvdWs
        #print " :::: zs      ", zs
        zs += RvdWs
        #print " :::: zs+Rvdw ", zs
    zs = zs - RvdW_H
    inds = np.argsort( -zs ) #.copy()
    #print " :::: inds ", inds
    xyzs_ = xyzs_[inds,:].copy()
    zs    = zs   [inds  ].copy()
    #zs         = zs   [inds]
    #mask =  xyzs_[:,2] > zmin
    mask  = zs > zmin
    xyzs_ = xyzs_[mask,:]
    #print  "inds new ", inds
    #print  "mask new ", mask
    #print  "xyzs_[:,:] new ", xyzs_[:,2]
    #print  "zs   new ", zs
    #print xyzs_.shape, mask.shape
    #print xyzs_
    #print mask
    if Zs is not None:
        Zs = Zs[inds]
        Zs = Zs[mask]
    return xyzs_[:Nmax,:], Zs[:Nmax] 

def getAtomsRotZminNsort_old( rot, xyzs, zmin, RvdWs=None, Zs=None, Nmax=30 ):
    xyzs_ = rotAtoms(rot, xyzs )
    inds = np.argsort( -xyzs_[:,2] )
    xyzs_[:,:] = xyzs_[inds,:]
    mask  = xyzs_[:,2] > zmin

    print  "inds old ", inds
    print  "mask old ", mask
    print  "zs   old ", xyzs_[:,2]
    xyzs_ = xyzs_[mask,:]
    #print xyzs_.shape, mask.shape
    #print xyzs_
    #print mask
    if Zs is not None:
        Zs = Zs[inds]
        Zs = Zs[mask]
    return xyzs_[:Nmax,:], Zs[:Nmax]

class Generator(Sequence,):

    bNoPoss   = True    # use algorithm which does not need to store array of FF_grid-sampling positions in memory (neither GPU-memory nor main-memory)
    bNoFFCopy = True    # should we copy Force-Field grid from GPU  to main_mem ?  ( Or we just copy it just internally withing GPU directly to image/texture used by RelaxedScanner )
    #bNoFFCopy = False
    bFEoutCopy = False  # should we copy un-convolved FEout from GPU to main_mem ? ( Or we should copy oly final convolved FEconv? ) 
    bMergeConv = False  # should we use merged kernel relaxStrokesTilted_convZ or two separated kernells  ( relaxStrokesTilted, convolveZ  )

    preName  = ""
    postName = ""

    n_channels = 1
    n_classes  = 10

    #Ymode = 'HeightMap'

    # --- ForceField
    #pixPerAngstrome = 10
    iZPP = 8
    Q    = 0.0

    bQZ = False
    Qs  = [100,-200,100,0]
    QZs = [0.1,0,-0.1,0]  

    # --- Relaxation
    scan_start = (-8.0,-8.0) 
    scan_end   = ( 8.0, 8.0)
    scan_dim   = ( 100, 100, 30)
    distAbove  =  6.5       # if only distAbove specified when calling generator it starts from center of top atom
    distAboveRange = None   # example of range: (6.0,6.4). If distAboveRange specified it starts from top sphere's shell: distAbove = distAbove + RvdW_top  
    #molCenterTopAtom   = False  # if setted molecule will appear not by top atom in center, but avereged center
    #molCenterBox       
    molCentering = 'topAtom'
    molCentering = 'box'
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
    randomize_enabled    = False
    randomize_nz         = True 
    randomize_parameters = True
    randomize_tip_tilt   = True
    randomize_distance   = True
    Yrange = 2

    zmin_xyz = -2.0
    Nmax_xyz = 30

    #npbc = None
    npbc = (1,1,1)

    debugPlots = False
    #debugPlots = True
    #debugPlotSlices   = [0,+2,+4,+6,+8,+10,+12,+14,+16]
    #debugPlotSlices   = [-1]
    #debugPlotSlices    = [-5,5]
    #debugPlotSlices    = [-5,5,10,15]
    debugPlotSlices    = [5,10,15]
    #debugPlotSlices   = [-10]
    #debugPlotSlices   = [-15]

    bOccl         = 0
    typeSelection =  [1,6,8]
    nChan = 8
    Rmin  = 1.4
    Rstep = 0.1

    nextMode =  1
    iZPP1    =  8
    Q1       = -0.5
    iZPP2    =  54
    Q2       = +0.5

    iepoch=0
    imol=0
    irot=0

    preHeight = False

    bDfPerMol = False
    nDfMin = 5
    nDfMax = 15

    rotJitter = None

    bRunTime = False

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
        if(verbose>0): print "Ymode", self.Ymode
        #if self.Ymode == 'Lorenzian' or self.Ymode == 'Spheres' or self.Ymode == 'SphereCaps' or self.Ymode == 'Disks' or self.Ymode == 'DisksOcclusion' or self.Ymode == 'QDisks' or self.Ymode == 'D-S-H' or self.Ymode == 'MultiMapSpheres' or self.Ymode == 'SpheresType':
        if self.Ymode in {'Lorenzian','Spheres','SphereCaps','Disks','DisksOcclusion','QDisks','D-S-H','MultiMapSpheres','SpheresType','Bonds','AtomRfunc','AtomsAndBonds'}:
            self.projector  = FFcl.AtomProcjetion()
        if self.Ymode == 'HeightMap' or self.Ymode == 'D-S-H' : 
            self.bZMap = True
        if self.Ymode == 'ElectrostaticMap':
            self.bZMap  = True
            self.bFEmap = True

        self.scanner = oclr.RelaxedScanner()
        self.scanner.relax_params = np.array( [ 0.1,0.9,0.1*0.2,0.1*5.0], dtype=np.float32 );
        self.scanner.stiffness    = np.array( [0.24,0.24,0.0, 30.0     ], dtype=np.float32 )/ -16.0217662
        #self.zWeight =  np.ones( self.scan_dim[2] )
        self.zWeight =  self.getZWeights();

    def initFF(self):
        if self.bNoPoss:
            self.forcefield.initSampling( self.lvec, pixPerAngstrome=self.pixPerAngstrome )
            #nDim = genFFSampling( self.lvec, pixPerAngstrome=self.pixPerAngstrome )
            #self.forcefield.setLvec(self.lvec, nDim=nDim )
        else:
            self.forcefield.initPoss( lvec=self.lvec, pixPerAngstrome=self.pixPerAngstrome )

        if self.bNoFFCopy:
            self.scanner.prepareBuffers( lvec=self.lvec, FEin_cl=self.forcefield.cl_FE, FEin_shape=self.forcefield.nDim,  scan_dim=self.scan_dim, 
                                         nDimConv=len(self.zWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), 
                                         bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in 
                                       )
            self.scanner.preparePosBasis( start=self.scan_start, end=self.scan_end )
        else:
            self.FEin = np.empty( self.forcefield.nDim, np.float32 )
            #print "self.FEin.shape() ", self.FEin.shape

        self.scanner.updateBuffers( WZconv=self.dfWeight )

        self.forcefield.setQs( Qs=[100,-200,100,0], QZs=[0.1,0,-0.1,0] )

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
            if self.randomize_enabled:
                np.random.shuffle( permut )
            self.molecules = [ self.molecules[i] for i in permut ]

    def getMolRotIndex(self, i):
        '''
        unfold iteration index to epoch, molecule, rotation
        '''
        #nrot = len(self.rotations)
        nrot = self.nBestRotations
        nmol = len(self.molecules)
        return i/(nrot*nmol), (i/nrot)%nmol, i%nrot

    def evalRotation(self, rot ):
        '''
        find closest atom of molecule with respect to camera rotation and respective entropy of the configuration
        '''
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

    def handleRotations(self):
        self.rotations_sorted = self.sortRotationsByEntropy()
        self.rotations_sorted = self.rotations_sorted[:self.nBestRotations]
        if self.shuffle_rotations and self.randomize_enabled:
            permut = np.array( range(len(self.rotations_sorted)) )
            np.random.shuffle( permut )
            self.rotations_sorted = [ self.rotations_sorted[i] for i in permut ]

    def next(self):
        '''
        callback for each iteration of generator
        '''
        if(bRunTime): t0=time.clock()
        if self.preHeight:
            self.bZMap  = True
        if   self.nextMode == 1:
            return self.next1()
        elif self.nextMode == 2:
            return self.next2()
        if(bRunTime): print "runTime(Generator.next()) [s]: ", time.clock()-t0

    def next1(self):
        '''
        Generate one batch of data
        for one input
        '''
        if(bRunTime): t0=time.clock()
        n  = self.batch_size
        Xs = np.empty( (n,)+ self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight),) )

        if self.Ymode == 'D-S-H':
            Ys = np.empty( (n,)+ self.scan_dim[:2] + (3,) )
        elif self.Ymode == 'MultiMapSpheres': 
            Ys = np.empty( (n,)+ self.scan_dim[:2] + (self.nChan,) )
        elif self.Ymode == 'SpheresType': 
            Ys = np.empty( (n,)+ self.scan_dim[:2] + (len(self.typeSelection),) )
        elif self.Ymode in {'ElectrostaticMap','AtomsAndBonds'}: 
            Ys = np.empty( (n,)+ self.scan_dim[:2] + (2,) )
        elif self.Ymode == 'xyz': 
            Ys = np.empty( (n,)+(self.Nmax_xyz,4) )
        else:
            Ys = np.empty( (n,)+ self.scan_dim[:2] )

        for ibatch in range(n):
            self.iepoch, self.imol, self.irot = self.getMolRotIndex( self.counter )
            if( self.irot == 0 ):# recalc FF

                if self.projector is not None:
                    self.projector.tryReleaseBuffers()

                self.molName =  self.molecules[self.imol]
                self.nextMolecule( self.molName ) 

                if self.bDfPerMol:
                    if self.randomize_nz and self.randomize_enabled : 
                        ndf = np.random.randint( self.nDfMin, self.nDfMax ) 
                    else:                      
                        ndf = self.nDfMax
                    if(verbose>0): print " ============= ndf ", ndf 
                    self.dfWeight = PPU.getDfWeight( ndf, dz=self.scanner.zstep ).astype(np.float32)

                if self.bNoFFCopy:
                    #self.scanner.prepareBuffers( lvec=self.lvec, FEin_cl=self.forcefield.cl_FE, FEin_shape=self.forcefield.nDim, 
                    #    scan_dim=self.scan_dim, nDimConv=len(self.zWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in )
                    #print "NO COPY scanner.updateFEin "
                    self.scanner.updateFEin( self.forcefield.cl_FE )
                else:
                    if(self.counter>0): # not first step
                        if(verbose>1): print "scanner.releaseBuffers()"
                        self.scanner.releaseBuffers()
                    self.scanner.prepareBuffers( self.FEin, self.lvec, scan_dim=self.scan_dim, nDimConv=len(self.zWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in )
                    self.scanner.preparePosBasis(self, start=self.scan_start, end=self.scan_end )

                self.handleRotations()
            #print " self.irot ", self.irot, len(self.rotations_sorted), self.nBestRotations

            rot = self.rotations_sorted[self.irot]
            self.nextRotation( Xs[ibatch], Ys[ibatch] )
            #self.nextRotation( self.rotations[self.irot], Xs[ibatch], Ys[ibatch] )
            self.counter +=1
        if(bRunTime): print "runTime(Generator_LJC.next1().tot        ) [s]: ", time.clock()-t0
        return Xs, Ys

    def next2(self):
        '''
        callback for each iteration of generator
        '''
        if(bRunTime): t0=time.clock()
        if self.projector is not None:
            self.projector.tryReleaseBuffers()

        self.Q    = self.Q1
        self.iZPP = self.iZPP1
        #print self.imol
        self.molName =  self.molecules[self.imol]
        self.nextMolecule( self.molName ) 
        self.handleRotations()
        #self.scanner.releaseBuffers()
        self.scanner.tryReleaseBuffers()
        self.scanner.prepareBuffers( self.FEin, self.lvec, scan_dim=self.scan_dim, nDimConv=len(self.zWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in )
        Xs1,Ys1   = self.nextRotBatch()

        self.iZPP = self.iZPP2
        self.Q    = self.Q2
        self.nextMolecule( self.molName ) 
        self.scanner.updateBuffers( FEin=self.FEin )
        Xs2,Ys2   = self.nextRotBatch()

        self.imol += 1
        self.imol =  self.imol % len(  self.molecules )
        if(bRunTime): print "runTime(Generator.next1()) [s]: ", time.clock()-t0
        return Xs1,Ys1,Xs2,Ys2

    def nextRotBatch(self):
        '''
        call per each batch
        '''
        if(bRunTime): t0=time.clock()
        n  = self.nBestRotations
        Xs = np.empty( (n,)+ self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight),) )

        if self.Ymode == 'D-S-H':
            Ys = np.empty( (n,)+ self.scan_dim[:2] + (3,) )
        elif self.Ymode == 'MultiMapSpheres': 
            Ys = np.empty( (n,)+ self.scan_dim[:2] + (self.nChan,) )
        elif self.Ymode == 'SpheresType': 
            Ys = np.empty( (n,)+ self.scan_dim[:2] + (len(self.typeSelection),) )
        elif self.Ymode in {'ElectrostaticMap','AtomsAndBonds'}: 
            Ys = np.empty( (n,)+ self.scan_dim[:2] + (2,) )
        elif self.Ymode == 'xyz': 
            Ys = np.empty( (n,)+(self.Nmax_xyz,4) )
        else:
            Ys = np.empty( (n,)+ self.scan_dim[:2] )

        self.irot = 0
        for irot in range(n):
            self.irot = irot
            rot = self.rotations_sorted[irot]
            self.nextRotation( Xs[irot], Ys[irot] )
        if(bRunTime): print "runTime(Generator.next2()) [s]: ", time.clock()-t0
        return Xs,Ys

    def calcPreHeight(self, scan_pos0s ):
        ''' 
            special where AFM tip follows path of pre-calculated heigh-map at equdistance
            should emulate https://pubs.acs.org/doi/pdf/10.1021/nl504182w 
            Imaging Three-Dimensional Surface Objects with SubmolecularResolution by Atomic Force Microscopy
            Cesar Moreno, Oleksandr Stetsovych, Tomoko K. Shimizu, Oscar Custance
        '''
        #print " self.scanner.tipRot ", self.scanner.tipRot
        #Hs = ( self.scanner.run_getZisoTilted( iso=0.1, nz=100 ) *-1 ).copy()
        dirFw = np.append( self.rot[2], [0] ); 
        poss_ = np.float32(  scan_pos0s - (dirFw*(self.distAbove-1.0))[None,None,:] )
        Hs = self.projector.run_evalSpheres( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0].copy()
        #Hs *= 1.0
        Hs = np.clip( Hs, -3.0, 0.0 )
        Hs -= 1.0

        self.scan_pos0s[:,:,0] += self.rot[2,0] * Hs
        self.scan_pos0s[:,:,1] += self.rot[2,1] * Hs
        self.scan_pos0s[:,:,2] += self.rot[2,2] * Hs
        cl.enqueue_copy( self.scanner.queue, self.scanner.cl_poss, self.scan_pos0s )
        return self.scan_pos0s

    def nextMolecule(self, fname ):
        '''
        call for each molecule
        '''
        if(bRunTime): t0=time.clock()
        fullname = self.preName+fname+self.postName
        if(verbose>0): print " ===== nextMolecule: ", fullname
        self.atom_lines = open( fullname ).readlines()
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( self.atom_lines )
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().1   ) [s]:  %0.6f" %(time.clock()-t0)    ," load atoms" 
        cog = (self.lvec[1,0]*0.5,self.lvec[2,1]*0.5,self.lvec[3,2]*0.5)
        setBBoxCenter( xyzs, cog )
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().2   ) [s]:  %0.6f" %(time.clock()-t0)    ," box,cog" 
        self.natoms0 = len(Zs)
        self.REAs = PPU.getAtomsREA(  self.iZPP, Zs, self.typeParams, alphaFac=-1.0 )
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().3   ) [s]:  %0.6f" %(time.clock()-t0)   ," REAs = getAtomsREA " 
        if self.randomize_parameters and self.randomize_enabled:
            self.modMolParams( Zs, qs, xyzs, self.REAs, self.rndQmax, self.rndRmax, self.rndEmax, self.rndAlphaMax )
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().4   ) [s]:  %0.6f" %(time.clock()-t0)    ," modMolParams  if randomize_parameters " 
        cLJs = PPU.REA2LJ( self.REAs )
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().5   ) [s]:  %0.6f" %(time.clock()-t0)    ," cLJs = REA2LJ(REAs) " 
        if( self.rotJitter is not None ):
            Zs, xyzs, qs, cLJs = PPU.multRot( Zs, xyzs, qs, cLJs, self.rotJitter, cog )
            basUtils.saveXyz( "test_____.xyz", Zs,  xyzs )
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().6   ) [s]:  %0.6f" %(time.clock()-t0)    ," rotJitter " 
        if( self.npbc is not None ):
            #Zs, xyzs, qs, cLJs = PPU.PBCAtoms3D( Zs, xyzs, qs, cLJs, self.lvec[1:], npbc=self.npbc )
            Zs, xyzqs, cLJs =  PPU.PBCAtoms3D_np( Zs, xyzs, qs, cLJs, self.lvec[1:], npbc=self.npbc )
        self.Zs = Zs
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().7   ) [s]:  %0.6f" %(time.clock()-t0)    ," pbc, PBCAtoms3D_np "    # ---- up to here it takes    ~0.012 second  for size=(150, 150, 150)
        
        if self.bNoFFCopy:
            #self.forcefield.makeFF( xyzs, qs, cLJs, FE=None, Qmix=self.Q, bRelease=False, bCopy=False, bFinish=True )
            self.forcefield.makeFF( atoms=xyzqs, cLJs=cLJs, FE=False, Qmix=self.Q, bRelease=False, bCopy=False, bFinish=True, bQZ=self.bQZ )
            self.atoms = self.forcefield.atoms
        else:
            #FF,self.atoms = self.forcefield.makeFF( xyzs, qs, cLJs, FE=None, Qmix=self.Q )       # ---- this takes   ~0.03 second  for size=(150, 150, 150)
            #FF,self.atoms  = self.forcefield.makeFF( xyzs, qs, cLJs, FE=self.FEin, Qmix=self.Q, bRelease=True, bCopy=True, bFinish=True )
            FF,self.atoms  = self.forcefield.makeFF( atoms=xyzqs, cLJs=cLJs, FE=self.FEin, Qmix=self.Q, bRelease=True, bCopy=True, bFinish=True )

            #import matplotlib.pyplot as plt
            #for i in range(0,150,5):
            #    print "save fig %i" %i, "  min,max  ",     FF[:,:,i,2].min(), FF[:,:,i,2].max()
            #    plt.imshow( FF[i,:,:,2] )
            #    plt.colorbar()
            #    plt.savefig("FF_%i.png" %i )
            #    plt.close()

        self.atomsNonPBC = self.atoms[:self.natoms0].copy()

        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().8   ) [s]:  %0.6f" %(time.clock()-t0)    ," forcefield.makeFF "
        if(bRunTime): t1 = time.clock()
                
        if( self.rotJitter is not None ):
            if self.bNoFFCopy: print "ERROR bNoFFCopy==True  is not compactible with rotJitter==True "
            FF[:,:,:,:] *= (1.0/len(self.rotJitter) )

        #self.FEin  = FF[:,:,:,:4] + self.Q*FF[:,:,:,4:];               # ---- this takes   0.05 second  for size=(150, 150, 150)
        #if(bRunTime): print "runTime(Generator_LJC.nextMolecule().5) [s]: ", time.clock()-t1

        if self.Ymode == 'ElectrostaticMap':
            if self.bNoFFCopy: print "ERROR bNoFFCopy==True is not compactible with Ymode=='ElectrostaticMap' "
            self.FE2in = FF[:,:,:,4:].copy();

        if self.projector is not None:
            na = len(self.atomsNonPBC)
            coefs=self.projector.makeCoefsZR( Zs[:na], elements.ELEMENTS )
            if   ( self.Ymode == 'MultiMapSpheres' ):
                self.projector.prepareBuffers( self.atomsNonPBC, self.scan_dim[:2]+(self.nChan,), coefs=coefs )
            elif ( self.Ymode == 'SpheresType' ):
                self.projector.prepareBuffers( self.atomsNonPBC, self.scan_dim[:2]+(len(self.typeSelection),), coefs=coefs )
                self.projector.setAtomTypes( self.Zs[:na], sel = self.typeSelection )
            elif ( self.Ymode in {'Bonds','AtomsAndBonds'} ):
                bonds2atoms = np.array( basUtils.findBonds_( self.atomsNonPBC, self.Zs, 1.2, ELEMENTS=elements.ELEMENTS ), dtype=np.int32 )
                # bonds = findBondsNP( atoms, fRcut=0.7, ELEMENTS=elements.ELEMENTS ) 
                self.projector.prepareBuffers( self.atomsNonPBC, self.scan_dim[:2]+(1,), coefs=coefs, bonds2atoms=bonds2atoms )
            else:
                self.projector.prepareBuffers( self.atomsNonPBC, self.scan_dim[:2]+(1,), coefs=coefs )

        #self.saveDebugXSF( self.preName+fname+"/FF_z.xsf", self.FEin[:,:,:,2], d=(0.1,0.1,0.1) )
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().8-9 ) [s]: ", time.clock()-t1    ," projector.prepareBuffers  "
        if(bRunTime): print "runTime(Generator_LJC.nextMolecule().tot ) [s]: ", time.clock()-t0,    " size ", self.forcefield.nDim

    #def nextRotation(self, rot, X,Y ):
    def nextRotation(self, X,Y ):
        '''
        for each rotation
        '''
        if(verbose>0): print " ----- nextRotation ", self.irot
        if(bRunTime): t0=time.clock()
        (entropy, self.pos0, self.rot) = self.rotations_sorted[self.irot]

        if(verbose>0):  print " imol, irot, entropy ", self.imol, self.irot, entropy
        zDir = self.rot[2].flat.copy()

        atoms_shifted_to_pos0 = self.atomsNonPBC[:,:3] - self.pos0[None,:]           #shift atoms coord to rotation center point of view            
        atoms_rotated_to_pos0 = rotAtoms(self.rot, atoms_shifted_to_pos0)            #rotate atoms coord to rotation center point of view
        if(verbose>1): print " atoms_rotated_to_pos0 ", atoms_rotated_to_pos0

        if(bRunTime): print "runTime(Generator_LJC.nextRotation().1   ) [s]:  %0.6f" %(time.clock()-t0)   ," atoms transform(shift,rot)  "

        # random uniform select distAbove in range distAboveRange and shift it up to radius vdW of top atom
        if self.distAboveRange is not None:
            if self.randomize_distance and self.randomize_enabled:
                self.distAbove=np.random.uniform(self.distAboveRange[0],self.distAboveRange[1])
            else:
                self.distAbove=  0.5*( self.distAboveRange[0] + self.distAboveRange[1] )
            RvdWs = self.REAs[:,0] - 1.6612  # real RvdWs of atoms after substraction of RvdW(O)
            zs = atoms_rotated_to_pos0[:,2].copy()
            zs += RvdWs  # z-coord of each atom with it's RvdW
            imax = np.argmax( zs ) 
            self.distAbove = self.distAbove + RvdWs[imax] # shifts distAbove for vdW-Radius of top atomic shell
            if(verbose>1): print "imax,distAbove ", imax, self.distAbove
        
        if(bRunTime): print "runTime(Generator_LJC.nextRotation().2   ) [s]:  %0.6f" %(time.clock()-t0)  ," top atom "

        # shift projection to molecule center but leave top atom still in the center
        AFM_window_shift=(0,0)
        if self.molCentering == 'topAtom':
            average_mol_pos = [np.mean(atoms_rotated_to_pos0[:,0]),np.mean(atoms_rotated_to_pos0[:,1])]
            if(verbose>1): print " : average_mol_pos", average_mol_pos
            top_atom_pos = atoms_rotated_to_pos0[:,[0,1]][atoms_rotated_to_pos0[:,2] == np.max(atoms_rotated_to_pos0[:,2]) ]
            if(verbose>1): print " : top_atom_pos", top_atom_pos
            #now we will move AFM window to the molecule center but still leave top atom inside window 
            AFM_window_shift = np.clip(average_mol_pos[:], a_min = top_atom_pos[:] + self.scan_start[:], a_max = top_atom_pos[:] + self.scan_end[:]) [0]
            if(verbose>1): print " : AFM_window_shift", AFM_window_shift
        elif self.molCentering == 'box':
            pmin,pmax = getBBox( atoms_rotated_to_pos0 )
            AFM_window_shift = (pmin+pmax)*0.5
      
        if(bRunTime): print "runTime(Generator_LJC.nextRotation().3   ) [s]:  %0.6f" %(time.clock()-t0)   ," molCenterAfm  "

        vtipR0    = np.zeros(3)
        if self.randomize_tip_tilt and self.randomize_enabled:
            vtipR0[0],vtipR0[1] = getRandomUniformDisk()
        else:
            vtipR0[0],vtipR0[1] = 0.0 , 0.0
        vtipR0    *= self.maxTilt0
        vtipR0[2]  = self.tipR0 

        if(bRunTime): print "runTime(Generator_LJC.nextRotation().4   ) [s]:  %0.6f" %(time.clock()-t0)   ," vtipR0  "

        #self.scanner.setScanRot( , rot=self.rot, start=self.scan_start, end=self.scan_end, tipR0=vtipR0  )
        pos0             = self.pos0+self.rot[2]*self.distAbove+np.dot((AFM_window_shift[0],AFM_window_shift[1],0),self.rot)
        self.scan_pos0s  = self.scanner.setScanRot(pos0, rot=self.rot, zstep=0.1, tipR0=vtipR0 )
        
        if(bRunTime): print "runTime(Generator_LJC.nextRotation().5   ) [s]:  %0.6f" %(time.clock()-t0)  ," scan_pos0s = scanner.setScanRot() "

        if self.preHeight: 
            self.scan_pos0s = self.calcPreHeight(self.scan_pos0s)

        if(bRunTime): print "runTime(Generator_LJC.nextRotation().6   ) [s]:  %0.6f" %(time.clock()-t0)  ," preHeight "

        if self.bMergeConv:
            FEout = self.scanner.run_relaxStrokesTilted_convZ()
            if(bRunTime): print "runTime(Generator_LJC.nextRotation().8   ) [s]:  %0.6f" %(time.clock()-t0)  ," scanner.run_relaxStrokesTilted_convZ() "
        else:
            if self.bFEoutCopy:
                FEout  = self.scanner.run_relaxStrokesTilted( bCopy=True, bFinish=True )
            else:
                #print "NO COPY scanner.run_relaxStrokesTilted "
                self.scanner.run_relaxStrokesTilted( bCopy=False, bFinish=True )
            if(bRunTime): print "runTime(Generator_LJC.nextRotation().7   ) [s]:  %0.6f" %(time.clock()-t0)  ," scanner.run_relaxStrokesTilted() "
            #print "FEout shape,min,max", FEout.shape, FEout.min(), FEout.max()
            if( len(self.dfWeight) != self.scanner.scan_dim[2] - self.scanner.nDimConvOut   ):
                print "len(dfWeight) must be scan_dim[2] - nDimConvOut ", len(self.dfWeight),  self.scanner.scan_dim[2], self.scanner.nDimConvOut
                exit()
            #self.scanner.updateBuffers( WZconv=self.dfWeight )
            FEout = self.scanner.run_convolveZ()
            if(bRunTime): print "runTime(Generator_LJC.nextRotation().8   ) [s]:  %0.6f" %(time.clock()-t0)  ," scanner.run_convolveZ() "

        nz = min( FEout.shape[2], X.shape[2] )
        X[:,:,:nz] = FEout[:,:,:nz,2]     #.copy()
        X[:,:,nz:] = 0

        if(bRunTime): print "runTime(Generator_LJC.nextRotation().9   ) [s]:  %0.6f" %(time.clock()-t0)  ," X = Fout.z  "

        dirFw = np.append( self.rot[2], [0] ); 
        if(verbose>0): print "dirFw ", dirFw
        poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAbove-1.0))[None,None,:] )

        if(bRunTime): print "runTime(Generator_LJC.nextRotation().10  ) [s]:  %0.6f" %(time.clock()-t0)  ," poss_ <- scan_pos0s  "

        # --- Different modes of output map
        if self.Ymode == 'HeightMap':
            Y[:,:] = ( self.scanner.run_getZisoTilted( iso=0.1, nz=100 ) *-1 ) . copy()
            Y *= (self.scanner.zstep)
            Ymin = max(Y[Y<=0].flatten().max() - self.Yrange, Y.flatten().min())
            Y[Y>0] = Ymin
            Y[Y<Ymin] = Ymin
            Y -= Ymin

        elif self.Ymode == 'ElectrostaticMap':
            zMap, feMap = self.scanner.run_getZisoFETilted( iso=0.1, nz=100 )
            Ye = ( feMap[:,:,2] ).copy() # Fel_z
            zMap *= -(self.scanner.zstep)
            zMin = max(zMap[zMap<=0].flatten().max() - self.Yrange, zMap.flatten().min())
            zMap[zMap>0] = zMin
            zMap[zMap<zMin] = zMin
            zMap -= zMin
            Ye[zMap == 0] = 0

            Y[:,:,0] = Ye
            Y[:,:,1] = zMap

        elif self.Ymode == 'SpheresType':
            Y[:,:,:] = self.projector.run_evalSpheresType( poss = poss_, tipRot=self.scanner.tipRot, bOccl=self.bOccl )
        elif self.Ymode == 'MultiMapSpheres':
            Y[:,:,:] = self.projector.run_evalMultiMapSpheres( poss = poss_, tipRot=self.scanner.tipRot, bOccl=self.bOccl, Rmin=self.Rmin, Rstep=self.Rstep )
        elif self.Ymode == 'Lorenzian':
            Y[:,:] =  self.projector.run_evalLorenz( poss = poss_ )[:,:,0]
        elif self.Ymode == 'Spheres':
            Y[:,:] = self.projector.run_evalSpheres( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'Bonds':
            Y[:,:] = self.projector.run_evalBondEllipses( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]   
        elif self.Ymode == 'AtomRfunc':
            Y[:,:] = self.projector.run_evalAtomRfunc( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]  
        elif self.Ymode == 'SphereCaps':
            Y[:,:] = self.projector.run_evalSphereCaps( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'Disks':
            Y[:,:] = self.projector.run_evaldisks( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'DisksOcclusion':
            Y[:,:] = self.projector.run_evaldisks_occlusion( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'QDisks':
            Y[:,:] = self.projector.run_evalQdisks( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'AtomsAndBonds':
            Y[:,:,0] = self.projector.run_evalAtomRfunc   ( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
            Y[:,:,1] = self.projector.run_evalBondEllipses( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
        elif self.Ymode == 'D-S-H':
            # Disks
            Y[:,:,0] = self.projector.run_evaldisks  ( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
            # Spheres
            Y[:,:,1] = self.projector.run_evalSpheres( poss = poss_, tipRot=self.scanner.tipRot )[:,:,0]
            # Height
            Y_  = ( self.scanner.run_getZisoTilted( iso=0.1, nz=100 ) *-1 ) . copy()
            Y_ *= (self.scanner.zstep)
            Ymin = max(Y_[Y_<=0].flatten().max() - self.Yrange, Y_.flatten().min())
            Y_[Y_>0] = Ymin
            Y_[Y_<Ymin] = Ymin
            Y_ -= Ymin
            Y[:,:,2] = Y_
        elif self.Ymode == 'xyz':
            Y[:,:] = 0.0
            Y[:,2] = self.zmin_xyz - 100.0
            xyzs = self.atomsNonPBC[:,:3] - self.pos0[None,:]
            xyzs_, Zs = getAtomsRotZminNsort( self.rot, xyzs, zmin=self.zmin_xyz, Zs=self.Zs[:self.natoms0], Nmax=self.Nmax_xyz, RvdWs = self.REAs[:,0] - 1.6612  )
            Y[:len(xyzs_),:3] = xyzs_[:,:]
            
            if self.molCenterAfm:    # shifts reference to molecule center            
                Y[:len(xyzs_),:3] -= (AFM_window_shift[0],AFM_window_shift[1],0)
                
            Y[:len(xyzs_), 3] = Zs

        if(bRunTime): print "runTime(Generator_LJC.nextRotation().tot ) [s]:  %0.6f" %(time.clock()-t0)  ," size ", FEout.shape

        if(self.debugPlots):
            print  "self.molName ", self.molName 
            list = os.listdir('model/predictions/') # dir is your directory path
            number_files = len(list)
            if (number_files < 100):
                self.plot( ("_rot%03i" % self.irot), self.molName ,  bPOVray=False, bXYZ=True , bRot=True)

    """
    # ============= Curently not used
    def getAFMinRot( self, rot, X ):
        '''
        #getAFMinRot - currently not used
        '''
        t1scan = time.clock();

        tipR0 = self.maxTilt0 * (np.random.rand(3) - 0.5); 
        tipR0[2]   = self.tipR0 # augumentation of data by varying tip

        self.scan_pos0s  = self.scanner.setScanRot( self.pos0+rot[2]*self.distAbove, rot=rot, start=self.scan_start, end=self.scan_end, tipR0=tipR0  )

        if(verbose>0): print  " >>>>>>> maxTilt0 ", self.maxTilt0, "tipR0 ", tipR0

        FEout  = self.scanner.run_relaxStrokesTilted()

        if( len(self.dfWeight) != self.scanner.scan_dim[2] - self.scanner.nDimConvOut   ):
            if(verbose>0): print "len(dfWeight) must be scan_dim[2] - nDimConvOut ", len(self.dfWeight),  self.scanner.scan_dim[2], self.scanner.nDimConvOut
            exit()
        self.scanner.updateBuffers( WZconv=self.dfWeight )
        FEout = self.scanner.run_convolveZ()
        X[:,:,:FEout.shape[2]] = FEout[:,:,:,2].copy()
    """

    def match( self, Xref ):
        return

    def saveDebugXSF( self, fname, F, d=(0.1,0.1,0.1) ):
        if hasattr(self, 'GridUtils'):
            GU = self.GridUtils
        else:
            import GridUtils as GU
            self.GridUtils = GU
        sh = F.shape
        #self.lvec_scan = np.array( [ [0.0,0.0,0.0],[self.scan_dim[0],0.0,0.0],[0.0,self.scan_dim[1],0.0],0.0,0.0, ] ] )
        lvec = np.array( [ [0.0,0.0,0.0],[sh[0]*d[0],0.0,0.0],[0.0,sh[1]*d[1],0.0], [ 0.0,0.0,sh[2]*d[2] ] ] )
        if(verbose>0): print "saveDebugXSF : ", fname
        GU.saveXSF( fname, F.transpose((2,1,0)), lvec )

    def plotGroups(self, plt, groups, xys):
        #print " >>> INSIDE :  plotGroups ", len(groups)
        plt.scatter(xys[:,0],xys[:,1],s=0.1,c="#ffffff",marker="x")
        #plt.plot(xys[:,0],xys[:,1],"o-w")
        
        for i in range(len(groups)):
            label = groups[i]
            if label != '':
                xy = xys[i]
                #print "annotate ", i, xy, label 
                plt.annotate(label, # this is the text
                    xy, # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,0), # distance from text to points (x,y)
                    ha='center')
        

    def plot(self, rotName, molName, X=None,Y=None,Y_=None, entropy=None, bXYZ=False, bPOVray=False, bRot=False, bGroups=False ):
        import matplotlib as mpl;  mpl.use('Agg');
        import matplotlib.pyplot as plt

        #extent=(self.scan_start + self.scan_end )
        extent=(self.scan_start[0],self.scan_end[0], self.scan_start[1],self.scan_end[1] )
        #print "extent: ", extent

        fname    = self.preName + molName + rotName
        #print " plot to file : ", fname

        if bXYZ:
            #self.saveDebugXSF( self.preName + self.molecules[imol] + ("/rot%03i_Y.xsf" %irot), Y_ )
             
            if bRot:
                atomsRot = rotAtoms(self.rot, self.atomsNonPBC)
                basUtils.writeDebugXYZ__('model/predictions/'+ molName + rotName+'.xyz', atomsRot, self.Zs )
                #print 'XYZ file: ', './predictions/'+ molName[6:] + rotName+'.xyz',' saved'
                #exit()
            else:
                basUtils.writeDebugXYZ_2('model/predictions/'+ molName + rotName+'.xyz', self.atoms, self.Zs, self.scan_pos0s[::40,::40,:].reshape(-1,4), pos0=self.pos0 )
                #print 'XYZ file: ', './model/predictions/'+ molName[6:] + rotName+'.xyz',' saved'
        if bPOVray:
            #basUtils.writeDebugXYZ__( self.preName + molName + rotName+".xyz", self.atomsNonPBC, self.Zs )
            bonds = basUtils.findBonds_( self.atomsNonPBC, self.Zs, 1.2, ELEMENTS=elements.ELEMENTS )
            #bonds = None
            #basUtils.writePov( self.preName + molName + rotName+".pov", self.atoms, self.Zs )
            #cam  = basUtils.makePovCam( [15,15,15], up=[0.0,10.0,0.0], rg=[-10.0, 0.0, 0.0])

            cam  = basUtils.makePovCam( self.pos0, rg=self.rot[0]*10.0, up= self.rot[1]*10.0, fw=self.rot[2]*-100.0, lpos = self.rot[2]*100.0 )
            cam += basUtils.DEFAULT_POV_HEAD_NO_CAM
            #print "makePovCam", cam
            #print "self.atomsNonPBC ", self.atomsNonPBC
            basUtils.writePov( self.preName + molName + rotName+".pov", self.atomsNonPBC, self.Zs, HEAD=cam, bonds=bonds, spherescale=0.5 )
            #basUtils.writeDebugXYZ( self.preName + molName + rotName, self.atom_lines, self.scan_pos0s[::10,::10,:].reshape(-1,4), pos0=self.pos0 )

        #self.saveDebugXSF(  self.preName + molName + rotName+"_Fz.xsf", X, d=(0.1,0.1,0.1) )

        cmap = 'viridis'

        title = "entropy  NA"
        if entropy is not None:
            title = "entropy %f" %entropy
        if Y is not None:
            plt.close()
            if self.Ymode == 'ElectrostaticMap':
                plt.figure(figsize=(5,5))
                vmax = max( Y.max(), -Y.min() )
                #plt.imshow( Y, vmin=-vmax, vmax=vmax, cmap='seismic', origin='image' );
                plt.imshow( Y, vmin=-vmax, vmax=vmax, cmap='bwr', origin='image', extent=extent );
                plt.title(title)
                plt.colorbar()
            elif self.Ymode == 'D-S-H':
                if(verbose>0):print "plot  D-S-H mode", fname, Y.shape
                plt.close()
                plt.figure(figsize=(15,5))
                #print "D-S-H Y.shape() ", Y.shape, Y[:,:,0].min(), Y[:,:,0].max(),  "  |  ",  Y[:,:,1].min(), Y[:,:,1].max(), "  |  ",   Y[:,:,2].min(), Y[:,:,2].max(),
                plt.subplot(1,3,1); plt.imshow( Y[:,:,0], origin='image', extent=extent, cmap=cmap ); plt.title("Disks");     plt.colorbar()
                plt.subplot(1,3,2); plt.imshow( Y[:,:,1], origin='image', extent=extent, cmap=cmap ); plt.title("Spheres");   plt.colorbar()
                plt.subplot(1,3,3); plt.imshow( Y[:,:,2], origin='image', extent=extent, cmap=cmap ); plt.title("HeightMap"); plt.colorbar()
            elif self.Ymode == 'AtomsAndBonds':
                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1); plt.imshow( Y[:,:,0], origin='image', extent=extent, cmap=cmap ); plt.title("AtomRfunc");     plt.colorbar()
                plt.subplot(1,2,2); plt.imshow( Y[:,:,1], origin='image', extent=extent, cmap=cmap ); plt.title("BondElipses");   plt.colorbar()
            else:
                plt.figure(figsize=(5,5))
                plt.imshow( Y, origin='image', extent=extent, cmap=cmap );
                plt.title(title)
                plt.colorbar()
            
            if bGroups:
                import chemistry as chem
                Zs = self.Zs[:self.natoms0]
                xyzs  = self.atomsNonPBC[:,:3] - self.pos0[None,:]
                xyzs_ = rotAtoms(self.rot,xyzs)
                #print "xyzs_.shape", xyzs_.shape
                bonds  = chem.findBonds( xyzs_, Zs, fR=1.3 )
                #print bonds
                neighs = chem.bonds2neighsZs(bonds, Zs )
                #print neighs
                groups = chem.neighs2str( Zs, neighs, bPreText=True )
                #print  groups
                self.plotGroups(plt, groups, xyzs_[:,:2] )
            
            #print "Y = ", Y
            #plt.imshow( Y, vmin=-5, vmax=5, origin='image', extent=extent );  
            #plt.close()
            #plt.figure()
            plt.savefig(  fname+"Dens.png", bbox_inches="tight"  );
            #plt.savefig(  fname+"Dens.png", bbox_inches="tight"  ); 
            plt.close()

        for isl in self.debugPlotSlices:
            #plt.imshow( FEout[:,:,isl,2], extent=extent )
            if (X is not None) and (Y_ is not None):
                plt.figure(figsize=(10,5))
                #print isl, np.min(X[:,:,isl]), np.max(X[:,:,isl])
                plt.subplot(1,2,2); plt.imshow (X [:,:,isl], origin='image', extent=extent, cmap=cmap );            plt.colorbar()
                #plt.subplot(1,2,1); plt.imshow (Y_[:,:,isl], origin='image', extent=extent, cmap=cmap );
                plt.subplot(1,2,1); plt.imshow (np.tanh(Y_[:,:,isl]), origin='image', extent=extent, cmap=cmap );   plt.colorbar()
                plt.title(title)
                plt.savefig( fname+( "FzFixRelax_iz%03i.png" %isl ), bbox_inches="tight"  ); 
                plt.close()
            else:
                if X is not None:
                    if(verbose>0): print isl, np.min(X[:,:,isl]), np.max(X[:,:,isl])
                    plt.imshow(  X[:,:,isl], origin='image', extent=extent, cmap=cmap );    plt.colorbar()
                    plt.savefig(  fname+( "Fz_iz%03i.png" %isl ), bbox_inches="tight"  ); 
                    plt.close()
                if Y_ is not None:
                    plt.imshow ( Y_[:,:,isl], origin='image', extent=extent, cmap=cmap );   plt.colorbar()
                    plt.savefig( fname+( "FzFix_iz%03i.png" %isl ), bbox_inches="tight"  ); 
                    plt.close()

    def getZWeights(self):
        '''
        generate mask for weighted average
        '''
        zs = np.mgrid[0:self.scan_dim[2]] * self.scanner.zstep * self.wz
        zWeights = self.zFunc( zs )
        zWeights = zWeights.astype(np.float32)
        return zWeights

    def getDistDens(self, atoms, poss, F):
        '''
        CPU calculation of distance density map ( used before OpenCL )
        '''
        F[:,:] = 0
        for atom in atoms[:1]:
            ## TODO - we should project it with respect to rotation
            dposs =  poss - atom[None,None,:]
            #F[:,:] += self.r2Func( (dposs[:,:,0]**2 + dposs[:,:,1]**2)/self.wr ) # * self.zFunc( dposs[:,:,2]/self.wz )
            F[:,:] += self.r2Func( (dposs[:,:,0]**2 + dposs[:,:,1]**2 + dposs[:,:,2]**2 )/self.wr )




# ==========================================================
# ==========================================================
# ====================== TEST RUN ==========================
# ==========================================================
# ==========================================================



if __name__ == "__main__":
    import matplotlib as mpl;    mpl.use('Agg');
    import matplotlib.pyplot as plt
    #import argparse
    #import datetime
    import os
    #from shutil import copyfile
    import subprocess
    from optparse import OptionParser

    #import sys
    #sys.path.append("/u/21/oinonen1/unix/PPMOpenCL")
    #sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")
    #sys.path.append("/home/prokop/git/ProbeParticleModel")
    #from   pyProbeParticle import basUtils
    #import pyProbeParticle.common    as PPU
    #import pyProbeParticle.GridUtils as GU
    #import pyopencl as cl
    #import pyProbeParticle.HighLevelOCL as hl
    #import pyProbeParticle.GeneratorOCL_LJC as PPGen
    #import pyProbeParticle.GeneratorOCL_LJC_RR as PPGen
    #from pyProbeParticle.GeneratorOCL_LJC import Generator
    PPGen = current_module = sys.modules[__name__]

    # ============ Setup Probe Particle

    batch_size = 1
    nRot           = 1
    nBestRotations = 1

    #molecules = ["out2", "out3","benzeneBrCl2"]
    molecules = ["benzeneBrCl2"]

    parser = OptionParser()
    parser.add_option( "-Y", "--Ymode", default='D-S-H', action="store", type="string", help="tip stiffenss [N/m]" )
    (options, args) = parser.parse_args()

    print "options.Ymode: ", options.Ymode

    #rotations = PPU.genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, nRot) )

    #rotations = PPU.sphereTangentSpace(n=nRot) # http://blog.marmakoide.org/?p=1
    rotations  = PPU.genRotations( np.array([0.,0.,1.]), np.arange( -np.pi, np.pi, 2*np.pi/nRot ) )
    #rotations = np.array( [ [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], ] )


    #import os
    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    bPlatformInfo = True
    if bPlatformInfo:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        print "######################################################################"
        print 
        env.printInfo()
        print 
        print "######################################################################"
        print 
        env.printPlatformInfo()
        print 
        print "######################################################################"

    #make data generator
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='HeightMap' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='ElectrostaticMap' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='Lorenzian' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='Disks' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='QDisks' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='DisksOcclusion' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='Spheres' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='Spheres' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='SphereCaps' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='D-S-H' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='xyz' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='MultiMapSpheres' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='SpheresType' )
    data_generator  = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode=options.Ymode  )


    #data_generator.use_rff      = True
    #data_generator.save_rff_xyz = True

    # --- 'MultiMapSpheres' and 'SpheresType' settings
    data_generator.bOccl = 1   # switch occlusion of atoms 0=off 1=on

    # --- 'SpheresType' setting
    #data_generator.typeSelection =  [1,6,8,16]  # select atom types for each output channel
    data_generator.typeSelection =  [1,6,7,8,16,33]  # select atom types for each output channel

    # --- 'MultiMapSpheres' settings  ztop[ichan] = (R - Rmin)/Rstep
    data_generator.nChan = 5      # number of channels, resp. atom size bins
    data_generator.Rmin  = 1.4    # minimum atom radius
    data_generator.Rstep = 0.2    # size range per bin (resp. cGenerator.nextRotationnel)

    data_generator.zmin_xyz = -2.0  # max depth for visible atoGenerator.nextRotation
    data_generator.Nmax_xyz = 3    # max number of visible atomGenerator.nextRotation

    #data_generator.preHeight = True

    data_generator.projector.Rpp  = -0.5
    '''
    #data_generator.projector.zmin = -3.0
    data_generator.projector.zmin  = -1.5
    data_generator.projector.dzmax = 2.0
    data_generator.projector.tgMax = 0.6
    data_generator.projector.tgWidth = 0.1
    '''

    xs = np.linspace(0.0,10.0,100)
    dx = xs[1]-xs[0];
    xs -= dx
    ys = np.exp( -5*xs )

    data_generator.projector.Rfunc   = ys.astype(np.float32)
    data_generator.projector.invStep = dx
    data_generator.projector.Rmax    = xs[-1] - 3*dx
    plt.figure()
    plt.plot(xs,data_generator.projector.Rfunc); plt.grid()
    plt.savefig( "Rfunc.png" )
    plt.close()
    

    #data_generator.rotJitter = PPU.makeRotJitter(10, 0.3)

    #data_generator.npbc = None    # switch of PeriodicBoundaryConditions


    # --- params randomization 
    data_generator.randomize_enabled    = False
    data_generator.randomize_nz         = True 
    data_generator.randomize_parameters = True
    data_generator.randomize_tip_tilt   = True
    data_generator.randomize_distance   = True
    data_generator.rndQmax     = 0.1    # charge += rndQmax * ( rand()-0.5 )  (negative is off)
    data_generator.rndRmax     = 0.2    # charge += rndRmax * ( rand()-0.5 )  (negative is off)
    data_generator.rndEmax     = 0.5    # charge *= (1 + rndEmax     * ( rand()-0.5 )) (negative is off)
    data_generator.rndAlphaMax = -0.1   # charge *= (1 + rndAlphaMax * ( rand()-0.5 )) (negative is off)
    #data_generator.modMolParams = modMolParams_def   # custom function to modify parameters

    #data_generator.debugPlots = True
    #data_generator.distAbove = 7.5
    #data_generator.distAbove = 8.0
    #data_generator.distAbove = 8.5
    #data_generator.distAbove = 9.0
    #data_generator.distAbove = 8.5
    #data_generator.distAbove = 9.0
    data_generator.distAboveRange = [7.0,7.01]


    data_generator.bQZ = False
    data_generator.Qs  = np.array([100,-200,100,0])*-0.2
    data_generator.QZs = [0.1,0.0,-0.1,0]  


    #data_generator.maxTilt0 = 0.0    # symmetric tip
    data_generator.maxTilt0 = 0.5     # asymetric tip  tilted max +/-1.0 Angstreom in random direction
    #data_generator.maxTilt0 = 2.0     # asymetric tip  tilted max +/-1.0 Angstreom in random direction

    data_generator.shuffle_rotations = False
    data_generator.shuffle_molecules = False
    data_generator.nBestRotations    = nBestRotations

    # molecule are read from filename =  preName + molecules[imol] + postName
    data_generator.preName    = ""           # appended befroe path filename
    data_generator.postName   = "/pos.xyz"

    #data_generator.debugPlots = True   # plotting dubug images? much slower when True
    #data_generator.Q = -0.5;
    #data_generator.Q = -0.3;
    data_generator.Q = -0.1;

    # z-weight exp(-wz*z)
    data_generator.wz      = 1.0    # deacay
    data_generator.zWeight =  data_generator.getZWeights();

    # weight-function for Fz -> df conversion ( oscilation amplitude 1.0Angstroem = 10 * 0.1 ( 10 n steps, dz=0.1 Angstroem step lenght ) )
    dfWeight = PPU.getDfWeight( 10, dz=0.1 ).astype(np.float32)
    data_generator.dfWeight = dfWeight

    # plot zWeights
    plt.figure()
    plt.plot(data_generator.zWeight);
    plt.savefig( "zWeights.png" )
    plt.close()

    # plot dfWeights
    plt.figure()
    plt.plot(data_generator.dfWeight);
    plt.savefig( "dfWeights.png" )
    plt.close()
    #plt.show()

    # print
    #data_generator.bDfPerMol = True
    #data_generator.nDfMin    = 5
    #data_generator.nDfMax    = 15


    #data_generator.scan_dim = ( 100, 100, 20)
    #data_generator.scan_dim = ( 128, 128, 30)
    data_generator.scan_dim   = ( 256, 256, 30)
    data_generator.scan_start = (-12.5,-12.5) 
    data_generator.scan_end   = ( 12.5, 12.5)

    bRunTime      = True
    FFcl.bRuntime = True

    data_generator            .verbose  = 1
    data_generator.forcefield .verbose  = 1
    data_generator.scanner    .verbose  = 1

    data_generator.initFF()

    # generate 10 batches
    for i in range(1):

        print "#### generate ", i 
        t1 = time.clock()
        Xs,Ys = data_generator[i]
        print "runTime(data_generator.next()) [s] : ", time.clock() - t1
        
        #continue

        #Xs=Xs[::2]; Ys=Ys[::2]

        '''
        print "Ys.shape ", Ys.shape

        for i in range( Ys[0].shape[2] ):
            plt.figure()
            plt.imshow( Ys[0][:,:,i] )
            plt.title( "img[%i]" %i )

        plt.show()
        '''
        #exit()

        #print "_0"
        
        data_generator.debugPlotSlices = range(0,Xs[0].shape[2],2)

        for j in range( len(Xs) ):
            #print "_1"
            #print "j ", j
            #np.save( "X_i%03i_j%03i.npy" %(i,j), Xs[j] )
            #np.save( "Y_i%03i_j%03i.npy" %(i,j), Ys[j] )
            #print "Ys[j].shape", Ys[j].shape
            fname = "batch_%03i_%03i_" %(i,j)

            #for ichan in range( Ys[j].shape[2] ):
            #    plt.figure()
            #    plt.imshow( Ys[j][:,:,ichan] )
            #    plt.title( "i %i j %i ichan %i" %(i,j,ichan) )

            #nch = Ys[j].shape[2]
            #plt.figure(figsize=(5*nch,5))
            #for ichan in range( nch ):
            #    plt.subplot(  1, nch, ichan+1 )
            #    plt.imshow( Ys[j][:,:,ichan] )
            #    plt.title( "i %i j %i ichan %i" %(i,j,ichan) )

            #data_generator.plot( "/"+fname, molecules[i*batch_size+j], X=Xs[j], Y=Ys[j], entropy=0.0, bXYZ=True )
            #data_generator.plot( "/"+fname, molecules[data_generator.imol], X=Xs[j], Y=Ys[j], entropy=0.0, bXYZ=True, bGroups=True )
            data_generator.plot( "/"+fname, molecules[data_generator.imol], X=Xs[j], Y=Ys[j], entropy=0.0, bXYZ=False, bGroups=False )
            #print "_2"
            #data_generator.plot( "/"+fname, molecules[data_generator.imol], X=None, Y=Ys[j], entropy=0.0, bXYZ=True )

            #print Ys[j]if __name__ == "__main__":

            '''
            fname = "batch_%03i_%03i_" %(i,j)
            data_generator.plot( "/"+fname, molecules[0], Y=Ys[j], entropy=0.0, bPOVray=True, bXYZ=True, bRot=True )
            #subprocess.run("povray Width=800 Height=800 Antialias=On Antialias_Threshold=0.3 Output_Alpha=true %s" %(fname+'.pov') )
            subprocess.call("povray Width=800 Height=800 Antialias=On Antialias_Threshold=0.3 Output_Alpha=true %s" %(fname+'.pov') )
            povname = "./"+molecules[0]+"/"+fname+'.pov'
            cwd = os.getcwd()
            print ">>>>> cwd = os.getcwd() ", cwd
            print ">>>>> povname : ", povname
            os.system( "povray Width=800 Height=800 Antialias=On Antialias_Threshold=0.3 Output_Alpha=true %s" %povname )
            '''
    plt.show()

    '''
    ====== Timing Results ======== ( Machine: GPU:   |  CPU: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz )
    runTime(Generator_LJC.nextMolecule().1   ) [s]:  0.000192  load atoms
    runTime(Generator_LJC.nextMolecule().2   ) [s]:  0.000264  box,cog
    runTime(Generator_LJC.nextMolecule().3   ) [s]:  0.000633  REAs = getAtomsREA 
    runTime(Generator_LJC.nextMolecule().4   ) [s]:  0.000640  modMolParams  if randomize_parameters 
    runTime(Generator_LJC.nextMolecule().5   ) [s]:  0.000684  cLJs = REA2LJ(REAs) 
    runTime(Generator_LJC.nextMolecule().6   ) [s]:  0.000690  rotJitter 
    runTime(Generator_LJC.nextMolecule().7   ) [s]:  0.000981  pbc, PBCAtoms3D_np 
    runTime(Generator_LJC.nextMolecule().8   ) [s]:  0.002383  forcefield.makeFF                 !!!!!!!!!!!!!!    
    runTime(Generator_LJC.nextMolecule().8-9 ) [s]:  0.001321  projector.prepareBuffers  
    runTime(Generator_LJC.nextMolecule().tot ) [s]:  0.003778  size  [150 150 150   4]
    runTime(Generator_LJC.nextRotation().1   ) [s]:  0.000124  atoms transform(shift,rot)  
    runTime(Generator_LJC.nextRotation().2   ) [s]:  0.000206  top atom 
    runTime(Generator_LJC.nextRotation().3   ) [s]:  0.000229  molCenterAfm  
    runTime(Generator_LJC.nextRotation().4   ) [s]:  0.000267  vtipR0  
    runTime(Generator_LJC.nextRotation().5   ) [s]:  0.002020  scan_pos0s = scanner.setScanRot() 
    runTime(Generator_LJC.nextRotation().6   ) [s]:  0.002110  preHeight 
    runTime(Generator_LJC.nextRotation().7   ) [s]:  0.003439  scanner.run_relaxStrokesTilted()  !!!!!!!!!!!!!!
    runTime(Generator_LJC.nextRotation().8   ) [s]:  0.007660  scanner.run_convolveZ()           !!!!!!!!!!!!!!
    runTime(Generator_LJC.nextRotation().9   ) [s]:  0.010177  X = Fout.z  
    runTime(Generator_LJC.nextRotation().10  ) [s]:  0.011105  poss_ <- scan_pos0s  
    runTime(Generator_LJC.nextRotation().tot ) [s]:  0.013272  size  (128, 128, 20, 4)
    runTime(Generator_LJC.next1().tot        ) [s]:  0.017792
    '''
