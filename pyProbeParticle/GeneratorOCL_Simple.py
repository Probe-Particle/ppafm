#!/usr/bin/python

# Refrences:
# - Keras Data Generator   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

import sys
import os
import shutil
import time
import random
import matplotlib;
import numpy as np
from enum import Enum

#import matplotlib.pyplot as plt

import pyopencl     as cl

from . import basUtils
from . import common    as PPU
from . import elements
from . import oclUtils     as oclu 
from . import fieldOCL     as FFcl 
from . import RelaxOpenCL  as oclr
from . import HighLevelOCL as hl
from . import AFMulatorOCL

import numpy as np

verbose  = 0
bRunTime = False

def getRandomUniformDisk():
    '''
    generate points unifromly distributed over disk
    # see: http://mathworld.wolfram.com/DiskPointPicking.html
    '''
    rnd = np.random.rand(2)
    rnd[0]    = np.sqrt( rnd[0] ) 
    rnd[1]   *= 2.0*np.pi
    return  rnd[0]*np.cos(rnd[1]), rnd[0]*np.sin(rnd[1])

def applyZWeith( F, zWeight ):
    '''
    Weighted average of 1D array
    '''
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

def getAtomsRotZmin( rot, xyzs, zmin, Zs=None ):
    '''
    get all atoms closer to camera than "zmin"
    '''
    xyzs_ = AFMulatorOCL.rotAtoms(rot, xyzs )
    mask  =  xyzs_[:,2] > zmin
    if Zs is not None:
        Zs = Zs[mask]
    return xyzs_[mask,:], Zs

class InverseAFMtrainer(AFMulatorOCL.AFMulator,):

    preName  = ""
    postName = ""
    iepoch=0
    imol=0
    irot=0
    nextMode =  1

    #n_channels = 1
    #n_classes  = 10
    #Ymode = 'HeightMap'

    distAboveDelta = None   # in example distAbovedelta=0.1 makes distAbove: rand(distAbove - 0.1,distAbove + starts from top sphere's shell: distAbove = distAbove + RvdW_top   
    molCentering = 'topAtom'
    molCentering = 'box'
    planeShift = -4.0

    maxTilt0 = 0.0
    tipR0    = 4.0
    wz = 1.0

    # ---- Atom Distance Density
    r2Func = staticmethod( lambda r2 : 1/(1.0+r2) )
    zFunc  = staticmethod( lambda x  : np.exp(-x)  )

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

    preHeight = False

    nDfMin = 5
    nDfMax = 15

    rotJitter = None
    bRunTime = False

    diskMode = 'center' # or 'sphere'

    def __init__(self, molecules, rotations, batch_size=32, pixPerAngstrome=10, lvec=None, Ymode='HeightMap' ):
        super().__init__( pixPerAngstrome=pixPerAngstrome, lvec=lvec, Ymode=Ymode )
        # --- params randomization
        self.rndQmax  = 0.1 
        self.rndRmax  = 0.2
        self.rndEmax  = 0.5
        self.rndAlphaMax = -0.1
        #self.modMolParams = staticmethod(modMolParams_def)
        self.modMolParams = modMolParams_def
        self.molecules  = molecules
        self.rotations  = rotations
        self.batch_size = batch_size
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int( np.ceil( len(self.molecules) * self.nBestRotations / float(self.batch_size) ) )

    def __getitem__(self, index):
        if(verbose>0): print("index ", index)
        return next(self)

    def on_epoch_end(self):
        if self.shuffle_molecules:
            permut = np.array( list(range(len(self.molecules))) )
            if self.randomize_enabled:
                np.random.shuffle( permut )
            self.molecules = [ self.molecules[i] for i in permut ]

    def getMolRotIndex(self, i):
        '''
        unfold iteration index to epoch, molecule, rotation
        '''
        nrot = self.nBestRotations
        nmol = len(self.molecules)
        if (nrot*nmol) == 0 : return 0,0,0
        return i//(nrot*nmol), (i//nrot)%nmol, i%nrot

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
            permut = np.array( list(range(len(self.rotations_sorted))) )
            np.random.shuffle( permut )
            self.rotations_sorted = [ self.rotations_sorted[i] for i in permut ]

    def __next__(self):
        '''
        callback for each iteration of generator
        '''
        if self.preHeight:
            self.bZMap  = True
        if   self.nextMode == 1:
            return self.next1()
        elif self.nextMode == 2:
            return self.next2()

    def next1(self):
        '''
        Generate one batch of data
        for one input
        '''
        n  = self.batch_size
        Xs = np.empty( (n,)+ self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight),) )
        Ys = np.empty( (n,) + self.getTsDim() )

        for ibatch in range(n):
            self.iepoch, self.imol, self.irot = self.getMolRotIndex( self.counter )
            if( self.irot == 0 ):# recalc FF

                if self.projector is not None:
                    self.projector.tryReleaseBuffers()

                print( "self.molecules[self.imol] ", self.molecules, self.imol )
                self.molName =  self.molecules[self.imol]
                self.nextMolecule( self.molName ) 

                if self.bDfPerMol:
                    if self.randomize_nz and self.randomize_enabled : 
                        ndf = np.random.randint( self.nDfMin, self.nDfMax ) 
                    else:                      
                        ndf = self.nDfMax
                    if(verbose>0): print(" ============= ndf ", ndf) 
                    self.dfWeight = PPU.getDfWeight( ndf, dz=self.scanner.zstep )[0].astype(np.float32)

                if self.bNoFFCopy:
                    #self.scanner.prepareBuffers( lvec=self.lvec, FEin_cl=self.forcefield.cl_FE, FEin_shape=self.forcefield.nDim, 
                    #    scan_dim=self.scan_dim, nDimConv=len(self.zWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in )
                    #print "NO COPY scanner.updateFEin "
                    self.scanner.updateFEin( self.forcefield.cl_FE )
                else:
                    if(self.counter>0): # not first step
                        if(verbose>1): print("scanner.releaseBuffers()")
                        self.scanner.releaseBuffers()
                    self.scanner.prepareBuffers( self.FEin, self.lvec, scan_dim=self.scan_dim, nDimConv=len(self.zWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in )
                    self.scanner.preparePosBasis(self, start=self.scan_start, end=self.scan_end )

                self.handleRotations()
            #print " self.irot ", self.irot, len(self.rotations_sorted), self.nBestRotations

            rot = self.rotations_sorted[self.irot]
            self.nextRotation( Xs[ibatch], Ys[ibatch] )
            #self.nextRotation( self.rotations[self.irot], Xs[ibatch], Ys[ibatch] )
            self.counter +=1
        return Xs, Ys

    def next2(self):
        '''
        callback for each iteration of generator
        '''
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
        return Xs1,Ys1,Xs2,Ys2

    def nextRotBatch(self):
        '''
        call per each batch
        '''
        n  = self.nBestRotations
        Xs = np.empty( (n,)+ self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight),) )
        Ys = np.empty( (n,) + self.getTsDim() )

        self.irot = 0
        for irot in range(n):
            self.irot = irot
            rot = self.rotations_sorted[irot]
            self.nextRotation( Xs[irot], Ys[irot] )
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
        poss_ = np.float32(  scan_pos0s - (dirFw*(self.distAboveActive-1.0))[None,None,:] )
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
        fullname = self.preName+fname+self.postName
        if(verbose>0): print(" ===== nextMolecule: ", fullname)
        self.atom_lines = open( fullname ).readlines()
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( self.atom_lines )

        # --------- Randomize molecular structure - ToDo: needs to execute within
        if( self.rotJitter is not None ):
            Zs, xyzs, qs, cLJs = PPU.multRot( Zs, xyzs, qs, cLJs, self.rotJitter, cog )
            basUtils.saveXyz( "test_____.xyz", Zs,  xyzs )
        if self.randomize_parameters and self.randomize_enabled:
            self.modMolParams( Zs, qs, xyzs, self.REAs, self.rndQmax, self.rndRmax, self.rndEmax, self.rndAlphaMax )
        REAs = PPU.getAtomsREA(  self.iZPP, Zs, self.typeParams, alphaFac=-1.0 )

        # TODO - RotJitter
        if( self.rotJitter is not None ):
            if self.bNoFFCopy: print("ERROR bNoFFCopy==True  is not compactible with rotJitter==True ")
            FF[:,:,:,:] *= (1.0/len(self.rotJitter) )

        self.prepareMolecule( xyzs,Zs,qs, REAs=REAs  )

    #def nextRotation(self, rot, X,Y ):
    def nextRotation(self, X,Y ):
        '''
        for each rotation
        '''
        (entropy, self.pos0, self.rot) = self.rotations_sorted[self.irot]
        if(verbose>0):  print(" imol, irot, entropy ", self.imol, self.irot, entropy)
        self.adjustImagingFrame( self.rot      )
        self.evalAFM           ( self.rot, X=X )
        self.evalAuxMap        ( self.rot, Y=Y )

    def adjustImagingFrame(self, rot):

        atoms_shifted_to_pos0 = self.atomsNonPBC[:,:3] - self.pos0[None,:]           #shift atoms coord to rotation center point of view            
        atoms_rotated_to_pos0 = AFMulatorOCL.rotAtoms(rot, atoms_shifted_to_pos0)            #rotate atoms coord to rotation center point of view
        if(verbose>1): print(" atoms_rotated_to_pos0 ", atoms_rotated_to_pos0)

        # select distAboveActive randomly uniform in range [distAbove-distAboveDelta,distAbove+distAboveDelta] and shift it up to radius vdW of top atom
        if self.randomize_distance and self.randomize_enabled and self.distAboveDelta:
            self.distAboveActive=np.random.uniform(self.distAbove - self.distAboveDelta,self.distAbove + self.distAboveDelta)
        else:
            self.distAboveActive = self.distAbove
         
        RvdWs = self.REAs[:,0] - 1.6612  # real RvdWs of atoms after substraction of RvdW(O)
        zs    = atoms_rotated_to_pos0[:,2].copy()
        zs   += RvdWs  # z-coord of each atom with it's RvdW
        imax  = np.argmax( zs ) 
        #poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAboveActive-RvdWs[imax]-self.projector.Rpp))[None,None,:] )
        #self.distAboveActive  = self.distAbove
        self.distAboveActive  = self.distAboveActive + RvdWs[imax] - self.projector.Rpp # shifts distAboveActive for vdW-Radius of top atomic shell
        if(verbose>1): print("imax,distAboveActive ", imax, self.distAboveActive)        
        atoms_rotated_to_pos0 = AFMulatorOCL.rotAtoms(rot, self.atomsNonPBC[:,:3] - self.atomsNonPBC[imax,:3])  #New top atom

        # shift projection to molecule center but leave top atom still in the center
        AFM_window_shift=(0,0)
        if self.molCentering == 'topAtom':
            average_mol_pos = [np.mean(atoms_rotated_to_pos0[:,0]),np.mean(atoms_rotated_to_pos0[:,1])]
            if(verbose>1): print(" : average_mol_pos", average_mol_pos)
            top_atom_pos = atoms_rotated_to_pos0[:,[0,1]][atoms_rotated_to_pos0[:,2] == np.max(atoms_rotated_to_pos0[:,2]) ]
            if(verbose>1): print(" : top_atom_pos", top_atom_pos)
            #now we will move AFM window to the molecule center but still leave top atom inside window 
            AFM_window_shift = np.clip(average_mol_pos[:], a_min = top_atom_pos[:] + self.scan_start[:], a_max = top_atom_pos[:] + self.scan_end[:]) [0]
            if(verbose>1): print(" : AFM_window_shift", AFM_window_shift)
        elif self.molCentering == 'box':
            pmin,pmax = getBBox( atoms_rotated_to_pos0 )
            AFM_window_shift = (pmin+pmax)*0.5
      
        vtipR0    = np.zeros(3)
        if self.randomize_tip_tilt and self.randomize_enabled:
            vtipR0[0],vtipR0[1] = getRandomUniformDisk()
        else:
            vtipR0[0],vtipR0[1] = 0.0 , 0.0
        vtipR0    *= self.maxTilt0
        vtipR0[2]  = self.tipR0 

        #self.scanner.setScanRot( , rot=rot, start=self.scan_start, end=self.scan_end, tipR0=vtipR0  )
        pos0             = self.atomsNonPBC[imax,:3]+rot[2]*self.distAboveActive+np.dot((AFM_window_shift[0],AFM_window_shift[1],0),rot)

        self.pos0 = pos0
        self.scan_pos0s  = self.scanner.setScanRot(pos0, rot=rot, zstep=0.1, tipR0=vtipR0 )
        if self.preHeight: 
            self.scan_pos0s = self.calcPreHeight(self.scan_pos0s)

    def match( self, Xref ):
        return

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
                atomsRot = AFMulatorOCL.rotAtoms(self.rot, self.atomsNonPBC)
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
                if(verbose>0):print("plot  D-S-H mode", fname, Y.shape)
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
                from . import chemistry as chem
                Zs = self.Zs[:self.natoms0]
                xyzs  = self.atomsNonPBC[:,:3] - self.pos0[None,:]
                xyzs_ = AFMulatorOCL.rotAtoms(self.rot,xyzs)
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
                    if(verbose>0): print(isl, np.min(X[:,:,isl]), np.max(X[:,:,isl]))
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

    PPGen = current_module = sys.modules[__name__]

    # ============ Setup Probe Particle

    batch_size = 1
    nRot           = 1
    nBestRotations = 1

    #molecules = ["formic_acid"]
    #molecules = ["out3"]
    molecules = ["out2", "out3","benzeneBrCl2"]
    #molecules = ["benzeneBrCl2"]

    parser = OptionParser()
    parser.add_option( "-Y", "--Ymode", default='D-S-H', action="store", type="string", help="tip stiffenss [N/m]" )
    (options, args) = parser.parse_args()

    print("options.Ymode: ", options.Ymode)

    #rotations = PPU.genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, nRot) )

    #rotations = PPU.sphereTangentSpace(n=nRot) # http://blog.marmakoide.org/?p=1
    #rotations  = PPU.genRotations( np.array([0.,0.,1.]), np.arange( -np.pi, np.pi, 2*np.pi/nRot ) )
    rotations = np.array( [ [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], ] )


    #import os
    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    '''
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
    '''

    '''
    lvec = np.array([
        [0.0,0.0,0.0],
        [15.0,0.0,0.0],
        [0.0,15.0,0.0],
        [0.0,0.0,15.0],
    ])
    '''
    lvec=None


    data_generator  = InverseAFMtrainer( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode=options.Ymode, lvec=lvec  )

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
    data_generator.distAbove = 7.0
    #data_generator.distAbove = 8.0
    #data_generator.distAbove = 8.5
    #data_generator.distAbove = 9.0
    #data_generator.distAbove = 8.5
    #data_generator.distAbove = 9.0
    data_generator.distAboveDelta = None  

    data_generator.bQZ = True
    #data_generator.QZs = [0.1,0.0,-0.1,0]
    #data_generator.Qs  = np.array([0,1,0,0]) * -1.0         # Monopole   Qs-1.0[e]
    #data_generator.Qs  = np.array([10,0,-10,0]) * +1.0      # Dipole     Qpz-1.0[e*A]
    data_generator.Qs  = np.array([100,-200,100,0]) * -0.2   # Quadrupole Qdz2-1.0[e*A^2]

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
    data_generator.Q = 0.0

    # z-weight exp(-wz*z)
    data_generator.wz      = 1.0    # deacay
    data_generator.zWeight =  data_generator.getZWeights();

    dz=0.1

    # weight-function for Fz -> df conversion ( oscilation amplitude 1.0Angstroem = 10 * 0.1 ( 10 n steps, dz=0.1 Angstroem step lenght ) )
    dfWeight = PPU.getDfWeight( 10, dz=dz )[0].astype(np.float32)
    #dfWeight, xs = PPU.getDfWeight( 10, dz=dz )
    #print " xs ", xs
    data_generator.dfWeight = dfWeight

    # plot zWeights
    plt.figure()
    plt.plot(data_generator.zWeight, '.-');
    plt.grid()
    plt.savefig( "zWeights.png" )
    plt.close()

    # plot dfWeights
    plt.figure()
    plt.plot( np.arange(len(data_generator.dfWeight))*dz , data_generator.dfWeight, '.-');
    plt.grid()
    plt.savefig( "dfWeights.png" )
    plt.close()
    #plt.show()

    # print
    #data_generator.bDfPerMol = True
    #data_generator.nDfMin    = 5
    #data_generator.nDfMax    = 15

    data_generator.bSaveFF = False
    data_generator.bMergeConv = True

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
    for i in range(9):

        print("#### generate ", i) 
        t1 = time.clock()
        Xs,Ys = data_generator[i]
        print("runTime(data_generator.next()) [s] : ", time.clock() - t1)
        
        '''
        print "Ys.shape ", Ys.shape

        for i in range( Ys[0].shape[2] ):
            plt.figure()
            plt.imshow( Ys[0][:,:,i] )
            plt.title( "img[%i]" %i )

        plt.show()
        '''
        
        #data_generator.debugPlotSlices = range(0,Xs[0].shape[2],2)
        data_generator.debugPlotSlices = list(range(0,Xs[0].shape[2],1))

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

            print(" Ys[j].shape",  Ys[j].shape)

            np.save(  "./"+molecules[data_generator.imol]+"/Atoms.npy", Ys[j][:,:,0] )
            np.save(  "./"+molecules[data_generator.imol]+"/Bonds.npy", Ys[j][:,:,1] )

            #continue

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