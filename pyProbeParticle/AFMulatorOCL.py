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

import numpy as np
#from keras.utils import Sequence

verbose  = 0
bRunTime = False

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
    return pmin, pmax, pc

def getAtomsRotZminNsort( rot, xyzs, zmin, RvdWs=None, Zs=None, Nmax=30, RvdW_H = 1.4870 ):
    '''
    get <=Nmax atoms closer to camera than "zmin" and sort by distance from camera
    '''
    xyzs_ = rotAtoms(rot, xyzs )
    zs = xyzs_[:,2].copy()
    if RvdWs is not None:
        zs += RvdWs
    zs = zs - RvdW_H
    inds = np.argsort( -zs ) #.copy()
    xyzs_ = xyzs_[inds,:].copy()
    zs    = zs   [inds  ].copy()
    mask  = zs > zmin
    xyzs_ = xyzs_[mask,:]
    if Zs is not None:
        Zs = Zs[inds]
        Zs = Zs[mask]
    return xyzs_[:Nmax,:], Zs[:Nmax] 

class Sequence:
    pass

class AFMulator(Sequence,):

    bNoPoss    = True    # use algorithm which does not need to store array of FF_grid-sampling positions in memory (neither GPU-memory nor main-memory)
    bNoFFCopy  = True    # should we copy Force-Field grid from GPU  to main_mem ?  ( Or we just copy it just internally withing GPU directly to image/texture used by RelaxedScanner )
    bFEoutCopy = False  # should we copy un-convolved FEout from GPU to main_mem ? ( Or we should copy oly final convolved FEconv? ) 
    bMergeConv = False  # should we use merged kernel relaxStrokesTilted_convZ or two separated kernells  ( relaxStrokesTilted, convolveZ  )

    #npbc = None
    npbc = (1,1,1)

    # --- ForceField
    iZPP = 8
    Q    = 0.0

    # --- multiplole tip
    bQZ = True
    QZs = [0.1,0,-0.1,0  ]     #  position tip charges along z-axis relative to probe-particle center
    Qs  = [100,-200,100,0]     #  values of tip charges in electrons, len()==4, some can be zero, 

    iZPP1    =  8
    Q1       = -0.5
    iZPP2    =  54
    Q2       = +0.5

    # --- Relaxation
    maxTilt0     = 0.0
    tipR0        = 4.0
    tipStiffness = [ 0.25, 0.25, 0.0, 30.0     ]    # [N/m]  (x,y,z,R) stiffness of harmonic springs anchoring probe-particle to the metalic tip-apex 
    relaxParams  = [ 0.5,0.1,  0.1*0.2,0.1*5.0 ]    # (dt,damp, .., .. ) parameters of relaxation, in the moment just dt and damp are used 
    scan_start   = (-8.0,-8.0) 
    scan_end     = ( 8.0, 8.0)
    scan_dim     = ( 100, 100, 30)
    distAbove    =  7.0       # distAbove starts from top sphere's shell: distAbove = distAbove + RvdW_top  

    # ---- Y-Modes (AuxMaps) specific
    zmin_xyz = -2.0
    Nmax_xyz = 30
    diskMode = 'center' # 'center' or 'sphere'

    # --- this atom-types recognition
    #n_channels = 1               #   ??
    #n_classes  = 10              #   ??
    bOccl         = 0             # switch occlusion of atoms 0=off 1=on
    typeSelection =  [1,6,8]      # for Ymode == ''SpheresType''
    nChan = 8                     # for Ymode == 'MultiMapSpheres'
    Rmin  = 1.4
    Rstep = 0.1

    # --- Debug
    bSaveFF    = False     # save debug ForceField as .xsf  ?
    debugPlots = False     # plot debug .png images of AFM and AuxMap ?
    debugPlotSlices      = [5,10,15]   #   which slices to plot ? 

    bDfPerMol = False     # recalculate dfWeight per each molecule ?
    #nDfMin = 5
    #nDfMax = 15

    # ==================== Methods =====================

    def __init__( self, pixPerAngstrome=10, lvec=None, Ymode=None ):
        'Initialization'
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

        self.typeParams = hl.loadSpecies('atomtypes.ini')
        self.ff_dim     = hl.genFFSampling( self.lvec, self.pixPerAngstrome );  #print "ff_dim ",     self.ff_dim
        #self.ff_poss    = FFcl.getposs    ( self.lvec, self.ff_dim );           print "poss.shape ", self.ff_poss.shape  # should we store this?

        self.forcefield = FFcl.ForceField_LJC()

        self.Ymode     = Ymode
        self.projector = None; 
        self.FE2in     = None
        self.bZMap     = False; 
        self.bFEmap    = False;
        if(verbose>0): print("Ymode", self.Ymode)
        #if self.Ymode == 'Lorenzian' or self.Ymode == 'Spheres' or self.Ymode == 'SphereCaps' or self.Ymode == 'Disks' or self.Ymode == 'DisksOcclusion' or self.Ymode == 'QDisks' or self.Ymode == 'D-S-H' or self.Ymode == 'MultiMapSpheres' or self.Ymode == 'SpheresType':
        if self.Ymode in {'Lorenzian','Spheres','SphereCaps','Disks','DisksOcclusion','QDisks','D-S-H','MultiMapSpheres','SpheresType','Bonds','AtomRfunc','AtomsAndBonds'}:
            self.projector  = FFcl.AtomProcjetion()
        if self.Ymode == 'HeightMap' or self.Ymode == 'D-S-H' : 
            self.bZMap = True
        if self.Ymode == 'ElectrostaticMap':
            self.bZMap  = True
            self.bFEmap = True
        self.scanner = oclr.RelaxedScanner()
        self.scanner.relax_params = np.array( self.relaxParams, dtype=np.float32 );
        self.scanner.stiffness    = np.array( self.tipStiffness, dtype=np.float32 )/ -16.0217662
        #self.zWeight =  self.getZWeights();
        self.counter = 0

    def initFF(self):
        if self.bNoPoss:
            self.forcefield.initSampling( self.lvec, pixPerAngstrome=self.pixPerAngstrome )
        else:
            self.forcefield.initPoss( lvec=self.lvec, pixPerAngstrome=self.pixPerAngstrome )
        if self.bNoFFCopy:
            self.scanner.prepareBuffers( lvec=self.lvec, FEin_cl=self.forcefield.cl_FE, FEin_shape=self.forcefield.nDim,  scan_dim=self.scan_dim, 
                                         nDimConv=len(self.dfWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), 
                                         bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in 
                                       )
            self.scanner.preparePosBasis( start=self.scan_start, end=self.scan_end )
        else:
            self.FEin = np.empty( self.forcefield.nDim, np.float32 )
        self.scanner.updateBuffers( WZconv=self.dfWeight )
        self.forcefield.setQs( Qs=self.Qs, QZs=self.QZs )

    def getTsDim(self, Ymode=None):
        if Ymode is None:
            Ymode=self.Ymode 
        if Ymode == 'D-S-H':
            return self.scan_dim[:2] + (3,)
        elif Ymode == 'MultiMapSpheres':
            return self.scan_dim[:2] + (self.nChan,)
        elif Ymode == 'SpheresType': 
            return self.scan_dim[:2] + (len(self.typeSelection),)
        elif Ymode in {'ElectrostaticMap','AtomsAndBonds'}: 
            return self.scan_dim[:2] + (2,)
        elif Ymode == 'xyz': 
            return (self.Nmax_xyz, 4 )
        else:
            return self.scan_dim[:2]

    def prepareMolecule_AFM(self, xyzs, Zs, qs ):
        cog = np.array([self.lvec[1,0]*0.5,self.lvec[2,1]*0.5,self.lvec[3,2]*0.5])
        print( " prepareMolecule cog ", cog )
        #setBBoxCenter( xyzs, cog ) #    ToDo : This introduces jitter !!!!!!
        xyzs = xyzs[:,:] + cog[None,:]

        self.natoms0 = len(Zs)
        self.REAs    = PPU.getAtomsREA(  self.iZPP, Zs, self.typeParams, alphaFac=-1.0 )
        cLJs = PPU.REA2LJ( self.REAs )
        if( self.npbc is not None ):
            #Zs, xyzs, qs, cLJs = PPU.PBCAtoms3D( Zs, xyzs, qs, cLJs, self.lvec[1:], npbc=self.npbc )
            Zs, xyzqs, cLJs =  PPU.PBCAtoms3D_np( Zs, xyzs, qs, cLJs, self.lvec[1:], npbc=self.npbc )
        self.Zs = Zs
        if self.bNoFFCopy:
            #self.forcefield.makeFF( xyzs, qs, cLJs, FE=None, Qmix=self.Q, bRelease=False, bCopy=False, bFinish=True )
            self.forcefield.makeFF( atoms=xyzqs, cLJs=cLJs, FE=False, Qmix=self.Q, bRelease=False, bCopy=False, bFinish=True, bQZ=self.bQZ )
            self.atoms = self.forcefield.atoms
            if self.bSaveFF:
                FF = self.forcefield.downloadFF( )
                FFx=FF[:,:,:,0]
                FFy=FF[:,:,:,1]
                FFz=FF[:,:,:,2]
                Fr = np.sqrt( FFx**2 + FFy**2 + FFz**2 )
                Fbound = 10.0
                mask = Fr.flat > Fbound
                FFx.flat[mask] *= (Fbound/Fr).flat[mask]
                FFy.flat[mask] *= (Fbound/Fr).flat[mask]
                FFz.flat[mask] *= (Fbound/Fr).flat[mask]
                print("FF.shape ", FF.shape)
                self.saveDebugXSF_FF( "FF_x.xsf", FFx )
                self.saveDebugXSF_FF( "FF_y.xsf", FFy )
                self.saveDebugXSF_FF( "FF_z.xsf", FFz )
                #self.saveDebugXSF_FF( "FF_E.xsf", FF[:,:,:,3] )
        else:
            FF,self.atoms  = self.forcefield.makeFF( atoms=xyzqs, cLJs=cLJs, FE=self.FEin, Qmix=self.Q, bRelease=True, bCopy=True, bFinish=True )
        self.atomsNonPBC = self.atoms[:self.natoms0].copy()

        if self.Ymode == 'ElectrostaticMap':
            if self.bNoFFCopy: print("ERROR bNoFFCopy==True is not compactible with Ymode=='ElectrostaticMap' ")
            self.FE2in = FF[:,:,:,4:].copy();

    def prepareMolecule_AuxMap(self, xyzs, Zs, qs ):

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

    #def prepareMolecule_AFMandAuxMap(self, xyzs, Zs, qs ):
    #    self.prepareMolecule_AFM    ( xyzs, Zs, qs )
    #    self.prepareMolecule_AuxMap ( xyzs, Zs, qs )
    def prepareImaging(self, xyzs, Zs, qs ):

        self.initFF()

        if self.projector is not None:
            self.projector.tryReleaseBuffers()

        self.prepareMolecule_AFM( xyzs, Zs, qs )
        if self.Ymode is not None:
            self.prepareMolecule_AuxMap( xyzs, Zs, qs )

        if self.bDfPerMol: 
            self.dfWeight = PPU.getDfWeight( ndf, dz=self.scanner.zstep )[0].astype(np.float32)

        if self.bNoFFCopy:
            self.scanner.updateFEin( self.forcefield.cl_FE )
        else:
            if(self.counter>0): # not first step
                if(verbose>1): print("scanner.releaseBuffers()")
                self.scanner.releaseBuffers()
            self.scanner.prepareBuffers( self.FEin, self.lvec, scan_dim=self.scan_dim, nDimConv=len(self.dfWeight), nDimConvOut=self.scan_dim[2]-len(self.dfWeight), bZMap=self.bZMap, bFEmap=self.bFEmap, FE2in=self.FE2in )
            self.scanner.preparePosBasis(self, start=self.scan_start, end=self.scan_end )
        
        self.scan_pos0s = None

    def prepare_Rotation( self, rot ):
        #if(verbose>0): print(" ----- imageRotation ", self.irot)
        zDir = rot[2].flat.copy()
        self.distAboveActive  = self.distAbove
        vtipR0    = np.zeros(3)
        vtipR0[2]  = self.tipR0 
        self.scan_pos0s  = self.scanner.setScanRot(self.pos0, rot=rot, zstep=0.1, tipR0=vtipR0 )

    def evalAFM( self, rot, X=None ):

        if self.scan_pos0s is None:
            self.prepare_Rotation(rot)

        if self.bMergeConv:
            FEout = self.scanner.run_relaxStrokesTilted_convZ()
        else:
            if self.bFEoutCopy:
                FEout  = self.scanner.run_relaxStrokesTilted( bCopy=True, bFinish=True )
            else:
                #print "NO COPY scanner.run_relaxStrokesTilted "
                self.scanner.run_relaxStrokesTilted( bCopy=False, bFinish=True )
            #print "FEout shape,min,max", FEout.shape, FEout.min(), FEout.max()
            if( len(self.dfWeight) != self.scanner.scan_dim[2] - self.scanner.nDimConvOut   ):
                print("len(dfWeight) must be scan_dim[2] - nDimConvOut ", len(self.dfWeight),  self.scanner.scan_dim[2], self.scanner.nDimConvOut)
                exit()
            #self.scanner.updateBuffers( WZconv=self.dfWeight )
            FEout = self.scanner.run_convolveZ()

        #print "DEBUG FEout.max,min ", FEout[:,:,:,:3].max(), FEout[:,:,:,:3].min() 

        nz = min( FEout.shape[2], X.shape[2] )
        X[:,:,:nz] = FEout[:,:,:nz,2]     #.copy()
        X[:,:,nz:] = 0
        self.counter += 1
        return X

    def evalAuxMap(self, rot, Y=None ):

        if(bRunTime): print(  " evalAuxMap Ymode = ", self.Ymode )

        if self.scan_pos0s is None:
            self.prepare_Rotation(rot)

        # shift projection to molecule center but leave top atom still in the center
        AFM_window_shift=(0,0)
        self.RvdWs = self.REAs[:,0] - 1.6612

        dirFw = np.append( rot[2], [0] ); 
        if(verbose>0): print("dirFw ", dirFw)
        if self.Ymode not in ['HeightMap', 'ElectrostaticMap', 'xyz']:
            #poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAboveActive-self.RvdWs[imax]-self.projector.Rpp))[None,None,:] )
            poss_ = np.float32(  self.scan_pos0s - (dirFw*(self.distAboveActive))[None,None,:] )

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
            #offset = np.dot( poss_[0,0,:3]-self.atomsNonPBC[np.argmax(atoms_rotated_to_pos0[:,2]),:3], rot[2]) # Distance from top atom to screen
            offset = np.dot( poss_[0,0,:3]-self.pos0, rot[2]) # Distance from top atom to screen
            if self.diskMode   == 'sphere':
                Y[:,:,0] = self.projector.run_evaldisks  ( poss = poss_, tipRot=self.scanner.tipRot, offset=offset )[:,:,0]
            elif self.diskMode == 'center':
                self.projector.dzmax_s = np.Inf
                #offset = np.dot(poss_[0,0,:3]-self.atomsNonPBC[np.argmax(atoms_rotated_to_pos0[:,2]),:3], rot[2]) - 1.0
                offset -= 1.0
                Y[:,:,0] = self.projector.run_evaldisks  ( poss = poss_, tipRot=self.scanner.tipRot, offset=offset )[:,:,0]
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
            xyzs_, Zs = getAtomsRotZminNsort( rot, xyzs, zmin=self.zmin_xyz, Zs=self.Zs[:self.natoms0], Nmax=self.Nmax_xyz, RvdWs = self.RvdWs  )
            Y[:len(xyzs_),:3] = xyzs_[:,:]
            
            if self.molCenterAfm:    # shifts reference to molecule center            
                Y[:len(xyzs_),:3] -= (AFM_window_shift[0],AFM_window_shift[1],0)
                
            Y[:len(xyzs_), 3] = Zs

        if(self.debugPlots):
            print("self.molName ", self.molName) 
            list = os.listdir('model/predictions/') # dir is your directory path
            number_files = len(list)
            if (number_files < 100):
                self.plot( ("_rot%03i" % self.irot), self.molName ,  bPOVray=False, bXYZ=True , bRot=True)
        
        return Y

    def evalAFMandAuxMap(self, rot, X=None, Y=None ):
        X = self.evalAFM   ( self, rot, X=None )
        Y = self.evalAuxMap( self, rot, Y=None )
        return X, Y

    #def performImaging_AFM(self, molecule, rotMat ):
    def performImaging_AFM(self, xyzs,Zs,qs, rotMat ):
        X = np.empty( self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight),) )
        self.prepareImaging(  xyzs,Zs,qs )
        #self.imageRotation_simple( X, Y, rotMat )
        self.evalAFM( rotMat, X )
        self.scan_pos0s = None
        return X
    
    def performImaging_AFM_(self, mol, rotMat ):
        return self.performImaging_AFM( mol.xyzs,mol.Zs,mol.qs, rotMat )

    #def performImaging_AFMandAuxMap(self, molecule, rotMat ):
    def performImaging_AFMandAuxMap(self, xyzs,Zs,qs, rotMat ):
        X = np.empty( self.scan_dim[:2] + (self.scan_dim[2] - len(self.dfWeight),) )
        Y = np.empty( self.getTsDim() )
        self.prepareImaging(  xyzs,Zs,qs )
        #self.evalAFMandAuxMap( rotMat, X, Y )
        self.evalAFM   ( rotMat, X=X )
        self.evalAuxMap( rotMat, Y=Y )
        #self.imageRotation_simple( X, Y, rotMat )
        self.scan_pos0s = None
        return X, Y

    def performImaging_AFMandAuxMap_(self, mol, rotMat ):
        return self.performImaging_AFMandAuxMap( mol.xyzs,mol.Zs,mol.qs, rotMat )

    # ================== Debug/Plot Misc.

    def saveDebugXSF( self, fname, F, d=(0.1,0.1,0.1) ):
        if hasattr(self, 'GridUtils'):
            GU = self.GridUtils
        else:
            from . import GridUtils as GU
            self.GridUtils = GU
        sh = F.shape
        #self.lvec_scan = np.array( [ [0.0,0.0,0.0],[self.scan_dim[0],0.0,0.0],[0.0,self.scan_dim[1],0.0],0.0,0.0, ] ] )
        lvec = np.array( [ [0.0,0.0,0.0],[sh[0]*d[0],0.0,0.0],[0.0,sh[1]*d[1],0.0], [ 0.0,0.0,sh[2]*d[2] ] ] )
        if(verbose>0): print("saveDebugXSF : ", fname)
        GU.saveXSF( fname, F.transpose((2,1,0)), lvec )

    def saveDebugXSF_FF( self, fname, F ):
        if hasattr(self, 'GridUtils'):
            GU = self.GridUtils
        else:
            from . import GridUtils as GU
            self.GridUtils = GU
        sh = F.shape
        #self.lvec_scan = np.array( [ [0.0,0.0,0.0],[self.scan_dim[0],0.0,0.0],[0.0,self.scan_dim[1],0.0],0.0,0.0, ] ] )
        #lvec = np.array( [ [0.0,0.0,0.0],[sh[0]*d[0],0.0,0.0],[0.0,sh[1]*d[1],0.0], [ 0.0,0.0,sh[2]*d[2] ] ] )
        if(verbose>0): print("saveDebugXSF : ", fname)
        #GU.saveXSF( fname, F.transpose((2,1,0)), self.lvec )
        GU.saveXSF( fname, F, self.lvec )

    """
    def getZWeights(self):
        '''
        generate mask for weighted average (convolution), e.g. for Fz -> df conversion 
        '''
        zs = np.mgrid[0:self.scan_dim[2]] * self.scanner.zstep * self.wz
        zWeights = self.zFunc( zs )
        zWeights = zWeights.astype(np.float32)
        return zWeights
    """

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

    batch_size     = 1
    nRot           = 1
    nBestRotations = 1
    molecules = ["out2", "out3","benzeneBrCl2"]

    parser = OptionParser()
    parser.add_option( "-Y", "--Ymode", default='D-S-H', action="store", type="string", help="tip stiffenss [N/m]" )
    (options, args) = parser.parse_args()

    print("options.Ymode: ", options.Ymode)

    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    lvec=None

    afmulator = AFMulator( pixPerAngstrome=5, Ymode=options.Ymode, lvec=lvec )

    afmulator.bOccl = 1   # switch occlusion of atoms 0=off 1=on
    #afmulator.typeSelection =  [1,6,8,16]  # select atom types for each output channel
    afmulator.typeSelection =  [1,6,7,8,16,33]  # select atom types for each output channel

    # --- 'MultiMapSpheres' settings  ztop[ichan] = (R - Rmin)/Rstep
    afmulator.nChan = 5      # number of channels, resp. atom size bins
    afmulator.Rmin  = 1.4    # minimum atom radius
    afmulator.Rstep = 0.2    # size range per bin (resp. cGenerator.nextRotationnel)

    afmulator.zmin_xyz = -2.0  # max depth for visible atoGenerator.nextRotation
    afmulator.Nmax_xyz = 3    # max number of visible atomGenerator.nextRotation

    #afmulator.preHeight = True
    afmulator.projector.Rpp  = -0.5

    xs = np.linspace(0.0,10.0,100)
    dx = xs[1]-xs[0];
    xs -= dx
    ys = np.exp( -5*xs )

    afmulator.projector.Rfunc   = ys.astype(np.float32)
    afmulator.projector.invStep = dx
    afmulator.projector.Rmax    = xs[-1] - 3*dx
    
    #afmulator.debugPlots = True
 
    afmulator.bQZ = True
    afmulator.Qs  = np.array([100,-200,100,0]) * -0.2   # Quadrupole Qdz2-1.0[e*A^2]
    afmulator.maxTilt0 = 0.5     # asymetric tip  tilted max +/-1.0 Angstreom in random direction

    # molecule are read from filename =  preName + molecules[imol] + postName
    afmulator.preName    = ""           # appended befroe path filename
    afmulator.postName   = "/pos.xyz"

    afmulator.Q = 0.0

    '''
    # z-weight exp(-wz*z)
    afmulator.wz      = 1.0    # deacay
    afmulator.zWeight =  afmulator.getZWeights();
    '''

    dz=0.1

    # weight-function for Fz -> df conversion ( oscilation amplitude 1.0Angstroem = 10 * 0.1 ( 10 n steps, dz=0.1 Angstroem step lenght ) )
    dfWeight = PPU.getDfWeight( 10, dz=dz )[0].astype(np.float32)
    #dfWeight, xs = PPU.getDfWeight( 10, dz=dz )
    #print " xs ", xs
    afmulator.dfWeight = dfWeight

    afmulator.bSaveFF    = False
    afmulator.bMergeConv = True

    afmulator.scan_dim   = ( 256, 256, 30)
    afmulator.scan_start = (-12.5,-12.5) 
    afmulator.scan_end   = ( 12.5, 12.5)

    bRunTime      = True
    FFcl.bRuntime = True

    afmulator            .verbose  = 1
    afmulator.forcefield .verbose  = 1
    afmulator.scanner    .verbose  = 1

    afmulator.initFF()

    # ----- Init Plots

    plt.figure()
    plt.plot(xs,afmulator.projector.Rfunc); plt.grid()
    plt.savefig( "Rfunc.png" )
    plt.close()

    # plot zWeights
    plt.figure()
    plt.plot(afmulator.zWeight, '.-');
    plt.grid()
    plt.savefig( "zWeights.png" )
    plt.close()

    # plot dfWeights
    plt.figure()
    plt.plot( np.arange(len(afmulator.dfWeight))*dz , afmulator.dfWeight, '.-');
    plt.grid()
    plt.savefig( "dfWeights.png" )
    plt.close()
    #plt.show()

    # ----- Loop over molecules
    for i in range(9):

        print("#### generate ", i) 
        t1 = time.clock()

        #Xs,Ys = afmulator.performImaging_AFMandAuxMap( xyzs, Zs, qs, rotMat )
        print("runTime(afmulator.next()) [s] : ", time.clock() - t1)
        #afmulator.debugPlotSlices = list(range(0,Xs[0].shape[2],1))
        afmulator.prepareImaging( xyzs, Zs, qs, )

        # ----- Loop over rotations
        for j in range( len(Xs) ):
            X,Y = afmulator.evalAFMandAuxMap( rotMat )

            fname = "batch_%03i_%03i_" %(i,j)
            print(" Ys[j].shape",  Ys[j].shape)
            #np.save(  "./"+molecules[afmulator.imol]+"/Atoms.npy", Ys[j][:,:,0] )
            #np.save(  "./"+molecules[afmulator.imol]+"/Bonds.npy", Ys[j][:,:,1] )
            #afmulator.plot( "/"+fname, molecules[afmulator.imol], X=Xs[j], Y=Ys[j], entropy=0.0, bXYZ=False, bGroups=False )


    plt.show()

