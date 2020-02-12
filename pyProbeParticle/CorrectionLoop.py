#!/usr/bin/python

'''
Idea is to improve prediction of gemeotry using physical generative model

1) CNN prediction of Bonds&Atoms
2) MM  relaxation (with predicted Bonds&Atoms as external potential)
3) PPM simulation (based on MM geometry)
4) recognize errors (reference AFM data-stack needed)
5) Generation of improved structure (How? Using CNN?)
    -> repeat from (2)

see:  https://mega.nz/#!KLoilKIB!NxxCRQ814xtCXfjy7mPFfmJTOL9TaTHbmPKSxn_0sFs

'''

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

from . import atomicUtils as au
from . import basUtils
from . import common    as PPU
from . import elements
from . import oclUtils     as oclu 
from . import fieldOCL     as FFcl 
from . import RelaxOpenCL  as oclr
from . import HighLevelOCL as hl

from . import GeneratorOCL_LJC #as Imager
from . import FARFF            #as Relaxer

verbose  = 0
bRunTime = False

# ========================================================================
class Sequence:
    pass

class SampleStructure():
    def __init__(self ):
        pass

class Critique():

    izPlot = -8
    logImgName = None

    def __init__(self ):
        self.best = None
        pass

    def modifyStructure(self, geomIn ):
        geom = geomIn.copy()
        geom[0,0] += 0.1
        geom[1,0] -= 0.1
        # --- ToDo: Relaxation Should be possible part of relaxation ????
        return geom

    def try_improve(self, geomIn, AFMs, AFMRef, itr=0 ):
        #print( " AFMs ", AFMs.shape, " AFMRef ", AFMRef.shape )
        AFMdiff = AFMs - AFMRef

        if self.logImgName is not None:
            plt.figure(figsize=(5*3,5))
            plt.subplot(1,3,1); plt.imshow( AFMs   [:,:,self.izPlot] ); plt.title("AFM[]"  )
            plt.subplot(1,3,2); plt.imshow( AFMRef [:,:,self.izPlot] ); plt.title("AFMref" )
            plt.subplot(1,3,3); plt.imshow( AFMdiff[:,:,self.izPlot] ); plt.title("AFMdiff")
            plt.savefig( self.logImgName+("_%03i.png" %itr), bbox_inches='tight')
            plt.close  ()

        Err  = np.sqrt( np.sum(AFMdiff**2) )  # root mean square error
        # ToDo : identify are of most difference and make random changes in that area
        if ( self.best is None ) or ( self.best.Err > Err ):
            self.best_structure = geomIn
        print( "Critique.try_improve Err2 ", Err  )
        geomOut = self.modifyStructure( self.best_structure )
        return Err, geomOut

# ========================================================================

def removeAtoms( xyzs, Zs, qs, p0, R=3.0, nmax=1 ):
    print( "----- removeAtoms  p0 ", p0 )
    rs   = (xyzs[:,0]-p0[0])**2 + (xyzs[:,1]-p0[1])**2
    #mask = rs<R
    sel = rs.argsort()[-3:] # [::-1]
    #print( "removeAtom.sel ", sel )
    #xyzs_ = xyzs[sel].copy()
    #Zs_   = Zs  [mask].copy()
    #qs_   = qs  [mask].copy()
    xyzs_ = np.delete( xyzs, sel, axis=0 )
    Zs_   = np.delete( Zs,   sel )
    qs_   = np.delete( qs,   sel )
    return xyzs_, Zs_, qs_

def addAtom( xyzs, Zs, qs, p0, R=0.0, Z0=1, q0=0.0, dq=0.0 ):
    print( "----- addAtom  p0 ", p0 )
    dp    = (np.random.rand(3)-0.5)*R
    dp[2] = 0
    Zs_   = np.append( Zs,   np.array([Z0,]),      axis=0 )
    xyzs_ = np.append( xyzs, np.array([p0 + dp,]), axis=0 )
    qs_   = np.append( qs,   np.array([q0 + (np.random.rand()-0.5)*dq,]), axis=0 )
    return xyzs_, Zs_, qs_

class Mutator():
    '''
    strategy of mutation:
    - change one atom
        - position
        - radius
        - height
    - change many atoms
    '''
    strategies = [ removeAtoms, addAtom ]
    probs      = np.array([ 0.5, 0.5 ])

    def __init__(self ):
        self.best = None
        self.setProbs()
        pass

    #def modAtom():

    def setProbs(self, probs=None ):
        if probs is not None:
            probs = np.array(probs)
            self.probs    = probs
        self.cumProbs = np.cumsum( self.probs )

    '''
    def randCarge(self, xyzs, sc=[0.2,0.2,0.1], sel=None ):

    def randXyzs(self, xyzs, sc=[0.2,0.2,0.1], sel=None ):
        if sel is not None:
            dxyzs = np.random.rand( xyzs.shape ) - 0.5
        else:
            dxyzs      = np.zeros( xyzs.shape ) - 0.5
            dxyzs[sel] = np.random.rand( xyzs[sel].shape ) - 0.5
        dxyzs[:,0] *= sc[0]
        dxyzs[:,1] *= sc[1]
        dxyzs[:,2] *= sc[2]
        return xyzs + dxyzs
    '''

    def mutate_local(self, xyzs, Zs, qs, p0, R ):
        toss = np.random.rand()
        i = np.searchsorted( self.cumProbs, toss )
        return self.strategies[i]( xyzs, Zs, qs,  p0, R )
        #return xyzs.copy(), Zs.copy(), qs.copy()

class CorrectorTrainer():

    restartProb = 0.1
    maxIndex    = 10000
    rotMat = np.array([[1.,0,0],[0.,1.,0],[0.,0,1.]])

    def __init__(self, simulator, mutator, molCreator=None ):
        self.index = 0
        self.simulator  = simulator
        self.mutator    = mutator
        self.molCreator = molCreator

    def start(self, xyzs=None, Zs=None, qs=None ):
        if xyzs is None:
            self.xyzs, self.Zs, self.qs = self.molCreator.create()
        else:
            self.xyzs = xyzs
            self.Zs   = Zs
            self.qs   = qs

    def generatePair(self):
        #Xs1,Ys1      = self.simulator.perform_imaging( self.xyzs.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        Xs1      = self.simulator.perform_just_AFM( self.xyzs.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        p0 = (np.random.rand(3) - 0.5) * 10.0 #
        R  = 3.0   
        self.xyzs, self.Zs, self.qs = self.mutator.mutate_local( xyzs, Zs, qs, p0, R )
        #Xs2,Ys2      = self.simulator.perform_imaging( self.xyzs.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        Xs2     = self.simulator.perform_just_AFM( self.xyzs.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        if self.molCreator is not None:
            if np.random.rand(1) < self.restartProb:
                self.xyzs, self.Zs, self.qs = self.molCreator.create()
        return Xs1, Xs2

    def __getitem__(self, index):
        self.index = index
        if(verbose>0): print("index ", index)
        return next(self)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index <= self.maxIndex:
            self.index += 1
            return self.generatePair()
        else:
            raise StopIteration

# ========================================================================

class CorrectionLoop():

    logAFMdataName = None
    logImgName = None
    logImgIzs  = [0,-8,-1] 

    rotMat = np.array([[1.,0,0],[0.,1.,0],[0.,0,1.]])

    def __init__(self, relaxator, simulator, critique ):
        self.simulator = simulator
        self.relaxator = relaxator
        self.critique  = critique
        self.xyzLogName = None

    def init(self):
        pass

    def startLoop(self, guess, Zs, qs, atomMap, bondMap, lvecMap, AFMRef ):
        self.xyzs    = xyzs
        self.Zs      = Zs
        self.qs      = qs
        self.atomMap = atomMap
        self.bondMap = bondMap
        self.mapLvec = lvecMap
        self.AFMRef  = AFMRef

    def iteration(self, itr=0 ):
        print( "### CorrectionLoop.iteration [1]" )
        print( "########### bond atom ---- min min ", self.atomMap.min(), self.bondMap.min() )
        self.relaxed    = self.relaxator.preform_relaxation ( self.xyzs, self.Zs, self.qs, self.mapLvec, self.atomMap, self.bondMap )
        if self.xyzLogFile is not None:
           # au.saveXYZ( self.Zs, self.xyzs, self.xyzLogName, qs=self.qs )
           au.writeToXYZ( self.xyzLogFile, self.Zs, self.xyzs, qs=self.qs, commet=("CorrectionLoop.iteration [%i] " %itr) )

        print( "### CorrectionLoop.iteration [%i]" %itr )
        AFMs,AuxMap     = self.simulator.perform_imaging( self.relaxed.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        if self.logAFMdataName:
            np.save( self.logAFMdataName+("%03i.dat" %itr), AFMs )
        if self.logImgName is not None:
            nz = len(self.logImgIzs)
            plt.figure(figsize=(5*nz,5))
            for iiz, iz in enumerate(self.logImgIzs):
                plt.subplot(1,nz,iiz+1); plt.imshow ( AFMs[:,:,iz] )
                plt.title( "iz %i" %iz )
            plt.savefig( self.logImgName+("_%03i.png" %itr), bbox_inches='tight')
            plt.close  ()

        print( "### CorrectionLoop.iteration [3]" )
        Err, self.xyzs = self.critique.try_improve( self.relaxed, AFMs, self.AFMRef, itr=itr )
        print( "### CorrectionLoop.iteration [4]" )
        return Err
        #Xs,Ys      = simulator.next1( self )

# ========================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print( " UNIT_TEST START : CorrectionLoop ... " )
    #import atomicUtils as au

    print("# ------ Init Generator   ")

    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    #GeneratorOCL_LJC.bRunTime = True

    lvec=None
    simulator  = GeneratorOCL_LJC.Generator( [], [], 1, pixPerAngstrome=5, Ymode='AtomsAndBonds', lvec=lvec  )

    simulator.bOccl = 1   # switch occlusion of atoms 0=off 1=on
    simulator.typeSelection =  [1,6,7,8,16,33]  # select atom types for each output channel
    simulator.nChan = 5      # number of channels, resp. atom size bins
    simulator.Rmin  = 1.4    # minimum atom radius
    simulator.Rstep = 0.2    # size range per bin (resp. cGenerator.nextRotationnel)
    simulator.zmin_xyz = -2.0  # max depth for visible atoGenerator.nextRotation
    simulator.Nmax_xyz = 3    # max number of visible atomGenerator.nextRotation

    simulator.projector.Rpp  = -0.5
    xs = np.linspace(0.0,10.0,100)
    dx = xs[1]-xs[0];
    xs -= dx
    ys = np.exp( -5*xs )

    simulator.projector.Rfunc   = ys.astype(np.float32)
    simulator.projector.invStep = dx
    simulator.projector.Rmax    = xs[-1] - 3*dx
    # --- params randomization 
    #simulator.randomize_enabled    = False
    #simulator.randomize_nz         = True 
    #simulator.randomize_parameters = True
    #simulator.randomize_tip_tilt   = True
    #simulator.randomize_distance   = True
    #simulator.rndQmax     = 0.1    # charge += rndQmax * ( rand()-0.5 )  (negative is off)
    #simulator.rndRmax     = 0.2    # charge += rndRmax * ( rand()-0.5 )  (negative is off)
    #simulator.rndEmax     = 0.5    # charge *= (1 + rndEmax     * ( rand()-0.5 )) (negative is off)
    #simulator.rndAlphaMax = -0.1   # charge *= (1 + rndAlphaMax * ( rand()-0.5 )) (negative is off)
    #simulator.modMolParams = modMolParams_def   # custom function to modify parameters

    simulator.randomize_enabled    = False
    simulator.randomize_nz         = False 
    simulator.randomize_parameters = False
    simulator.randomize_tip_tilt   = False
    simulator.randomize_distance   = False

    #simulator.debugPlots = True
    simulator.distAbove = 7.0
    simulator.distAboveDelta = None  

    simulator.bQZ = True
    simulator.Qs  = np.array([100,-200,100,0]) * -0.2   # Quadrupole Qdz2-1.0[e*A^2]
    simulator.maxTilt0 = 0.5     # asymetric tip  tilted max +/-1.0 Angstreom in random direction

    #simulator.shuffle_rotations = False
    #simulator.shuffle_molecules = False
    #simulator.nBestRotations    = 0

    #simulator.preName    = ""           # appended befroe path filename
    #simulator.postName   = "/pos.xyz"
    simulator.Q = 0.0
    # z-weight exp(-wz*z)
    simulator.wz      = 1.0    # deacay
    simulator.zWeight =  simulator.getZWeights();
    dz=0.1
    dfWeight = PPU.getDfWeight( 10, dz=dz )[0].astype(np.float32)
    simulator.dfWeight = dfWeight

    #simulator.pos0 = np.array([0.0,0.0,0.0])
    simulator.pos0 = np.array([14.0, 18.0, 24.0])

    iz = -8
    mutator = Mutator()
    trainer = CorrectorTrainer( simulator, mutator, molCreator=None )
    xyzs,Zs,elems,qs = au.loadAtomsNP("pos_out3.xyz")
    xyzs[:,0] -= 10.0 
    print("xyzs ", xyzs) 
    trainer.start( xyzs=xyzs, Zs=Zs, qs=qs )
    for itr in range(10):
        Xs1,Xs2  = trainer[itr]
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1); plt.imshow(Xs1[:,:,iz])
        plt.subplot(1,2,2); plt.imshow(Xs2[:,:,iz])
        plt.savefig( "CorrectorTrainAFM_%03i.png" %itr )
        plt.close()


    '''
    print( "# ------ Init Relaxator  ")

    relaxator = FARFF.EngineFARFF()

    print( "# ------ Init Critique   ")

    critique = Critique()
    critique.logImgName = "AFM_Err"

    print( "# ------ Init Looper     ")

    nscan = simulator.scan_dim; nscan = ( nscan[0], nscan[1], nscan[2]- len(simulator.dfWeight) )
    np.save( 'AFMref.npy', np.zeros(nscan) )
    AFMRef = np.load('AFMref.npy')
    AFMRef = np.roll( AFMRef, 5, axis=0 );
    AFMRef = np.roll( AFMRef, -6, axis=1 ); 

    looper = CorrectionLoop(relaxator,simulator,critique)
    looper.xyzLogFile = open( "CorrectionLoopLog.xyz", "w")
    looper.logImgName = "CorrectionLoopAFMLog"
    looper.logAFMdataName = "AFMs"
    #xyzs,Zs,elems,qs = au.loadAtomsNP("input.xyz")
    xyzs,Zs,elems,qs = au.loadAtomsNP("pos_out3.xyz")
    atomMap, bondMap, lvecMap = FARFF.makeGridFF( FARFF,  fname_atom='./Atoms.npy', fname_bond='./Bonds.npy',   dx=0.1, dy=0.1 )

    looper.startLoop( xyzs, Zs, qs, atomMap, bondMap, lvecMap, AFMRef )
    ErrConv = 0.1
    print( "# ------ To Loop    ")
    for itr in range(1000):
        print( "# ======= CorrectionLoop[ %i ] ", itr )
        Err = looper.iteration(itr=itr)
        if Err < ErrConv:
            break

    looper.xyzLogFile.close()
    '''

    #print( "UNIT_TEST is not yet written :-( " )
    print( " UNIT_TEST CorrectionLoop DONE !!! " )