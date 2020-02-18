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

class Molecule():

    def __init__(self, xyzs, Zs, qs ):
        self.xyzs = xyzs
        self.Zs   = Zs
        self.qs   = qs

    def clone(self):
        xyzs = self.xyzs 
        Zs   = self.Zs
        qs   = self.qs
        return Molecule(xyzs,Zs,qs)

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

def removeAtoms( molecule, p0, R=3.0, nmax=1 ):
    #print( "----- removeAtoms  p0 ", p0 )
    xyzs = molecule.xyzs 
    Zs   = molecule.Zs
    qs   = molecule.qs
    rs   = (xyzs[:,0]-p0[0])**2 + (xyzs[:,1]-p0[1])**2
    sel = rs.argsort()[:nmax] # [::-1]
    xyzs_ = np.delete( xyzs, sel, axis=0 )
    Zs_   = np.delete( Zs,   sel )
    qs_   = np.delete( qs,   sel )
    return Molecule(xyzs_, Zs_, qs_)

def addAtom( molecule, p0, R=0.0, Z0=1, q0=0.0, dq=0.0 ):
    # ToDo : it would be nice to add atoms in a more physicaly reasonable way - not overlaping, proper bond-order etc.
    # ToDo : currently we add always hydrogen - should we somewhere randomly pick different atom types ?
    xyzs = molecule.xyzs 
    Zs   = molecule.Zs
    qs   = molecule.qs
    #print( "----- addAtom  p0 ", p0 )
    dp    = (np.random.rand(3)-0.5)*R
    dp[2] = 0
    Zs_   = np.append( Zs,   np.array([Z0,]),      axis=0 )
    xyzs_ = np.append( xyzs, np.array([p0 + dp,]), axis=0 )
    qs_   = np.append( qs,   np.array([q0 + (np.random.rand()-0.5)*dq,]), axis=0 )
    return Molecule(xyzs_, Zs_, qs_)

def moveAtom( molecule, p0, R=0.0, dpMax=np.array([1.0,1.0,0.25]), nmax=1 ):
    # ToDo : it would be nice to add atoms in a more physicaly reasonable way - not overlaping, proper bond-order etc.
    # ToDo : currently we add always hydrogen - should we somewhere randomly pick different atom types ?
    xyzs         = molecule.xyzs.copy() 
    Zs           = molecule.Zs.copy()
    qs           = molecule.qs.copy()
    rs           = (xyzs[:,0]-p0[0])**2 + (xyzs[:,1]-p0[1])**2
    sel          = rs.argsort()[:nmax] # [::-1]
    xyzs[sel,:] += (np.random.rand(len(sel),3)-0.5)*dpMax[None,:]
    return Molecule(xyzs, Zs, qs)

class Mutator():
    # Strtegies contain 
    strategies = [ 
        (0.1,removeAtoms,{}), 
        (0.1,addAtom,{}), 
        (0.5,moveAtom,{}) 
    ]

    def __init__(self ):
        self.setStrategies()
        pass

    #def modAtom():
    def setStrategies(self, strategies=None):
        if strategies is not None: self.strategies = strategies
        self.cumProbs = np.cumsum( [ it[0] for it in self.strategies ] )
        #print( self.cumProbs ); exit()  

    def mutate_local(self, molecule, p0, R ):
        toss = np.random.rand()*self.cumProbs[-1]
        i = np.searchsorted( self.cumProbs, toss )
        print( "mutate_local ", i, toss, self.cumProbs[-1] )
        args = self.strategies[i][2]
        args['R'] = R
        return self.strategies[i][1]( molecule, p0, **args )
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

    #def start(self, xyzs=None, Zs=None, qs=None ):
    def start(self, molecule=None ):
        if molecule is None:
            #self.xyzs, self.Zs, self.qs = self.molCreator.create()
            self.molecule = self.molCreator.create()
        else:
            self.molecule = molecule
            #self.xyzs = xyzs
            #self.Zs   = Zs
            #self.qs   = qs

    def generatePair(self):
        if self.molCreator is not None:
            if np.random.rand(1) < self.restartProb:
                #self.xyzs, self.Zs, self.qs = self.molCreator.create()
                self.molecule = self.molCreator.create()
        #Xs1,Ys1      = self.simulator.perform_imaging( self.xyzs.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        #Xs1      = self.simulator.perform_just_AFM( self.xyzs.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        mol1     = self.molecule
        Xs1      = self.simulator.perform_just_AFM( mol1, self.rotMat )
        p0 = (np.random.rand(3) - 0.5)
        p0[0:2] *= 10.0
        p0[  2] *= 1.0
        R  = 3.0   
        #self.xyzs, self.Zs, self.qs = self.mutator.mutate_local( self.xyzs, self.Zs, self.qs, p0, R )
        mol2 = self.mutator.mutate_local( self.molecule, p0, R )
        #Xs2,Ys2      = self.simulator.perform_imaging( self.xyzs.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        #Xs2     = self.simulator.perform_just_AFM( self.xyzs.copy(), self.Zs.copy(), self.qs.copy(), self.rotMat )
        Xs2     = self.simulator.perform_just_AFM( mol2, self.rotMat )
        self.molecule = mol2
        return Xs1, Xs2, mol1, mol2

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

    def startLoop(self, molecule, bondMap, lvecMap, AFMRef ):
        #self.xyzs    = xyzs
        #self.Zs      = Zs
        #self.qs      = qs
        self.molecule = molecule
        self.atomMap = atomMap
        self.bondMap = bondMap
        self.mapLvec = lvecMap
        self.AFMRef  = AFMRef

    def iteration(self, itr=0 ):
        print( "### CorrectionLoop.iteration [1]" )
        print( "########### bond atom ---- min min ", self.atomMap.min(), self.bondMap.min() )
        self.relaxed    = self.relaxator.preform_relaxation ( self.molecule, self.mapLvec, self.atomMap, self.bondMap )
        if self.xyzLogFile is not None:
           # au.saveXYZ( self.Zs, self.xyzs, self.xyzLogName, qs=self.qs )
           au.writeToXYZ( self.xyzLogFile, self.molecule.Zs, self.molecule.xyzs, qs=self.molecule.qs, commet=("CorrectionLoop.iteration [%i] " %itr) )

        print( "### CorrectionLoop.iteration [%i]" %itr )
        AFMs,AuxMap     = self.simulator.perform_imaging( self.molecule, self.qs.copy(), self.rotMat )
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
        Err, self.molecule = self.critique.try_improve( self.molecule, AFMs, self.AFMRef, itr=itr )
        print( "### CorrectionLoop.iteration [4]" )
        return Err
        #Xs,Ys      = simulator.next1( self )

def Job_trainCorrector( simulator, geom_fname="input.xyz", nstep=10 ):
    iz = -10
    mutator = Mutator()
    trainer = CorrectorTrainer( simulator, mutator, molCreator=None )
    xyzs,Zs,elems,qs = au.loadAtomsNP(geom_fname)
    mol = Molecule(xyzs,Zs,qs)

    GeneratorOCL_LJC.setBBoxCenter( xyzs, [0.0,0.0,0.0] )
    #xyzs[:,0] -= 10.0 
    print("xyzs ", xyzs) 
    trainer.start( mol )
    #extent = ( simulator.scan_start, simulator.scan_end, simulator.scan_start, simulator.scan_end )
    extent=( simulator.scan_start[0], simulator.scan_end[0], simulator.scan_start[1], simulator.scan_end[1] )
    sc = 3.0

    xyzfile = open("geomMutations.xyz","w")
    au.writeToXYZ( xyzfile, mol.Zs, mol.xyzs, qs=mol.qs, commet="# start " )
    for itr in range(nstep):
        Xs1,Xs2,mol1,mol2  = trainer[itr]
        au.writeToXYZ( xyzfile, mol2.Zs,  mol2.xyzs, qs=mol2.qs, commet=("# mutation %i " %itr) )
        #print( " mol1 ", mol1.xyzs, "\n mol2 ", mol2.xyzs)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1); plt.imshow(Xs1[:,:,iz], origin='image', extent=extent); plt.scatter( mol1.xyzs[:,0], mol1.xyzs[:,1], s=mol1.Zs*sc, c=cm.rainbow( mol1.xyzs[:,2] )  )
        plt.subplot(1,2,2); plt.imshow(Xs2[:,:,iz], origin='image', extent=extent); plt.scatter( mol2.xyzs[:,0], mol2.xyzs[:,1], s=mol2.Zs*sc, c=cm.rainbow( mol2.xyzs[:,2] ) )
        plt.savefig( "CorrectorTrainAFM_%03i.png" %itr )
        plt.close()
    xyzfile.close()

def Job_CorrectionLoop( simulator, geom_fname="input.xyz", nstep=10 ):
    relaxator = FARFF.EngineFARFF()
    critique = Critique()
    critique.logImgName = "AFM_Err"
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
    xyzs,Zs,elems,qs = au.loadAtomsNP(geom_fname)
    molecule = Molecule(xyzs,Zs,qs)
    atomMap, bondMap, lvecMap = FARFF.makeGridFF( FARFF,  fname_atom='./Atoms.npy', fname_bond='./Bonds.npy',   dx=0.1, dy=0.1 )

    looper.startLoop( molecule, atomMap, bondMap, lvecMap, AFMRef )
    ErrConv = 0.1
    print( "# ------ To Loop    ")
    for itr in range(nstep):
        print( "# ======= CorrectionLoop[ %i ] ", itr )
        Err = looper.iteration(itr=itr)
        if Err < ErrConv:
            break

    looper.xyzLogFile.close()


# ========================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option( "-j", "--job", action="store", type="string", help="[train/loop]")
    (options, args) = parser.parse_args()

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
    #simulator.pos0 = np.array([14.0, 18.0, 24.0])
    simulator.pos0 = np.array([15.0, 15.0, 24.0])

    if options.job == "loop":
        Job_CorrectionLoop( simulator, geom_fname="pos_out3.xyz" )
    elif options.job == "train":
        Job_trainCorrector( simulator, geom_fname="pos_out3.xyz", nstep=50 )        
    else:
        print("ERROR : invalid job ", options.job )

    #print( "UNIT_TEST is not yet written :-( " )
    print( " UNIT_TEST CorrectionLoop DONE !!! " )