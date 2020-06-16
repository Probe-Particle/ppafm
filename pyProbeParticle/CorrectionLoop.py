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

from . import AFMulatorOCL_Simple
from . import AuxMap
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

    def __init__(self, simulator, mutator, molCreator=None ):
        self.index = 0
        self.simulator  = simulator
        self.mutator    = mutator
        self.molCreator = molCreator

    def start(self, molecule=None ):
        if molecule is None:
            self.molecule = self.molCreator.create()
        else:
            self.molecule = molecule

    def generatePair(self):
        
        if self.molCreator is not None:
            if np.random.rand(1) < self.restartProb:
                self.molecule = self.molCreator.create()

        # Get AFM
        mol1 = self.molecule
        Xs1 = self.simulator.eval_(mol1)

        # Mutate and get new AFM
        p0       = (np.random.rand(3))
        p0[0:2] *= 20.0
        p0[  2] *= 1.0
        R        = 3.0   
        mol2     = self.mutator.mutate_local( self.molecule, p0, R )
        Xs2      = self.simulator.eval_(mol2)

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

class Corrector():

    izPlot     = -8
    logImgName = None

    def __init__(self ):
        self.best = None
        pass

    def modifyStructure(self, molIn ):
        mol = molIn.clone()
        #molOut.xyzs[0,0] += 0.1
        #molOut.xyzs[1,0] -= 0.1
        p0 = (np.random.rand(3))
        p0[0:2] *= 20.0
        p0[  2] *= 1.0
        R  = 3.0   
        molOut = moveAtom( molIn, p0, R=0.0, nmax=1 )   # ToDo : This is just token example - later need more sophisticated Correction strategy
        # --- ToDo: Relaxation Should be possible part of relaxation ????
        return molOut

    def try_improve(self, molIn, AFMs, AFMRef, itr=0 ):
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
            self.best_structure = molIn
        print( "Corrector.try_improve Err2 ", Err  )
        molOut = self.modifyStructure( self.best_structure )
        return Err, molOut

# ========================================================================

class CorrectionLoop():

    logAFMdataName = None
    logImgName = None
    logImgIzs  = [0,-8,-1] 

    rotMat = np.array([[1.,0,0],[0.,1.,0],[0.,0,1.]])

    def __init__(self, relaxator, simulator, atoms, bonds, corrector ):
        self.simulator  = simulator
        self.relaxator  = relaxator
        self.atoms      = atoms
        self.bonds      = bonds
        self.corrector  = corrector
        self.xyzLogName = None

    def init(self):
        pass

    def startLoop(self, molecule, atomMap, bondMap, lvecMap, AFMRef ):
        self.molecule = molecule
        self.atomMap = atomMap
        self.bondMap = bondMap
        self.mapLvec = lvecMap
        self.AFMRef  = AFMRef

    def iteration(self, itr=0 ):
        print( "### CorrectionLoop.iteration [1]" )
        print( "########### bond atom ---- min min ", self.atomMap.min(), self.bondMap.min() )
        
        # ToDo : Use relaxation later
        #self.relaxed    = self.relaxator.preform_relaxation( self.molecule, self.mapLvec, self.atomMap, self.bondMap )
        if self.xyzLogFile is not None:
           au.writeToXYZ( self.xyzLogFile, self.molecule.Zs, self.molecule.xyzs, qs=self.molecule.qs, commet=("CorrectionLoop.iteration [%i] " %itr) )

        print( "### CorrectionLoop.iteration [%i]" %itr )

        # Get AFM
        xyzs, qs, Zs = self.molecule.xyzs, self.molecule.qs, self.molecule.Zs
        AFMs = self.simulator(xyzs, Zs, qs)

        # Get Atoms and Bonds AuxMaps
        xyzqs = np.concatenate([xyzs, qs[:,None]], axis=1)
        atoms_map = self.atoms(xyzqs, Zs)
        bonds_map = self.bonds(xyzqs, Zs)
        AuxMaps = np.stack([atoms_map, bonds_map], axis=-1)

        if self.logAFMdataName:
            np.save( self.logAFMdataName+("%03i.dat" %itr), AFMs )
        if self.logImgName is not None:
            nz  = len(self.logImgIzs)
            nch = AuxMaps.shape[2]
            #print( "DEBUG nz nch ", nz, nch )
            plt.figure(figsize=(5*(nz+nch),5))
            for ich in range(nch):
                #print( "DEBUG ich ", ich )
                plt.subplot(1,nz+nch,ich+1); plt.imshow ( AuxMaps[:,:,ich] )
            for iiz, iz in enumerate(self.logImgIzs):
                plt.subplot(1,nz+nch,iiz+nch+1); plt.imshow ( AFMs[:,:,iz] )
                plt.title( "iz %i" %iz )
            plt.savefig( self.logImgName+("_%03i.png" %itr), bbox_inches='tight')
            plt.close  ()

        print( "### CorrectionLoop.iteration [3]" )
        Err, self.molecule = self.corrector.try_improve( self.molecule, AFMs, self.AFMRef, itr=itr )
        print( "### CorrectionLoop.iteration [4]" )
        return Err
        #Xs,Ys      = simulator.next1( self )

def Job_trainCorrector( simulator, geom_fname="input.xyz", nstep=10 ):
    iz = -10
    mutator = Mutator()
    trainer = CorrectorTrainer( simulator, mutator, molCreator=None )
    xyzs,Zs,elems,qs = au.loadAtomsNP(geom_fname)

    # AFMulatorOCL.setBBoxCenter( xyzs, [0.0,0.0,0.0] )
    sw = simulator.scan_window
    scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
    xyzs[:,:2] += scan_center - xyzs[:,:2].mean(axis=0)
    xyzs[:,2] += (sw[1][2] - 9.0) - xyzs[:,2].max()
    print("xyzs ", xyzs) 
    mol = Molecule(xyzs,Zs,qs)

    #xyzs[:,0] -= 10.0 
    trainer.start( mol )
    extent=( simulator.scan_window[0][0], simulator.scan_window[1][0], simulator.scan_window[0][1], simulator.scan_window[1][1] )
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

def Job_CorrectionLoop( simulator, atoms, bonds, geom_fname="input.xyz", nstep=10 ):
    relaxator = FARFF.EngineFARFF()
    corrector = Corrector()
    corrector.logImgName = "AFM_Err"
    nscan = simulator.scan_dim; nscan = ( nscan[0], nscan[1], nscan[2]- len(simulator.dfWeight) )
    np.save( 'AFMref.npy', np.zeros(nscan) )
    AFMRef = np.load('AFMref.npy')
    AFMRef = np.roll( AFMRef,  5, axis=0 );
    AFMRef = np.roll( AFMRef, -6, axis=1 ); 

    looper = CorrectionLoop(relaxator, simulator, atoms, bonds, corrector)
    looper.xyzLogFile = open( "CorrectionLoopLog.xyz", "w")
    looper.logImgName = "CorrectionLoopAFMLog"
    looper.logAFMdataName = "AFMs"
    #xyzs,Zs,elems,qs = au.loadAtomsNP("input.xyz")
    xyzs,Zs,elems,qs = au.loadAtomsNP(geom_fname)

    # AFMulatorOCL.setBBoxCenter( xyzs, [0.0,0.0,0.0] )
    sw = simulator.scan_window
    scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
    xyzs[:,:2] += scan_center - xyzs[:,:2].mean(axis=0)
    xyzs[:,2] += (sw[1][2] - 9.0) - xyzs[:,2].max()
    print("xyzs ", xyzs) 

    xyzqs = np.concatenate([xyzs, qs[:,None]], axis=1)
    np.save('./Atoms.npy', atoms(xyzqs, Zs))
    np.save('./Bonds.npy', bonds(xyzqs, Zs))

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

    afmulator = AFMulatorOCL_Simple.AFMulator(
        pixPerAngstrome = 10,
        lvec            = np.array([
                            [ 0.0,  0.0, 0.0],
                            [20.0,  0.0, 0.0],
                            [ 0.0, 20.0, 0.0],
                            [ 0.0,  0.0, 5.0]
                          ]),
        scan_window     = ((2.0, 2.0, 5.0), (18.0, 18.0, 8.0)),
    )

    atoms = AuxMap.AtomRfunc(scan_dim=(128, 128), scan_window=((2,2),(18,18)))
    bonds = AuxMap.Bonds(scan_dim=(128, 128), scan_window=((2,2),(18,18)))

    if options.job == "loop":
        Job_CorrectionLoop( afmulator, atoms, bonds, geom_fname="pos_out3.xyz" )
    elif options.job == "train":
        Job_trainCorrector( afmulator, geom_fname="pos_out3.xyz", nstep=10 )        
    else:
        print("ERROR : invalid job ", options.job )

    #print( "UNIT_TEST is not yet written :-( " )
    print( " UNIT_TEST CorrectionLoop DONE !!! " )