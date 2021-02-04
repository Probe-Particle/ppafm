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
import matplotlib
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
#from . import FARFF            #as Relaxer
from . import GeneratorOCL_Simple2

verbose  = 0
bRunTime = False

# ========================================================================
class Sequence:
    pass

class Molecule():
    """
    A simple class for Molecule data containing atom xyz positions, element types and charges.
    Arguments:
        xyzs: list, xyz positions of atoms in the molecule
        Zs: list, element types
        qs: list, charges
    """

    def __init__(self, xyzs, Zs, qs ):
        self.xyzs = xyzs
        self.Zs   = Zs
        self.qs   = qs

    def clone(self):
        xyzs = self.xyzs
        Zs   = self.Zs
        qs   = self.qs
        return Molecule(xyzs,Zs,qs)

    def __str__(self):
        return np.c_[self.xyzs, self.Zs, self.qs].__str__()

    def __len__(self):
        return len(self.xyzs)

# ========================================================================

def removeAtoms( molecule, zmin=0.7, nmax=1 ):
    #print( "----- removeAtoms  p0 ", p0 )
    xyzs = molecule.xyzs
    Zs   = molecule.Zs
    qs   = molecule.qs
    top_atom_idx = np.where(np.greater_equal(xyzs[:,2],xyzs[:,2].max()+zmin))[0]

    nmax = min(nmax, len(top_atom_idx))
    rem_idx = np.random.randint(nmax+1)
    sel = np.random.choice(top_atom_idx, rem_idx, replace=False)

    xyzs_ = np.delete( xyzs, sel, axis=0 )
    Zs_   = np.delete( Zs,   sel )
    qs_   = np.delete( qs,   sel )
    return Molecule(xyzs_, Zs_, qs_), sel

def addAtom( molecule, p0, R=0.0, Z0=1, q0=0.0, dq=0.0 ):
    # ToDo : it would be nice to add atoms in a more physicaly reasonable way - not overlaping, proper bond-order etc.
    # ToDo : currently we add always hydrogen - should we somewhere randomly pick different atom types ?
    xyzs = molecule.xyzs
    Zs   = molecule.Zs
    qs   = molecule.qs
    #print( "----- addAtom  p0 ", p0 )
    #dp    = (np.random.rand(3)-0.5)*R
    #dp[2] = 0
    min_x = xyzs.min(axis=0)
    max_x = xyzs.max(axis=0)
    min_x[2] = max_x[2]-1.0
    nx = np.random.uniform(min_x, max_x)[np.newaxis, ...]

    min_q = qs.min(axis=0)
    max_q = qs.max(axis=0)
    nq = np.random.uniform(min_q, max_q, (1,))

    Zs_ = np.append(Zs, np.array([Z0,]), axis=0)
    xyzs_ = np.append(xyzs, nx, axis=0 )
    qs_   = np.append(qs,   nq, axis=0 )
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

def moveMultipleAtoms(molecule: Molecule, scale=0.5, nmax=10):
    """
    A function for slightly shifting multiple atoms in the molecule.
    """
    nmax = nmax if nmax < len(molecule) else len(molecule)

    move_n = np.random.randint(nmax+1)
    to_be_moved = np.random.choice(len(molecule), move_n, replace=False)
    magnitudes = np.random.uniform(-scale, scale, (move_n, 3))

    xyzs_ = molecule.xyzs.copy()
    Zs_   = molecule.Zs.copy()
    qs_   = molecule.qs.copy()

    xyzs_[to_be_moved] = xyzs_[to_be_moved] + magnitudes

    return Molecule(xyzs_, Zs_, qs_)

class Mutator():
    """
    A class for creating simple mutations to a molecule. The current state of the class allows for adding,
    removing and moving atoms. The mutations are not necessarily physically sensible at this point.
    Parameters:
        maxMutations: Integer. Maximum number of mutations per mutant. The number of mutations is chosen
        randomly between [1, maxMutations].
    """
    # Strtegies contain
    strategies = [
        (0.1,removeAtoms,{}),
        (0.0,addAtom,{}),
        (0.0,moveAtom,{})
    ]

    def __init__(self, maxMutations=10, zmin=0.7):
        self.setStrategies()
        self.maxMutations = maxMutations
        self.zmin = zmin

    def setStrategies(self, strategies=None):
        if strategies is not None: self.strategies = strategies
        self.cumProbs = np.cumsum( [ it[0] for it in self.strategies ] )
        #print( self.cumProbs ); exit()

    def mutate_local(self, molecule):
        molecule, removed = removeAtoms(molecule, zmin=self.zmin, nmax=self.maxMutations,)
        return molecule, removed

        #toss = np.random.rand(n_mutations)*self.cumProbs[-1]
        #i = np.searchsorted( self.cumProbs, toss )

        #print( "mutate_local ", i, toss, self.cumProbs[-1] )

        #molecule = moveMultipleAtoms(molecule, 1.0, 10)
        #for strat_nr in i:
        #    args = self.strategies[strat_nr][2]
        #    args['R'] = R
        #    molecule = self.strategies[strat_nr][1](molecule, p0, **args)
        #    #print(n_mutations, "mutations: ", self.strategies[strat_nr][1].__name__)

        #return molecule

        #args = self.strategies[i][2]
        #args['R'] = R
        #return self.strategies[i][1]( molecule, p0, **args )
        #return xyzs.copy(), Zs.copy(), qs.copy()

class CorrectorTrainer(GeneratorOCL_Simple2.InverseAFMtrainer):
    """
    A class for creating a batch of a data set for the CorrectionLoop. Iterable.
    Arguments:
        :param afmulator: instance of AFMulatorOCL_Simple.AFMulator
        :param mutator:   instance of CorrectionLoop.Mutator
        :param paths: array, paths to the molecules
        :param nMutants: integer, number of different mutants per molecule
    """

    def __init__(self, afmulator, mutator, paths, nMutants, **gen_args):
        self.index        = 0
        self.molIndex     = 0
        self.afmulator    = afmulator
        self.mutator      = mutator
        self.paths        = paths
        self.nMutants     = nMutants
        super().__init__(afmulator, None, paths, **gen_args)

        self.extend_molecules_with_mutants()
        self.check_empty_paths()

    def generatePair(self):
        """
        Generates a mutant of the current molecule.
        :return: Two molecules in np array format
        """

        # Get current molecule
        mol1_xyz = self.molecules[self.index]
        xyzs = mol1_xyz[:, :3]
        qs = mol1_xyz[:, 3]
        Zs = mol1_xyz[:, 4].astype(np.int32)

        mol1 = Molecule(xyzs, Zs, qs)

        # Mutate
        p0       = (np.random.rand(3))
        p0[0] *= (self.afmulator.scan_window[1][0]+self.afmulator.scan_window[0][0])*0.5
        p0[1] *= (self.afmulator.scan_window[1][1]+self.afmulator.scan_window[0][1])*0.5
        p0[2] *= 1.0
        R        = 1.0
        mol2, removed     = self.mutator.mutate_local(mol1)

        return mol1, mol2, removed

    def __getitem__(self, index):
        self.index = index
        if(verbose>0): print("index ", index)
        return next(self)

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return len(self.molecules)//self.batch_size

    def __next__(self):
        if self.index < len(self.molecules):

            # Callback
            self.on_batch_start()

            mol1s = []
            mol2s = []
            removed = []
            X1s   = [[] for _ in self.iZPPs]
            X2s   = [[] for _ in self.iZPPs]

            for b in range(self.batch_size):
                # Get original and mutant
                mol1, mol2, rem = self.generatePair()
                self.xyzs = mol1.xyzs
                self.qs   = mol1.qs
                self.Zs   = mol1.Zs

                # Callback
                self.on_sample_start()

                # Make sure the molecule is in right position
                self.handle_positions()

                # Get AFM images
                for i, (iZPP, Q, Qz) in enumerate(zip(self.iZPPs, self.Qs, self.QZs)):  # Loop over different tips
                    # Set tip parameters
                    self.afmulator.iZPP = iZPP
                    self.afmulator.setQs(Q, Qz)
                    self.REAs = PPU.getAtomsREA(self.afmulator.iZPP, self.Zs, self.afmulator.typeParams, alphaFac=-1.0)

                    # Make sure tip-sample distance is right
                    tot_dist, z_max = self.handle_distance_and_return()

                    # Callback
                    self.on_afm_start()

                    # Evaluate 1st AFM
                    X1 = self.afmulator(self.xyzs, self.Zs, self.qs, self.REAs)
                    X1s[i].append(X1)

                    # Set mutant as current molecule
                    self.xyzs = mol2.xyzs
                    self.qs   = mol2.qs
                    self.Zs   = mol2.Zs

                    # Make sure the molecule is in right position
                    self.handle_positions()

                    # Calculate new interaction parameters for the mutant
                    self.REAs = PPU.getAtomsREA(self.afmulator.iZPP, self.Zs, self.afmulator.typeParams, alphaFac=-1.0)

                    # Make sure tip-sample distance is right
                    self.handle_distance_mutant(tot_dist, z_max)

                    # Evaluate 2nd AFM
                    X2 = self.afmulator(self.xyzs, self.Zs, self.qs, self.REAs)
                    X2s[i].append(X2)

                mol1s.append(mol1)
                mol2s.append(mol2)
                removed.append(rem)

                self.index += 1

            for i in range(len(self.iZPPs)):
                X1s[i] = np.stack(X1s[i], axis=0)
                X2s[i] = np.stack(X2s[i], axis=0)

        else:
            raise StopIteration

        mol1s = [np.c_[(mol1.xyzs, mol1.qs, mol1.Zs)] for mol1 in mol1s]
        mol2s = [np.c_[(mol2.xyzs, mol2.qs, mol2.Zs)] for mol2 in mol2s]
        return X1s, X2s, mol1s, mol2s, removed

    def handle_distance_mutant(self, total_distance, zs_max):
        '''
        Set correct distance from scan region for the current mutant. Get distance information from original molecule
        Arguments:
        '''
        self.xyzs[:,2] += (self.afmulator.scan_window[1][2] - total_distance) - zs_max

    def handle_distance_and_return(self):
        '''
        Set correct distance from scan region for the current molecule for the original molecule and return highest atom
        Returns:

        '''
        RvdwPP = self.afmulator.typeParams[self.afmulator.iZPP-1][0]
        Rvdw = self.REAs[:,0] - RvdwPP
        zs = self.xyzs[:,2].copy()
        imax = np.argmax(zs + Rvdw)
        total_distance = self.distAboveActive + Rvdw[imax] + RvdwPP - (zs.max() - zs[imax])
        self.xyzs[:,2] += (self.afmulator.scan_window[1][2] - total_distance) - zs.max()
        return total_distance, zs.max()

    def extend_molecules_with_mutants(self):
        """
        Replicate each molecule in self.molecules 'nMutants' times
        """
        self.molecules *= self.nMutants

    def match_batch_size(self):
        """
        Select random molecules to extend the list of molecules to make batch sizes even.
        This should be called after augmenting data with rotations.
        """
        remainder = self.batch_size - len(self.molecules) % self.batch_size
        self.molecules.extend(np.random.choice(self.molecules, remainder))

    def check_empty_paths(self):
        """
        If self.paths is empty, raise ValueError
        """
        if not self.paths: raise ValueError("No molecules selected")


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
