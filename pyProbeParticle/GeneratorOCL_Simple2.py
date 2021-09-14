
import time
import random
import numpy as np

from . import basUtils
from . import common as PPU

import numpy as np

class InverseAFMtrainer:
    '''
    A data generator for training machine learning models. Generates batches of input/output pairs.
    An iterator.
    Arguments:
        afmulator: An instance of AFMulatorOCL_Simple.AFMulator.
        auxmaps: list of AuxMap objects.
        paths: list of paths to xyz files of molecules. The molecules are saved to the "molecules" attribute
               in np.ndarrays of shape (num_atoms, 5) with [x, y, z, charge, element] for each atom.
        batch_size: int. Number of samples per batch.
        distAbove: float. Tip-sample distance parameter.
        iZPPs: list of ints. Elements for AFM tips. Image is produced with every tip for each sample.
        Qs: list of arrays of length 4. Charges for tips.
        QZS list of arrays of length 4. Positions of tip charges.
    '''

    # Print timings during excecution
    bRuntime = False

    def __init__(self, 
        afmulator, aux_maps, paths,
        batch_size = 30,
        distAbove  = 5.3,
        iZPPs      = [8],
        Qs         = [[ -10, 20,  -10, 0 ]],
        QZs        = [[ 0.1,  0, -0.1, 0 ]],
        ):

        assert len(iZPPs) == len(Qs) and len(Qs) == len(QZs)

        self.afmulator = afmulator
        self.aux_maps = aux_maps
        self.paths = paths
        self.batch_size = batch_size
        self.distAbove = distAbove
        self.distAboveActive = distAbove

        self.iZPPs = iZPPs
        self.Qs = Qs
        self.QZs = QZs

        self.read_xyzs()
        self.counter = 0

    def __next__(self):


        if self.counter < len(self.molecules):

            # Callback
            self.on_batch_start()
            
            mols = []
            Xs = [[] for _ in range(len(self.iZPPs))]
            Ys = [[] for _ in range(len(self.aux_maps))]
            batch_size = min(self.batch_size, len(self.molecules) - self.counter)

            if self.bRuntime: batch_start = time.time()

            for s in range(batch_size):

                if self.bRuntime: sample_start = time.time()

                # Load molecule
                mol = self.molecules[self.counter]
                mols.append(mol)
                self.xyzs = mol[:,:3]
                self.qs   = mol[:,3]
                self.Zs   = mol[:,4].astype(np.int32)

                # Make sure the molecule is in right position
                self.handle_positions()

                # Callback
                self.on_sample_start()

                # Get AFM
                for i, (iZPP, Q, Qz) in enumerate(zip(self.iZPPs, self.Qs, self.QZs)): # Loop over different tips

                    # Set interaction parameters
                    self.afmulator.iZPP = iZPP
                    self.afmulator.setQs(Q, Qz)
                    self.REAs = PPU.getAtomsREA( self.afmulator.iZPP, self.Zs, self.afmulator.typeParams, alphaFac=-1.0 )

                    # Make sure tip-sample distance is right
                    self.handle_distance()
                    
                    # Callback
                    self.on_afm_start()

                    # Evaluate AFM
                    if self.bRuntime: afm_start = time.time()
                    Xs[i].append(self.afmulator(self.xyzs, self.Zs, self.qs, self.REAs))
                    if self.bRuntime: print(f'AFM {i} runtime [s]: {time.time() - afm_start}')
                
                    self.Xs = Xs[i][-1]   
                    # Callback
                    self.on_afm_end()

                    
                            #cut_x_min = max(px_x - px_radius, 0) 
                            #cut_x_max = min(px_x + px_radius, scan_dim[0]) 
                            #cut_y_min = max(px_y - px_radius, 0) 
                            #cut_y_max = min(px_y + px_radius, scan_dim[1]) 
                            #Xs[i][-1][cut_y_min: cut_y_max,cut_x_min:cut_x_max,iz] = 0 
                # Get AuxMaps
                for i, aux_map in enumerate(self.aux_maps):
                    if self.bRuntime: aux_start = time.time()
                    xyzqs = np.concatenate([self.xyzs, self.qs[:,None]], axis=1)
                    Ys[i].append(aux_map(xyzqs, self.Zs))
                    if self.bRuntime: print(f'AuxMap {i} runtime [s]: {time.time() - aux_start}')


                
                if self.bRuntime: print(f'Sample {s} runtime [s]: {time.time() - sample_start}')

                self.counter += 1

            for i in range(len(self.iZPPs)):
                Xs[i] = np.stack(Xs[i], axis=0)

            for i in range(len(self.aux_maps)):
                Ys[i] = np.stack(Ys[i], axis=0)

            if self.bRuntime: print(f'Batch runtime [s]: {time.time() - batch_start}')

        else:
            raise StopIteration

        return Xs, Ys, mols
    
    def __iter__(self):
        self.counter = 0
        return self

    def __len__(self):
        '''
        Returns the number of batches that will be generated with the current molecules.
        '''
        return int(np.floor(len(self.molecules)/self.batch_size))

    def read_xyzs(self):
        '''
        Read molecule xyz files from selected paths.
        '''
        self.molecules = []
        for path in self.paths:
            with open(path, 'r') as f:
                xyzs, Zs, _, qs = basUtils.loadAtomsLines(f.readlines())
                self.molecules.append(np.concatenate([xyzs, qs[:,None], Zs[:,None]], axis=1))
    
    def handle_positions(self):
        '''
        Set current molecule to the center of the scan window.
        '''
        sw = self.afmulator.scan_window
        scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
        self.xyzs[:,:2] += scan_center - self.xyzs[:,:2].mean(axis=0)

    def handle_distance(self):
        '''
        Set correct distance from scan region for the current molecule.
        '''
        RvdwPP = self.afmulator.typeParams[self.afmulator.iZPP-1][0]
        Rvdw = self.REAs[:,0] - RvdwPP
        zs = self.xyzs[:,2]
        imax = np.argmax(zs + Rvdw)
        total_distance = self.distAboveActive + Rvdw[imax] + RvdwPP - (zs.max() - zs[imax])
        self.xyzs[:,2] += (self.afmulator.scan_window[1][2] - total_distance) - zs.max()

    # ======== Augmentation =========

    def shuffle_molecules(self):
        '''
        Shuffle list of molecules.
        '''
        random.shuffle(self.molecules)

    def augment_with_rotations(self, rotations):
        '''
        Augment molecule list with rotations of the molecules.
        Arguments:
            rotations: list of np.ndarray. Rotation matrices.
        '''
        molecules = self.molecules
        self.molecules = []
        for mol in molecules:
            xyzs = mol[:,:3]
            qs   = mol[:,3]
            Zs   = mol[:,4]
            for xyzs_rot in rotate(xyzs, rotations):
                self.molecules.append(np.concatenate([xyzs_rot, qs[:,None], Zs[:,None]], axis=1))

    def augment_with_rotations_entropy(self, rotations, n_best_rotations=30):
        '''
        Augment molecule list with rotations of the molecules. Rotations are sorted in terms of their "entropy".
        Arguments:
            rotations: list of np.ndarray. Rotation matrices.
            n_best_rotations: int. Only the first n_best_rotations with the highest "entropy" will be taken.
        '''
        molecules = self.molecules
        self.molecules = []
        for mol in molecules:
            xyzs = mol[:,:3]
            qs   = mol[:,3]
            Zs   = mol[:,4]
            rots = sortRotationsByEntropy(mol[:,:3], rotations)[:n_best_rotations]
            for xyzs_rot in rotate(xyzs, rots):
                self.molecules.append(np.concatenate([xyzs_rot, qs[:,None], Zs[:,None]], axis=1))

    def randomize_tip(self, max_tilt=0.5):
        '''
        Randomize tip tilt to simulate asymmetric adsorption of particle on tip apex.
        Arguments:
            max_tilt: float. Maximum deviation in xy plane in angstroms.
        '''
        self.afmulator.tipR0[:2] = np.array(getRandomUniformDisk())*max_tilt

    def randomize_distance(self, delta=0.25):
        '''
        Randomize tip-sample distance.
        Arguments:
            delta: float. Maximum deviation from original value in angstroms.
        '''
        self.distAboveActive = np.random.uniform(self.distAbove - delta, self.distAbove + delta)

    def randomize_mol_parameters(self, rndQmax=0.0, rndRmax=0.0, rndEmax=0.0, rndAlphaMax=0.0):
        '''
        Randomize various interaction parameters for current molecule.
        '''
        num_atoms = len(self.qs)
        if rndQmax > 0:
            self.qs[:]     += rndQmax * ( np.random.rand( num_atoms ) - 0.5 )
        if rndRmax > 0:
            self.REAs[:,0] += rndRmax * ( np.random.rand( num_atoms ) - 0.5 )
        if rndEmax > 0:
            self.REAs[:,1] *= ( 1 + rndEmax * ( np.random.rand( num_atoms ) - 0.5 ) )
        if rndAlphaMax > 0:
            self.REAs[:,2] *= ( 1 + rndAlphaMax * ( np.random.rand( num_atoms ) - 0.5 ) )

    # ====== Callback methods =======

    def on_batch_start(self):
        '''
        Excecuted right at the start of each batch. Override to modify parameters for each batch.
        '''
        pass

    def on_sample_start(self):
        '''
        Excecuted right before evaluating first AFM image. Override to modify the parameters for each sample.
        '''
        pass

    def on_afm_start(self):
        '''
        Excecuted right before every AFM image evalution. Override to modify the parameters for each AFM image.
        '''
        pass

        
    def on_afm_end(self):
        '''
        Excecuted right after evaluating AFM image. Override to modify the parameters for each sample.
        '''
        pass
    
def sortRotationsByEntropy(xyzs, rotations):
    rots = []
    for rot in rotations:
        zDir = rot[2].flat.copy()
        _, _, entropy = PPU.maxAlongDirEntropy( xyzs, zDir )
        rots.append( (entropy, rot) )
    rots.sort( key=lambda item: -item[0] )
    rots = [rot[1] for rot in rots]
    return rots

def rotate(xyzs, rotations):
    rotated_xyzs = []
    for rot in rotations:
        rotated_xyzs.append(np.dot(xyzs, rot.T))
    return rotated_xyzs

def getRandomUniformDisk():
    '''
    generate points unifromly distributed over disk
    # see: http://mathworld.wolfram.com/DiskPointPicking.html
    '''
    rnd = np.random.rand(2)
    rnd[0]    = np.sqrt( rnd[0] ) 
    rnd[1]   *= 2.0*np.pi
    return  rnd[0]*np.cos(rnd[1]), rnd[0]*np.sin(rnd[1])
