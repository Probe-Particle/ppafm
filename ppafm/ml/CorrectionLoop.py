#!/usr/bin/python

"""
Idea is to improve prediction of gemeotry using physical generative model

1) CNN prediction of Bonds&Atoms
2) MM  relaxation (with predicted Bonds&Atoms as external potential)
3) PPM simulation (based on MM geometry)
4) recognize errors (reference AFM data-stack needed)
5) Generation of improved structure (How? Using CNN?)
    -> repeat from (2)

see:  https://mega.nz/#!KLoilKIB!NxxCRQ814xtCXfjy7mPFfmJTOL9TaTHbmPKSxn_0sFs

"""


import matplotlib
import numpy as np
import pyopencl as cl

from .. import atomicUtils as au
from .. import common as PPU
from .. import elements, io
from ..dev import SimplePot as sp
from ..ocl import AFMulator
from ..ocl import field as FFcl
from ..ocl import oclUtils as oclu
from ..ocl import relax as oclr
from . import AuxMap, Generator
from .Corrector import Corrector, Molecule, Mutator

verbose = 0
bRunTime = False


# ========================================================================
class Sequence:
    pass


class CorrectorTrainer(Generator.InverseAFMtrainer):
    """
    A class for creating a batch of a data set for the CorrectionLoop. Iterable.
    Arguments:
        :param afmulator: instance of AFMulator.AFMulator
        :param mutator:   instance of CorrectionLoop.Mutator
        :param paths: array, paths to the molecules
        :param nMutants: integer, number of different mutants per molecule
    """

    def __init__(
        self,
        afmulator,
        mutator,
        paths,
        nMutants,
        zmin,
        potential=True,
        added_types=[],
        **gen_args,
    ):
        self.index = 0
        self.molIndex = 0
        self.afmulator = afmulator
        self.mutator = mutator
        self.paths = paths
        self.nMutants = nMutants
        self.zmin = zmin
        self.potential = potential
        super().__init__(afmulator, None, paths, **gen_args)

        self.extend_molecules_with_mutants()
        self.check_empty_paths()
        if self.potential:
            mins = self.afmulator.scan_window[0][:2] + (self.zmin,)
            maxs = self.afmulator.scan_window[1][:2] + (0.0,)
            self.pot_sw = (mins, maxs)
            self.pot_dim = (100, 100, 20)
            self.pot = sp.SimplePotential(self.pot_dim, self.pot_sw)
            self.added_types = [added_types] if isinstance(added_types, int) else added_types

    def generatePair(self):
        """
        Generates a mutant of the current molecule.
        :return: Two molecules in np array format
        """

        # Get current molecule
        mol1_xyz_all = self.molecules[self.index]
        mol1_xyz_top = mol1_xyz_all[mol1_xyz_all[:, 2] >= mol1_xyz_all[:, 2].max() + self.zmin]

        xyzs = mol1_xyz_top[:, :3]
        qs = mol1_xyz_top[:, 3]
        Zs = mol1_xyz_top[:, 4].astype(np.int32)
        mol1_top = Molecule(xyzs, Zs, qs)

        xyzs_all = mol1_xyz_all[:, :3]
        qs_all = mol1_xyz_all[:, 3]
        Zs_all = mol1_xyz_all[:, 4].astype(np.int32)
        mol1_all = Molecule(xyzs_all, Zs_all, qs_all)

        mol2, removed = self.mutator.mutate_local(mol1_top)

        return mol1_all, mol2, removed

    def __getitem__(self, index):
        self.index = index
        if verbose > 0:
            print("index ", index)
        return next(self)

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return len(self.molecules) // self.batch_size

    def __next__(self):
        if self.index < len(self.molecules):
            # Callback
            self.on_batch_start()

            mol1s = []
            mol2s = []
            removed = []
            X1s = [[] for _ in self.iZPPs]
            X2s = [[] for _ in self.iZPPs]
            potMaps = []

            for b in range(self.batch_size):
                # Get original and mutant
                mol1_a, mol2, rem = self.generatePair()

                # Callback
                self.on_sample_start()

                # Get AFM images
                for i, (iZPP, Q, Qz) in enumerate(zip(self.iZPPs, self.Qs, self.QZs)):  # Loop over different tips
                    self.xyzs = mol1_a.xyzs
                    self.qs = mol1_a.qs
                    self.Zs = mol1_a.Zs

                    # Set tip parameters
                    self.afmulator.iZPP = iZPP
                    self.afmulator.setQs(Q, Qz)
                    self.REAs = PPU.getAtomsREA(
                        self.afmulator.iZPP,
                        self.Zs,
                        self.afmulator.typeParams,
                        alphaFac=-1.0,
                    )

                    # Make sure the molecule is in right position
                    center = self.handle_positions_and_return()
                    # Make sure tip-sample distance is right
                    tot_dist, z_max = self.handle_distance_and_return()

                    # Callback
                    self.on_afm_start()

                    # Evaluate 1st AFM
                    X1 = self.afmulator(self.xyzs, self.Zs, self.qs, self.REAs)
                    X1s[i].append(X1)

                    # fmt: off
                    mol1_tx = self.xyzs[mol1_a.array[:, 2] >= mol1_a.array[:, 2].max() + self.zmin]
                    mol1_tq = self.qs  [mol1_a.array[:, 2] >= mol1_a.array[:, 2].max() + self.zmin]
                    mol1_tz = self.Zs  [mol1_a.array[:, 2] >= mol1_a.array[:, 2].max() + self.zmin]
                    # fmt: on
                    mol1_t = Molecule(mol1_tx, mol1_tz, mol1_tq)

                    if mol2.xyzs.size > 0:
                        # Set mutant as current molecule
                        self.xyzs = mol2.xyzs
                        self.qs = mol2.qs
                        self.Zs = mol2.Zs

                        # Calculate new interaction parameters for the mutant
                        self.REAs = PPU.getAtomsREA(
                            self.afmulator.iZPP,
                            self.Zs,
                            self.afmulator.typeParams,
                            alphaFac=-1.0,
                        )

                        # Make sure the molecule is in right position
                        self.handle_positions_mutant(center)
                        # Make sure tip-sample distance is right
                        self.handle_distance_mutant(tot_dist, z_max)

                        # Evaluate 2nd AFM
                        X2 = self.afmulator(self.xyzs, self.Zs, self.qs, self.REAs)
                        X2s[i].append(X2)

                        mol2 = Molecule(self.xyzs, self.Zs, self.qs)

                    else:
                        X2 = np.random.normal(0, 1e-7, (128, 128, 10)).astype(np.float32)
                        X2s[i].append(X2)

                self.top_atom_to_zero(mol1_t, mol2)
                mol1s.append(mol1_t)
                mol2s.append(mol2)
                removed.append(rem)

                if self.potential:
                    self.pot.init_molecule(mol2.xyzs, mol2.Zs)
                    for z in self.added_types:
                        e = -self.pot.calc_potential(z_added=z)
                        e = e.reshape((self.pot_dim[2], self.pot_dim[1], self.pot_dim[0]))

                    potMaps.append(e)

                self.index += 1

            for i in range(len(self.iZPPs)):
                X1s[i] = np.stack(X1s[i], axis=0)
                X2s[i] = np.stack(X2s[i], axis=0)

        else:
            raise StopIteration

        potMaps = np.transpose(np.stack(potMaps, axis=0), (0, 2, 3, 1))
        mol1s = [np.c_[(mol1.xyzs, mol1.qs, mol1.Zs)] for mol1 in mol1s]
        mol2s = [np.c_[(mol2.xyzs, mol2.qs, mol2.Zs)] for mol2 in mol2s]

        if self.potential:
            out = X1s, X2s, mol1s, mol2s, removed, potMaps
        else:
            out = X1s, X2s, mol1s, mol2s, removed

        return out

    def handle_distance_mutant(self, total_distance, zs_max):
        """
        Set correct distance from scan region for the current mutant. Get distance information from original molecule
        Arguments:
        """
        self.xyzs[:, 2] += (self.afmulator.scan_window[1][2] - total_distance) - zs_max

    def handle_distance_and_return(self):
        """
        Set correct distance from scan region for the current molecule for the original molecule and return highest atom
        Returns:

        """
        RvdwPP = self.afmulator.typeParams[self.afmulator.iZPP - 1][0]
        Rvdw = self.REAs[:, 0] - RvdwPP
        zs = self.xyzs[:, 2].copy()
        imax = np.argmax(zs + Rvdw)
        total_distance = self.distAboveActive + Rvdw[imax] + RvdwPP - (zs.max() - zs[imax])
        self.xyzs[:, 2] += (self.afmulator.scan_window[1][2] - total_distance) - zs.max()
        return total_distance, zs.max()

    def handle_positions_and_return(self):
        """
        Set current molecule to the center of the scan window and return center coordinates
        """
        sw = self.afmulator.scan_window
        scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
        out = self.xyzs[:, :2].mean(axis=0)
        self.xyzs[:, :2] += scan_center - self.xyzs[:, :2].mean(axis=0)
        return out

    def handle_positions_mutant(self, center):
        """
        Set current molecule to the center of the scan window.
        """
        sw = self.afmulator.scan_window
        scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
        self.xyzs[:, :2] += scan_center - center

    def top_atom_to_zero(self, mol1, mol2):
        """
        Set the z coordinate of the highest atom to 0. Keep molecule and mutant in same coordinates
        """
        top = mol1.xyzs[:, 2].max()
        mol1.xyzs[:, 2] -= top
        mol2.xyzs[:, 2] -= top

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
        if not self.paths:
            raise ValueError("No molecules selected")


class CorrectionLoop:
    def __init__(self, relaxator, simulator, atoms, bonds, corrector):
        self.rotMat = np.array([[1.0, 0, 0], [0.0, 1.0, 0], [0.0, 0, 1.0]])
        self.logAFMdataName = None
        self.logImgName = None
        self.logImgIzs = [0, -8, -1]
        self.xyzLogFile = None

        self.simulator = simulator
        self.relaxator = relaxator
        self.atoms = atoms
        self.bonds = bonds
        self.corrector = corrector
        self.xyzLogName = None

    def init(self):
        pass

    def startLoop(self, molecule, atomMap, bondMap, lvecMap, AFMRef):
        self.molecule = molecule
        self.atomMap = atomMap
        self.bondMap = bondMap
        self.mapLvec = lvecMap
        self.AFMRef = AFMRef
        self.bAuxMap = (self.atomMap is not None) and (self.bondMap is not None)

    def debug_plot(self, AFMs, AuxMaps=None):
        if self.logImgName is not None:
            plt = self.plt
            nz = len(self.logImgIzs)
            if self.bAuxMap:
                nch = AuxMaps.shape[2]
                plt.figure(figsize=(5 * (nz + nch), 5))
                for ich in range(nch):
                    plt.subplot(1, nz + nch, ich + 1)
                    plt.imshow(AuxMaps[:, :, ich])
            else:
                nch = 0
                plt.figure(figsize=(5 * (nz + nch), 5))

            for iiz, iz in enumerate(self.logImgIzs):
                plt.subplot(1, nz + nch, iiz + nch + 1)
                plt.imshow(AFMs[:, :, iz])
                plt.title("iz %i" % iz)
            plt.savefig(self.logImgName + ("_%03i.png" % itr), bbox_inches="tight")
            plt.close()

    def iteration(self, itr=0):
        if self.xyzLogFile is not None:
            io.saveXYZ(
                self.xyzLogFile,
                self.molecule.xyzs,
                self.molecule.Zs,
                qs=self.molecule.qs,
                comment=(f"CorrectionLoop.iteration [{itr}] "),
            )
        # Get AFM
        xyzs, qs, Zs = self.molecule.xyzs, self.molecule.qs, self.molecule.Zs
        AFMs = self.simulator(xyzs, Zs, qs)
        xyzqs = np.concatenate([xyzs, qs[:, None]], axis=1)
        # Get Atoms and Bonds AuxMaps
        AuxMaps = None
        if self.bAuxMap:
            atoms_map = self.atoms(xyzqs, Zs)
            bonds_map = self.bonds(xyzqs, Zs)
            AuxMaps = np.stack([atoms_map, bonds_map], axis=-1)
        if self.logAFMdataName:
            np.save(self.logAFMdataName + ("%03i.dat" % itr), AFMs)
        self.debug_plot(AFMs, AuxMaps)
        sw = self.simulator.scan_window
        Err, self.molecule = self.corrector.try_improve(self.molecule, AFMs, self.AFMRef, sw, itr=itr)
        return Err


def Job_trainCorrector(simulator, geom_fname="input.xyz", nstep=10):
    iz = -10
    mutator = Mutator()
    trainer = CorrectorTrainer(simulator, mutator, molCreator=None)
    xyzs, Zs, qs, _ = io.loadXYZ(geom_fname)

    sw = simulator.scan_window
    scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
    xyzs[:, :2] += scan_center - xyzs[:, :2].mean(axis=0)
    xyzs[:, 2] += (sw[1][2] - 9.0) - xyzs[:, 2].max()
    print("xyzs ", xyzs)
    mol = Molecule(xyzs, Zs, qs)

    trainer.start(mol)
    extent = (
        simulator.scan_window[0][0],
        simulator.scan_window[1][0],
        simulator.scan_window[0][1],
        simulator.scan_window[1][1],
    )
    sc = 3.0

    xyzfile = "geomMutations.xyz"
    io.saveXYZ(xyzfile, mol.xyzs, mol.Zs, qs=mol.qs, comment="# start ")
    for itr in range(nstep):
        Xs1, Xs2, mol1, mol2 = trainer[itr]
        io.saveXYZ(
            xyzfile,
            mol2.xyzs,
            mol2.Zs,
            qs=mol2.qs,
            comment=(f"# mutation {itr} "),
            append=True,
        )
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Xs1[:, :, iz], origin="upper", extent=extent)
        plt.scatter(
            mol1.xyzs[:, 0],
            mol1.xyzs[:, 1],
            s=mol1.Zs * sc,
            c=cm.rainbow(mol1.xyzs[:, 2]),
        )
        plt.subplot(1, 2, 2)
        plt.imshow(Xs2[:, :, iz], origin="upper", extent=extent)
        plt.scatter(
            mol2.xyzs[:, 0],
            mol2.xyzs[:, 1],
            s=mol2.Zs * sc,
            c=cm.rainbow(mol2.xyzs[:, 2]),
        )
        plt.savefig("CorrectorTrainAFM_%03i.png" % itr)
        plt.close()


def Job_CorrectionLoop(simulator, atoms, bonds, geom_fname="input.xyz", nstep=10):
    relaxator = FARFF.EngineFARFF()
    corrector = Corrector()
    corrector.logImgName = "AFM_Err"
    nscan = simulator.scan_dim
    nscan = (nscan[0], nscan[1], nscan[2] - len(simulator.dfWeight))
    np.save("AFMref.npy", np.zeros(nscan))
    AFMRef = np.load("AFMref.npy")

    looper = CorrectionLoop(relaxator, simulator, atoms, bonds, corrector)
    looper.xyzLogFile = "CorrectionLoopLog.xyz"
    looper.logImgName = "CorrectionLoopAFMLog"
    looper.logAFMdataName = "AFMs"
    xyzs, Zs, qs, _ = io.loadXYZ(geom_fname)

    sw = simulator.scan_window
    scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
    xyzs[:, :2] += scan_center - xyzs[:, :2].mean(axis=0)
    xyzs[:, 2] += (sw[1][2] - 9.0) - xyzs[:, 2].max()
    print("xyzs ", xyzs)

    xyzqs = np.concatenate([xyzs, qs[:, None]], axis=1)
    np.save("./Atoms.npy", atoms(xyzqs, Zs))
    np.save("./Bonds.npy", bonds(xyzqs, Zs))

    molecule = Molecule(xyzs, Zs, qs)
    atomMap, bondMap, lvecMap = FARFF.makeGridFF(FARFF, fname_atom="./Atoms.npy", fname_bond="./Bonds.npy", dx=0.1, dy=0.1)

    looper.startLoop(molecule, atomMap, bondMap, lvecMap, AFMRef)
    ErrConv = 0.1
    print("# ------ To Loop    ")
    for itr in range(nstep):
        print("# ======= CorrectionLoop[ %i ] ", itr)
        Err = looper.iteration(itr=itr)
        if Err < ErrConv:
            break


# ========================================================================

if __name__ == "__main__":
    from optparse import OptionParser

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    parser = OptionParser()
    parser.add_option("-j", "--job", action="store", type="string", help="[train/loop]")
    (options, args) = parser.parse_args()

    print(" UNIT_TEST START : CorrectionLoop ... ")

    print("# ------ Init Generator   ")

    i_platform = 0
    env = oclu.OCLEnvironment(i_platform=i_platform)
    FFcl.init(env)
    oclr.init(env)

    # fmt: off
    afmulator = AFMulator.AFMulator(
        pixPerAngstrome=10,
        lvec=np.array([
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 5.0]
            ]),
        scan_window=((2.0, 2.0, 5.0), (18.0, 18.0, 8.0)),
    )
    # fmt: on

    atoms = AuxMap.AtomRfunc(scan_dim=(128, 128), scan_window=((2, 2), (18, 18)))
    bonds = AuxMap.Bonds(scan_dim=(128, 128), scan_window=((2, 2), (18, 18)))

    if options.job == "loop":
        Job_CorrectionLoop(afmulator, atoms, bonds, geom_fname="pos_out3.xyz")
    elif options.job == "train":
        Job_trainCorrector(afmulator, geom_fname="pos_out3.xyz", nstep=10)
    else:
        print("ERROR : invalid job ", options.job)

    print(" UNIT_TEST CorrectionLoop DONE !!! ")
