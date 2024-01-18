import random
import time

import numpy as np

from .. import common as PPU
from .. import io


class InverseAFMtrainer:
    """
    A data generator for training machine learning models. Generates batches of input/output pairs.
    An iterator.

    Yields batches of samples (Xs, Ys, mols), where Xs are AFM images, Ys are aux map descriptors, and mols
    are molecules. Xs is a list of np.ndarray of shape (n_batch, sx, sy, sz), Ys is a list of np.ndarray of
    shape (n_batch, sx, sy), and mols is a list of length n_batch of np.ndarray of shape (n_atoms, 5), where
    n_batch is the batch size, sx, sy, and sz are the scan sizes in x, y, and z dimensions, respectively,
    and n_atoms is the number of atoms. The outer lists correspond to the tip number in Xs, the aux map number
    in Ys. In mols the rows of the arrays correspond to the x, y, and z coordinates, the charge, and the element
    of each atom.

    Arguments:
        afmulator: An instance of AFMulator.
        auxmaps: list of :class:`.AuxMapBase`.
        paths: list of paths to xyz files of molecules. The molecules are saved to the "molecules" attribute
               in np.ndarrays of shape (num_atoms, 5) with [x, y, z, charge, element] for each atom.
        batch_size: int. Number of samples per batch.
        distAbove: float. Tip-sample distance parameter.
        iZPPs: list of ints. Elements for AFM tips. Image is produced with every tip for each sample.
        Qs: list of arrays of length 4. Charges for tips.
        QZS list of arrays of length 4. Positions of tip charges.
    """

    # Print timings during excecution
    bRuntime = False

    def __init__(
        self,
        afmulator,
        aux_maps,
        paths,
        batch_size=30,
        distAbove=5.3,
        iZPPs=[8],
        Qs=[[-10, 20, -10, 0]],
        QZs=[[0.1, 0, -0.1, 0]],
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

            if self.bRuntime:
                batch_start = time.time()

            for s in range(batch_size):
                if self.bRuntime:
                    sample_start = time.time()

                # Load molecule
                mol = self.molecules[self.counter]
                mols.append(mol)
                self.xyzs = mol[:, :3]
                self.qs = mol[:, 3]
                self.Zs = mol[:, 4].astype(np.int32)

                # Make sure the molecule is in right position
                self.handle_positions()

                # Callback
                self.on_sample_start()

                # Get AFM
                for i, (iZPP, Q, Qz) in enumerate(zip(self.iZPPs, self.Qs, self.QZs)):  # Loop over different tips
                    # Set interaction parameters
                    self.afmulator.iZPP = iZPP
                    self.afmulator.setQs(Q, Qz)
                    self.REAs = PPU.getAtomsREA(
                        self.afmulator.iZPP,
                        self.Zs,
                        self.afmulator.typeParams,
                        alphaFac=-1.0,
                    )

                    # Make sure tip-sample distance is right
                    self.handle_distance()

                    # Callback
                    self.on_afm_start()

                    # Evaluate AFM
                    if self.bRuntime:
                        afm_start = time.time()
                    Xs[i].append(self.afmulator(self.xyzs, self.Zs, self.qs, REAs=self.REAs))
                    if self.bRuntime:
                        print(f"AFM {i} runtime [s]: {time.time() - afm_start}")

                    self.Xs = Xs[i][-1]
                    # Callback
                    self.on_afm_end()

                # Get AuxMaps
                for i, aux_map in enumerate(self.aux_maps):
                    if self.bRuntime:
                        aux_start = time.time()
                    xyzqs = np.concatenate([self.xyzs, self.qs[:, None]], axis=1)
                    Ys[i].append(aux_map(xyzqs, self.Zs))
                    if self.bRuntime:
                        print(f"AuxMap {i} runtime [s]: {time.time() - aux_start}")

                if self.bRuntime:
                    print(f"Sample {s} runtime [s]: {time.time() - sample_start}")
                self.counter += 1

            for i in range(len(self.iZPPs)):
                Xs[i] = np.stack(Xs[i], axis=0)

            for i in range(len(self.aux_maps)):
                Ys[i] = np.stack(Ys[i], axis=0)

            if self.bRuntime:
                print(f"Batch runtime [s]: {time.time() - batch_start}")

        else:
            raise StopIteration

        return Xs, Ys, mols

    def __iter__(self):
        self.counter = 0
        return self

    def __len__(self):
        """
        Returns the number of batches that will be generated with the current molecules.
        """
        return int(np.floor(len(self.molecules) / self.batch_size))

    def read_xyzs(self):
        """
        Read molecule xyz files from selected paths.
        """
        self.molecules = []
        for path in self.paths:
            xyzs, Zs, qs, _ = io.loadXYZ(path)
            self.molecules.append(np.concatenate([xyzs, qs[:, None], Zs[:, None]], axis=1))

    def handle_positions(self):
        """
        Set current molecule to the center of the scan window.
        """
        sw = self.afmulator.scan_window
        scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
        self.xyzs[:, :2] += scan_center - self.xyzs[:, :2].mean(axis=0)

    def handle_distance(self):
        """
        Set correct distance from scan region for the current molecule.
        """
        RvdwPP = self.afmulator.typeParams[self.afmulator.iZPP - 1][0]
        Rvdw = self.REAs[:, 0] - RvdwPP
        zs = self.xyzs[:, 2]
        imax = np.argmax(zs + Rvdw)
        total_distance = self.distAboveActive + Rvdw[imax] + RvdwPP - (zs.max() - zs[imax])
        self.xyzs[:, 2] += (self.afmulator.scan_window[1][2] - total_distance) - zs.max()

    # ======== Augmentation =========

    def shuffle_molecules(self):
        """
        Shuffle list of molecules.
        """
        random.shuffle(self.molecules)

    def augment_with_rotations(self, rotations):
        """
        Augment molecule list with rotations of the molecules.

        Arguments:
            rotations: list of np.ndarray. Rotation matrices.
        """
        molecules = self.molecules
        self.molecules = []
        for mol in molecules:
            xyzs = mol[:, :3]
            qs = mol[:, 3]
            Zs = mol[:, 4]
            for xyzs_rot in rotate(xyzs, rotations):
                self.molecules.append(np.concatenate([xyzs_rot, qs[:, None], Zs[:, None]], axis=1))

    def augment_with_rotations_entropy(self, rotations, n_best_rotations=30):
        """
        Augment molecule list with rotations of the molecules. Rotations are sorted in terms of their "entropy".

        Arguments:
            rotations: list of np.ndarray. Rotation matrices.
            n_best_rotations: int. Only the first n_best_rotations with the highest "entropy" will be taken.
        """
        molecules = self.molecules
        self.molecules = []
        for mol in molecules:
            xyzs = mol[:, :3]
            qs = mol[:, 3]
            Zs = mol[:, 4]
            rots = sortRotationsByEntropy(mol[:, :3], rotations)[:n_best_rotations]
            for xyzs_rot in rotate(xyzs, rots):
                self.molecules.append(np.concatenate([xyzs_rot, qs[:, None], Zs[:, None]], axis=1))

    def randomize_tip(self, max_tilt=0.5):
        """
        Randomize tip tilt to simulate asymmetric adsorption of particle on tip apex.

        Arguments:
            max_tilt: float. Maximum deviation in xy plane in angstroms.
        """
        self.afmulator.tipR0[:2] = np.array(getRandomUniformDisk()) * max_tilt

    def randomize_distance(self, delta=0.25):
        """
        Randomize tip-sample distance.

        Arguments:
            delta: float. Maximum deviation from original value in angstroms.
        """
        self.distAboveActive = np.random.uniform(self.distAbove - delta, self.distAbove + delta)

    def randomize_mol_parameters(self, rndQmax=0.0, rndRmax=0.0, rndEmax=0.0, rndAlphaMax=0.0):
        """
        Randomize various interaction parameters for current molecule.
        """
        num_atoms = len(self.qs)
        if rndQmax > 0:
            self.qs[:] += rndQmax * (np.random.rand(num_atoms) - 0.5)
        if rndRmax > 0:
            self.REAs[:, 0] += rndRmax * (np.random.rand(num_atoms) - 0.5)
        if rndEmax > 0:
            self.REAs[:, 1] *= 1 + rndEmax * (np.random.rand(num_atoms) - 0.5)
        if rndAlphaMax > 0:
            self.REAs[:, 2] *= 1 + rndAlphaMax * (np.random.rand(num_atoms) - 0.5)

    # ====== Callback methods =======

    def on_batch_start(self):
        """
        Excecuted right at the start of each batch. Override to modify parameters for each batch.
        """

    def on_sample_start(self):
        """
        Excecuted right before evaluating first AFM image. Override to modify the parameters for each sample.
        """

    def on_afm_start(self):
        """
        Excecuted right before every AFM image evalution. Override to modify the parameters for each AFM image.
        """

    def on_afm_end(self):
        """
        Excecuted right after evaluating AFM image. Override to modify the parameters for each sample.
        """


class HartreeAFMtrainer(InverseAFMtrainer):
    """
    Generate batches of input/output pairs for machine learning based on Hartree potentials. An iterator.

    Arguments:
        afmulator: An instance of AFMulator.
        auxmaps: list of :class:`.AuxMapBase`.
        sample_generator: Iterable. An iterable that returns tuples (hartree, xyzs, Zs, rotations), where
            hartree is a HartreePotential, xyzs is a np.ndarray of shape (n_atoms, 3), Zs is a np.ndarray of
            shape (n_atoms,), and rotations is a list of np.ndarray of shape (3, 3). Input/output samples will be
            generated for each rotation in rotations of the molecule specified by the atomic coordinates in
            xyzs, elements in Zs, and Hartree potential in hartree.
        batch_size: int. Number of samples per batch.
        distAbove: float. Tip-sample distance parameter.
        iZPPs: list of int. Elements for AFM tips. Image is produced with every tip for each sample.
        rhos: list of dict or TipDensity. Tip charge densities.
    """

    def __init__(
        self,
        afmulator,
        aux_maps,
        sample_generator,
        batch_size=30,
        distAbove=5.3,
        iZPPs=[8],
        rhos=[{"dz2": -0.1}],
    ):
        assert len(iZPPs) == len(rhos)

        Qs = QZs = [[0, 0, 0, 0] for _ in range(len(iZPPs))]
        super().__init__(afmulator, aux_maps, [], batch_size, distAbove, iZPPs, Qs, QZs)

        self.sample_generator = sample_generator
        self._prepareBuffers(rhos)

        sw = self.afmulator.scan_window
        self.scan_window = sw
        self.scan_size = (sw[1][0] - sw[0][0], sw[1][1] - sw[0][1], sw[1][2] - sw[0][2])
        self.scan_dim = self.afmulator.scan_dim
        self.df_steps = self.afmulator.df_steps
        self.z_size = self.scan_dim[2] - self.df_steps + 1

    def _prepareBuffers(self, rhos):
        self.rhos = []
        self.ffts = []
        for rho in rhos:
            self.afmulator.setRho(rho)
            self.rhos.append(self.afmulator.forcefield.rho)
            self.ffts.append(self.afmulator.forcefield.fft_conv)

    def __next__(self):
        if self.iteration_done:
            raise StopIteration

        # Callback
        self.on_batch_start()

        mols = []
        Xs = [[] for _ in range(len(self.iZPPs))]
        Ys = [[] for _ in range(len(self.aux_maps))]
        scan_windows = [[] for _ in range(len(self.iZPPs))]

        if self.bRuntime:
            batch_start = time.perf_counter()

        for s in range(self.batch_size):
            if self.bRuntime:
                sample_start = time.perf_counter()

            # Load sample
            if len(self.rots) == 0:
                if self.pot:
                    self.pot.release()
                try:
                    self.pot, self.xyzs, self.Zs, self.rots = next(self.sample_iterator)
                    self.qs = np.zeros(len(self.xyzs))
                except StopIteration:
                    self.iteration_done = True
                    break

            # Get rotation
            rot = self.rots.pop(0)
            xyz_center = self.xyzs.mean(axis=0)
            self.xyzs_rot = np.dot(self.xyzs - xyz_center, rot.T) + xyz_center
            mol = np.concatenate([self.xyzs_rot, self.qs[:, None], self.Zs[:, None]], axis=1)
            mols.append(mol)

            # Make sure the molecule is in right position
            self.handle_positions()

            # Callback
            self.on_sample_start()

            if self.bRuntime:
                print(f"Sample {s} preparation time [s]: {time.perf_counter() - sample_start}")

            # Get AFM
            for i, (iZPP, rho, fft) in enumerate(zip(self.iZPPs, self.rhos, self.ffts)):  # Loop over different tips
                # Set interaction parameters
                self.afmulator.iZPP = iZPP
                self.afmulator.forcefield.rho = rho
                self.afmulator.forcefield.fft_conv = fft
                self.REAs = PPU.getAtomsREA(
                    self.afmulator.iZPP,
                    self.Zs,
                    self.afmulator.typeParams,
                    alphaFac=-1.0,
                )

                # Make sure tip-sample distance is right
                self.handle_distance()

                # Set AFMulator scan window and force field lattice vectors
                self.afmulator.setScanWindow(self.scan_window, self.scan_dim, df_steps=self.df_steps)
                self.afmulator.setLvec()

                # Callback
                self.on_afm_start()

                # Evaluate AFM
                if self.bRuntime:
                    afm_start = time.perf_counter()
                Xs[i].append(self.afmulator(self.xyzs, self.Zs, self.pot, rot=rot, REAs=self.REAs))
                if self.bRuntime:
                    print(f"AFM {i} runtime [s]: {time.perf_counter() - afm_start}")

                scan_windows[i].append(np.array(self.scan_window))

            # Get AuxMaps
            for i, aux_map in enumerate(self.aux_maps):
                if self.bRuntime:
                    aux_start = time.perf_counter()
                xyzqs = np.concatenate([self.xyzs, self.qs[:, None]], axis=1)
                Ys[i].append(aux_map(xyzqs, self.Zs, self.pot, rot))
                if self.bRuntime:
                    print(f"AuxMap {i} runtime [s]: {time.perf_counter() - aux_start}")

            if self.bRuntime:
                print(f"Sample {s} runtime [s]: {time.perf_counter() - sample_start}")

        if len(mols) == 0:  # Sample iterator was empty
            raise StopIteration

        Xs = [np.stack(x, axis=0) for x in Xs]
        Ys = [np.stack(y, axis=0) for y in Ys]
        scan_windows = [np.stack(sw, axis=0) for sw in scan_windows]

        if self.bRuntime:
            print(f"Batch runtime [s]: {time.perf_counter() - batch_start}")

        return Xs, Ys, mols, scan_windows

    def __iter__(self):
        self.pot = None
        self.xyzs = None
        self.Zs = None
        self.rots = []
        self.sample_iterator = iter(self.sample_generator)
        self.iteration_done = False
        return self

    def __len__(self):
        """
        Returns the number of batches that will be generated. Requires for the sample generator
        to have attribute __len__ that returns the total number of samples (including rotations).
        """
        if not hasattr(self.sample_generator, "__len__"):
            raise RuntimeError("Cannot infer the number of batches because sample generator does not have length attribute.")
        return int(np.floor(len(self.sample_generator) / self.batch_size))

    def handle_positions(self):
        """
        Shift scan window laterally to center on the molecule.
        """
        ss = self.scan_size
        sw = self.scan_window
        xy_center = self.xyzs[:, :2].mean(axis=0)
        sw = (
            (xy_center[0] - ss[0] / 2, xy_center[1] - ss[1] / 2, sw[0][2]),
            (xy_center[0] + ss[0] / 2, xy_center[1] + ss[1] / 2, sw[1][2]),
        )
        self.scan_window = sw
        for aux_map in self.aux_maps:
            aux_map.scan_window = ((sw[0][0], sw[0][1]), (sw[1][0], sw[1][1]))

    def handle_distance(self):
        """
        Set correct distance of the scan window from the current molecule.
        """
        RvdwPP = self.afmulator.typeParams[self.afmulator.iZPP - 1][0]
        Rvdw = self.REAs[:, 0] - RvdwPP
        zs = self.xyzs_rot[:, 2]
        imax = np.argmax(zs + Rvdw)
        total_distance = self.distAboveActive + Rvdw[imax] + RvdwPP - (zs.max() - zs[imax])
        z_min = self.xyzs_rot[:, 2].max() + total_distance
        sw = self.scan_window
        self.scan_window = (
            (sw[0][0], sw[0][1], z_min),
            (sw[1][0], sw[1][1], z_min + self.scan_size[2]),
        )

    def randomize_df_steps(self, minimum=4, maximum=20):
        """Randomize oscillation amplitude by randomizing the number of steps in df convolution.

        Chosen number of df steps is uniform random between minimum and maximum. Modifies self.scan_dim and
        self.scan_size to retain same output z dimension and same dz step for the chosen number of df steps.

        Arguments:
            minimum: int. Minimum number of df steps (inclusive).
            maximum: int. Maximum number of df steps (inclusive).
        """
        self.df_steps = np.random.randint(minimum, maximum + 1)
        self.scan_dim = (
            self.scan_dim[0],
            self.scan_dim[1],
            self.z_size + self.df_steps - 1,
        )
        self.scan_size = (
            self.scan_size[0],
            self.scan_size[1],
            self.afmulator.dz * self.scan_dim[2],
        )


def sortRotationsByEntropy(xyzs, rotations):
    rots = []
    for rot in rotations:
        zDir = rot[2].flat.copy()
        _, _, entropy = PPU.maxAlongDirEntropy(xyzs, zDir)
        rots.append((entropy, rot))
    rots.sort(key=lambda item: -item[0])
    rots = [rot[1] for rot in rots]
    return rots


def rotate(xyzs, rotations):
    rotated_xyzs = []
    for rot in rotations:
        rotated_xyzs.append(np.dot(xyzs, rot.T))
    return rotated_xyzs


def getRandomUniformDisk():
    """
    generate points unifromly distributed over disk
    # see: http://mathworld.wolfram.com/DiskPointPicking.html
    """
    rnd = np.random.rand(2)
    rnd[0] = np.sqrt(rnd[0])
    rnd[1] *= 2.0 * np.pi
    return rnd[0] * np.cos(rnd[1]), rnd[0] * np.sin(rnd[1])
