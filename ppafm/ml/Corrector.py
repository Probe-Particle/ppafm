import matplotlib
import numpy as np
import pyopencl as cl

from .. import atomicUtils as au
from .. import common as PPU
from .. import elements, io
from ..dev import SimplePot as pot

# ======================================================
# ================== Class  Molecule
# ======================================================


class Molecule:
    """
    A simple class for Molecule data containing atom xyz positions, element types and charges.
    Arguments:
        xyzs: list, xyz positions of atoms in the molecule
        Zs: list, element types
        qs: list, charges
    """

    def __init__(self, xyzs, Zs, qs):
        self.xyzs = xyzs
        self.Zs = Zs
        self.qs = qs
        self.array = np.c_[self.xyzs, self.Zs, self.qs]

    def clone(self):
        xyzs = self.xyzs
        Zs = self.Zs
        qs = self.qs
        return Molecule(xyzs, Zs, qs)

    def __str__(self):
        return np.c_[self.xyzs, self.Zs, self.qs].__str__()

    def __len__(self):
        return len(self.xyzs)

    def cog(self):
        n = len(self.xyzs)
        x = self.xyzs[:, 0].sum()
        y = self.xyzs[:, 1].sum()
        z = self.xyzs[:, 2].sum()
        return np.array([x, y, z]) / n

    def toXYZ(self, xyzfile, comment="#comment"):
        io.saveXYZ(xyzfile, self.xyzs, self.Zs, qs=self.qs, comment=comment)


# ======================================================
# ================== free function operating on Molecule
# ======================================================


def removeAtoms(molecule, nmax=1):
    xyzs = molecule.xyzs
    Zs = molecule.Zs
    qs = molecule.qs

    nmax = min(nmax, xyzs.shape[0])
    rem_idx = np.random.randint(nmax + 1)
    sel = np.random.choice(np.arange(xyzs.shape[0]), rem_idx, replace=False)

    xyzs_ = np.delete(xyzs, sel, axis=0)
    Zs_ = np.delete(Zs, sel)
    qs_ = np.delete(qs, sel)
    return Molecule(xyzs_, Zs_, qs_), sel


def addAtom_bare(mol, xyz, Z, q):
    # fmt: off
    Zs_ = np.append(mol.Zs, np.array([Z,]), axis=0,)
    xyzs_ = np.append(mol.xyzs, xyz[None, :], axis=0)
    qs_ = np.append(mol.qs, np.array([q,]), axis=0)
    # fmt: on
    return Molecule(xyzs_, Zs_, qs_)


def addAtom(molecule, p0, R=0.0, Z0=1, q0=0.0, dq=0.0):
    # ToDo : it would be nice to add atoms in a more physicaly reasonable way - not overlaping, proper bond-order etc.
    # ToDo : currently we add always hydrogen - should we somewhere randomly pick different atom types ?
    xyzs = molecule.xyzs
    Zs = molecule.Zs
    qs = molecule.qs
    min_x = xyzs.min(axis=0)
    max_x = xyzs.max(axis=0)
    min_x[2] = max_x[2] - 1.0
    nx = np.random.uniform(min_x, max_x)[np.newaxis, ...]

    min_q = qs.min(axis=0)
    max_q = qs.max(axis=0)
    nq = np.random.uniform(min_q, max_q, (1,))

    # fmt: off
    Zs_ = np.append(Zs, np.array([Z0,]), axis=0)
    # fmt: on
    xyzs_ = np.append(xyzs, nx, axis=0)
    qs_ = np.append(qs, nq, axis=0)
    return Molecule(xyzs_, Zs_, qs_)


def moveAtom(molecule, ia, dpMax=np.array([1.0, 1.0, 0.25])):
    xyzs = molecule.xyzs.copy()
    Zs = molecule.Zs.copy()
    qs = molecule.qs.copy()
    xyzs[ia, :] += (np.random.rand(3) - 0.5) * dpMax
    return Molecule(xyzs, Zs, qs)


def moveAtoms(molecule, p0, R=0.0, dpMax=np.array([1.0, 1.0, 0.25]), nmax=1):
    # ToDo : it would be nice to add atoms in a more physicaly reasonable way - not overlaping, proper bond-order etc.
    # ToDo : currently we add always hydrogen - should we somewhere randomly pick different atom types ?
    xyzs = molecule.xyzs.copy()
    Zs = molecule.Zs.copy()
    qs = molecule.qs.copy()
    # --- choose nmax closest atoms
    rs = (xyzs[:, 0] - p0[0]) ** 2 + (xyzs[:, 1] - p0[1]) ** 2
    sel = rs.argsort()[:nmax]  # [::-1]
    # --- move them randomly
    xyzs[sel, :] += (np.random.rand(len(sel), 3) - 0.5) * dpMax[None, :]
    return Molecule(xyzs, Zs, qs)


def moveMultipleAtoms(molecule: Molecule, scale=0.5, nmax=10):
    """
    A function for slightly shifting multiple atoms in the molecule.
    """
    nmax = nmax if nmax < len(molecule) else len(molecule)

    move_n = np.random.randint(nmax + 1)
    to_be_moved = np.random.choice(len(molecule), move_n, replace=False)
    magnitudes = np.random.uniform(-scale, scale, (move_n, 3))

    xyzs_ = molecule.xyzs.copy()
    Zs_ = molecule.Zs.copy()
    qs_ = molecule.qs.copy()

    xyzs_[to_be_moved] = xyzs_[to_be_moved] + magnitudes

    return Molecule(xyzs_, Zs_, qs_)


def paretoNorm(Err1, Err2):
    """
    this norm evaluate ares where Error got better (sum1), and where error got worse (sum2)
    we only accept moves where sum1>0 and sum2~0
    motivation is to make sure our move does not make things worse
    """
    diff = Err1 - Err2
    Ebetter = diff[diff > 0].sum()  # got better
    Eworse = -diff[diff < 0].sum()  # got worse
    return Ebetter, Eworse


def paretoNorm_(Err1, Err2, trash=0.0000001):
    """
    this norm evaluate ares where Error got better (sum1), and where error got worse (sum2)
    we only accept moves where sum1>0 and sum2~0
    motivation is to make sure our move does not make things worse
    """
    diff = Err1 - Err2  # positive diff is improvement
    Ebetter = diff - trash
    Ebetter[Ebetter < 0] = 0
    Eworse = -diff - trash
    Eworse[Eworse < 0] = 0
    return Ebetter, Eworse


def blur(F):
    return (F[:-1, :-1, :] + F[1:, :-1, :] + F[:-1, 1:, :] + F[1:, 1:, :]) * 0.25


def halfRes(F):
    return (F[:-1:2, :-1:2, :] + F[:-1:2, 1::2, :] + F[1::2, :-1:2, :] + F[1::2, 1::2, :]) * 0.25


def lowResErrorMap(Err3D):
    Err2D = np.empty(Err3D.shape[:2] + (2,))
    Err2D[:, :, 0] = Err3D.sum(axis=2) / Err3D.shape[2]
    Err2D[:, :, 1] = Err2D[:, :, 0]
    ErrLo = halfRes(Err2D)
    ErrLo = halfRes(ErrLo)
    ErrLo = halfRes(ErrLo)
    return blur(ErrLo).astype(np.float64)


# ======================================================
# ================== Class  Corrector
# ======================================================


class Corrector:
    def __init__(self):
        self.izPlot = -8
        self.logImgName = None
        self.xyzLogFile = None

        self.best_E = None
        self.best_mol = None
        self.best_diff = None
        self.dpMax = np.array([0.5, 0.5, 0.15])

    def modifyStructure(self, molIn):
        if self.best_ps_i < len(self.best_ps):
            p = self.best_ps[self.best_ps_i]
        else:
            self.best_ps = pot.genAtoms(npick=10, Rcov=0.7, kT=self.kT)
            self.best_ps_i = 1
            p = self.best_ps[0]
        self.best_ps_i += 1
        molOut = addAtom_bare(molIn, p, 6, 0)
        return molOut

    def debug_plot(self, itr, AFM_Err, AFM_ErrSub, AFMs, AFMRef, Err):
        if self.logImgName is not None:
            plt = self.plt
            plt.figure(figsize=(5 * 4, 5))
            vmax = AFMRef[:, :, self.izPlot].max()
            vmin = AFMRef[:, :, self.izPlot].min()
            v2max = np.maximum(vmin**2, vmax**2)
            plt.subplot(1, 4, 1)
            plt.imshow(AFMRef[:, :, self.izPlot])
            # plt.title("AFMref" ); plt.grid()
            plt.subplot(1, 4, 2)
            plt.imshow(AFMs[:, :, self.izPlot], vmin=vmin, vmax=vmax)
            # plt.title("AFM[]"  ); plt.grid()
            plt.subplot(1, 4, 3)
            plt.imshow(
                np.sqrt(AFM_Err[:, :, self.izPlot]),
                vmin=0,
                vmax=np.sqrt(v2max) * 0.2,
                interpolation="nearest",
            )
            # plt.title("AFMdiff"); plt.grid()
            plt.subplot(1, 4, 4)
            plt.imshow(
                np.sqrt(AFM_ErrSub[:, :, self.izPlot]),
                vmin=0,
                vmax=np.sqrt(v2max) * 0.2,
                interpolation="bilinear",
            )
            plt.title("Error=%g" % Err)
            plt.savefig(self.logImgName + ("_%03i.png" % itr), bbox_inches="tight")
            plt.close()

    def debug_prob_map(self, molIn):
        ps, Ws = pot.genAtomWs(kT=2.0e-5, Rcov=0.7, natom=10000)
        na0 = len(molIn.Zs)
        mask = Ws > 0
        ps = ps[mask]
        Ws = Ws[mask]
        nps = len(ps)
        print("na0,nps", na0, nps)
        Zs2 = np.concatenate([molIn.Zs, np.ones(nps, dtype=np.int32)])
        Rs = np.concatenate([np.ones(na0), Ws])
        xyzs2 = np.concatenate([molIn.xyzs, ps], axis=0)
        print("xyzs2.shape ", xyzs2.shape)
        _saveXYZDebug(Zs2, xyzs2, "debug_genAtomWs.xyz", qs=([0.0] * (na0 + nps)), Rs=Rs)

    def try_improve(self, molIn, AFMs, AFMRef, span, itr=0):
        AFMdiff = AFMs - AFMRef
        AFMdiff = blur(AFMdiff)
        AFMdiff = blur(AFMdiff)  # BLUR
        AFMdiff2 = AFMdiff**2
        Err = np.sqrt(AFMdiff2.sum())  # root mean square error

        # ToDo : identify are of most difference and make random changes in that area
        Eworse = 0
        bBetter = False
        if self.best_E is not None:
            if self.best_E > Err:
                ErrB, ErrW = paretoNorm_(self.best_diff, AFMdiff2)
                Eworse = ErrW.sum()
                ErrPar = Err + 2.0 * Eworse
                print("\nmaybe better ? ", self.best_E, " <? ", ErrPar, " Eworse ", Eworse)
                if self.best_E > ErrPar:  # check if some areas does not got worse
                    bBetter = True
                    print(
                        "[%i]SUCCESS : Err:" % itr,
                        Err,
                        " best: ",
                        self.best_E,
                        " Eworse ",
                        Eworse,
                    )
            if not bBetter:
                print("*", end="", flush=True)

        if (self.best_E is None) or bBetter:
            ErrLo = lowResErrorMap(AFMdiff2).astype(np.float64)

            self.debug_plot(itr, AFMdiff2, ErrLo, AFMs, AFMRef, Err)
            self.best_mol = molIn
            self.best_E = Err
            self.best_diff = AFMdiff2
            self.best_ErrMap = ErrLo
            pot.setGridSize(ErrLo.shape, span[0], span[1])
            pot.setGridPointer(ErrLo)
            pot.init(self.best_mol.xyzs)
            self.kT = 2.0e-5
            self.best_ps = pot.genAtoms(npick=10, Rcov=0.7, kT=self.kT)
            self.best_ps_i = 0
            if self.xyzLogFile is not None:
                self.best_mol.toXYZ(
                    self.xyzLogFile,
                    comment=("Corrector [%i] Err %g " % (itr, self.best_E)),
                )
        molOut = self.modifyStructure(self.best_mol)
        return Err, molOut


def _saveXYZDebug(es, xyzs, fname, qs, Rs):
    with open(fname, "w") as fout:
        fout.write(f"{len(xyzs)}\n\n")
        for i, xyz in enumerate(xyzs):
            fout.write(f"{es[i]} {xyz[0]} {xyz[1]} {xyz[2]} {qs[i]} {Rs[i]} \n")


# ======================================================
# ================== Class  Mutator
# ======================================================


class Mutator:
    """
    A class for creating simple mutations to a molecule. The current state of the class allows for adding,
    removing and moving atoms. The mutations are not necessarily physically sensible at this point.
    Parameters:
        maxMutations: Integer. Maximum number of mutations per mutant. The number of mutations is chosen
        randomly between [1, maxMutations].
    """

    # Strtegies contain
    strategies = [(0.1, removeAtoms, {}), (0.0, addAtom, {}), (0.0, moveAtom, {})]

    def __init__(self, maxMutations=10):
        self.setStrategies()
        self.maxMutations = maxMutations

    def setStrategies(self, strategies=None):
        if strategies is not None:
            self.strategies = strategies
        self.cumProbs = np.cumsum([it[0] for it in self.strategies])

    def mutate_local(self, molecule):
        molecule, removed = removeAtoms(molecule, nmax=self.maxMutations)
        return molecule, removed
