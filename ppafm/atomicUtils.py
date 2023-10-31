#!/usr/bin/python

import math

import numpy as np

from . import elements


def neighs(natoms, bonds):
    neighs = [{} for i in range(natoms)]
    for ib, b in enumerate(bonds):
        i = b[0]
        j = b[1]
        neighs[i][j] = ib
        neighs[j][i] = ib
    return neighs


def findTypeNeigh(atoms, neighs, typ, neighTyps=[(1, 2, 2)]):
    typ_mask = atoms[:, 0] == typ
    satoms = atoms[typ_mask]
    iatoms = np.arange(len(atoms), dtype=int)[typ_mask]
    selected = []
    for i, atom in enumerate(satoms):
        iatom = iatoms[i]
        count = {}
        for jatom in neighs[iatom]:
            jtyp = atoms[jatom, 0]
            count[jtyp] = count.get(jtyp, 0) + 1
        for jtyp, (nmin, nmax) in list(neighTyps.items()):
            n = count.get(jtyp, 0)
            if (n >= nmin) and (n <= nmax):
                selected.append(iatom)
    return selected


def getAllNeighsOfSelected(selected, neighs, atoms, typs={1}):
    result = {}
    for iatom in selected:
        for jatom in neighs[iatom]:
            if atoms[jatom, 0] in typs:
                if jatom in result:
                    result[jatom].append(iatom)
                else:
                    result[jatom] = [iatom]
    return result


def findPairs(select1, select2, atoms, Rcut=2.0):
    ps = atoms[select2, 1:]
    Rcut2 = Rcut * Rcut
    pairs = []
    select2 = np.array(select2)
    for iatom in select1:
        p = atoms[iatom, 1:]
        rs = np.sum((ps - p) ** 2, axis=1)
        for jatom in select2[rs < Rcut2]:
            pairs.append((iatom, jatom))
    return pairs


def findPairs_one(select1, atoms, Rcut=2.0):
    ps = atoms[select1, 1:]
    Rcut2 = Rcut * Rcut
    pairs = []
    select1 = np.array(select1)
    for i, iatom in enumerate(select1):
        p = atoms[iatom, 1:]
        rs = np.sum((ps - p) ** 2, axis=1)
        for jatom in select1[:i][rs[:i] < Rcut2]:
            pairs.append((iatom, jatom))
    return pairs


def pairsNotShareNeigh(pairs, neighs):
    pairs_ = []
    for pair in pairs:
        ngis = neighs[pair[0]]
        ngjs = neighs[pair[1]]
        share_ng = False
        for ngi in ngis:
            if ngi in ngjs:
                share_ng = True
                break
        if not share_ng:
            pairs_.append(pair)
    return pairs_


def makeRotMat(fw, up):
    fw = fw / np.linalg.norm(fw)
    up = up - fw * np.dot(up, fw)
    up = up / np.linalg.norm(up)
    left = np.cross(fw, up)
    left = left / np.linalg.norm(left)
    return np.array([left, up, fw])


def groupToPair(p1, p2, group, up, up_by_cog=False):
    center = (p1 + p2) * 0.5
    fw = p2 - p1
    if up_by_cog:
        up = center - up
    rotmat = makeRotMat(fw, up)
    ps = group[:, 1:]
    ps_ = np.dot(ps, rotmat)
    group[:, 1:] = ps_ + center
    return group


def replacePairs(pairs, atoms, group, up_vec=(np.array((0.0, 0.0, 0.0)), 1)):
    replaceDict = {}
    for ipair, pair in enumerate(pairs):
        for iatom in pair:
            replaceDict[iatom] = 1
    atoms_ = []
    for iatom, atom in enumerate(atoms):
        if iatom in replaceDict:
            continue
        atoms_.append(atom)
    for pair in pairs:
        group_ = groupToPair(atoms[pair[0], 1:], atoms[pair[1], 1:], group.copy(), up_vec[0], up_vec[1])
        for atom in group_:
            atoms_.append(atom)
    return atoms_


def findNearest(p, ps, rcut=1e9):
    rs = np.sum((ps - p) ** 2, axis=1)
    imin = np.argmin(rs)
    if rs[imin] < (rcut**2):
        return imin
    else:
        return -1


def countTypeBonds(atoms, ofAtoms, rcut):
    bond_counts = np.zeros(len(atoms), dtype=int)
    ps = ofAtoms[:, 1:]
    for i, atom in enumerate(atoms):
        p = atom[1:]
        rs = np.sum((ps - p) ** 2, axis=1)
        bond_counts[i] = np.sum(rs < (rcut**2))
    return bond_counts


def replace(atoms, found, to=17, bond_length=2.0, radial=0.0, prob=0.75):
    replace_mask = np.random.rand(len(found)) < prob
    for i, foundi in enumerate(found):
        if replace_mask[i]:
            iatom = foundi[0]
            bvec = foundi[1]
            rb = np.linalg.norm(bvec)
            bvec *= (bond_length - rb) / rb
            atoms[iatom, 0] = to
            atoms[iatom, 1:] += bvec
    return atoms


def loadCoefs(characters=["s"]):
    dens = None
    coefs = []
    for char in characters:
        fname = "phi_0000_%s.dat" % char
        print(fname)
        raw = np.genfromtxt(fname, skip_header=1)
        Es = raw[:, 0]
        cs = raw[:, 1:]
        sh = cs.shape
        print(("shape : ", sh))
        cs = cs.reshape(sh[0], sh[1] // 2, 2)
        d = cs[:, :, 0] ** 2 + cs[:, :, 1] ** 2
        coefs.append(cs[:, :, 0] + 1j * cs[:, :, 1])
        if dens is None:
            dens = d
        else:
            dens += d
    return dens, coefs, Es


def findCOG(ps, byBox=False):
    if byBox:
        # fmt: off
        xmin=ps[:,0].min(); xmax=ps[:,0].max();
        ymin=ps[:,1].min(); ymax=ps[:,1].max();
        zmin=ps[:,2].min(); zmax=ps[:,2].max();
        # fmt: on
        return np.array((xmin + xmax, ymin + ymax, zmin + zmax)) * 0.5
    else:
        cog = np.sum(ps, axis=0)
        cog *= 1.0 / len(ps)
        return cog


def histR(ps, dbin=None, Rmax=None, weights=None):
    rs = np.sqrt(np.sum((ps * ps), axis=1))
    bins = 100
    if dbin is not None:
        if Rmax is None:
            Rmax = rs.max() + 0.5
        bins = np.linspace(0, Rmax, int(Rmax / (dbin)) + 1)
    print((rs.shape, weights.shape))
    return np.histogram(rs, bins, weights=weights)


def ZsToElems(Zs):
    """Convert atomic numbers to element symbols."""
    return [elements.ELEMENTS[Z - 1][1] for Z in Zs]


def findBonds(atoms, iZs, sc, ELEMENTS=elements.ELEMENTS, FFparams=None):
    bonds = []
    xs = atoms[1]
    ys = atoms[2]
    zs = atoms[3]
    n = len(xs)
    for i in range(n):
        for j in range(i):
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            dz = zs[j] - zs[i]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            ii = iZs[i] - 1
            jj = iZs[j] - 1
            bondlength = ELEMENTS[ii][6] + ELEMENTS[jj][6]
            print(" find bond ", i, j, bondlength, r, sc, (xs[i], ys[i], zs[i]), (xs[j], ys[j], zs[j]))
            if r < (sc * bondlength):
                bonds.append((i, j))
    return bonds


def findBonds_(atoms, iZs, sc, ELEMENTS=elements.ELEMENTS):
    bonds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i):
            d = atoms[i] - atoms[j]
            r = math.sqrt(np.dot(d, d))
            ii = iZs[i] - 1
            jj = iZs[j] - 1
            bondlength = ELEMENTS[ii][6] + ELEMENTS[jj][6]
            if r < (sc * bondlength):
                bonds.append((i, j))
    return bonds


def getAtomColors(iZs, ELEMENTS=elements.ELEMENTS, FFparams=None):
    colors = []
    for e in iZs:
        colors.append(ELEMENTS[FFparams[e - 1][3] - 1][8])
    return colors
