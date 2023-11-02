#!/usr/bin/python


import numpy as np

from . import elements

# ===========================
#      Molecular Topology
# ===========================

iDebug = 0


def findBonds(xyzs, Rs, fR=1.3):
    n = len(xyzs)
    bonds = []
    inds = np.indices((n,))[0]
    for i in range(1, n):
        ds = xyzs[:i, :] - xyzs[i, :][None, :]
        r2s = np.sum(ds**2, axis=1)
        mask = r2s < ((Rs[:i] + Rs[i]) * fR) ** 2
        sel = inds[:i][mask]
        bonds += [(i, j) for j in sel]
    return bonds


def bonds2neighs(bonds, na):
    ngs = [[] for i in range(na)]
    for i, j in bonds:
        ngs[i].append(j)
        ngs[j].append(i)
    return ngs


def bonds2neighsZs(bonds, Zs):
    ngs = [[] for i in Zs]
    for i, j in bonds:
        ngs[i].append((j, Zs[j]))
        ngs[j].append((i, Zs[i]))
    return ngs


def neighs2str(Zs, neighs, ELEMENTS=elements.ELEMENTS, bPreText=False):
    groups = ["" for i in Zs]
    for i, ngs in enumerate(neighs):
        nng = len(ngs)
        if nng > 1:
            # s = ELEMENTS[Zs[i]-1][1] + ("(%i):" %nng)
            if bPreText:
                s = ELEMENTS[Zs[i] - 1][1] + ":"
            else:
                s = ""
            dct = {}
            for j, jz in ngs:
                if jz in dct:
                    dct[jz] += 1
                else:
                    dct[jz] = 1
            for k in sorted(dct.keys()):
                s += ELEMENTS[k - 1][1] + str(dct[k])
            groups[i] = s
    return groups


def findTris(bonds, neighs):
    tris = set()
    tbonds = []
    for b in bonds:
        a_ngs = neighs[b[0]]
        b_ngs = neighs[b[1]]
        common = []
        for i in a_ngs:
            if i in b_ngs:
                common.append(i)
        # print "bond ",b," common ",common
        ncm = len(common)
        if ncm > 2:
            if iDebug > 0:
                print("WARRNING: bond ", b, " common neighbors ", common)
            continue
        elif ncm < 1:
            if iDebug > 0:
                print("WARRNING: bond ", b, " common neighbors ", common)
            continue
        tri0 = tuple(sorted(b + (common[0],)))
        tris.add(tri0)
        if len(common) == 2:
            tri1 = tuple(sorted(b + (common[1],)))
            tris.add(tri1)
            tbonds.append((tri0, tri1))
    return tris, tbonds


def findTris_(bonds, neighs):
    tris = set()
    tbonds = []
    for b in bonds:
        a_ngs = neighs[b[0]]
        b_ngs = neighs[b[1]]
        common = []
        for i in a_ngs:
            if i in b_ngs:
                common.append(i)
        ncm = len(common)
        if ncm > 2:
            continue
        elif ncm < 1:
            continue
        tri0 = tuple(sorted(b + (common[0],)))
        tris.add(tri0)
        if len(common) == 2:
            tri1 = tuple(sorted(b + (common[1],)))
            tris.add(tri1)
            tbonds.append((tri0, tri1))
    return tris, tbonds


def getRingNatom(atom2ring, nr):
    # nr = len(ringNeighs)
    nra = np.zeros(nr, dtype=int)
    for r1, r2, r3 in atom2ring:
        nra[r1] += 1
        nra[r2] += 1
        nra[r3] += 1
    return nra


def tris2num_(tris, tbonds):
    t2i = {k: i for i, k in enumerate(tris)}
    tbonds_ = [(t2i[i], t2i[j]) for i, j in tbonds]
    return tbonds_, t2i


def trisToPoints(tris, ps):
    ops = np.empty((len(tris), 2))
    for i, t in enumerate(tris):
        ops[i, :] = (ps[t[0], :] + ps[t[1], :] + ps[t[2], :]) / 3.0
        # ops.append()
    return ops


def removeBorderAtoms(ps, cog, R):
    rnds = np.random.rand(len(ps))
    r2s = np.sum((ps - cog[None, :]) ** 2, axis=1)  # ;print "r2s ", r2s, R*R
    mask = rnds > r2s / (R * R)
    return mask


def validBonds(bonds, mask, na):
    a2a = np.cumsum(mask) - 1
    bonds_ = []
    # print mask
    for i, j in bonds:
        # print i,j
        if mask[i] and mask[j]:
            bonds_.append((a2a[i], a2a[j]))
    return bonds_


def removeAtoms(atom_pos, bonds, atom2ring, cog, Lrange=10):
    # --- remove some atoms
    mask = removeBorderAtoms(atom_pos, cog, Lrange)
    bonds = validBonds(bonds, mask, len(atom_pos))
    atom_pos = atom_pos[mask, :]
    atom2ring = atom2ring[mask, :]
    return atom_pos, bonds, atom2ring


def ringsToMolecule(ring_pos, ring_Rs, Lrange=6.0):
    Nring = len(ring_pos)
    cog = np.sum(ring_pos, axis=0) / Nring
    ring_bonds = findBonds(ring_pos, ring_Rs, fR=1.0)
    ring_neighs = bonds2neighs(ring_bonds, Nring)
    ring_nngs = np.array([len(ng) for ng in ring_neighs], dtype=int)

    tris, bonds_ = findTris(ring_bonds, ring_neighs)
    atom2ring = np.array(list(tris), dtype=int)

    atom_pos = (ring_pos[atom2ring[:, 0]] + ring_pos[atom2ring[:, 1]] + ring_pos[atom2ring[:, 2]]) / 3.0
    bonds, _ = tris2num_(tris, bonds_)

    (
        atom_pos,
        bonds,
        atom2ring,
    ) = removeAtoms(atom_pos, bonds, atom2ring, cog, Lrange)
    bonds = np.array(bonds)

    # fmt: off
    # --- select aromatic hexagons as they have more pi-character
    ring_natm   = getRingNatom(atom2ring,len(ring_neighs))
    ring_N6mask = np.logical_and( ring_natm[:]==6, ring_nngs[:]==6 )
    atom_N6mask = np.logical_or( ring_N6mask[atom2ring[:,0]],
                  np.logical_or( ring_N6mask[atom2ring[:,1]],
                                 ring_N6mask[atom2ring[:,2]]  ) )
    # fmt: on

    neighs = bonds2neighs(bonds, len(atom_pos))  # ;print neighs
    nngs = np.array([len(ngs) for ngs in neighs], dtype=int)

    atypes = nngs.copy() - 1
    atypes[atom_N6mask] = 3

    return atom_pos, bonds, atypes, nngs, neighs, ring_bonds, atom_N6mask


# ===========================
#   Atom Types and groups
# ===========================


def speciesToPLevels(species):
    levels = []
    for l in species:
        l_ = [s[1] * 1.0 for s in l]
        l_ = np.cumsum(l_)
        l_ *= 1.0 / l_[-1]
        levels.append(l_)
    return levels


def selectRandomElements(nngs, species, levels):
    rnds = np.random.rand(len(nngs))
    elist = []
    for i, nng in enumerate(nngs):
        ing = nng - 1
        il = np.searchsorted(levels[ing], rnds[i])
        elist.append(species[ing][il][0])
    return elist


def makeGroupLevels(groupDict):
    for k, groups in groupDict.items():
        vsum = 0
        l = np.empty(len(groups))
        for i, (g, v) in enumerate(groups):
            vsum += v
            l[i] = vsum
        l /= vsum
        groupDict[k] = [
            l,
        ] + groups
    return groupDict


def selectRandomGroups(an, ao, groupDict):
    na = len(an)
    rnds = np.random.rand(na)
    out = []
    for i in range(na):
        k = (an[i], ao[i])
        if k in groupDict:
            groups = groupDict[k]
            levels = groups[0]
            il = np.searchsorted(levels, rnds[i])
            out.append(groups[il + 1][0])
        else:
            out.append(None)
        print(k, out[-1])
    return out


# fmt: off
group_definition = {
# name   Center,  ndir,  nb,nsigma,npi  nt,nH,ne
#         0   1  2 3 4  5 6 7
"-CH3" :("C" ,4, 1,1,0, 3,3,0),
"-NH2" :("N" ,4, 1,1,0, 3,2,1),
"-OH"  :("O" ,4, 1,1,0, 3,1,2),
"-F"   :("F" ,4, 1,1,0, 3,0,3),
"-Cl"  :("Cl",4, 1,1,0, 3,0,3),

"-CH2-":("C", 4, 2,2,0, 2,2,0),
"-NH-" :("N", 4, 2,2,0, 2,1,1),
"-O-"  :("O", 4, 2,2,0, 2,0,2),

"*CH"  :("C", 4, 3,3,0, 1,1,0),
"*N"   :("N", 4, 3,3,0, 1,0,1),   # what about N+ ?

"=CH-" :("C", 3, 3,2,1, 1,1,0),
"=N-"  :("N", 3, 3,2,1, 1,0,1),

"=CH2" :("C" ,3, 2,1,1, 2,2,0),
"=NH"  :("N" ,3, 2,1,1, 2,1,1),
"=O"   :("O" ,3, 2,1,1, 2,0,2),

"*C"   :("C", 3, 4,3,1, 0,0,0),
"*N+"  :("N", 3, 4,3,1, 0,0,0),

"#CH"  :("C", 2, 3,1,2, 1,1,0),
"#N"   :("N", 2, 3,1,2, 1,0,1),
}
# fmt: on

"""
Simplified

nt = nH
nb = nsigma + npi

4 = nb + nt
4 = nsigma + npi + nH + ne

ndir = nsigma + nH + ne
ndir = nsigma + nt

"""


def normalize(v):
    l = np.sqrt(np.dot(v, v))
    v /= l
    return v, l


def makeTetrahedron(db, up):
    normalize(db)
    side = np.cross(db, up)
    # https://en.wikipedia.org/wiki/Tetrahedron#Formulas_for_a_regular_tetrahedron
    a = 0.81649658092  # sqrt(2/3)
    b = 0.47140452079  # sqrt(2/9)
    c = 0.33333333333  # 1/3
    return np.array(
        [
            db * c + up * (b * 2),
            db * c - up * b - side * a,
            db * c - up * b + side * a,
        ]
    )


def makeTetrahedronFork(d1, d2):
    up = np.cross(d1, d2)
    normalize(up)
    db = d1 + d2
    normalize(db)
    a = 0.81649658092  # sqrt(2/3)
    b = 0.57735026919  # sqrt(1/3)
    return np.array(
        [
            db * b + up * a,
            db * b - up * a,
        ]
    )


def makeTriFork(db, up):
    normalize(db)
    side = np.cross(db, up)
    a = 0.87758256189  # 1/2
    b = 0.5  # sqrt(1/8)
    return np.array(
        [
            db * b + side * a,
            db * b - side * a,
        ]
    )


def groups2atoms(groupNames, neighs, ps):
    def appendHs(txyz, Hmask, elems, xyzs, e1="H", e2="He"):
        Hm = Hmask[np.random.randint(len(Hmask))]
        for ih in range(len(Hm)):
            if Hm[ih] == 1:
                elems.append(e1)
                xyzs.append(txyz[ih])
            else:
                elems.append(e2)
                xyzs.append(txyz[ih])

    # fmt: off
    up=np.array((0.,0.,1.))
    Hmasks3=[ [(0,0,0)],
              [(1,0,0),(0,1,0),(0,0,1)],
              [(0,1,1),(1,0,1),(1,1,0)],
              [(1,1,1)]]
    Hmasks2=[[(0,0)],
             [(1,0),(1,0)],
             [(1,1)]]
    # fmt: on
    elems = []
    xyzs = []
    for ia, name in enumerate(groupNames):
        if name in group_definition:
            ngs = neighs[ia]
            pi = ps[ia]
            g = group_definition[name]
            ndir = g[1]
            nsigma = g[3]
            nH = g[6]
            elems.append(g[0])
            xyzs.append(pi.copy())

            if ndir == 4:  # ==== tetrahedral
                flip = np.random.randint(2) * 2 - 1
                if nsigma == 1:  # like -CH3
                    print(name, ndir, nsigma, nH)
                    txyz = makeTetrahedron(pi - ps[ngs[0]], up * flip) + pi[None, :]
                    appendHs(txyz, Hmasks3[nH], elems, xyzs)
                elif nsigma == 2:  # like -CH2-
                    txyz = makeTetrahedronFork(pi - ps[ngs[0]], pi - ps[ngs[1]]) + pi[None, :]
                    appendHs(txyz, Hmasks2[nH], elems, xyzs)
                elif nsigma == 3:  # like *CH
                    if nH == 1:
                        elems.append("H")
                        xyzs.append(pi + up * flip)

            elif ndir == 3:  # ==== triangular
                if nsigma == 1:  # like =CH2
                    txyz = makeTriFork(pi - ps[ngs[0]], up) + pi[None, :]
                    appendHs(txyz, Hmasks2[nH], elems, xyzs)
                elif nsigma == 2:  # like  =CH-
                    appendHs(normalize(pi * 2 - ps[ngs[0]] - ps[ngs[1]])[0] + pi[None, :], [(1,)], elems, xyzs)

            elif ndir == 2:  # ==== linear
                appendHs(normalize(pi - ps[ngs[0]])[0] + pi[None, :], [(1,)], elems, xyzs)

        else:
            print("Group >>%s<< not known" % name)
    print("len(xyzs), len(elems) ", len(xyzs), len(elems))
    for xyz in xyzs:
        print(len(xyz), end=" ")
        if len(xyz) != 3:
            print(xyz)
    return np.array(xyzs), elems


# ===========================
#           FIRE
# ===========================


class FIRE:
    v = None
    minLastNeg = 5
    t_inc = 1.1
    t_dec = 0.5
    falpha = 0.98
    kickStart = 1.0

    def __init__(self, dt_max=0.2, dt_min=0.01, damp_max=0.2, f_limit=10.0, v_limit=10.0):
        # fmt: off
        self.dt       = dt_max
        self.dt_max   = dt_max
        self.dt_min   = dt_min
        self.damp     = damp_max
        self.damp_max = damp_max
        self.v_limit  = v_limit
        self.f_limit  = f_limit
        self.bFIRE    = True
        self.lastNeg  = 0
        # fmt: on

    def move(self, p, f):
        if self.v is None:
            self.v = np.zeros(len(p))
        v = self.v

        f_norm = np.sqrt(np.dot(f, f))
        v_norm = np.sqrt(np.dot(v, v))
        vf = np.dot(v, f)
        dt_sc = min(min(1.0, self.f_limit / (f_norm + 1e-32)), min(1.0, self.v_limit / (v_norm + 1e-32)))

        if self.bFIRE:
            if (vf < 0.0) or (dt_sc < 0.0):
                self.dt = max(self.dt * self.t_dec, self.dt_min)
                self.damp = self.damp_max
                self.lastNeg = 0
                v[:] = f[:] * self.dt * dt_sc
            else:
                v[:] = v[:] * (1 - self.damp) + f[:] * (self.damp * v_norm / (f_norm + 1e-32))
                if self.lastNeg > self.minLastNeg:
                    self.dt = min(self.dt * self.t_inc, self.dt_max)
                    self.damp = self.damp * self.falpha
                self.lastNeg += 1
        else:
            v[:] *= 1 - self.damp_max

        dt_ = self.dt * dt_sc
        v[:] += f[:] * dt_
        p[:] += v[:] * dt_

        return f_norm


# ===========================
#           Bond-Order Opt
# ===========================


def simpleAOEnergies(Eh2=-4, Eh3=4, E12=0, E13=0, E22=0, E23=1, E32=0, E33=4, Ex1=0.0, Ex4=10.0, Ebound=20.0):
    typeEs = np.array(
        [
            [Ebound, Ex1, E12, E13, Ex4, Ebound],  # nng=1
            [Ebound, Ex1, E22, E23, Ex4, Ebound],  # nng=2
            [Ebound, Ex1, E32, E33, Ex4, Ebound],  # nng=3
            [Ebound, Ex1, Eh2, Eh3, Ex4, Ebound],  # hex
        ]
    )
    return typeEs


def assignAtomBOFF(atypes, typeEs):
    from scipy.interpolate import Akima1DInterpolator

    nt = len(typeEs)
    na = len(atypes)
    typeMasks = np.empty((nt, na), dtype=bool)
    typeFFs = []
    Xs = np.array([-1, 0, 1, 2, 3, 4])
    for it in range(nt):
        typeMasks[it, :] = atypes[:] == it
        Efunc = Akima1DInterpolator(Xs, typeEs[it])
        Ffunc = Efunc.derivative()
        typeFFs.append(Ffunc)
    return typeMasks, typeFFs


def relaxBondOrder(bonds, typeMasks, typeFFs, fConv=0.01, nMaxStep=1000, EboStart=0.0, EboEnd=10.0, boStart=None, optimizer=None):
    nt = typeMasks.shape[0]
    na = typeMasks.shape[1]
    nb = len(bonds)
    if boStart is None:
        bo = np.zeros(nb) + 0.33  # initial guess
    else:
        bo = boStart.copy()
    fb = np.empty(nb)
    fa = np.empty(na)
    ao = np.empty(na)  # + 0.5 # initial guess

    if optimizer is None:
        optimizer = FIRE()

    for itr in range(nMaxStep):
        # -- update Atoms
        ao[:] = 0
        for ib, (i, j) in enumerate(bonds):
            boi = bo[ib]
            ao[i] += boi
            ao[j] += boi

        for it in range(nt):
            Ffunc = typeFFs[it]
            mask = typeMasks[it]
            fa[mask] = Ffunc(ao[mask])
        fb = fa[bonds[:, 0]] + fa[bonds[:, 1]]
        Ebo = (EboEnd - EboStart) * (itr / float(nMaxStep - 1)) + EboStart
        fb += Ebo * np.sin(bo * np.pi * 2)  # force integer forces

        fb[:] *= -1
        f_norm = optimizer.move(bo, fb)

        if f_norm < fConv:
            break

    return bo, ao


def estimateBondOrder(atypes, bonds, E12=0.5, E22=+0.5, E32=+0.5):
    typeEs = simpleAOEnergies(E12=E12, E22=E22, E32=E32)
    typeMasks, typeFFs = assignAtomBOFF(atypes, typeEs)
    opt = FIRE(dt_max=0.1, damp_max=0.25)
    bo, ao = relaxBondOrder(bonds, typeMasks, typeFFs, fConv=0.0001, optimizer=opt, EboStart=0.0, EboEnd=0.0)  # relax delocalized pi-bond superposition
    bo, ao = relaxBondOrder(
        bonds, typeMasks, typeFFs, fConv=-1.0, nMaxStep=100, optimizer=opt, EboStart=0.0, EboEnd=10.0, boStart=bo
    )  # gradually increase integer-discretization strenght 'Ebo'
    bo, ao = relaxBondOrder(
        bonds, typeMasks, typeFFs, fConv=0.0001, optimizer=opt, EboStart=10.0, EboEnd=10.0, boStart=bo
    )  # relax finaly with maximum discretization strenght 'Ebo'
    return bo, ao, typeEs


# ===========================
#       Geometry Opt
# ===========================


def getForceIvnR24(ps, Rs):
    r2safe = 1e-4
    na = len(ps)
    ds = np.zeros(ps.shape)
    fs = np.zeros(ps.shape)
    ir2s = np.zeros(na)
    R2ijs = np.zeros(na)
    for i in range(na):
        R2ijs = Rs[:] + Rs[i]
        R2ijs[:] *= R2ijs[:]
        ds[:, :] = ps - ps[i][None, :]
        ir2s[:] = 1 / (np.sum(ds**2, axis=1) + r2safe)
        ir2s[i] = 0
        fs[:, :] += ds[:, :] * ((R2ijs * ir2s - 1) * R2ijs * ir2s * ir2s)[:, None]
    return fs


def relaxAtoms(ps, aParams, FFfunc=getForceIvnR24, fConv=0.001, nMaxStep=1000, optimizer=None):
    if optimizer is None:
        optimizer = FIRE()

    f_debug = []
    for itr in range(nMaxStep):
        fs = FFfunc(ps, aParams)
        f_norm = optimizer.move(ps.flat, fs.flat)
        if f_norm < fConv:
            break
        f_debug.append(f_norm)

    return ps
