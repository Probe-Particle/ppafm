#!/usr/bin/python

import copy
import os
import re

import numpy as np

from . import elements
from .GridUtils import readNumsUpTo

bohrRadius2angstroem = 0.5291772109217
Hartree2eV = 27.211396132

verbose = 0


def loadXYZ(fname):
    """
    Read the contents of an xyz file.

    The standard xyz file format only has per-atom elements and xyz positions. In Probe-Particle
    we also use the per-atom charges, which can be written as an extra column into the xyz file.
    By default the fifth column is interpreted as the charges, but if the file is written in the
    extended xyz format used by ASE, the relevant column indicated in the comment line is used.

    Arguments:
        fname: str. Path to file.

    Returns:
        xyzs: np.ndarray of shape (N_atoms, 3). Atom xyz positions.
        Zs: np.ndarray of shape (N_atoms,). Atomic numbers.
        qs: np.ndarray of shape (N_atoms). Per-atom charges. All zeros if no charges are present in the file.
        comment: str. The contents of the second line of the xyz file.
    """

    xyzs = []
    Zs = []
    extra_cols = []

    with open(fname) as f:
        line = f.readline().strip()
        try:
            N = int(line)
        except ValueError:
            raise ValueError(f"The first line of an xyz file should have the number of atoms, but got `{line}`")

        comment = f.readline().strip()

        for i, line in enumerate(f):
            if i >= N:
                break
            wds = line.split()
            try:
                Z = wds[0]
                if Z in elements.ELEMENT_DICT:
                    Z = elements.ELEMENT_DICT[Z][0]
                else:
                    Z = int(Z)
                xyzs.append((float(wds[1]), float(wds[2]), float(wds[3])))
                Zs.append(Z)
                extra_cols.append(wds[4:])
            except (ValueError, IndexError):
                raise ValueError(f"Could not interpret line in xyz file: `{line}`")

    xyzs = np.array(xyzs, dtype=np.float64)
    Zs = np.array(Zs, dtype=np.int32)

    if len(extra_cols[0]) > 0:
        qs = _getCharges(comment, extra_cols)
    else:
        qs = np.zeros(len(Zs), dtype=np.float64)

    return xyzs, Zs, qs, comment


def _getCharges(comment, extra_cols):
    match = re.match(r".*Properties=(\S*) ", comment)
    if match:
        # ASE format, check if one of the columns has charges
        props = match.group(1).split(":")[6:]  # [6:] is for skipping over elements and positions
        col = 0
        for name, size in zip(props[::3], props[2::3]):
            if name in ["charge", "initial_charges"]:
                qs = np.array([float(ex[col]) for ex in extra_cols], dtype=np.float64)
                break
            col += int(size)
        else:
            qs = np.zeros(len(extra_cols), dtype=np.float64)
    else:
        # Not ASE format, so just take first column
        qs = np.array([float(ex[0]) for ex in extra_cols], dtype=np.float64)
    return qs


def saveXYZ(fname, xyzs, Zs, qs=None, comment="", append=False):
    """
    Save atom types, positions, and, (optionally) charges to an xyz file.

    Arguments:
        fname: str. Path to file.
        xyzs: np.ndarray of shape (N_atoms, 3). Atom xyz positions.
        Zs: np.ndarray of shape (N_atoms,). Atom atomic numbers.
        qs: np.ndarray of shape (N_atoms) or None. If not None, the partial charges of atoms written as
            the fifth column into the xyz file.
        comment: str. Comment string written to the second line of the xyz file.
        append: bool. Append to file instead of overwriting if it already exists. Useful for creating
            movies of changing structures.
    """
    N = len(xyzs)
    mode = "a" if append else "w"
    file_exists = os.path.exists(fname)
    with open(fname, mode) as f:
        if append and file_exists:
            f.write("\n")
        f.write(f"{N}\n{comment}\n")
        for i in range(N):
            f.write(f"{Zs[i]} {xyzs[i, 0]} {xyzs[i, 1]} {xyzs[i, 2]}")
            if qs is not None:
                f.write(f" {qs[i]}")
            if i < (N - 1):
                f.write("\n")


def loadGeometryIN(fname):
    Zs = []
    xyzs = []
    lvec = []
    with open(fname) as f:
        for line in f:
            ws = line.strip().split()
            if len(ws) > 0 and ws[0][0] != "#":
                if ws[0] == "atom":
                    xyzs.append([float(ws[1]), float(ws[2]), float(ws[3])])
                    Zs.append(elements.ELEMENT_DICT[ws[4]][0])
                elif ws[0] == "lattice_vector":
                    lvec.append([float(ws[1]), float(ws[2]), float(ws[3])])
                elif ws[0] == "atom_frac" and len(lvec) == 3:
                    xyzs.append(np.array(lvec) @ np.array([float(ws[1]), float(ws[2]), float(ws[3])]))
                    Zs.append(elements.ELEMENT_DICT[ws[4]][0])
                elif ws[0] == "trust_radius":
                    break

    xyzs = np.array(xyzs)
    Zs = np.array(Zs, dtype=np.int32)
    if lvec != []:
        lvec = np.stack([[0.0, 0.0, 0.0]] + lvec, axis=0)
    else:
        lvec = None
    return xyzs, Zs, lvec


def loadPOSCAR(file_path):
    with open(file_path) as f:
        f.readline()  # Comment line
        scale = float(f.readline())  # Scaling constant

        # Lattice
        lvec = np.zeros((4, 3))
        for i in range(3):
            lvec[i + 1] = np.array([float(v) for v in f.readline().strip().split()])

        # Elements
        elems = [elements.ELEMENT_DICT[e][0] for e in f.readline().strip().split()]
        Zs = []
        for e, n in zip(elems, f.readline().strip().split()):
            Zs += [e] * int(n)
        Zs = np.array(Zs, dtype=np.int32)

        # Coordinate type
        line = f.readline()
        if line[0] in "Ss":
            line = f.readline()  # Ignore optional selective dynamics line
        if line[0] in "CcKk":
            coord_type = "cartesian"
        else:
            coord_type = "direct"

        # Atom coordinates
        xyzs = []
        for i in range(len(Zs)):
            xyzs.append([float(v) for v in f.readline().strip().split()[:3]])
        xyzs = np.array(xyzs)

        # Scale coordinates
        lvec *= scale
        if coord_type == "cartesian":
            xyzs *= scale
        else:
            xyzs = np.outer(xyzs[:, 0], lvec[1]) + np.outer(xyzs[:, 1], lvec[2]) + np.outer(xyzs[:, 2], lvec[3])

    return xyzs, Zs, lvec


def writeMatrix(fout, mat):
    for v in mat:
        for num in v:
            fout.write(" %f " % num)
        fout.write("\n")


def saveGeomXSF(fname, elems, xyzs, primvec, convvec=None, bTransposed=False):
    if convvec is None:
        primvec = convvec
    with open(fname, "w") as f:
        f.write("CRYSTAL\n")
        f.write("PRIMVEC\n")
        writeMatrix(f, primvec)
        f.write("CONVVEC\n")
        writeMatrix(f, convvec)
        f.write("PRIMCOORD\n")
        f.write("%i %i\n" % (len(elems), 1))
        if bTransposed:
            xs = xyzs[0]
            ys = xyzs[1]
            zs = xyzs[2]
            for i in range(len(elems)):
                f.write(str(elems[i]))
                f.write(f" {xs[i]:10.10f} {ys[i]:10.10f} {zs[i]:10.10f}\n")
        else:
            for i in range(len(elems)):
                xyzsi = xyzs[i]
                f.write(str(elems[i]))
                f.write(f" {xyzsi[0]:10.10f} {xyzsi[1]:10.10f} {xyzsi[2]:10.10f}\n")
        f.write("\n")


def loadXSFGeom(fname):
    f = open(fname)
    e = []
    x = []
    y = []
    z = []
    q = []
    nDim = []
    lvec = []
    for i in range(10000):
        if "PRIMCOORD" in f.readline():
            break
    n = int(f.readline().split()[0])
    for j in range(n):
        ws = f.readline().split()
        e.append(int(ws[0]))
        x.append(float(ws[1]))
        y.append(float(ws[2]))
        z.append(float(ws[3]))
        q.append(0)
    for i in range(10000):
        line = f.readline()
        if ("BEGIN_DATAGRID_3D") in line:
            break
        elif ("DATAGRID_3D_DENSITY") in line:
            break
        elif ("BEGIN_DATAGRID_3D_CONTCAR_v2xsf") in line:
            break
    ws = f.readline().split()
    nDim = [int(ws[0]), int(ws[1]), int(ws[2])]
    for j in range(4):
        ws = f.readline().split()
        lvec.append([float(ws[0]), float(ws[1]), float(ws[2])])
    f.close()
    if verbose > 0:
        print("nDim+1", nDim)
    nDim = (nDim[0] - 1, nDim[1] - 1, nDim[2] - 1)
    if verbose > 0:
        print("lvec", lvec)
    if verbose > 0:
        print("reading ended")
    return [e, x, y, z, q], nDim, lvec


def loadNPYGeom(fname):
    if verbose > 0:
        print("loading atoms")
    tmp = np.load(fname + "_atoms.npy")
    e = tmp[0]
    x = tmp[1]
    y = tmp[2]
    z = tmp[3]
    q = tmp[4]
    del tmp
    if verbose > 0:
        print("loading lvec")
    lvec = np.load(fname + "_vec.npy")
    if verbose > 0:
        print("loading nDim")
    tmp = np.load(fname + "_z.npy")
    nDim = tmp.shape
    del tmp
    if verbose > 0:
        print("nDim", nDim)
    if verbose > 0:
        print("lvec", lvec)
    if verbose > 0:
        print("e,x,y,z", e, x, y, z)
    return [e, x, y, z, q], nDim, lvec


def loadAtomsCUBE(fname):
    bohrRadius2angstroem = 0.5291772109217  # find a good place for this
    e = []
    x = []
    y = []
    z = []
    q = []
    f = open(fname)
    # First two lines of the header are comments
    f.readline()
    f.readline()
    # The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    sth0 = f.readline().split()
    # The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
    f.readline().split()
    f.readline().split()
    f.readline().split()

    # origin = [float(sth0[1]), float(sth0[2]), float(sth0[3])]
    nlines = int(sth0[0])
    for i in range(nlines):
        l = f.readline().split()
        r = [float(l[2]), float(l[3]), float(l[4])]
        x.append(r[0] * bohrRadius2angstroem)
        y.append(r[1] * bohrRadius2angstroem)
        z.append(r[2] * bohrRadius2angstroem)
        e.append(int(l[0]))
        q.append(0.0)
    f.close()
    return [e, x, y, z, q]


def primcoords2Xsf(iZs, xyzs, lvec):
    import io as SIO

    if verbose > 0:
        print("lvec: ", lvec)
    sio = SIO.StringIO()
    sio.write("CRYSTAL\n")
    sio.write("PRIMVEC\n")
    sio.write(f"{lvec[1][0]:f} {lvec[1][1]:f} {lvec[1][2]:f}\n")
    sio.write(f"{lvec[2][0]:f} {lvec[2][1]:f} {lvec[2][2]:f}\n")
    sio.write(f"{lvec[3][0]:f} {lvec[3][1]:f} {lvec[3][2]:f}\n")
    sio.write("CONVVEC\n")
    sio.write(f"{lvec[1][0]:f} {lvec[1][1]:f} {lvec[1][2]:f}\n")
    sio.write(f"{lvec[2][0]:f} {lvec[2][1]:f} {lvec[2][2]:f}\n")
    sio.write(f"{lvec[3][0]:f} {lvec[3][1]:f} {lvec[3][2]:f}\n")
    sio.write("PRIMCOORD\n")
    n = len(iZs)
    sio.write("%i 1\n" % n)
    for i in range(n):
        sio.write("%i %5.6f %5.6f %5.6f\n" % (iZs[i], xyzs[0][i], xyzs[1][i], xyzs[2][i]))
    sio.write("\n")
    sio.write("BEGIN_BLOCK_DATAGRID_3D\n")
    sio.write("some_datagrid\n")
    sio.write("BEGIN_DATAGRID_3D_whatever\n")
    s = sio.getvalue()
    # print s; exit()
    return s


def loadCellCUBE(fname):
    bohrRadius2angstroem = 0.5291772109217  # find a good place for this
    f = open(fname)
    # First two lines of the header are comments
    f.readline()
    f.readline()
    # The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    line = f.readline().split()
    int(line[0])
    c0 = [float(s) for s in line[1:4]]

    # The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
    line = f.readline().split()
    n1 = int(line[0])
    c1 = [float(s) for s in line[1:4]]

    line = f.readline().split()
    n2 = int(line[0])
    c2 = [float(s) for s in line[1:4]]

    line = f.readline().split()
    n3 = int(line[0])
    c3 = [float(s) for s in line[1:4]]

    cell0 = [c0[0] * bohrRadius2angstroem, c0[1] * bohrRadius2angstroem, c0[2] * bohrRadius2angstroem]
    cell1 = [c1[0] * n1 * bohrRadius2angstroem, c1[1] * n1 * bohrRadius2angstroem, c1[2] * n1 * bohrRadius2angstroem]
    cell2 = [c2[0] * n2 * bohrRadius2angstroem, c2[1] * n2 * bohrRadius2angstroem, c2[2] * n2 * bohrRadius2angstroem]
    cell3 = [c3[0] * n3 * bohrRadius2angstroem, c3[1] * n3 * bohrRadius2angstroem, c3[2] * n3 * bohrRadius2angstroem]
    f.close()
    return [cell0, cell1, cell2, cell3]


def loadNCUBE(fname):
    bohrRadius2angstroem = 0.5291772109217  # find a good place for this
    f = open(fname)
    # First two lines of the header are comments
    f.readline()
    f.readline()
    # The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    f.readline().split()
    # The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
    sth1 = f.readline().split()
    sth2 = f.readline().split()
    sth3 = f.readline().split()
    f.close()
    return [int(sth1[0]), int(sth2[0]), int(sth3[0])]


def loadGeometry(fname=None, format=None, params=None):
    if verbose > 0:
        print("loadGeometry ", fname)
    if fname == None:
        raise ValueError("Please provide the name of the file with coordinates")
    if params == None:
        raise ValueError("Please provide the parameters dictionary here")

    if format == None or format == "":
        if fname.lower().endswith(".xyz"):
            format = "xyz"
        elif fname.lower().endswith(".cube"):
            format = "cube"
        elif fname.lower().endswith(".xsf"):
            format = "xsf"
        elif fname.lower().endswith(".npy"):
            format = "npy"
        elif fname.lower().endswith(".in") or fname.lower().endswith(".in.next_step"):
            format = "in"
        if fname.startswith("POSCAR") or fname.startswith("CONTCAR"):
            format = "poscar"
    else:
        format = format.lower()  # prevent format from being case sensitive, e.g. "XYZ" and "xyz" should be the same

    xyzs = Zs = qs = None
    if format == "xyz":
        xyzs, Zs, qs, comment = loadXYZ(fname)
        lvec = parseLvecASE(comment)
    elif format == "cube":
        atoms = loadAtomsCUBE(fname)
        lvec = loadCellCUBE(fname)
        nDim = loadNCUBE(fname)
    elif format == "xsf":
        atoms, nDim, lvec = loadXSFGeom(fname)
    elif format == "npy":
        atoms, nDim, lvec = loadNPYGeom(fname)  # under development
    elif format == "poscar":
        xyzs, Zs, lvec = loadPOSCAR(fname)
    elif format == "in":
        xyzs, Zs, lvec = loadGeometryIN(fname)
    # TODO: introduce a function which reads the geometry from the .npy file
    else:
        if format is None:
            raise ValueError("ERROR!!! Input geometry format was neither specified nor could it be determined automatically.")
        else:
            raise ValueError("ERROR!!! Unknown format %s of input geometry." % (format))

    if xyzs is not None:
        # Some of the load functions return the result in a different form. Really the return values should be unified.
        assert Zs is not None, "Somehow loaded xyzs without Zs"
        if qs is None:
            qs = np.zeros(len(Zs))
        atoms = [list(Zs), list(xyzs[:, 0]), list(xyzs[:, 1]), list(xyzs[:, 2]), list(qs)]
        nDim = copy.copy(params["gridN"])

    # Make sure lvec is a 4x3 array. Use default grid parameters if needed
    if (lvec is None) or (len(lvec) == 0):
        lvec = np.zeros((4, 3))
        lvec[0, :] = params["gridO"].copy()
        lvec[1, :] = params["gridA"].copy()
        lvec[2, :] = params["gridB"].copy()
        lvec[3, :] = params["gridC"].copy()
    else:
        if len(lvec) == 1:
            lvec.append(params["gridA"].copy())
        if len(lvec) == 2:
            lvec.append(params["gridB"].copy())
        if len(lvec) == 3:
            lvec.append(params["gridC"].copy())
    lvec = np.array(lvec)

    # Shift scanning coordinates to such that actually make sense (being directly comparable to coordinates of atoms)
    probe_min = params["scanMin"].copy()
    probe_max = params["scanMax"].copy()
    probe_min[0] += params["r0Probe"][0]
    probe_min[1] += params["r0Probe"][1]
    probe_min[2] -= params["r0Probe"][2]
    probe_max[0] += params["r0Probe"][0]
    probe_max[1] += params["r0Probe"][1]
    probe_max[2] -= params["r0Probe"][2]

    # Generate automatic lattice vectors and grid dimensions if needed
    pad = 3.0
    default_grid_step = 0.1
    for i in range(3):
        # Zero lattice vector is considered undefined and triggers creation of an automatic one.
        # The automatically generated lattice vector should enclose the whole scanning area plus the default padding.
        if np.allclose(lvec[i + 1, :], 0):
            params["PBC"] = False
            lvec[i + 1, i] = probe_max[i] - probe_min[i] + 2 * pad
            lvec[0, i] = probe_min[i] - pad

        # Generate automatic grid using the default step if the grid dimension specified so far is not a positive number
        if not nDim[i] > 0:
            nDim[i] = round(np.linalg.norm(lvec[i + 1, :]) / default_grid_step)

    # Copy lattice vectors and grid dimensions to the global parameters
    # so as to guarantee compatibility between the local variables and global parameters

    for i in range(3):
        params["gridO"][i] = lvec[0][i]
        params["gridA"][i] = lvec[1][i]
        params["gridB"][i] = lvec[2][i]
        params["gridC"][i] = lvec[3][i]
    params["gridN"] = nDim

    return atoms, nDim, lvec


def parseLvecASE(comment):
    """
    Try to parse the lattice vectors in an xyz file comment line according to the extended xyz
    file format used by ASE. The origin is always at zero.

    Arguments:
        comment: str. Comment line to parse.

    Returns:
        lvec: np.array of shape (4, 3) or None. The found lattice vectors or None if the
            comment line does not match the extended xyz file format.
    """
    match = re.match('.*Lattice="\\s*((?:[+-]?(?:[0-9]*\\.)?[0-9]+\\s*){9})"', comment)
    if match:
        lvec = np.zeros(12, dtype=np.float32)
        lvec[3:] = np.array([float(s) for s in match.group(1).split()], dtype=np.float32)
        lvec = lvec.reshape(4, 3)
    else:
        lvec = None
    return lvec


# =================== XSF


def _readUpTo(filein, keyword):
    i = 0
    linelist = []
    while True:
        line = filein.readline()
        linelist.append(line)
        i = i + 1
        if (not line) or (keyword in line):
            break
    return i, linelist


def _readmat(filein, n):
    temp = []
    for i in range(n):
        temp.append([float(iii) for iii in filein.readline().split()])
    return np.array(temp)


def _writeArr(f, arr):
    f.write(" ".join(str(x) for x in arr) + "\n")


def _writeArr2D(f, arr):
    for vec in arr:
        _writeArr(f, vec)


def _orthoLvec(sh, dd):
    return [[0, 0, 0], [sh[2] * dd[0], 0, 0], [0, sh[1] * dd[1], 0], [0, 0, sh[0] * dd[2]]]


XSF_HEAD_DEFAULT = headScan = """
ATOMS
 1   0.0   0.0   0.0

BEGIN_BLOCK_DATAGRID_3D
   some_datagrid
   BEGIN_DATAGRID_3D_whatever
"""


def saveXSF(fname, data, lvec=None, dd=None, head=XSF_HEAD_DEFAULT, verbose=1):
    if verbose > 0:
        print("Saving xsf", fname)
    fileout = open(fname, "w")
    if lvec is None:
        if dd is None:
            dd = [1.0, 1.0, 1.0]
        lvec = _orthoLvec(data.shape, dd)
    for line in head:
        fileout.write(line)
    nDim = np.shape(data)
    _writeArr(fileout, (nDim[2] + 1, nDim[1] + 1, nDim[0] + 1))
    _writeArr2D(fileout, lvec)
    data2 = np.zeros(np.array(nDim) + 1)
    # These crazy 3 lines are here since the first and the last cube in XSF in every direction is the same
    data2[:-1, :-1, :-1] = data
    data2[-1, :, :] = data2[0, :, :]
    data2[:, -1, :] = data2[:, 0, :]
    data2[:, :, -1] = data2[:, :, 0]
    for r in data2.flat:
        fileout.write("%10.5e\n" % r)
    fileout.write("   END_DATAGRID_3D\n")
    fileout.write("END_BLOCK_DATAGRID_3D\n")


def loadXSF(fname, xyz_order=False, verbose=True):
    filein = open(fname)
    startline, head = _readUpTo(filein, "BEGIN_DATAGRID_3D")  # startline - number of the line with DATAGRID_3D_. Dinensions are located in the next line
    nDim = [int(iii) for iii in filein.readline().split()]  # reading 1 line with dimensions
    nDim.reverse()
    nDim = np.array(nDim)
    lvec = _readmat(filein, 4)  # reading 4 lines where 1st line is origin of datagrid and 3 next lines are the cell vectors
    filein.close()
    if verbose:
        print("nDim xsf (= nDim + [1,1,1] ):", nDim)
    if verbose:
        print("io | Load " + fname + " using readNumsUpTo ")
    F = readNumsUpTo(fname, nDim.astype(np.int32).copy(), startline + 5)
    if verbose:
        print("io | Done")
    FF = np.reshape(F, nDim)[:-1, :-1, :-1]
    if xyz_order:
        FF = FF.transpose((2, 1, 0))
    # FF is not C_CONTIGUOUS without copy
    FF = FF.copy()
    return FF, lvec, nDim - 1, head


def getFromHead_PRIMCOORD(head):
    Zs = None
    Rs = None
    for i, line in enumerate(head):
        if "PRIMCOORD" in line:
            natoms = int(head[i + 1].split()[0])
            Zs = np.zeros(natoms, dtype="int32")
            Rs = np.zeros((natoms, 3))
            for j in range(natoms):
                words = head[i + j + 2].split()
                Zs[j] = int(words[0])
                Rs[j, 0] = float(words[1])
                Rs[j, 1] = float(words[2])
                Rs[j, 2] = float(words[3])
    return Zs, Rs


# =================== Cube


def loadCUBE(fname, xyz_order=False, verbose=True):
    filein = open(fname)
    # First two lines of the header are comments
    filein.readline()
    filein.readline()
    # The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    sth0 = filein.readline().split()
    # The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
    sth1 = filein.readline().split()
    sth2 = filein.readline().split()
    sth3 = filein.readline().split()
    filein.close()
    nDim = np.array([int(sth1[0]), int(sth2[0]), int(sth3[0])])
    lvec = np.zeros((4, 3))
    for jj in range(3):
        lvec[0, jj] = float(sth0[jj + 1]) * bohrRadius2angstroem
        lvec[1, jj] = float(sth1[jj + 1]) * int(sth1[0]) * bohrRadius2angstroem  # bohr_radius ?
        lvec[2, jj] = float(sth2[jj + 1]) * int(sth2[0]) * bohrRadius2angstroem
        lvec[3, jj] = float(sth3[jj + 1]) * int(sth3[0]) * bohrRadius2angstroem

    if verbose:
        print("io | Load " + fname + " using readNumsUpTo")
    noline = 6 + int(sth0[0])
    F = readNumsUpTo(fname, nDim.astype(np.int32).copy(), noline)
    if verbose:
        print("io | np.shape(F): ", np.shape(F))
    if verbose:
        print("io | nDim: ", nDim)

    FF = np.reshape(F, nDim)
    if not xyz_order:
        FF = FF.transpose((2, 1, 0)).copy()  # Transposition of the array to have the same order of data as in XSF file

    nDim = [nDim[2], nDim[1], nDim[0]]  # Setting up the corresponding dimensions.
    head = []
    head.append("BEGIN_BLOCK_DATAGRID_3D \n")
    head.append("g98_3D_unknown \n")
    head.append("DATAGRID_3D_g98Cube \n")
    FF *= Hartree2eV
    return FF, lvec, nDim, head


# ================ WSxM output


def saveWSxM_2D(name_file, data, Xs, Ys):
    tmp_data = data.flatten()
    out_data = np.zeros((len(tmp_data), 3))
    out_data[:, 0] = Xs.flatten()
    out_data[:, 1] = Ys.flatten()
    out_data[:, 2] = tmp_data  # .copy()
    f = open(name_file, "w")
    print("WSxM file copyright Nanotec Electronica", file=f)
    print("WSxM ASCII XYZ file", file=f)
    print("X[A]  Y[A]  df[Hz]", file=f)
    print("", file=f)
    np.savetxt(f, out_data)
    f.close()


def saveWSxM_3D(prefix, data, extent, slices=None):
    nDim = np.shape(data)
    if slices is None:
        slices = list(range(nDim[0]))
    xs = np.linspace(extent[0], extent[1], nDim[2])
    ys = np.linspace(extent[2], extent[3], nDim[1])
    Xs, Ys = np.meshgrid(xs, ys)
    for i in slices:
        print("slice no: ", i)
        fname = prefix + "_%03d.xyz" % i
        saveWSxM_2D(fname, data[i], Xs, Ys)


# ================ Npy


def saveNpy(fname, data, lvec, atomic_info):
    """
    Function for saving scalar grid data, together with its lattice_vector and information about original atoms and the original lattice vector (lvec0) in numpy format

    Arguments:
        fname: str. Path to file. fname should be without the npz expension, which is added by this function.
        data: np.ndarray of shape (n_z, n_y, n_x) with scallar data
        lvec: np.ndarray of shape (4, 3). Lattice vector of the data
        atomic_info: tuple of shape (2). First part is [e, x, y, z] of atoms, the second is lvec of the atoms from the original geometry file, named as lvec0;
    """
    np.savez(fname + ".npz", data=data, lvec=lvec, atoms=atomic_info[0], lvec0=atomic_info[1])


def loadNpy(fname):
    """
    Function for loading scalar grid data, together with its lattice_vector and information about original atoms and the original lattice vector (lvec0) in numpy format

    Arguments:
        fname: str. name of the npz file. fname should be without the npz expension, which is added by this function.

    Returns:
        data: np.array of shape(nz, ny, nx) with volumetric (scaler data) we want to save
        lvec: np.array of shape(4,3) with lattice vector of the volumetric data
        atomic_info tuple of shape (2) with 2 np.arrays - one is np.array([e,x,y,z]) with atoms positions and the second one is np.array(lvec0) of shape (4,3) with saved information about lattice vector.
    """
    tmp_input = np.load(fname + ".npz")
    data = tmp_input["data"]
    lvec = tmp_input["lvec"]
    atomic_info = (tmp_input["atoms"], tmp_input["lvec0"])
    return data.copy(), lvec, atomic_info
    # necessary for being 'C_CONTINUOS'


# =============== Vector Field


def packVecGrid(Fx, Fy, Fz, FF=None):
    if FF is None:
        nDim = np.shape(Fx)
        FF = np.zeros((nDim[0], nDim[1], nDim[2], 3))
    FF[:, :, :, 0] = Fx
    FF[:, :, :, 1] = Fy
    FF[:, :, :, 2] = Fz
    return FF


def unpackVecGrid(FF):
    return FF[:, :, :, 0].copy(), FF[:, :, :, 1].copy(), FF[:, :, :, 2].copy()


def loadVecFieldXsf(fname, FF=None):
    Fx, lvec, nDim, head = loadXSF(fname + "_x.xsf")
    Fy, lvec, nDim, head = loadXSF(fname + "_y.xsf")
    Fz, lvec, nDim, head = loadXSF(fname + "_z.xsf")
    FF = packVecGrid(Fx, Fy, Fz, FF)
    del Fx, Fy, Fz
    return FF, lvec, nDim, head


def loadVecFieldNpy(fname, FF=None):
    """
    Function for loading vector grid data, together with its lattice_vector and information about original atoms and the original lattice vector (lvec0) in numpy format.

    Arguments:
        fname: str. name of the npz file.

    Returns:
        FF: np.array of shape(nz, ny, nx, 3) with volumetric (vector data) we want to load.
        lvec: np.array of shape(4,3) with lattice vector of the volumetric data.
        atomic_info: tuple of shape (2) with 2 np.arrays, one is np.array([e,x,y,z]) with atoms positions and the second one is np.array(lvec0) of shape (4,3) with saved information about lattice vector.
    """
    tmp_input = np.load(fname + ".npz")
    FF = tmp_input["FF"]
    lvec = tmp_input["lvec"]
    atomic_info = (tmp_input["atoms"], tmp_input["lvec0"])
    return FF, lvec, atomic_info


def saveVecFieldXsf(fname, FF, lvec, head=XSF_HEAD_DEFAULT):
    saveXSF(fname + "_x.xsf", FF[:, :, :, 0], lvec, head=head)
    saveXSF(fname + "_y.xsf", FF[:, :, :, 1], lvec, head=head)
    saveXSF(fname + "_z.xsf", FF[:, :, :, 2], lvec, head=head)


def saveVecFieldNpy(fname, FF, lvec, atomic_info):
    """
    Function for saving vector grid data, together with its lattice_vector and information about original atoms and the original lattice vector (lvec0) in numpy format.

    Arguments:
        fname: str. name of the npz file. fname should be without the npz expension, which is added by this function.
        FF: np.array of shape(nz, ny, nx, 3) with volumetric (vector data) we want to load.
        lvec: np.array of shape(4,3) with lattice vector of the volumetric data.
        atomic_info: tuple of shape (2) with 2 np.arrays, one is np.array([e,x,y,z]) with atoms positions and the second one is np.array(lvec0) of shape (4,3) with saved information about lattice vector.
    """
    np.savez(fname + ".npz", FF=FF, lvec=lvec, atoms=atomic_info[0], lvec0=atomic_info[1])


def limit_vec_field(FF, Fmax=100.0):
    """
    remove too large values; preserves direction of vectors.

    Arguments:
        FF: np.array of shape(nz, ny, nx, 3) with volumetric (vector data) we want to be limited.
        Fmax: float maximum value to which all the larger values will be lowered to.
    """
    FR = np.sqrt(FF[:, :, :, 0] ** 2 + FF[:, :, :, 1] ** 2 + FF[:, :, :, 2] ** 2).flat
    mask = FR > Fmax
    FF[:, :, :, 0].flat[mask] *= Fmax / FR[mask]
    FF[:, :, :, 1].flat[mask] *= Fmax / FR[mask]
    FF[:, :, :, 2].flat[mask] *= Fmax / FR[mask]


def save_vec_field(fname, data, lvec, data_format="xsf", head=XSF_HEAD_DEFAULT, atomic_info=None):
    """
    Saving vector fields into xsf, or npy

    Arguments:
        fname: str. name of the npz or xsf file. fname should be without any extension, which is added later automatically based on the format (data_format).
        data: np.array of shape(nz, ny, nx, 3) with volumetric (vector data) we want to save; note [:,:,:,0] are x part, [:,:,:,1] is the y part and [:,:,:,2] is the z part of the vector
        lvec: np.array of shape (4,3) with lattice vector of the volumetric data
        data_format: string "xsf" or "npy"
        head: string header of the XSF file
        atomic_info: tuple of shape (2) with 2 np.arrays - one is np.array([e,x,y,z]) with atoms positions and the second one is np.array(lvec) of shape (4,3) with saved information about lattice vector.
    """
    if data_format == "xsf":
        saveVecFieldXsf(fname, data, lvec, head=head)
    elif data_format == "npy":
        atomic_info = atomic_info if atomic_info is not None else (np.zeros((4, 1)), lvec)
        saveVecFieldNpy(fname, data, lvec, atomic_info)
    else:
        print("I cannot save this format!")


def load_vec_field(fname, data_format="xsf"):
    """
    Loading Vector fields from xsf, or npy

    Arguments:
        fname: str. name of the npz or xsf file. fname should be without any extension, which is added later automatically based on the format (data_format).
        data_fromat: str "xsf" or "npy"

    Returns:
        data: np.array of shape(nz, ny, nx, 3) with volumetric (vector data) we want to load.
        lvec: np.array of shape(4,3) with lattice vector of the volumetric data.
        ndim: tupple of lenght 4 with dimmensions of the vector data.
        atomic_info_or_head: tuple or string. If tupple then shape (2) with 2 np.arrays,
             one is np.array([e,x,y,z]) with atoms positions and the second one is np.array(lvec0) of shape (4,3) with saved information about lattice vector.
             if string, the same information is basically stored as the header of xsf
    """
    atomic_info_or_head = None
    if data_format == "xsf":
        data, lvec, ndim, atomic_info_or_head = loadVecFieldXsf(fname)
    elif data_format == "npy":
        data, lvec, atomic_info_or_head = loadVecFieldNpy(fname)
        ndim = data.shape
    else:
        print("I cannot load this format!")
    return data.copy(), lvec, ndim, atomic_info_or_head


# =============== Scalar Fields


def save_scal_field(fname, data, lvec, data_format="xsf", head=XSF_HEAD_DEFAULT, atomic_info=None):
    """
    Saving scalar fields into xsf, or npy

    Arguments:
        fname: str. name of the npz or xsf file. fname should be without any extension, which is added later automatically based on the format (data_format).
        data: np.array of shape(nz, ny, nx, 3) with volumetric (scalar data) we want to save.
        lvec: np.array of shape(4,3) with lattice vector of the volumetric data.
        data_format: str "xsf" or "npy".
        head: string header of the XSF file
        atomic_info: tuple of shape (2) with 2 np.arrays - one is np.array([e,x,y,z]) with atoms positions and the second one is np.array(lvec) of shape (4,3) with saved information about lattice vector.
    """
    if data_format == "xsf":
        saveXSF(fname + ".xsf", data, lvec, head=head)
    elif data_format == "npy":
        atomic_info = atomic_info if atomic_info is not None else (np.zeros((4, 1)), lvec)
        saveNpy(fname, data, lvec, atomic_info)
    else:
        print("I cannot save this format!")


def load_scal_field(fname, data_format="xsf"):
    """
    Loading Vector fields from xsf, or npy

    Arguments:
        fname: str. name of the npz or xsf file. fname should be without any extension, which is added later automatically based on the format (data_format).
        data_fromat: str "xsf" or "npy"

    Returns:
        data: np.array of shape(nz, ny, nx) with volumetric (scalar data) we want to load.
        lvec: np.array of shape(4,3) with lattice vector of the volumetric data.
        atomic_info_or_head: tuple or string. If tupple then shape (2) with 2 np.arrays,
             one is np.array([e,x,y,z]) with atoms positions and the second one is np.array(lvec0) of shape (4,3) with saved information about lattice vector.
             if string, the same information is basically stored as the header of xsf
    """
    atomic_info_or_head = None
    if data_format == "xsf":
        data, lvec, ndim, atomic_info_or_head = loadXSF(fname + ".xsf")
    elif data_format == "npy":
        data, lvec, atomic_info_or_head = loadNpy(fname)
        ndim = data.shape
    elif data_format == "cube":
        data, lvec, ndim, atomic_info_or_head = loadCUBE(fname + ".cube")
    else:
        print("I cannot load this format!")
    return data.copy(), lvec, ndim, atomic_info_or_head


# ================ POV-Ray

DEFAULT_POV_HEAD_NO_CAM = """
background      { color rgb <1.0,1.0,1.0> }
//background      { color rgb <0.5,0.5,0.5> }
//global_settings { ambient_light rgb< 0.2, 0.2, 0.2> }
// ***********************************************
// macros for common shapes
// ***********************************************
#default { finish {
  ambient 0.45
  diffuse 0.84
  specular 0.22
  roughness .00001
  metallic
  phong 0.9
  phong_size 120
}
}
#macro translucentFinish(T)
 finish {
  ambient 0.45
  diffuse 0.84
  specular 0.22
  roughness .00001
  metallic 1.0
  phong 0.9
  phong_size 120
}#end
#macro a(X,Y,Z,RADIUS,R,G,B,T)
 sphere{<X,Y,Z>,RADIUS
  pigment{rgbt<R,G,B,T>}
  translucentFinish(T)
  no_shadow  // comment this out if you want include shadows
  }
#end
#macro b(X1,Y1,Z1,RADIUS1,X2,Y2,Z2,RADIUS2,R,G,B,T)
 cone{<X1,Y1,Z1>,RADIUS1,<X2,Y2,Z2>,RADIUS2
  pigment{rgbt<R,G,B,T>  }
  translucentFinish(T)
  no_shadow // comment this out if you want include shadows
  }
#end
"""

DEFAULT_POV_HEAD = (
    """
// ***********************************************
// Camera & other global settings
// ***********************************************
#declare Zoom = 30.0;
#declare Width = 800;
#declare Height = 800;
camera{
  orthographic
  location < 0,  0,  -100>
  sky      < 0, -1,    0 >
  right    < -Zoom, 0, 0>
  up       < 0, Zoom, 0 >
  look_at  < .0.0,  0.0,  0.0 >
}
"""
    + DEFAULT_POV_HEAD_NO_CAM
)


def makePovCam(pos, up=[0.0, 1.0, 0.0], rg=[-1.0, 0.0, 0.0], fw=[0.0, 0.0, 100.0], lpos=[0.0, 0.0, -100.0], W=10.0, H=10.0):
    return """
    // ***********************************************
    // Camera & other global settings
    // ***********************************************
    #declare Zoom   = 30.0;
    #declare Width  = 800;
    #declare Height = 800;
    camera{{
      orthographic
      right    {:f}
      up       {:f}
      sky      < {:f}, {:f}, {:f} >
      location < {:f}, {:f}, {:f} >
      look_at  < {:f}, {:f}, {:f} >
    }}
    light_source    {{ < {:f},{:f},{:f}>  rgb <0.5,0.5,0.5> }}
    """.format(
        W,
        H,
        up[0],
        up[1],
        up[2],
        pos[0] - fw[0],
        pos[1] - fw[1],
        pos[2] - fw[2],
        pos[0],
        pos[1],
        pos[2],
        lpos[0],
        lpos[1],
        lpos[2],
    )


def writePov(fname, xyzs, Zs, bonds=None, HEAD=DEFAULT_POV_HEAD, bondw=0.1, spherescale=0.25, ELEMENTS=elements.ELEMENTS):
    fout = open(fname, "w")
    n = len(xyzs)
    fout.write(HEAD)
    for i in range(n):
        clr = ELEMENTS[Zs[i] - 1][8]
        R = ELEMENTS[Zs[i] - 1][7]
        s = "a( {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f} ) \n".format(
            xyzs[i][0],
            xyzs[i][1],
            xyzs[i][2],
            spherescale * R,
            clr[0] / 255.0,
            clr[1] / 255.0,
            clr[2] / 255.0,
            0.0,
        )
        fout.write(s)
    if bonds is not None:
        for b in bonds:
            i = b[0]
            j = b[1]
            clr = [128, 128, 128]
            s = "b( {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f},0.0 ) \n".format(
                xyzs[i][0],
                xyzs[i][1],
                xyzs[i][2],
                bondw,
                xyzs[j][0],
                xyzs[j][1],
                xyzs[j][2],
                bondw,
                clr[0] / 255.0,
                clr[1] / 255.0,
                clr[2] / 255.0,
            )
            fout.write(s)
    fout.close()
