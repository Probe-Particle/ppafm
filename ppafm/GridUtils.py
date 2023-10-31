#!/usr/bin/python

from ctypes import c_char_p, c_double, c_int

import numpy as np

from . import cpp_utils

# ============================== interface to C++ core

lib = cpp_utils.get_cdll("GU")

# define used numpy array types for interfacing with C++
array1i = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="CONTIGUOUS")
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags="CONTIGUOUS")
array4d = np.ctypeslib.ndpointer(dtype=np.double, ndim=4, flags="CONTIGUOUS")


# ============== Filters


def renorSlice(F):
    vranges = []
    for i in range(len(F)):
        Fi = F[i]
        vmin = np.nanmin(Fi)
        vmax = np.nanmax(Fi)
        F[i] -= vmin
        F[i] /= vmax - vmin
        vranges.append((vmin, vmax))
    return vranges


# ==============  Cutting, Sampling, Interpolation ...

# 	void interpolate_gridCoord( int n, Vec3d * pos_list, double * data )
lib.interpolate_gridCoord.argtypes = [c_int, array2d, array3d, array1d]
lib.interpolate_gridCoord.restype = None
interpolate_gridCoord = lib.interpolate_gridCoord

# 	void interpolateLine_gridCoord( int n, Vec3d * p1, Vec3d * p2, double * data, double * out )
lib.interpolateLine_gridCoord.argtypes = [c_int, array1d, array1d, array3d, array1d]
lib.interpolateLine_gridCoord.restype = None
# interpolateLine_gridCoord                   = lib.interpolateLine_gridCoord

# 	void interpolateLine_gridCoord( int n, Vec3d * p1, Vec3d * p2, double * data, double * out )
lib.interpolateLine_cartes.argtypes = [c_int, array1d, array1d, array3d, array1d]
lib.interpolateLine_cartes.restype = None
# interpolateLine_gridCoord                   = lib.interpolateLine_gridCoord

# 	void interpolateQuad_gridCoord( int * nij, Vec3d * p00, Vec3d * p01, Vec3d * p10, Vec3d * p11, double * data, double * out )
lib.interpolateQuad_gridCoord.argtypes = [array1i, array1d, array1d, array1d, array1d, array3d, array2d]
lib.interpolateQuad_gridCoord.restype = None
# interpolateQuad_gridCoord              = lib.interpolateQuad_gridCoord

# 	void interpolate_cartesian( int n, Vec3d * pos_list, double * data, double * out )
lib.interpolate_cartesian.argtypes = [c_int, array4d, array3d, array3d]
lib.interpolate_cartesian.restype = None
# interpolate_cartesian               = lib.interpolate_cartesian

# 	void setGridCell( double * cell )
lib.setGridCell.argtypes = [array2d]
lib.setGridCell.restype = None
setGridCell = lib.setGridCell

# 	void setGridN( int * n )
lib.setGridN.argtypes = [array1i]
lib.setGridN.restype = None
setGridN = lib.setGridN


def interpolateLine(F, p1, p2, sz=500, cartesian=False):
    result = np.zeros(sz)
    p00 = np.array(p1, dtype="float64")
    p01 = np.array(p2, dtype="float64")
    if cartesian:
        lib.interpolateLine_cartes(sz, p00, p01, F, result)
    else:
        lib.interpolateLine_gridCoord(sz, p00, p01, F, result)
    return result


def interpolateQuad(F, p00, p01, p10, p11, sz=(500, 500)):
    result = np.zeros(sz)
    npxy = np.array(sz, dtype="int32")
    p00 = np.array(p00, dtype="float64")
    p01 = np.array(p01, dtype="float64")
    p10 = np.array(p10, dtype="float64")
    p11 = np.array(p11, dtype="float64")
    lib.interpolateQuad_gridCoord(npxy, p00, p01, p10, p11, F, result)
    return result


def interpolate_cartesian(F, pos, cell=None, result=None):
    if cell is not None:
        setGridCell(cell)
    nDim = np.array(pos.shape)
    print(nDim)
    if result is None:
        result = np.zeros((nDim[0], nDim[1], nDim[2]))
    n = nDim[0] * nDim[1] * nDim[2]
    lib.interpolate_cartesian(n, pos, F, result)
    return result


def verticalCut(F, p1, p2, sz=(500, 500)):
    result = np.zeros(sz)
    npxy = npxy = np.array(sz, dtype="int32")
    p00 = np.array((p1[0], p1[1], p1[2]), dtype="float64")
    p01 = np.array((p2[0], p2[1], p1[2]), dtype="float64")
    p10 = np.array((p1[0], p1[1], p2[2]), dtype="float64")
    p11 = np.array((p2[0], p2[1], p2[2]), dtype="float64")
    lib.interpolateQuad_gridCoord(npxy, p00, p01, p10, p11, F, result)
    return result


def dens2Q_CHGCARxsf(data, lvec):
    nDim = data.shape
    Ntot = nDim[0] * nDim[1] * nDim[2]
    Vtot = np.linalg.det(lvec[1:])
    print("dens2Q Volume    : ", Vtot)
    print("dens2Q Ntot      : ", Ntot)
    print("dens2Q Vtot/Ntot : ", Vtot / Ntot)
    # Qsum = rho1.sum()
    return Vtot / Ntot


# double cog( double * data_, double* center ){
lib.cog.argtypes = [array3d, array1d]
lib.cog.restype = c_double


def cog(data):
    center = np.zeros(3)
    Hsum = lib.cog(data, center)
    return center, Hsum


# sphericalHist( double * data_, double* center, double dr, int n, double* Hs, double* Ws ){
lib.sphericalHist.argtypes = [array3d, array1d, c_double, c_int, array1d, array1d]
lib.sphericalHist.restype = None


def sphericalHist(data, center, dr, n):
    Hs = np.zeros(n)
    Ws = np.zeros(n)
    rs = np.arange(0, n) * dr
    lib.sphericalHist(data, center, dr, n, Hs, Ws)
    return rs, Hs, Ws


lib.ReadNumsUpTo_C.argtypes = [c_char_p, array1d, array1i, c_int]
lib.ReadNumsUpTo_C.restype = c_int


def readNumsUpTo(filename, dimensions, noline):
    N_arry = np.zeros((dimensions[0] * dimensions[1] * dimensions[2]), dtype=np.double)
    lib.ReadNumsUpTo_C(filename.encode(), N_arry, dimensions, noline)
    return N_arry
