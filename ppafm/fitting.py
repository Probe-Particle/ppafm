import ctypes
from ctypes import c_double, c_int

import numpy as np

from . import cpp_utils

c_double_p = ctypes.POINTER(c_double)
c_int_p = ctypes.POINTER(c_int)


def _np_as(arr, atype):
    if arr is None:
        return None
    else:
        return arr.ctypes.data_as(atype)


cpp_utils.s_numpy_data_as_call = "_np_as(%s,%s)"

# ===== To generate Interfaces automatically from headers call:
header_strings = [
    "void setPBC( int* npbc_, double* cell ){",
    "void setSplines( int ntypes, int npts, double invStep, double Rcut, double* RFuncs  ){",
    "void getProjections( int nps, int ncenters, double*  ps, double* yrefs, double* centers, int* types, int* ncomps,double* By, double* BB ){",
    "void project( int nps, int ncenters, double*  ps, double* Youts, double* centers, int* types, int* ncomps, double* coefs ){",
    "void debugGeomPBC_xsf( int ncenters, double* centers )",
]

lib = cpp_utils.get_cdll("fitting")

# ========= C functions

#  void setPBC( int * npbc_, double * cell ){
lib.setPBC.argtypes = [c_int_p, c_double_p]
lib.setPBC.restype = None


def setPBC(lvec, npbc=[1, 1, 1]):
    if len(lvec) != 3:
        lvec[:3] = lvec[:3]
    lvec = np.array(lvec, dtype=np.float64)
    npbc = np.array(npbc, np.int32)
    return lib.setPBC(_np_as(npbc, c_int_p), _np_as(lvec, c_double_p))


#  void setSplines( int ntypes, int npts, double invStep, double Rcut, double* RFuncs  ){
lib.setSplines.argtypes = [c_int, c_int, c_double, c_double, c_double_p]
lib.setSplines.restype = None


def setSplines(step, Rcut, RFuncs):
    rfsh = RFuncs.shape
    return lib.setSplines(rfsh[0], rfsh[1], 1 / step, Rcut, _np_as(RFuncs, c_double_p))


#  void getProjections( int nps, int ncenters, double*  ps, double* yrefs, double* centers, int* types, int* ncomps,double * By, double * BB ){
lib.getProjections.argtypes = [c_int, c_int, c_double_p, c_double_p, c_double_p, c_int_p, c_int_p, c_double_p, c_double_p]
lib.getProjections.restype = None


def getProjections(ps, Yrefs, centers, types, ncomps, By=None, BB=None):
    ndim = Yrefs.shape
    nps = 1
    if len(ndim) > 1:
        print("ndim ", ndim)
        for ni in ndim:
            nps *= ni
    else:
        nps = ndim[0]
    ncenters = len(centers)
    if By is None:
        nbas = ncomps.sum()
        By = np.zeros(nbas)
        BB = np.zeros((nbas, nbas))
    lib.getProjections(
        nps,
        ncenters,
        _np_as(ps, c_double_p),
        _np_as(Yrefs, c_double_p),
        _np_as(centers, c_double_p),
        _np_as(types, c_int_p),
        _np_as(ncomps, c_int_p),
        _np_as(By, c_double_p),
        _np_as(BB, c_double_p),
    )
    return By, BB


#  void project( int nps, int ncenters, double*  ps, double* Youts, double* centers, int* types, int* ncomps, double* coefs ){
lib.project.argtypes = [c_int, c_int, c_double_p, c_double_p, c_double_p, c_int_p, c_int_p, c_double_p]
lib.project.restype = None


def project(ps, Youts, centers, types, ncomps, coefs):
    ndim = Youts.shape
    nps = 1
    if len(ndim) > 1:
        print("ndim ", ndim)
        for ni in ndim:
            nps *= ni
    else:
        nps = ndim[0]
    ncenters = len(centers)
    return lib.project(
        nps, ncenters, _np_as(ps, c_double_p), _np_as(Youts, c_double_p), _np_as(centers, c_double_p), _np_as(types, c_int_p), _np_as(ncomps, c_int_p), _np_as(coefs, c_double_p)
    )


#  void debugGeomPBC_xsf( int ncenters, double* centers )
lib.debugGeomPBC_xsf.argtypes = [c_int, c_double_p]
lib.debugGeomPBC_xsf.restype = None


def debugGeomPBC_xsf(centers):
    ncenters = len(centers)
    return lib.debugGeomPBC_xsf(ncenters, _np_as(centers, c_double_p))


# ========= Python

if __name__ == "__main__":
    np.set_printoptions(precision=None, linewidth=200)

    from . import common as PPU
    from . import io

    fext = "xsf"
    fname = "CHGCAR"
    fname_ext = fname + "." + fext

    atoms, nDim, lvec = io.loadGeometry(fname_ext, params=PPU.params)
    centers = np.array(atoms[1:4]).transpose().copy()
    print("centers \n", centers)

    import sys

    fitting = sys.modules[__name__]

    data = np.genfromtxt(fname + "_zlines_type.dat").transpose()

    zs = data[0, :]
    RFuncs = data[1:, :].copy()

    rfsh = RFuncs.shape
    print("RFunc.shape() ", rfsh)
    fitting.setSplines(zs[1] - zs[0], 5.0, RFuncs)

    print("nDim ", nDim)
    fitting.setPBC(lvec[1:], npbc=[1, 1, 1])

    types_header = [1, 6, 7]
    typedict = {k: i for i, k in enumerate(types_header)}
    types = np.array([typedict[elem] for elem in atoms[0]], dtype=np.int32)

    print("types ", types)
    ncomps = np.ones(len(types), dtype=np.int32)

    Yrefs, lvec, nDim, head = io.loadXSF(fname_ext)
    gridPoss = PPU.getPos_Vec3d(np.array(lvec), nDim)

    print("gridPoss.shape, yrefs.shape, centers.shape ", gridPoss.shape, Yrefs.shape, centers.shape)

    coefs = np.ones(len(centers)) * 1.2

    print(">>>>>> Yrefs -= project( coefs ) ")
    fitting.project(gridPoss, Yrefs, centers, types, ncomps, coefs * -1.0)
    io.saveXSF("Yresidual.xsf", Yrefs, lvec)
    exit()

    print(" **** ALL DONE *** ")
