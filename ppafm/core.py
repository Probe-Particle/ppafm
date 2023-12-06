#!/usr/bin/python

from ctypes import c_double, c_int

import numpy as np

from . import common as PPU
from . import cpp_utils
from .defaults import d3
from .io import bohrRadius2angstroem

# ==============================
# ============================== interface to C++ core
# ==============================

lib = cpp_utils.get_cdll("PP")

# define used numpy array types for interfacing with C++
array1i = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="CONTIGUOUS")
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags="CONTIGUOUS")
array4d = np.ctypeslib.ndpointer(dtype=np.double, ndim=4, flags="CONTIGUOUS")

# ========
# ======== Python warper function for C++ functions
# ========

# void setGridN( int * n ){
lib.setGridN.argtypes = [array1i]
lib.setGridN.restype = None


def setGridN(n):
    n = np.array(n, dtype=np.int32).copy()
    lib.setGridN(n)


# void setGridCell( double * cell ){
lib.setGridCell.argtypes = [array2d]
lib.setGridCell.restype = None


def setGridCell(cell=None):
    if cell is None:
        cell = np.array(
            [
                PPU.params["gridA"],
                PPU.params["gridB"],
                PPU.params["gridC"],
            ],
            dtype=np.float64,
        ).copy()
    cell = np.array(cell, dtype=np.float64)
    if cell.shape == (3, 3):
        cell = cell.copy()
    elif cell.shape == (4, 3):
        cell = cell[1:, :].copy()
    else:
        raise ValueError("cell has wrong format")
        exit()
    print("cell", cell)
    lib.setGridCell(cell)


def setFF_shape(n_, cell):
    n = np.array(n_).astype(np.int32)
    lib.setGridN(n)
    setGridCell(cell)


# void setFF_pointer( double * gridF, double * gridE  )
lib.setFF_Fpointer.argtypes = [array4d]
lib.setFF_Fpointer.restype = None


def setFF_Fpointer(gridF):
    lib.setFF_Fpointer(gridF)


# void setFF_pointer( double * gridF, double * gridE  )
lib.setFF_Epointer.argtypes = [array3d]
lib.setFF_Epointer.restype = None


def setFF_Epointer(gridE):
    lib.setFF_Epointer(gridE)


def setFF(cell=None, gridF=None, gridE=None):
    n_ = None
    if gridF is not None:
        setFF_Fpointer(gridF)
        n_ = np.shape(gridF)
    if gridE is not None:
        setFF_Epointer(gridE)
        n_ = np.shape(gridF)
    if cell is None:
        cell = np.array(
            [
                PPU.params["gridA"],
                PPU.params["gridB"],
                PPU.params["gridC"],
            ]
        ).copy()
    if n_ is not None:
        print("setFF() n_ : ", n_)
        setFF_shape(n_, cell)
    else:
        "Warrning : setFF shape not set !!!"


# void setRelax( int maxIters, double convF2, double dt, double damping )
lib.setRelax.argtypes = [c_int, c_double, c_double, c_double]
lib.setRelax.restype = None


def setRelax(maxIters=1000, convF2=1.0e-4, dt=0.1, damping=0.1):
    lib.setRelax(maxIters, convF * convF, dt, damping)


# void setFIRE( double finc, double fdec, double falpha )
lib.setFIRE.argtypes = [c_double, c_double, c_double]
lib.setFIRE.restype = None


def setFIRE(finc=1.1, fdec=0.5, falpha=0.99):
    lib.setFIRE(finc, fdec, falpha)


# void setTip( double lRad, double kRad, double * rPP0, double * kSpring )
lib.setTip.argtypes = [c_double, c_double, array1d, array1d]
lib.setTip.restype = None


def setTip(lRadial=None, kRadial=None, rPP0=None, kSpring=None):
    if lRadial is None:
        lRadial = PPU.params["r0Probe"][2]
    if kRadial is None:
        kRadial = PPU.params["krad"] / -PPU.eVA_Nm
    if rPP0 is None:
        rPP0 = np.array((PPU.params["r0Probe"][0], PPU.params["r0Probe"][1], 0.0))
    if kSpring is None:
        kSpring = np.array((PPU.params["klat"], PPU.params["klat"], 0.0)) / -PPU.eVA_Nm
    print(" IN setTip !!!!!!!!!!!!!! ")
    print(" lRadial ", lRadial)
    print(" kRadial ", kRadial)
    print(" rPP0 ", rPP0)
    print(" kSpring ", kSpring)
    lib.setTip(lRadial, kRadial, rPP0, kSpring)


# void setTipSpline( int n, double * xs, double * ydys ){
lib.setTipSpline.argtypes = [c_int, array1d, array2d]
lib.setTipSpline.restype = None


def setTipSpline(xs, ydys):
    n = len(xs)
    lib.setTipSpline(n, xs, ydys)


# void getInPoints_LJ( int npoints, double * points, double * FEs, int natoms_, double * Ratoms_, double * cLJs ){
lib.getInPoints_LJ.argtypes = [c_int, array2d, array2d, c_int, array2d, array2d]
lib.getInPoints_LJ.restype = None


def getInPoints_LJ(ps, Rs, cLJs, FEs=None):
    nats = len(Rs)
    npts = len(ps)
    if FEs is None:
        FEs = np.zeros((npts, 4))
    lib.getInPoints_LJ(npts, ps, FEs, nats, Rs, cLJs)
    return FEs


# void evalRadialFF( int n, double* rs, double* coefs, double* Es, double* Fs, int kind ){
lib.evalRadialFF.argtypes = [c_int, array1d, array1d, array1d, array1d, c_int, c_double]
lib.evalRadialFF.restype = None


def evalRadialFF(rs, coefs, Es=None, Fs=None, kind=1, ADamp=-1):
    n = len(rs)
    if Es is None:
        Es = np.zeros(n)
    if Fs is None:
        Fs = np.zeros(n)
    coefs = np.array(coefs, dtype=np.float64)
    lib.evalRadialFF(n, rs, coefs, Es, Fs, kind, ADamp)
    return Es, Fs


# void getClassicalFF       (    int natom,   double * Rs_, double * cLJs )
lib.getLennardJonesFF.argtypes = [c_int, array2d, array2d]
lib.getLennardJonesFF.restype = None


def getLennardJonesFF(Rs, cLJs):
    natom = len(Rs)
    lib.getLennardJonesFF(natom, Rs, cLJs)


# void getClassicalFF       (    int natom,   double * Rs_, double * cLJs )
lib.getVdWFF.argtypes = [c_int, array2d, array2d]
lib.getVdWFF.restype = None


def getVdWFF(Rs, cLJs):
    natom = len(Rs)
    lib.getVdWFF(natom, Rs, cLJs)


# void getVdWFF_RE( int natoms_, double * Ratoms_, double * REs, int kind, double ADamp_=-1.0 ){
lib.getVdWFF_RE.argtypes = [c_int, array2d, array2d, c_int, c_double]
lib.getVdWFF_RE.restype = None


def getVdWFF_RE(Rs, REs, kind=0, ADamp=-1.0):
    natom = len(Rs)
    lib.getVdWFF_RE(natom, Rs, REs, kind, ADamp)


# void getDFTD3FF(int natoms_, double * Ratoms_, double *d3_coeffs)
lib.getDFTD3FF.argtypes = [c_int, array2d, array2d]
lib.getDFTD3FF.restype = None


def getDFTD3FF(Rs, d3_coeffs):
    natom = len(Rs)
    lib.getDFTD3FF(natom, Rs, d3_coeffs)


# void computeD3Coeffs(
#    const int natoms_, const double *rs, const double *elems, const double *r_cov, const double *r_cut, const double *ref_cn,
#    const double *ref_c6, const double *r4r2, const double *k, const double *params, const int elem_pp, double *coeffs
# )
lib.computeD3Coeffs.argtypes = [c_int, array2d, array1i, array1d, array1d, array1d, array1d, array1d, array1d, array1d, c_int, array1d]
lib.computeD3Coeffs.restype = None


def computeD3Coeffs(Rs, iZs, iZPP, df_params):
    natom = len(Rs)
    Rs = np.array(Rs, dtype=np.float64)
    iZs = np.array(iZs, dtype=np.int32)
    r_cov = d3.R_COV.astype(np.float64)
    r_cut = d3.load_R0().astype(np.float64).flatten()
    ref_cn = d3.REF_CN.astype(np.float64).flatten()
    r4r2 = d3.R4R2.astype(np.float64)
    ref_c6 = d3.load_ref_c6().astype(np.float64).flatten()
    k = np.array([d3.K1, d3.K2, d3.K3], dtype=np.float64)
    df_params = np.array([df_params["s6"], df_params["s8"], df_params["a1"], df_params["a2"] * bohrRadius2angstroem], dtype=np.float64)
    coeffs = np.empty(4 * natom, dtype=np.float64)
    lib.computeD3Coeffs(natom, Rs, iZs, r_cov, r_cut, ref_cn, ref_c6, r4r2, k, df_params, iZPP, coeffs)
    return coeffs.reshape((natom, 4))


# void getClassicalFF       (    int natom,   double * Rs_, double * cLJs )
lib.getMorseFF.argtypes = [c_int, array2d, array2d, c_double]
lib.getMorseFF.restype = None


def getMorseFF(Rs, REs, alpha=None):
    if alpha is None:
        alpha = PPU.params["aMorse"]
    print("getMorseFF: alpha: %g [1/A] ", alpha)
    natom = len(Rs)
    lib.getMorseFF(natom, Rs, REs, alpha)


# void getCoulombFF       (    int natom,   double * Rs_, double * C6, double * C12 )
lib.getCoulombFF.argtypes = [c_int, array2d, array1d, c_int]
lib.getCoulombFF.restype = None


def getCoulombFF(Rs, kQQs, kind=0):
    natom = len(Rs)
    lib.getCoulombFF(natom, Rs, kQQs, kind)


# void getGaussDensity( int natoms_, double * Ratoms_, double * cRAs ){
lib.getGaussDensity.argtypes = [c_int, array2d, array2d]
lib.getGaussDensity.restype = None


def getGaussDensity(Rs, cRAs):
    natom = len(Rs)
    lib.getGaussDensity(natom, Rs, cRAs)


# void getSlaterDensity( int natoms_, double * Ratoms_, double * cRAs ){
lib.getSlaterDensity.argtypes = [c_int, array2d, array2d]
lib.getSlaterDensity.restype = None


def getSlaterDensity(Rs, cRAs):
    natom = len(Rs)
    lib.getSlaterDensity(natom, Rs, cRAs)


# void getDensityR4spline( int natoms_, double * Ratoms_, double * cRAs ){
lib.getDensityR4spline.argtypes = [c_int, array2d, array2d]
lib.getDensityR4spline.restype = None


def getDensityR4spline(Rs, cRAs, bNormalize=True):
    if bNormalize:
        cRAs[:, 0] /= ((np.pi * 32) / 105) * cRAs[:, 1] ** 3  # see https://www.wolframalpha.com/input/?i=4*pi*x%5E2*%281-x%5E2%29%5E2+integrate+from+0+to+1
    natom = len(Rs)
    lib.getDensityR4spline(natom, Rs, cRAs)


# int relaxTipStroke ( int probeStart, int nstep, double * rTips_, double * rs_, double * fs_ )
lib.relaxTipStroke.argtypes = [c_int, c_int, c_int, array2d, array2d, array2d]
lib.relaxTipStroke.restype = c_int


def relaxTipStroke(rTips, rs, fs, probeStart=1, relaxAlg=1):
    n = len(rTips)
    return lib.relaxTipStroke(probeStart, relaxAlg, n, rTips, rs, fs)


# int relaxTipStrokes ( int nx, int ny, int probeStart, int nstep, double * rTips_, double * rs_, double * fs_ )
lib.relaxTipStrokes_omp.argtypes = [c_int, c_int, c_int, c_int, c_int, array4d, array4d, array4d]
lib.relaxTipStrokes_omp.restype = c_int


def relaxTipStrokes_omp(rTips, rs, fs, probeStart=1, relaxAlg=1):
    nx, ny, nz, _ = rTips.shape
    return lib.relaxTipStrokes_omp(nx, ny, probeStart, relaxAlg, nz, rTips, rs, fs)


# void stiffnessMatrix( double ddisp, int which, int n,  double * rTips_, double * rs_,    double * eigenvals_, double * evec1_, double * evec2_, double * evec3_ ){
lib.stiffnessMatrix.argtypes = [c_double, c_int, c_int, array2d, array2d, array2d, array2d, array2d, array2d]
lib.stiffnessMatrix.restype = None


def stiffnessMatrix(rTips, rPPs, which=0, ddisp=0.05):
    print("py.core.stiffnessMatrix() ")
    n = len(rTips)
    eigenvals = np.zeros((n, 3))
    # this is really stupid solution because we cannot simply pass null pointer by ctypes; see :
    # https://github.com/numpy/numpy/issues/6239
    # http://stackoverflow.com/questions/32120178/how-can-i-pass-null-to-an-external-library-using-ctypes-with-an-argument-decla
    evecs = [None, None, None]
    for i in range(which):
        evecs[i] = np.zeros((n, 3))
    print("py.core.stiffnessMatrix() 1 ")

    lib.stiffnessMatrix(ddisp, which, n, rTips, rPPs, eigenvals, evecs[0], evecs[1], evecs[2])
    return eigenvals, evecs


# void subsample_uniform_spline( double x0, double dx, int n, double * ydys, int m, double * xs_, double * ys_ )
lib.subsample_uniform_spline.argtypes = [c_double, c_double, c_int, array2d, c_int, array1d, array1d]
lib.subsample_uniform_spline.restype = None


def subsample_uniform_spline(x0, dx, ydys, xs_, ys_=None):
    n = len(ydys)
    m = len(xs_)
    if ys_ is None:
        ys_ = np.zeros(m)
    lib.subsample_uniform_spline(x0, dx, n, ydys, m, xs_, ys_)
    return ys_


# void subsample_nonuniform_spline( int n, double * xs, double * ydys, int m, double * xs_, double * ys_ )
lib.subsample_nonuniform_spline.argtypes = [c_int, array1d, array2d, c_int, array1d, array1d]
lib.subsample_nonuniform_spline.restype = None


def subsample_nonuniform_spline(xs, ydys, xs_, ys_=None):
    n = len(xs)
    m = len(xs_)
    if ys_ is None:
        ys_ = np.zeros(m)
    lib.subsample_nonuniform_spline(n, xs, ydys, m, xs_, ys_)
    return ys_


# void test_force( int type, int n, double * r0_, double * dr_, double * R_, double * fs_ ){
lib.test_force.argtypes = [c_int, c_int, array1d, array1d, array1d, array2d]
lib.test_force.restype = None


def test_force(typ, r0, dr, R, fs):
    n = len(fs)
    lib.test_force(typ, n, r0, dr, R, fs)
    return fs


# void test_eigen3x3( double * mat, double * evs ){
lib.test_eigen3x3.argtypes = [array2d, array2d]
lib.test_eigen3x3.restype = None


def test_eigen3x3(mat):
    evs = np.zeros((4, 3))
    lib.test_eigen3x3(mat, evs)
    return evs
