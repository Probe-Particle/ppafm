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
    "void convolve1D(int m, int n, double* coefs, double* x, double* y )",
    "void convolve2D_tensorProduct( int ord, int nx, int ny, double* coefs, double* x, double* y )",
    "void convolve3D_tensorProduct( int ord, int di, int nx, int ny, int nz, double* coefs, double* x, double* y ){",
    "void solveCG( int n, double* A, double* b, double* x, int maxIters, double maxErr ){",
    "void fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* Yref, double* Ycoefs, int maxIters, double maxErr2 )",
    "void fit_tensorProd_3D( int ord, int nx, int ny, int nz, double* kernel_coefs_, double* BYref, double* Ycoefs, int maxIters, double maxErr, int nConvPerCG_ ){",
    "void setup_fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs ){",
    "void step_fit_tensorProd( ){",
]

lib = cpp_utils.get_cdll("fitSpline")

# ========= C functions

#  void convolve1D( const int m, const int n, const double* coefs, const double* x, double* y )
lib.convolve1D.argtypes = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p]
lib.convolve1D.restype = None


def convolve1D(coefs, x, y=None, di=1):
    m = len(coefs) / 2
    nx = len(x)
    if y is None:
        if di < 0:
            y = np.zeros(nx * -di)
            m /= -di
        else:
            nx /= di
            y = np.zeros(nx)
    print("di ", di, m)
    lib.convolve1D(m, di, nx, _np_as(coefs, c_double_p), _np_as(x, c_double_p), _np_as(y, c_double_p))
    return y


#  void convolve2D_tensorProduct( int nx, int ny, int ord, double* x, double* y, const double* coefs )
lib.convolve2D_tensorProduct.argtypes = [c_int, c_int, c_int, c_int, c_double_p, c_double_p, c_double_p]
lib.convolve2D_tensorProduct.restype = None


def convolve2D_tensorProduct(coefs, x, y=None, di=1):
    m = len(coefs) / 2
    nx, ny = x.shape
    if y is None:
        if di < 0:
            y = np.zeros((nx * -di, ny * -di))
            m /= -di
        else:
            nx /= di
            ny /= di
            y = np.zeros((nx, ny))
    lib.convolve2D_tensorProduct(m, di, nx, ny, _np_as(coefs, c_double_p), _np_as(x, c_double_p), _np_as(y, c_double_p))
    # print ( "DONE 3 \n" );
    return y


#  void convolve3D_tensorProduct( int ord, int di, int nx, int ny, int nz, double* coefs, double* x, double* y ){
lib.convolve3D_tensorProduct.argtypes = [c_int, c_int, c_int, c_int, c_int, c_double_p, c_double_p, c_double_p]
lib.convolve3D_tensorProduct.restype = None


def convolve3D_tensorProduct(
    coefs,
    x,
    y=None,
    di=1,
):
    m = len(coefs) / 2
    nx, ny, nz = x.shape
    if y is None:
        if di < 0:
            y = np.zeros((nx * -di, ny * -di, nz * -di))
            m /= -di
        else:
            nx /= di
            ny /= di
            nz /= di
            y = np.zeros((nx, ny, nz))
    lib.convolve3D_tensorProduct(m, di, nx, ny, nz, _np_as(coefs, c_double_p), _np_as(x, c_double_p), _np_as(y, c_double_p))
    return y


#  void solveCG( int n, double* A, double* b, double* x, int maxIters, double maxErr ){
lib.solveCG.argtypes = [c_int, c_double_p, c_double_p, c_double_p, c_int, c_double]
lib.solveCG.restype = None


def solveCG(A, b, x=None, maxIters=100, maxErr=1e-6):
    n = len(b)
    if x is None:
        x = np.zeros(n)
    lib.solveCG(n, _np_as(A, c_double_p), _np_as(b, c_double_p), _np_as(x, c_double_p), maxIters, maxErr)
    return x


#  void fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* Yref, double* Ycoefs, int maxIters, double maxErr2 )
lib.fit_tensorProd_2D.argtypes = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p, c_int, c_double, c_int]
lib.fit_tensorProd_2D.restype = None


def fit_tensorProd_2D(BYref=None, Yref=None, basis_coefs=None, kernel_coefs=None, Ycoefs=None, maxIters=100, maxErr=1e-6, di=1, nConvPerCG=1):
    if BYref is None:
        print(" fit_tensorProd BYref = Basis * Yref ")
        BYref = convolve2D_tensorProduct(basis_coefs, Yref, di=di)
    if Ycoefs is None:
        Ycoefs = np.zeros(BYref.shape)
    nx, ny = BYref.shape
    print(" >> fit_tensorProd ... ")
    if kernel_coefs is None:
        print(" NO KERNEL => use basis with nConvPerCG==2")
        lib.fit_tensorProd_2D(len(basis_coefs) / 2, nx, ny, _np_as(basis_coefs, c_double_p), _np_as(BYref, c_double_p), _np_as(Ycoefs, c_double_p), maxIters, maxErr, 2)
    else:
        nker = len(kernel_coefs)
        lib.fit_tensorProd_2D(nker / 2, nx, ny, _np_as(kernel_coefs, c_double_p), _np_as(BYref, c_double_p), _np_as(Ycoefs, c_double_p), maxIters, maxErr, nConvPerCG)
    return Ycoefs


#  void fit_tensorProd_3D( int ord, int nx, int ny, int nz, double* kernel_coefs_, double* BYref, double* Ycoefs, int maxIters, double maxErr, int nConvPerCG_ ){
lib.fit_tensorProd_3D.argtypes = [c_int, c_int, c_int, c_int, c_double_p, c_double_p, c_double_p, c_int, c_double, c_int]
lib.fit_tensorProd_3D.restype = None


def fit_tensorProd_3D(BYref=None, Yref=None, basis_coefs=None, kernel_coefs=None, Ycoefs=None, maxIters=100, maxErr=1e-6, di=1, nConvPerCG=1):
    if BYref is None:
        print(" fit_tensorProd BYref = Basis * Yref ")
        BYref = convolve3D_tensorProduct(basis_coefs, Yref, di=di)
    if Ycoefs is None:
        Ycoefs = np.zeros(BYref.shape)
    nx, ny, nz = BYref.shape
    print(" >> fit_tensorProd ... ")
    if kernel_coefs is None:
        print(" NO KERNEL => use basis with nConvPerCG==2")
        lib.fit_tensorProd_3D(len(basis_coefs) / 2, nx, ny, nz, _np_as(basis_coefs, c_double_p), _np_as(BYref, c_double_p), _np_as(Ycoefs, c_double_p), maxIters, maxErr, 2)
    else:
        nker = len(kernel_coefs)
        lib.fit_tensorProd_3D(nker / 2, nx, ny, nz, _np_as(kernel_coefs, c_double_p), _np_as(BYref, c_double_p), _np_as(Ycoefs, c_double_p), maxIters, maxErr, nConvPerCG)
    return Ycoefs


#  void setup_fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs ){
lib.setup_fit_tensorProd.argtypes = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p, c_double_p, c_int]
lib.setup_fit_tensorProd.restype = None


def setup_fit_tensorProd(kernel_coefs, BYref, Ycoefs, Wprecond=None, nConvPerCG=1):
    nx, ny = BYref.shape
    if Wprecond is not None:
        Wprecond = _np_as(Wprecond, c_double_p)
    return lib.setup_fit_tensorProd(len(kernel_coefs) / 2, nx, ny, _np_as(kernel_coefs, c_double_p), _np_as(BYref, c_double_p), _np_as(Ycoefs, c_double_p), Wprecond, nConvPerCG)


#  void step_fit_tensorProd( ){
lib.step_fit_tensorProd.argtypes = []
lib.step_fit_tensorProd.restype = None


def step_fit_tensorProd():
    return lib.step_fit_tensorProd()


# ========= Python


def upSwizzle(coefs, di):
    n = int(np.ceil(float(len(coefs)) / di))
    cs = np.zeros((di, n))
    print("cs.shape ", cs.shape)
    for i in range(di):
        csi = coefs[i::di]
        cs[i, -len(csi) :] = csi
    return cs.flat.copy()


"""
https://math.stackexchange.com/questions/746939/2d-cubic-b-splines
https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/index.html
Box Spline https://en.wikipedia.org/wiki/Box_spline

Thin plate spline
https://en.wikipedia.org/wiki/Polyharmonic_spline
"""

# corel coefs
#                   0                1                 2                3
corel_coefs = [1.72571429e01, 8.50714286e00, 8.57142857e-01, 7.14285714e-03]


import numpy as np


def genSplineBasis(x, x0s):
    Bs = np.empty((len(x0s), len(x)))
    for i, x0 in enumerate(x0s):
        Bs[i, :] = BsplineCubic(x - x0)
    return Bs


def BsplineCubic(x):
    absx = abs(x)
    x2 = x * x
    y = 3 * absx * x2 - 6 * x2 + 4
    mask = absx > 1
    y[mask] = (-absx * (x2 + 12) + 6 * x2 + 8)[mask]
    mask = absx > 2
    y[mask] = 0
    return y


def conv1D(xs, ys):
    nx = len(xs)
    ny = len(ys)
    ntot = nx + ny + 2
    xs_ = np.zeros(ntot)
    ys_ = np.zeros(ntot)
    dnx = (nx / 2) + 1
    dny = (ny / 2) + 1
    print(nx, ny, ntot, dnx, dny, len(xs_[dny:-dny]))
    xs_[dny : -dny - 1] = xs
    ys_[dnx : -dnx - 1] = ys
    conv = np.real(np.fft.ifft(np.fft.fft(xs_) * np.fft.fft(ys_)))  # Keep it Real !
    conv = np.roll(conv, ntot / 2)
    return conv


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def imfig(data, title):
        plt.figure()
        plt.imshow(data)
        plt.title(title)
        plt.colorbar()

    sp = BsplineCubic(np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]))
    print(sp / sp.sum())
    coefs3_2 = np.array([0.01041667, 0.08333333, 0.23958333, 0.33333333, 0.23958333, 0.08333333, 0.01041667])  # *2
    print("np.outer(coefs3_2,coefs3_2).sum() ", np.outer(coefs3_2, coefs3_2).sum())

    np.set_printoptions(precision=None, linewidth=200)

    coefs3 = np.array([1.0, 4.0, 1.0]) / 6

    coefs6 = np.array([7.14285714e-03, 8.57142857e-01, 8.50714286e00, 1.72571429e01, 8.50714286e00, 8.57142857e-01, 7.14285714e-03]) / (2 * 1.72571429e01)  # ;print coefs6

    coefs5 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    coefs3_ker = conv1D(coefs3_2, coefs3_2)
    print(coefs3_ker)
    coefs3_ker_down = coefs3_ker[:-2:2].copy()
    plt.plot(coefs3_2, ".-")
    plt.plot(coefs3_ker, ".-")
    plt.plot(coefs3_ker_down, ".-")

    # ======  3D
    N = 10
    y3D = np.zeros((N, N, N))
    y3D[:, :] = 1

    y3D[5, 7, 2] = 0
    y3D[3:5, 3, 4:7] = 0

    print("coefs3 ", coefs3)
    y3D_conv = convolve3D_tensorProduct(coefs3, y3D, di=1)
    print("min,max y3d_conv ", y3D_conv.min(), y3D_conv.max())

    iz_view = 4

    interp = "nearest"
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(y3D[iz_view], interpolation=interp)
    plt.title("input: coefs")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(y3D_conv[iz_view], interpolation=interp)
    plt.title("input: Yfunc")
    plt.colorbar()

    from ppafm import io

    lvec0 = [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]
    io.saveXSF("y2d.xsf", y3D, lvec0)
    io.saveXSF("y2d_conv.xsf", y3D_conv, lvec0)

    # ====== Fit 3D

    xs = np.linspace(-np.pi, np.pi, N)
    Xs, Ys = np.meshgrid(xs, xs)

    Yref = y3D_conv

    Ycoefs = fit_tensorProd_3D(Yref=Yref, basis_coefs=coefs3, maxIters=50, maxErr=1e-6, di=1, nConvPerCG=2)
    Yfit = convolve3D_tensorProduct(coefs3, Ycoefs)

    imfig(Ycoefs[iz_view], "Fitted: Ycoefs")
    imfig(Yfit[iz_view], "Fitted: Yfit")
    io.saveXSF("Ycoefs.xsf", Ycoefs, lvec0)
    io.saveXSF("Yfit.xsf", Yfit, lvec0)

    plt.show()
