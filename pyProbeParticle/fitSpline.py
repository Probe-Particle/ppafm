import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os
import cpp_utils

c_double_p = ctypes.POINTER(c_double)
c_int_p    = ctypes.POINTER(c_int)

def _np_as(arr,atype):
    if arr is None:
        return None
    else: 
        return arr.ctypes.data_as(atype)

cpp_utils.s_numpy_data_as_call = "_np_as(%s,%s)"

# ===== To generate Interfaces automatically from headers call:
header_strings = [
    "void convolve1D(int m, int n, double* coefs, double* x, double* y )",
    "void convolve2D_tensorProduct( int ord, int nx, int ny, double* coefs, double* x, double* y )",
    "void fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* Yref, double* Ycoefs, int maxIters, double maxErr2 )"
]
#cpp_utils.writeFuncInterfaces( header_strings );        exit()     #   uncomment this to re-generate C-python interfaces

cpp_name='fitSpline'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

# ========= C functions

#  void convolve1D( const int m, const int n, const double* coefs, const double* x, double* y )
lib.convolve1D.argtypes  = [c_int, c_int, c_double_p, c_double_p, c_double_p] 
lib.convolve1D.restype   =  None
def convolve1D(coefs, x, y=None):
    if y is None: y = np.zeros(x.shape)
    lib.convolve1D( len(coefs)/2, len(x), _np_as(coefs,c_double_p), _np_as(x,c_double_p), _np_as(y,c_double_p)) 
    return y

#  void convolve2D_tensorProduct( int nx, int ny, int ord, double* x, double* y, const double* coefs )
lib.convolve2D_tensorProduct.argtypes  = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p] 
lib.convolve2D_tensorProduct.restype   =  None
def convolve2D_tensorProduct( coefs, x, y=None ):
    if y is None: y = np.zeros(x.shape)
    nx,ny=x.shape
    lib.convolve2D_tensorProduct( len(coefs)/2, nx, ny, _np_as(coefs,c_double_p), _np_as(x,c_double_p), _np_as(y,c_double_p) ) 
    return y

#  void fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* Yref, double* Ycoefs, int maxIters, double maxErr2 )
lib.fit_tensorProd.argtypes  = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p, c_int, c_double] 
lib.fit_tensorProd.restype   =  None
def fit_tensorProd( kernel_coefs, BYref=None, Yref=None, basis_coefs=None, Ycoefs=None, maxIters=100, maxErr=1e-6 ):
    if BYref is None:
        print " fit_tensorProd BYref = Basis * Yref "
        BYref = convolve2D_tensorProduct( basis_coefs, Yref )
    if Ycoefs is None: Ycoefs = np.zeros(BYref.shape)
    nx,ny=BYref.shape
    print " >> fit_tensorProd ... "
    lib.fit_tensorProd( len(kernel_coefs)/2, nx, ny, _np_as(kernel_coefs,c_double_p), _np_as(BYref,c_double_p), _np_as(Ycoefs,c_double_p), maxIters, maxErr )
    print "... DONE "
    return Ycoefs 

# ========= Python

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions( precision=None, linewidth=200 )

    coefs3 = np.array([1.,4.,1.])/6

    coefs6 = np.array( [ 7.14285714e-03, 8.57142857e-01, 8.50714286e+00, 1.72571429e+01,  8.50714286e+00,  8.57142857e-01,  7.14285714e-03 ] )/(2*1.72571429e+01) ;print coefs6

    '''
    # ====== 1D
    y1d      = np.zeros(20)
    y1d[5]   = 1    

    y1d_conv  = convolve1D( coefs3, y1d,     )  ;print "y1d_conv  ", y1d_conv
    y1d_conv2 = convolve1D( coefs3, y1d_conv )  ;print "y1d_conv2 ", y1d_conv2
    y1d_conv3 = convolve1D( coefs6, y1d      )  ;print "y1d_conv2 ", y1d_conv3

    plt.plot( y1d     ,  ".-", label='y1d' )
    plt.plot( y1d_conv,  ".-", label='y1d_conv' )
    plt.plot( y1d_conv2, ".-", label='y1d_conv2' )
    plt.plot( y1d_conv3, ".-", label='y1d_conv3' )
    '''

    '''
    # ======  2D
    y2d = np.zeros((20,20))
    y2d[10,10] = 1

    #y2d_conv = convolve2D_tensorProduct( coefs3, y2d )
    y2d_conv = convolve2D_tensorProduct( coefs6, y2d )

    interp = 'nearest'

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow( y2d     , interpolation=interp ); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow( y2d_conv, interpolation=interp ); plt.colorbar()
    '''

    N = 20
    xs    = np.linspace(-5,5)
    Xs,Ys = np.meshgrid( xs, xs )

    Yref = 1 +  0.5* np.sin(Xs) * np.cos(Ys)

    Ycoefs = fit_tensorProd( kernel_coefs=coefs6, Yref=Yref, basis_coefs=coefs3, maxIters=50, maxErr=1e-6 )

    plt.imshow(Yref)

    plt.show()



