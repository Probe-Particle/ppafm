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
    "void solveCG( int n, double* A, double* b, double* x, int maxIters, double maxErr ){",
    "void fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* Yref, double* Ycoefs, int maxIters, double maxErr2 )",
    "void setup_fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs ){",
    "void step_fit_tensorProd( ){",
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

#  void solveCG( int n, double* A, double* b, double* x, int maxIters, double maxErr ){
lib.solveCG.argtypes  = [c_int, c_double_p, c_double_p, c_double_p, c_int, c_double] 
lib.solveCG.restype   =  None
def solveCG( A, b, x=None, maxIters=100, maxErr=1e-6 ):
    n = len(b)
    if x is None: x = np.zeros(n)
    lib.solveCG(n, _np_as(A,c_double_p), _np_as(b,c_double_p), _np_as(x,c_double_p), maxIters, maxErr)
    return x

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

#  void setup_fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs ){
lib.setup_fit_tensorProd.argtypes  = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p, c_int ] 
lib.setup_fit_tensorProd.restype   =  None
def setup_fit_tensorProd( kernel_coefs, BYref, Ycoefs, nConvPerCG=1 ):
    nx,ny=BYref.shape
    return lib.setup_fit_tensorProd( len(kernel_coefs)/2, nx, ny, _np_as(kernel_coefs,c_double_p), _np_as(BYref,c_double_p), _np_as(Ycoefs,c_double_p), nConvPerCG ) 

#  void step_fit_tensorProd( ){
lib.step_fit_tensorProd.argtypes  = [] 
lib.step_fit_tensorProd.restype   =  None
def step_fit_tensorProd():
    return lib.step_fit_tensorProd() 

# ========= Python

'''
https://math.stackexchange.com/questions/746939/2d-cubic-b-splines
https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/index.html
Box Spline https://en.wikipedia.org/wiki/Box_spline

Thin plate spline
https://en.wikipedia.org/wiki/Polyharmonic_spline
'''

# corel coefs
#                   0                1                 2                3
corel_coefs = [ 1.72571429e+01,  8.50714286e+00,  8.57142857e-01,  7.14285714e-03 ]


import numpy as np

'''
def testFunc(x, params, b=1 ):
    y = np.zeros(x.shape)
    for pi in params:
        y += pi[0]/(1.+((x-pi[1])/b)**2)
    return y
'''

def genSplineBasis( x, x0s ):
    Bs = np.empty( ( len(x0s), len(x) ) )
    for i,x0 in enumerate(x0s):
        Bs[i,:] = BsplineCubic( x-x0 )
    return Bs

def BsplineCubic(x):
    absx = abs(x)
    x2   = x*x
    y                    = ( 3*absx*x2      - 6*x2 + 4)
    mask=absx>1; y[mask] = ( (-absx*(x2+12) + 6*x2 + 8) )[mask]
    mask=absx>2; y[mask] = 0
    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions( precision=None, linewidth=200 )

    coefs3 = np.array([1.,4.,1.])/6

    coefs6 = np.array( [ 7.14285714e-03, 8.57142857e-01, 8.50714286e+00, 1.72571429e+01,  8.50714286e+00,  8.57142857e-01,  7.14285714e-03 ] )/(2*1.72571429e+01) #;print coefs6

    coefs5 = np.array( [0.1, 0.2, 0.4, 0.2, 0.1] );
    
    '''
    # ====== 1D
    y1d      = np.zeros(10)
    #y1d[5]   = 1    
    y1d[8]   = 1
    y1d[1]   = 1
    y1d[4]   = 1
    y1d[5]   = 1

    print "y1d  ", y1d

    #y1d_conv  = convolve1D( coefs3, y1d,     )  ;print "y1d_conv  ", y1d_conv
    y1d_conv  = convolve1D( coefs5, y1d,     )  ;print "y1d_conv  ", y1d_conv
    #y1d_conv2 = convolve1D( coefs3, y1d_conv )  ;print "y1d_conv2 ", y1d_conv2
    #y1d_conv3 = convolve1D( coefs6, y1d      )  ;print "y1d_conv3 ", y1d_conv3

    plt.plot( y1d     ,  ".-", label='y1d' )
    plt.plot( y1d_conv,  ".-", label='y1d_conv' )
    #plt.plot( y1d_conv2, ".-", label='y1d_conv2' )
    #plt.plot( y1d_conv3, ".-", label='y1d_conv3' )
    
    exit()
    '''

    '''
    # ======  2D
    y2d = np.zeros((20,20))
    y2d[:,:] = 1
    y2d[0,10] = 0
    y2d[10,10] = 0
    y2d[19,10] = 0

    y2d[0,0] = 0
    y2d[19,19] = 0

    #y2d_conv = convolve2D_tensorProduct( coefs3, y2d )
    #y2d_conv = convolve2D_tensorProduct( coefs6, y2d )
    y2d_conv = convolve2D_tensorProduct( coefs5, y2d )

    interp = 'nearest'

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow( y2d     , interpolation=interp ); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow( y2d_conv, interpolation=interp ); plt.colorbar()
    
    plt.show();    exit()
    '''

    '''
    import CG

    N = 100
    nbas = 5
    x = np.linspace(-5,5,N)
    y_ref = np.sin( x*.3+6 )**5.0
    plt.plot(x,y_ref)

    x0s = range(-nbas,nbas+1)
    Bs = genSplineBasis( x, x0s )
    By = np.dot( Bs, y_ref )/N   ;print "By \n",  By
    BB = np.dot( Bs, Bs.T )/N    ;print "BB \n",  BB

    #coefs0 = np.zeros(len(By))
    #coefs  = CG.CG( BB, By, coefs0, nMaxIter=20, Econv=1e-10 )  ; print coefs

    coefs = solveCG( BB, By )

    y_fit = np.dot( Bs.T, coefs )
    plt.plot(x,y_fit, ':')

    plt.show()
    exit()
    '''

    N = 20
    xs    = np.linspace(-np.pi,np.pi)
    Xs,Ys = np.meshgrid( xs, xs )

    #Yref = 1 +  0.5* np.sin(Xs*2) * np.cos(Ys*3) * (1+np.cos(Ys))*(1+np.cos(Xs))

    Yref = Xs*0;
    green = np.outer( coefs3, coefs3 )
    Yref[10:13,9:12] += green
    Yref[6:9,9:12] += green
    Yref[12:15,9:12] += green


    #Ycoefs = fit_tensorProd( kernel_coefs=coefs6, Yref=Yref, basis_coefs=coefs3, maxIters=50, maxErr=1e-6 )

    BYref = convolve2D_tensorProduct( coefs3, Yref )

    plt.figure(); plt.imshow(Yref);  plt.title( 'Yref  ' ); plt.colorbar()
    plt.figure(); plt.imshow(BYref); plt.title( 'BYref ' ); plt.colorbar()

    Ycoefs = np.zeros(BYref.shape)
    #setup_fit_tensorProd( coefs6, BYref, Ycoefs, nConvPerCG=1 )
    setup_fit_tensorProd( coefs3, BYref, Ycoefs, nConvPerCG=2 )
    for iter in range(50):
        step_fit_tensorProd ()
        if iter%10==0:
            plt.figure()
            plt.imshow(Ycoefs)
            plt.title( 'x CG[%i]' %iter )
            plt.colorbar()

    plt.show()



