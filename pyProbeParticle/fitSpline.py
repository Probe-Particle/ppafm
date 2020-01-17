import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os
from . import cpp_utils
#import cpp_utils

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
    "void convolve3D_tensorProduct( int ord, int di, int nx, int ny, int nz, double* coefs, double* x, double* y ){",
    "void solveCG( int n, double* A, double* b, double* x, int maxIters, double maxErr ){",
    "void fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* Yref, double* Ycoefs, int maxIters, double maxErr2 )",
    "void fit_tensorProd_3D( int ord, int nx, int ny, int nz, double* kernel_coefs_, double* BYref, double* Ycoefs, int maxIters, double maxErr, int nConvPerCG_ ){",
    "void setup_fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs ){",
    "void step_fit_tensorProd( ){",
]
#cpp_utils.writeFuncInterfaces( header_strings );        exit()     #   uncomment this to re-generate C-python interfaces

cpp_name='fitSpline'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

# ========= C functions

#  void convolve1D( const int m, const int n, const double* coefs, const double* x, double* y )
lib.convolve1D.argtypes  = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p] 
lib.convolve1D.restype   =  None
def convolve1D(coefs, x, y=None, di=1 ):
    m = len(coefs)/2
    nx=len(x)
    if y is None:
        if di<0:
            y = np.zeros( nx*-di )
            m /= -di
        else: 
            nx/=di
            y = np.zeros( nx )
    print("di ", di, m)
    lib.convolve1D( m, di, nx, _np_as(coefs,c_double_p), _np_as(x,c_double_p), _np_as(y,c_double_p)) 
    return y

#  void convolve2D_tensorProduct( int nx, int ny, int ord, double* x, double* y, const double* coefs )
lib.convolve2D_tensorProduct.argtypes  = [c_int, c_int, c_int, c_int, c_double_p, c_double_p, c_double_p] 
lib.convolve2D_tensorProduct.restype   =  None
def convolve2D_tensorProduct( coefs, x, y=None, di=1 ):
    m = len(coefs)/2
    nx,ny=x.shape;   
    if y is None: 
        if di<0:
            y = np.zeros( (nx*-di,ny*-di) )
            m /= -di 
        else:
            nx/=di; ny/=di;
            y = np.zeros( (nx,ny) )
    lib.convolve2D_tensorProduct( m, di, nx, ny, _np_as(coefs,c_double_p), _np_as(x,c_double_p), _np_as(y,c_double_p) )
    #print ( "DONE 3 \n" );
    return y

#  void convolve3D_tensorProduct( int ord, int di, int nx, int ny, int nz, double* coefs, double* x, double* y ){
lib.convolve3D_tensorProduct.argtypes  = [c_int, c_int, c_int, c_int, c_int, c_double_p, c_double_p, c_double_p] 
lib.convolve3D_tensorProduct.restype   =  None
def convolve3D_tensorProduct( coefs, x, y=None, di=1,):
    m = len(coefs)/2
    nx,ny,nz=x.shape;   
    if y is None: 
        if di<0:
            y = np.zeros( (nx*-di,ny*-di,nz*-di) )
            m /= -di 
        else:
            nx/=di; ny/=di; nz/=di;
            y = np.zeros( (nx,ny,nz) )
    lib.convolve3D_tensorProduct( m, di, nx, ny, nz, _np_as(coefs,c_double_p), _np_as(x,c_double_p), _np_as(y,c_double_p)) 
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
lib.fit_tensorProd_2D.argtypes  = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p, c_int, c_double, c_int ] 
lib.fit_tensorProd_2D.restype   =  None
def fit_tensorProd_2D( BYref=None, Yref=None, basis_coefs=None, kernel_coefs=None, Ycoefs=None, maxIters=100, maxErr=1e-6, di=1, nConvPerCG=1 ):
    if BYref is None:
        print(" fit_tensorProd BYref = Basis * Yref ")
        BYref = convolve2D_tensorProduct( basis_coefs, Yref, di=di )
    if Ycoefs is None: Ycoefs = np.zeros(BYref.shape)
    nx,ny=BYref.shape
    print(" >> fit_tensorProd ... ")
    if kernel_coefs is None:
        print(" NO KERNEL => use basis with nConvPerCG==2")
        lib.fit_tensorProd_2D( len(basis_coefs)/2, nx, ny, _np_as(basis_coefs,c_double_p), _np_as(BYref,c_double_p), _np_as(Ycoefs,c_double_p), maxIters, maxErr, 2 )
    else:
        nker = len(kernel_coefs)
        lib.fit_tensorProd_2D( nker/2, nx, ny, _np_as(kernel_coefs,c_double_p), _np_as(BYref,c_double_p), _np_as(Ycoefs,c_double_p), maxIters, maxErr, nConvPerCG )
    #print "... DONE "
    return Ycoefs 

#  void fit_tensorProd_3D( int ord, int nx, int ny, int nz, double* kernel_coefs_, double* BYref, double* Ycoefs, int maxIters, double maxErr, int nConvPerCG_ ){
lib.fit_tensorProd_3D.argtypes  = [c_int, c_int, c_int, c_int, c_double_p, c_double_p, c_double_p, c_int, c_double, c_int] 
lib.fit_tensorProd_3D.restype   =  None
def fit_tensorProd_3D( BYref=None, Yref=None, basis_coefs=None, kernel_coefs=None, Ycoefs=None, maxIters=100, maxErr=1e-6, di=1, nConvPerCG=1  ):
    if BYref is None:
        print(" fit_tensorProd BYref = Basis * Yref ")
        BYref = convolve3D_tensorProduct( basis_coefs, Yref, di=di )
    if Ycoefs is None: Ycoefs = np.zeros(BYref.shape)
    nx,ny,nz=BYref.shape
    print(" >> fit_tensorProd ... ")
    if kernel_coefs is None:
        print(" NO KERNEL => use basis with nConvPerCG==2")
        lib.fit_tensorProd_3D( len(basis_coefs)/2, nx, ny, nz, _np_as(basis_coefs,c_double_p), _np_as(BYref,c_double_p), _np_as(Ycoefs,c_double_p), maxIters, maxErr, 2 )
    else:
        nker = len(kernel_coefs)
        lib.fit_tensorProd_3D( nker/2, nx, ny, nz, _np_as(kernel_coefs,c_double_p), _np_as(BYref,c_double_p), _np_as(Ycoefs,c_double_p), maxIters, maxErr, nConvPerCG )
    #return lib.fit_tensorProd_3D(ord, nx, ny, nz, _np_as(kernel_coefs_,c_double_p), _np_as(BYref,c_double_p), _np_as(Ycoefs,c_double_p), maxIters, maxErr, nConvPerCG_) 
    return Ycoefs

#  void setup_fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs ){
lib.setup_fit_tensorProd.argtypes  = [c_int, c_int, c_int, c_double_p, c_double_p, c_double_p, c_double_p, c_int ] 
lib.setup_fit_tensorProd.restype   =  None
def setup_fit_tensorProd( kernel_coefs, BYref, Ycoefs, Wprecond=None, nConvPerCG=1 ):
    nx,ny=BYref.shape
    if Wprecond is not None:
        Wprecond = _np_as(Wprecond,c_double_p)
    return lib.setup_fit_tensorProd( len(kernel_coefs)/2, nx, ny, _np_as(kernel_coefs,c_double_p), _np_as(BYref,c_double_p), _np_as(Ycoefs,c_double_p), Wprecond, nConvPerCG ) 

#  void step_fit_tensorProd( ){
lib.step_fit_tensorProd.argtypes  = [] 
lib.step_fit_tensorProd.restype   =  None
def step_fit_tensorProd():
    return lib.step_fit_tensorProd() 

# ========= Python

def upSwizzle( coefs, di ):
    n = int( np.ceil(float(len(coefs))/di) )
    cs = np.zeros((di,n))
    print("cs.shape ", cs.shape)
    for i in range(di):
        csi     = coefs[i::di]
        cs[i,-len(csi):] = csi 
    return cs.flat.copy()

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

def conv1D(xs,ys):
    nx=len(xs)
    ny=len(ys)
    #dn=(nx-ny)/2
    #res = np.emppty(nx+ny)
    ntot=nx+ny+2    
    xs_ = np.zeros(ntot)
    ys_ = np.zeros(ntot)
    dnx=(nx/2)+1
    dny=(ny/2)+1
    print(nx,ny,ntot, dnx, dny, len(xs_[dny:-dny]))
    xs_[dny:-dny-1] = xs
    ys_[dnx:-dnx-1] = ys
    conv = np.real(np.fft.ifft((np.fft.fft(xs_)*np.fft.fft(ys_))))  # Keep it Real !
    conv = np.roll(conv,ntot/2)
    #for i in range(nx,ny):
    #    res[i] = np.dot( xs[dn:-dn], ys[] )
    return conv




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def imfig( data, title ):
        plt.figure()
        plt.imshow(data)
        plt.title( title )
        plt.colorbar()

    #sp=BsplineCubic(np.array([-2,-1.,0.,1.,2.])); print sp, sp.sum()
    sp=BsplineCubic(np.array([-2.0,-1.5,-1.0,-0.5,0.,0.5,1.,1.5,2.])) ; print(sp/sp.sum())
    coefs3_2 = np.array([ 0.01041667, 0.08333333, 0.23958333, 0.33333333, 0.23958333, 0.08333333, 0.01041667 ])  # *2
    print("np.outer(coefs3_2,coefs3_2).sum() ", np.outer(coefs3_2,coefs3_2).sum())
    #coefs3_2 = sp/sp.sum()

    np.set_printoptions( precision=None, linewidth=200 )

    coefs3 = np.array([1.,4.,1.])/6

    coefs6 = np.array( [ 7.14285714e-03, 8.57142857e-01, 8.50714286e+00, 1.72571429e+01,  8.50714286e+00,  8.57142857e-01,  7.14285714e-03 ] )/(2*1.72571429e+01) #;print coefs6

    coefs5 = np.array( [0.1, 0.2, 0.4, 0.2, 0.1] );
    
    


    coefs3_ker = conv1D(coefs3_2,coefs3_2)    ; print(coefs3_ker) ; 
    coefs3_ker_down = coefs3_ker[:-2:2].copy()
    plt.plot( coefs3_2  ,'.-' )
    plt.plot( coefs3_ker,'.-' )
    plt.plot( coefs3_ker_down,'.-' )
    #plt.show()
    #exit()




    '''
    # ====== 1D
    y1d      = np.zeros(10)
    #y1d[5]   = 1    
    #y1d[8]   = 1
    #y1d[1]   = 1
    #y1d[4]   = 1
    y1d[0]   = 1
    y1d[5]   = 1
    y1d[-1]  = 1

    #print "y1d  ", y1d

    coefs3_2_up = upSwizzle( coefs3_2, di=2 )

    print "coefs3_2    ", len(coefs3_2   ), coefs3_2
    print "coefs3_2_up ", len(coefs3_2_up),"\n", coefs3_2_up

    y1d_conv  = convolve1D( coefs3, y1d,     )  ;print "y1d_conv  ", y1d_conv
    #y1d_conv  = convolve1D( coefs5, y1d,     )  ;print "y1d_conv  ", y1d_conv
    y1d_conv2  = convolve1D( coefs3_2_up, y1d,  di=-2   )  ;print "y1d_conv2  ", y1d_conv2
    #y1d_conv2 = convolve1D( coefs3, y1d_conv )  ;print "y1d_conv2 ", y1d_conv2
    #y1d_conv3 = convolve1D( coefs6, y1d      )  ;print "y1d_conv3 ", y1d_conv3

    plt.plot( np.arange(0,len(y1d),1),y1d     ,  "o-", label='y1d' )
    plt.plot( np.arange(0,len(y1d),1),y1d_conv,  ".-", label='y1d_conv' )
    plt.plot( np.arange(0,len(y1d),0.5),y1d_conv2,  ".-", label='y1d_conv2' )
    #plt.plot( y1d_conv2, ".-", label='y1d_conv2' )
    #plt.plot( y1d_conv3, ".-", label='y1d_conv3' )
    
    plt.show(); exit()
    '''

    '''
    # ======  2D
    y2d = np.zeros((20,20))
    y2d[:,:]   = 1

    
    y2d[15,16] = 0
    y2d[10:13,10] = 0
    y2d[13,11] = 0
    y2d[5:7,6:8] = 0

    #y2d[0,10]  = 0
    #y2d[19,10] = 0

    #y2d[0,0]   = 0
    #y2d[19,19] = 0

    y2d_conv = convolve2D_tensorProduct( coefs3, y2d )
    #y2d_conv = convolve2D_tensorProduct( coefs6, y2d )
    #y2d_conv  = convolve2D_tensorProduct( coefs5, y2d, di=2 )
    #y2d_conv  = convolve2D_tensorProduct( coefs3_2, y2d, di=2 )

    #coefs3_2_up = upSwizzle( coefs3_2, di=2 )
    #y2d_conv  = convolve2D_tensorProduct( coefs3_2_up, y2d, di=-2 )
    #print "DONE 4 "
    print  y2d_conv 

    interp = 'nearest'
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow( y2d     , interpolation=interp ); plt.title("input: coefs"); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow( y2d_conv, interpolation=interp );plt.title("input: Yfunc");  plt.colorbar()
    
    plt.show();    exit()
    '''

    
    # ======  3D
    N = 10
    y3D = np.zeros((N,N,N))
    y3D[:,:]   = 1

    y3D[5  ,7,  2] = 0
    y3D[3:5,3,4:7] = 0

    print("coefs3 ", coefs3)
    y3D_conv  = convolve3D_tensorProduct( coefs3, y3D, di=1 )
    print("min,max y3d_conv ", y3D_conv.min(), y3D_conv.max())


    iz_view = 4 

    interp = 'nearest'
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow( y3D     [iz_view], interpolation=interp ); plt.title("input: coefs"); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow( y3D_conv[iz_view], interpolation=interp ); plt.title("input: Yfunc"); plt.colorbar()
    
    from . import GridUtils as GU
    lvec0 = [[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1]]
    GU.saveXSF( "y2d.xsf",      y3D, lvec0 )
    GU.saveXSF( "y2d_conv.xsf", y3D_conv, lvec0 )
    #exit()
    #plt.show();    exit()

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

    """
    # ====== Fit 2D

    N = 20
    xs    = np.linspace(-np.pi,np.pi,N)
    Xs,Ys = np.meshgrid( xs, xs )

    #Yref = 1 +  0.5* np.sin(Xs*2) * np.cos(Ys*3) * (1+np.cos(Ys))*(1+np.cos(Xs))

    Yref = Xs*0;
    green = np.outer( coefs3, coefs3 )

    '''
    Yref[0:3,0:3]    += green
    Yref[N-3:N,N-3:] += green
    Yref[0:3,9:12]   += green
    Yref[16:19,16:19] += green
    Yref[10:13,9:12] += green
    Yref[6:9,9:12]   += green
    Yref[12:15,9:12] += green
    '''

    #Yref[5:10,5:10] = 1.0

    #Ycoefs = fit_tensorProd( kernel_coefs=coefs6, Yref=Yref, basis_coefs=coefs3, maxIters=50, maxErr=1e-6 )
    #Ycoefs = fit_tensorProd( Yref=Yref, basis_coefs=coefs3, maxIters=50, maxErr=1e-6 )

    #Ycoefs = fit_tensorProd( Yref=y2d_conv, basis_coefs=coefs3_2, kernel_coefs=coefs3,  maxIters=50, maxErr=1e-6, di=2, nConvPerCG=2 )
    Ycoefs = fit_tensorProd_2D( Yref=y2d_conv, basis_coefs=coefs3_2, kernel_coefs=coefs3_ker_down,  maxIters=50, maxErr=1e-6, di=2, nConvPerCG=1 )
    Yfit  = convolve2D_tensorProduct( coefs3_2_up, Ycoefs, di=-2 )

    #Ycoefs = fit_tensorProd( Yref=Yref, basis_coefs=coefs3, maxIters=50, maxErr=1e-6, di=2 )
    #Yfit   = convolve2D_tensorProduct( coefs3, Ycoefs )

    #imfig( Yref,   "Yref"   )
    #imfig( Yref,   "Yref"   )
    imfig( Ycoefs, "Fitted: Ycoefs" )
    #imfig( Yfit,   "Yfit"   )
    imfig( Yfit, "Fitted: Yfit" )
    

    '''
    BYref = convolve2D_tensorProduct( coefs3, Yref )
    plt.figure(); plt.imshow(Yref);  plt.title( 'Yref  ' ); plt.colorbar()
    plt.figure(); plt.imshow(BYref); plt.title( 'BYref ' ); plt.colorbar()
    Wprecond = None
    #c = coefs3[1]
    #K = 0.99
    #print " coefs3[1], coefs3 ", c, K, coefs3
    #Wprecond = np.ones( BYref.shape ) * K
    Ycoefs = np.zeros(BYref.shape)
    #setup_fit_tensorProd( coefs6, BYref, Ycoefs, nConvPerCG=1 )
    setup_fit_tensorProd( coefs3, BYref, Ycoefs, Wprecond, nConvPerCG=2 )
    for iter in range(50):
        step_fit_tensorProd ()
        if iter%10==0:
            plt.figure()
            plt.imshow(Ycoefs)
            plt.title( 'x CG[%i]' %iter )
            plt.colorbar()
    '''
    """


    # ====== Fit 3D

    xs    = np.linspace(-np.pi,np.pi,N)
    Xs,Ys = np.meshgrid( xs, xs )

    Yref = y3D_conv

    Ycoefs = fit_tensorProd_3D       ( Yref=Yref, basis_coefs=coefs3, maxIters=50, maxErr=1e-6, di=1, nConvPerCG=2 )
    Yfit   = convolve3D_tensorProduct( coefs3, Ycoefs )

    imfig( Ycoefs[iz_view], "Fitted: Ycoefs" )
    imfig( Yfit  [iz_view],   "Fitted: Yfit" )
    GU.saveXSF( "Ycoefs.xsf", Ycoefs, lvec0 )
    GU.saveXSF( "Yfit.xsf",   Yfit  , lvec0 )

    plt.show()



