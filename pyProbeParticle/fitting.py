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
    "void setPBC( int* npbc_, double* cell ){",
    "void setSplines( int ntypes, int npts, double invStep, double Rcut, double* RFuncs  ){",
    "void getProjections( int nps, int ncenters, double*  ps, double* yrefs, double* centers, int* types, int* ncomps,double* By, double* BB ){",
    "void project( int nps, int ncenters, double*  ps, double* Youts, double* centers, int* types, int* ncomps, double* coefs ){",
    "void debugGeomPBC_xsf( int ncenters, double* centers )",
]
#cpp_utils.writeFuncInterfaces( header_strings );        exit()     #   uncomment this to re-generate C-python interfaces

cpp_name='fitting'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

# ========= C functions

#  void setPBC( int * npbc_, double * cell ){
lib.setPBC.argtypes  = [c_int_p, c_double_p] 
lib.setPBC.restype   =  None
def setPBC(lvec, npbc=[1,1,1]):
    if len(lvec)!=3:
        lvec[:3] = lvec[:3]
    lvec = np.array(lvec,dtype=np.float64)
    npbc=np.array(npbc,np.int32)
    return lib.setPBC(_np_as(npbc,c_int_p), _np_as(lvec,c_double_p)) 

#  void setSplines( int ntypes, int npts, double invStep, double Rcut, double* RFuncs  ){
lib.setSplines.argtypes  = [c_int, c_int, c_double, c_double, c_double_p] 
lib.setSplines.restype   =  None
def setSplines( step, Rcut, RFuncs ):
    rfsh = RFuncs.shape
    return lib.setSplines( rfsh[0], rfsh[1],  1/step, Rcut, _np_as(RFuncs,c_double_p) ) 

#  void getProjections( int nps, int ncenters, double*  ps, double* yrefs, double* centers, int* types, int* ncomps,double * By, double * BB ){
lib.getProjections.argtypes  = [c_int, c_int, c_double_p, c_double_p, c_double_p, c_int_p, c_int_p, c_double_p, c_double_p] 
lib.getProjections.restype   =  None
def getProjections( ps, Yrefs, centers, types, ncomps, By=None, BB=None ):
    #nps      = len(ps)
    ndim = Yrefs.shape
    nps=1
    if( len(ndim)>1):
        print("ndim ", ndim)
        for ni in ndim: nps*=ni
    else:
        nps=ndim[0]
    ncenters = len(centers)
    if By is None:
        nbas  = ncomps.sum()
        By = np.zeros( nbas )
        BB = np.zeros( (nbas,nbas) )
    lib.getProjections(nps, ncenters, _np_as(ps,c_double_p), _np_as(Yrefs,c_double_p), _np_as(centers,c_double_p), _np_as(types,c_int_p), _np_as(ncomps,c_int_p), _np_as(By,c_double_p), _np_as(BB,c_double_p)) 
    return By, BB

#  void project( int nps, int ncenters, double*  ps, double* Youts, double* centers, int* types, int* ncomps, double* coefs ){
lib.project.argtypes  = [c_int, c_int, c_double_p, c_double_p, c_double_p, c_int_p, c_int_p, c_double_p] 
lib.project.restype   =  None
def project( ps, Youts, centers, types, ncomps, coefs):
    ndim = Youts.shape
    nps=1
    if( len(ndim)>1):
        print("ndim ", ndim)
        for ni in ndim: nps*=ni
    else:
        nps=ndim[0]
    ncenters = len(centers)
    return lib.project(nps, ncenters, _np_as(ps,c_double_p), _np_as(Youts,c_double_p), _np_as(centers,c_double_p), _np_as(types,c_int_p), _np_as(ncomps,c_int_p), _np_as(coefs,c_double_p)) 

#  void debugGeomPBC_xsf( int ncenters, double* centers )
lib.debugGeomPBC_xsf.argtypes  = [c_int, c_double_p] 
lib.debugGeomPBC_xsf.restype   =  None
def debugGeomPBC_xsf(centers):
    ncenters = len(centers)
    return lib.debugGeomPBC_xsf(ncenters, _np_as(centers,c_double_p)) 

# ========= Python

if __name__ == "__main__":

    np.set_printoptions( precision=None, linewidth=200 )

    from . import GridUtils  as GU 
    from . import basUtils   as BU
    from . import common     as PPU

    fext  = "xsf" 
    fname = "CHGCAR"
    fname_ext = fname+"."+fext

    atoms,nDim,lvec = BU.loadGeometry   ( fname_ext, params=PPU.params )
    #F,lvec,nDim     = GU.load_scal_field( fname, data_format=fext )
    centers = np.array( atoms[1:4] ).transpose().copy()
    print("centers \n", centers)

    import sys
    fitting = sys.modules[__name__]

    data   = np.genfromtxt( fname+"_zlines_type.dat" ).transpose()

    zs     = data[0, :]
    RFuncs = data[1:,:].copy()
    #for i in range(len(RFuncs)): RFuncs[i] *= ( 1/RFuncs[i,0] )

    rfsh   = RFuncs.shape
    print("RFunc.shape() ", rfsh)
    fitting.setSplines( zs[1]-zs[0], 5.0, RFuncs )

    print("nDim ", nDim)
    fitting.setPBC(lvec[1:], npbc=[1,1,1])
    #fitting.setPBC(lvec[1:], npbc=[0,0,0])

    types_header = [1, 6, 7]
    typedict     = { k:i for i,k in enumerate(types_header) }
    types  = np.array( [ typedict[elem] for elem in atoms[0] ], dtype=np.int32)

    print("types ", types)    #;exit() 
    ncomps = np.ones( len(types), dtype=np.int32  )

    #fitting.debugGeomPBC_xsf(centers);
    #exit();


    #yrefs,lvec,nDim = GU.load_scal_field( fname, data_format=fext )
    Yrefs,lvec,nDim,head = GU.loadXSF( fname_ext )
    gridPoss = PPU.getPos_Vec3d( np.array(lvec), nDim )

    #gridPoss = gridPoss[::8,::8,::8,:].copy()
    #Yrefs    = Yrefs   [::8,::8,::8].copy()

    #DEBUG 2 nbas 11 nsel 11 ps[42437](9.71429,12.5714,19.4286)
    #DEBUG 2 nbas 11 nsel 7 ps[42443](13.1429,12.5714,19.4286) 

    #gridPoss = np.array( [ [9.71429,12.5714,19.4286], [13.1429,12.5714,19.4286] ] )
    #Yrefs    = np.array( [ 1.0, 1.5 ] )

    print("gridPoss.shape, yrefs.shape, centers.shape ", gridPoss.shape, Yrefs.shape, centers.shape)

    #fitting.debugGeomPBC_xsf(centers);

    '''
    Youts = np.zeros( Yrefs.shape )
    coefs = np.ones( len(centers) )
    fitting.project( gridPoss, Youts, centers, types, ncomps, coefs );
    GU.saveXSF( "Youts.xsf", Youts, lvec )
    exit();
    '''

    '''
    print ">>>>>> By,BB = getProjections( Yref ) "
    By,BB = fitting.getProjections( gridPoss, Yrefs, centers, types, ncomps )
    print "By   ", By
    print "BB \n", BB
    print ">>>>>> Solve(   BB c = B y ) "
    coefs = np.linalg.solve( BB, By )
    print "coefs = ", coefs
    '''

    coefs = np.ones( len(centers) )*1.2

    print(">>>>>> Yrefs -= project( coefs ) ")
    #Youts = np.zeros( Yrefs.shape )
    fitting.project( gridPoss, Yrefs, centers, types, ncomps, coefs*-1.0 );
    GU.saveXSF( "Yresidual.xsf", Yrefs, lvec )
    exit();

    print(" **** ALL DONE *** ")

