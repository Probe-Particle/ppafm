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
    "void setPBC( int* npbc_, double* cell ){",
    "void setSplines( int ntypes, int npts, double invStep, double Rcut, double* RFuncs  ){",
    "void getProjections( int nps, int ncenters, double*  ps, double* yrefs, double* centers, int* types, int* ncomps,double* By, double* BB ){",
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
def getProjections( ps, yrefs, centers, types, ncomps, By=None, BB=None ):
    #nps      = len(ps)
    ndim = yrefs.shape
    nps=1
    if( len(ndim)>1):
        print "ndim ", ndim
        for ni in ndim: nps*=ni
    else:
        nps=ndim[0]
    ncenters = len(centers)
    if By is None:
        nbas  = ncomps.sum()
        By = np.zeros( nbas )
        BB = np.zeros( (nbas,nbas) )
    lib.getProjections(nps, ncenters, _np_as(ps,c_double_p), _np_as(yrefs,c_double_p), _np_as(centers,c_double_p), _np_as(types,c_int_p), _np_as(ncomps,c_int_p), _np_as(By,c_double_p), _np_as(BB,c_double_p)) 
    return By, BB

# ========= Python


if __name__ == "__main__":

    import GridUtils  as GU 
    import basUtils   as BU
    import common     as PPU

    fext  = "xsf" 
    fname = "CHGCAR"
    fname_ext = fname+"."+fext

    atoms,nDim,lvec = BU.loadGeometry   ( fname_ext, params=PPU.params )
    #F,lvec,nDim     = GU.load_scal_field( fname, data_format=fext )

    import sys
    fitting = sys.modules[__name__]

    data   = np.genfromtxt( fname+"_zlines_type.dat" ).transpose()

    zs     = data[0, :]
    RFuncs = data[1:,:].copy()

    rfsh   = RFuncs.shape
    print "RFunc.shape() ", rfsh
    fitting.setSplines( zs[1]-zs[0], 5.0, RFuncs )

    print "nDim ", nDim
    fitting.setPBC(lvec, npbc=[1,1,1])

    types_header = [1, 6, 7]
    typedict     = { k:i for i,k in enumerate(types_header) }
    #types  = np.array( [ typedict[elem] for elem in atoms[0] ], dtype=np.int32)
    ncomps = np.ones( len(types), dtype=np.int32  )

    gridPoss = PPU.getPos_Vec3d( np.array(lvec), nDim )
    yrefs,lvec,nDim = GU.load_scal_field( fname, data_format=fext )
    centers = np.array( atoms[1:] ).transpose().copy()

    print "gridPoss.shape, yrefs.shape, centers.shape ", gridPoss.shape, yrefs.shape, centers.shape

    print "GOTO getProjections "
    By,BB = fitting.getProjections( gridPoss, yrefs, centers, types, ncomps )
    print "By   ", By
    print "BB \n", BB
    print " **** ALL DONE *** "

