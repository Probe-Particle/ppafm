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
    "int insertAtomType( int nbond, int ihyb, double rbond0, double aMorse, double bMorse, double c6, double R2vdW, double Epz ){",
    "void reallocFF(int natom){",
    "int*    getTypes (){",
    "double* getDofs  (){",
    "double* getFDofs (){",
    "double* getEDofs (){",
    "void setupFF( int natom, int* types ){",
    "double setupOpt( double dt, double damp, double f_limit, double l_limit ){",
    "void setBox(double* pmin, double* pmax, double* k){",
    "double relaxNsteps( int nsteps, double Fconv, int ialg ){",
]
# cpp_utils.writeFuncInterfaces( header_strings );        exit()     #   uncomment this to re-generate C-python interfaces

cpp_name='FARFF'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

# ========= C functions

#  int insertAtomType( int nbond, int ihyb, double rbond0, double aMorse, double bMorse, double c6, double R2vdW, double Epz ){
lib.insertAtomType.argtypes  = [c_int, c_int, c_double, c_double, c_double, c_double, c_double, c_double] 
lib.insertAtomType.restype   =  c_int
def insertAtomType(nbond, ihyb, rbond0, aMorse, bMorse, c6, R2vdW, Epz):
    return lib.insertAtomType(nbond, ihyb, rbond0, aMorse, bMorse, c6, R2vdW, Epz) 

#  void reallocFF(int natom){
lib.reallocFF.argtypes  = [c_int] 
lib.reallocFF.restype   =  c_int
def reallocFF(natom):
    return lib.reallocFF(natom) 

#  int*    getTypes (){
#lib.   getTypes .argtypes  = [] 
#lib.   getTypes .restype   =  c_int_p
#def    getTypes (natoms):
#    return np.ctypeslib.as_array( lib.getTypes(), shape=(natoms,))

#  double* getDofs  (){
lib.getDofs  .argtypes  = [] 
lib.getDofs  .restype   =  c_double_p
def getDofs  (ndofs):
    return np.ctypeslib.as_array( lib.getDofs(), shape=(ndofs,3))

#  double* getFDofs (){
lib.getFDofs .argtypes  = [] 
lib.getFDofs .restype   =  c_double_p
def getFDofs (ndofs):
    return np.ctypeslib.as_array( lib.getFDofs(), shape=(ndofs,3)) 

#  double* getEDofs (){
lib.getEDofs .argtypes  = [] 
lib.getEDofs .restype   =  c_double_p
def getEDofs (ndofs):
    return np.ctypeslib.as_array( lib.getEDofs(), shape=(ndofs,)) 

#  double setupOpt( double dt, double damp, double f_limit, double l_limit ){
lib.setupFF.argtypes  = [c_int, c_int_p] 
lib.setupFF.restype   =  None
def setupFF( n=None, itypes=None ):
    if itypes is None:
        itypes = np.zeros( n, dtype=np.int32)
    else:
        n = len(itypes)
    print "type(itypes)",type(itypes),"itypes",itypes
    return lib.setupFF( n, _np_as(itypes,c_int_p) ) 

#  double setupOpt( double dt, double damp, double f_limit, double l_limit ){
lib.setupOpt.argtypes  = [c_double, c_double, c_double, c_double] 
lib.setupOpt.restype   =  c_double
def setupOpt(dt=0.2, damp=0.2, f_limit=10.0, l_limit=0.2 ):
    return lib.setupOpt(dt, damp, f_limit, l_limit) 

#  void setBox(double* pmin, double* pmax, double* k){
lib.setBox.argtypes  = [c_double_p, c_double_p, c_double_p] 
lib.setBox.restype   =  None
def setBox(pmin, pmax, k):
    return lib.setBox(_np_as(pmin,c_double_p), _np_as(pmax,c_double_p), _np_as(k,c_double_p)) 

#  double relaxNsteps( int nsteps, double Fconv, int ialg ){
lib.relaxNsteps.argtypes  = [c_int, c_double, c_int ] 
lib.relaxNsteps.restype   =  c_double
def relaxNsteps(nsteps, Fconv=1e-6, ialg=0):
    print nsteps
    return lib.relaxNsteps( nsteps, Fconv, ialg) 

if __name__ == "__main__":
    #import basUtils as bu
    import atomicUtils as au
    import sys
    fff = sys.modules[__name__]
    xyzs,Zs,elems,qs = au.loadAtomsNP("input.xyz")     #; print xyzs

    #fff.insertAtomType(nbond, ihyb, rbond0, aMorse, bMorse, c6, R2vdW, Epz)

    natom  = len(xyzs)
    ndof   = fff.reallocFF(natom)
    norb   = ndof - natom
    #atypes = fff.getTypes (natom)    ; print "atypes.shape ", atypes.shape
    dofs   = fff.getDofs(ndof)       ; print "dofs.shape ", dofs.shape
    apos   = dofs[:natom]            ; print "apos.shape ", apos.shape
    opos   = dofs[natom:]            ; print "opos.shape ", opos.shape

    #atypes[:] = 0         # use default atom type
    apos[:,:] = xyzs[:,:] #
    opos[:,:] = np.random.rand( norb, 3 ); print "opos.shape ", opos #   exit()

    fff.setupFF(n=natom)   # use default atom type

    fff.setupOpt(dt=0.2, damp=0.2, f_limit=10.0, l_limit=0.2 )
    #fff.relaxNsteps(50, Fconv=1e-6, ialg=0)

    def animRelax(i, perFrame=1):
        fff.relaxNsteps(perFrame, Fconv=1e-6, ialg=0)
        return apos,None
    au.makeMovie( "movie.xyz", 100, elems, animRelaxs )