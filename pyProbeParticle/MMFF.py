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


'''

NOTE: 

Nice QM/MM interface with own simple QM solver : Sire:
https://github.com/michellab/Sire
/home/prokop/git_SW/_Chemistry/Sire/corelib/src/libs

'''

# ===== To generate Interfaces automatically from headers call:
'''
cpp_utils.writeFuncInterfaces([
"void addAtoms( int n, double* pos_, int* npe_ ){",
"void addBonds( int n, int* bond2atom_, double* l0s, double* ks ){",
"double* setNonBonded( int n, double* REQs){",
"int getAtomTypes( int nmax, int* types ){",
"void setBox(double* pmin, double* pmax, double* k){",
"bool buildFF( bool bAutoHydrogens, bool bAutoAngles, bool bSortBonds ){",
"double setupOpt( double dt, double damp, double f_limit, double l_limit ){",
"double relaxNsteps( int ialg, int nsteps, double F2conf ){",
])
exit()
'''

cpp_name='MMFF'
#cpp_utils.compile_lib( cpp_name  )
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

# ========= C functions

#double* getPos  (){ 
lib.getPos.argtypes = []
lib.getPos.restype  = ctypes.POINTER(c_double)
def getPos(n):
    return np.ctypeslib.as_array( lib.getPos(), shape=(n,3) )  

#double* getForce  (){ 
lib.getForce.argtypes = []
lib.getForce.restype  = ctypes.POINTER(c_double)
def getForce(n):
    return np.ctypeslib.as_array( lib.getForce(), shape=(n,3) ) 

#int getAtomTypes( int nmax, int* types ){
lib.getAtomTypes.argtypes = [c_int, c_int_p ]
lib.getAtomTypes.restype  = c_int
def getAtomTypes(n=1000,types=None):
    if types is None:
        types = np.zeros(n,dtype=np.int32)
    na = lib.getAtomTypes(len(types),  _np_as(types,c_int_p))
    return types[:na]

#  void addAtoms( int n, double* pos_, int* npe_ ){
lib.addAtoms.argtypes  = [c_int, c_double_p, c_int_p] 
lib.addAtoms.restype   =  None
def addAtoms(pos_, npe_=None):
    return lib.addAtoms( len(pos_), _np_as(pos_,c_double_p), _np_as(npe_,c_int_p)) 

#  void addBonds( int n, int* bond2atom_, double* l0s, double* ks ){
lib.addBonds.argtypes  = [c_int, c_int_p, c_double_p, c_double_p] 
lib.addBonds.restype   =  None
def addBonds(bond2atom_, l0s=None, ks=None):
    return lib.addBonds(len(bond2atom_), _np_as(bond2atom_,c_int_p), _np_as(l0s,c_double_p), _np_as(ks,c_double_p)) 



#  double* setNonBonded( int n, double* REQs){
lib.setNonBonded.argtypes  = [c_int, c_double_p] 
lib.setNonBonded.restype   =  c_double_p
def setNonBonded(REQs):
    n=0
    if REQs is not None: n=len(REQs) 
    return lib.setNonBonded(n, _np_as(REQs,c_double_p)) 



#  bool buildFF( bool bAutoHydrogens, bool bAutoAngles, bool bSortBonds ){
lib.buildFF.argtypes  = [c_bool, c_bool, c_bool] 
lib.buildFF.restype   =  c_int
def buildFF(bAutoHydrogens, bAutoAngles, bSortBonds):
    return lib.buildFF(bAutoHydrogens, bAutoAngles, bSortBonds) 



#  double setupOpt( double dt, double damp, double f_limit, double l_limit ){
lib.setupOpt.argtypes  = [c_double, c_double, c_double, c_double] 
lib.setupOpt.restype   =  c_double
def setupOpt(dt=0.2, damp=0.2, f_limit=10.0, l_limit=0.2 ):
    return lib.setupOpt(dt, damp, f_limit, l_limit) 

#  void setBox(double* pmin, double* pmax, double* k){
lib.setBox.argtypes  = [c_double_p, c_double_p, c_double_p] 
lib.setBox.restype   =  None
def setBox(pmin, pmax, k):
    return lib.setBox(_np_as(np.array(pmin),c_double_p), _np_as(np.array(pmax),c_double_p), _np_as(np.array(k),c_double_p)) 


#  double relaxNsteps( int ialg, int nsteps, double Fconv ){
lib.relaxNsteps.argtypes  = [c_int, c_int, c_double] 
lib.relaxNsteps.restype   =  c_double
def relaxNsteps(nsteps, Fconv=1e-6, ialg=0 ):
    return lib.relaxNsteps(ialg, nsteps, Fconv)


#addAtoms( np.array([[1,2,3],[4,5,6]],dtype=np.float), None  )
#addAtoms( np.array([[1,2,3],[4,5,6]],dtype=np.float), np.array([[1,0],[0,1]]) )

