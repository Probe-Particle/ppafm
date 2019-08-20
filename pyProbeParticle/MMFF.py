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






'''

# ===== To generate Interfaces automatically from headers call:

cpp_utils.writeFuncInterfaces([
"void addAtoms( int n, double* pos_, int* npe_ ){",
"void addBonds( int n, int* bond2atom_, double* l0s, double* ks ){",
"double* setNonBonded( int n, double* REQs){",
"bool buildSystem( bool bAutoHydrogens, bool bAutoAngles, bool bSortBonds ){",
"double setupOpt( double dt, double damp, double f_limit, double l_limit ){",
"double relaxNsteps( int ialg, int nsteps, double F2conf ){",
])
'''

#exit()

cpp_name='MMFF'
#cpp_utils.compile_lib( cpp_name  )
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

# ========= C functions



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



#  bool buildSystem( bool bAutoHydrogens, bool bAutoAngles, bool bSortBonds ){
lib.buildSystem.argtypes  = [c_bool, c_bool, c_bool] 
lib.buildSystem.restype   =  c_bool
def buildSystem(bAutoHydrogens, bAutoAngles, bSortBonds):
    return lib.buildSystem(bAutoHydrogens, bAutoAngles, bSortBonds) 



#  double setupOpt( double dt, double damp, double f_limit, double l_limit ){
lib.setupOpt.argtypes  = [c_double, c_double, c_double, c_double] 
lib.setupOpt.restype   =  c_double
def setupOpt(dt=0.25, damp=0.1, f_limit=20.0, l_limit=0.2 ):
    return lib.setupOpt(dt, damp, f_limit, l_limit) 



#  double relaxNsteps( int ialg, int nsteps, double F2conf ){
lib.relaxNsteps.argtypes  = [c_int, c_int, c_double] 
lib.relaxNsteps.restype   =  c_double
def relaxNsteps(nsteps, F2conf=1e-6, ialg=0 ):
    return lib.relaxNsteps(ialg, nsteps, F2conf)


#addAtoms( np.array([[1,2,3],[4,5,6]],dtype=np.float), None  )
#addAtoms( np.array([[1,2,3],[4,5,6]],dtype=np.float), np.array([[1,0],[0,1]]) )

