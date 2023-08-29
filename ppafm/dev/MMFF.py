import ctypes
import os
from ctypes import c_bool, c_double, c_int

import numpy as np

from . import cpp_utils, io

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

cpp_name='MMFF'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes

# ========= C functions

#void clear(){
lib.clear.argtypes = []
lib.clear.restype  = None
def clear():
    return lib.clear()

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

#  bool buildFF( bool bAutoHydrogens, bool bAutoAngles, bool bSortBonds, bDummyPi, bDummyEpair ){
lib.buildFF.argtypes  = [c_bool, c_bool, c_bool, c_bool, c_bool ]
lib.buildFF.restype   =  c_int
def buildFF(bAutoHydrogens, bAutoAngles, bSortBonds, bDummyPi, bDummyEpair=False):
    return lib.buildFF(bAutoHydrogens, bAutoAngles, bSortBonds, bDummyPi, bDummyEpair)

#  double setupOpt( double dt, double damp, double f_limit, double l_limit ){
lib.setupOpt.argtypes  = [c_double, c_double, c_double, c_double]
lib.setupOpt.restype   =  None
def setupOpt(dt=0.2, damp=0.2, f_limit=10.0, l_limit=0.2 ):
    lib.setupOpt(dt, damp, f_limit, l_limit)

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

# ============= Pure python

def assignAtomTypes( ao, nngs, epairProb=0.4, bNitrogenFix=True, ne2elem=[ 'C', 'N', 'O' ] ):
    na=len(ao)
    # --- electron pairs
    npis      = np.round(ao).astype(np.int32).copy()
    nemax     = (4-nngs-npis)
    ne_rnd    =  epairProb * np.random.rand( na )
    nepairs   =  np.round( nemax*ne_rnd ).astype(np.int32)
    # --- make low-bond order aromatic atoms to nitrogens with pi-orbital (to prevent non-planar wrikles of pi-system)
    if bNitrogenFix:
        mask_N = np.logical_and( (nngs==3) , (npis==0) )
        nepairs[mask_N] = 1
    # select elemtns by number of free electron pairs
    elems = [ ne2elem[ne] for ne in nepairs ]
    # --- convert electron pairt to pi-orbital for special nitrogens inside pi-system
    if bNitrogenFix:
        npis   [mask_N] = 1
        nepairs[mask_N] = 0
    return npis, nepairs, elems

def relaxMolecule(
        apos, bonds, npis, nepairs, elems,
        bAutoHydrogens=True, bAutoAngles=True, bSortBonds=True, bNonBonded=True, bDummyPi=True, bBox=True, bMovie=False, bMaskDummy=True,
        box_k=[0.,0.,0.05], box_pmin=[.0,0,0], box_pmax=[.0,0,0],
        type2elem = ['U','U','He','Ne','H'],
        Fconv=1e-6, Nmax=1000, perSave=10,
        fname=None,
    ):

    clear()

    bDummyEpair = False  # not yet tested/implemented

    # --- set atomic configurations ( set number of pi-orbitals and electron-pairs for each atom )
    len(apos)
    bonds = bonds.astype(np.int32).copy()
    aconf      = np.zeros( (len(apos),2), dtype=np.int32 )
    aconf[:,0] = npis                                                   # pi
    aconf[:,1] = nepairs
    # --- insert atoms, bonds and configurations (npi,nepair)
    addAtoms(apos, aconf )
    addBonds(bonds, l0s=None, ks=None)
    # --- build the molecule (including hydrogen passivations, pi-bonds and automatically asigned angular force-field )
    natom = buildFF(bAutoHydrogens,bAutoAngles,bSortBonds, bDummyPi, bDummyEpair )
    types = getAtomTypes(natom)            #;print types
    # --- assign types for newly created atoms (passivation, dummy-pi, dummy-epair)
    mask_dummy = (types==types)
    if bAutoHydrogens:
        if bMaskDummy:
            mask_dummy = np.logical_and( (types!=-2), (types!=-3) )
        elems += [  type2elem[-t] for t in types[len(elems):] ]
        elems = np.array(elems)
    # --- non covalent force field
    if(bNonBonded): setNonBonded (None)
    if bBox:        setBox       ( box_pmin,box_pmax, box_k)

    setupOpt()
    pos = getPos (natom)

    if bMovie:
        if fname is None:
            movie_name = "movie_MMFF.xyz"
        else:
            movie_name = "movie_"+fname
        if os.path.exists(movie_name): os.remove(movie_name)
        for i in range(Nmax/perSave):
            f = relaxNsteps(perSave, Fconv=Fconv)
            io.saveXYZ(movie_name, pos[mask_dummy], elems[mask_dummy], append=True)
            if(f<1e-6): break
    else:
        relaxNsteps(Nmax,Fconv=Fconv)
    if fname is not None:
        io.saveXYZ(fname, pos[mask_dummy], elems[mask_dummy])
    return pos.copy(),types.copy(),elems
