'''

Flexible Atom sp-hybridization forcefield
 - each atom has 4 atomic orbitals of 3 types { sigma-bond, election-pair, pi-pond }
 - there are two kinds of interaction "onsite" and "bond"
 - "onsite" interaction
    - sigma and e-pair try to maximize angle between them ( leadign to tetrahedral configuration )
    - pi orbitals try to orhogonalize
 - "bond" interaction
    - spring constant for bonds
    - pi-pi alignment

see.

cpp/FARFF.h
cpp/FARFF.cpp


# === problem with relative imports in python-3     see :  https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py/27876800#27876800
'''


import ctypes
import sys
from ctypes import c_double, c_int

import numpy as np

from .. import cpp_utils

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
    "double* getAtomMapStrenghs(){",
    "double* getBondMapStrenghs(){",
    "void setupFF( int natom, int* types ){",
    "void setGridShape( int* n, double* cell ){",
    "void bindGrids( double* atomMap, double*  bondMap ){",
    "void setupOpt( double dt, double damp, double f_limit, double l_limit ){",
    "void setBox(double* pmin, double* pmax, double* k){",
    "double relaxNsteps( int nsteps, double Fconv, int ialg ){",
]

libSDL = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libSDL2.so", ctypes.RTLD_GLOBAL )
libGL  = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libGL.so",   ctypes.RTLD_GLOBAL )

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

#  double* getAtomMapStrenghs(){
lib.getAtomMapStrenghs.argtypes  = []
lib.getAtomMapStrenghs.restype   =  c_double_p
def getAtomMapStrenghs(natom):
    return np.ctypeslib.as_array(  lib.getAtomMapStrenghs(), shape=(natom,))

#  double* getBondMapStrenghs(){
lib.getBondMapStrenghs.argtypes  = []
lib.getBondMapStrenghs.restype   =  c_double_p
def getBondMapStrenghs(nbond):
    return np.ctypeslib.as_array(  lib.getBondMapStrenghs(), shape=(nbond,))

#  void setupFF( int natom, int* types ){
lib.setupFF.argtypes  = [c_int, c_int_p]
lib.setupFF.restype   =  None
def setupFF( n=None, itypes=None ):
    if itypes is None:
        itypes = np.zeros( n, dtype=np.int32)
    else:
        n = len(itypes)
    print("type(itypes)",type(itypes),"itypes",itypes)
    return lib.setupFF( n, _np_as(itypes,c_int_p) )

#  void setGridShape( int * n, double * cell ){
lib.setGridShape.argtypes  = [c_int_p, c_double_p]
lib.setGridShape.restype   =  None
def setGridShape( n, cell):
    n=np.array(n,dtype=np.int32);
    return lib.setGridShape(_np_as(n,c_int_p), _np_as(cell,c_double_p))

#  void bindGrids( double* atomMap, double*  bondMap ){
lib.bindGrids.argtypes  = [c_double_p, c_double_p]
lib.bindGrids.restype   =  None
def bindGrids(atomMap, bondMap):
    return lib.bindGrids(_np_as(atomMap,c_double_p), _np_as(bondMap,c_double_p))

#  double setupOpt( double dt, double damp, double f_limit, double l_limit ){
lib.setupOpt.argtypes  = [c_double, c_double, c_double, c_double]
lib.setupOpt.restype   =  None
def setupOpt(dt=0.2, damp=0.2, f_limit=10.0, l_limit=0.2 ):
    lib.setupOpt(dt, damp, f_limit, l_limit)

#  void setBox(double* pmin, double* pmax, double* k){
lib.setBox.argtypes  = [c_double_p, c_double_p, c_double_p]
lib.setBox.restype   =  None
def setBox(pmin, pmax, k):
    return lib.setBox(_np_as(pmin,c_double_p), _np_as(pmax,c_double_p), _np_as(k,c_double_p))

#  double relaxNsteps( int nsteps, double Fconv, int ialg ){
lib.relaxNsteps.argtypes  = [c_int, c_double, c_int ]
lib.relaxNsteps.restype   =  c_double
def relaxNsteps(nsteps, Fconv=1e-6, ialg=0):
    return lib.relaxNsteps( nsteps, Fconv, ialg)

# ================= Python Functions

def blur( E ):
    return (E[:-1,:-1] + E[1:,:-1] + E[:-1,1:] + E[1:,1:])*0.25

def derivGrid_25D( E, F, dx, dy ):
    F[:,:,:,1] = ( E[2:,1:-1,None] - E[:-2,1:-1,None] )*(.5/dx)
    F[:,:,:,0] = ( E[1:-1,2:,None] - E[1:-1,:-2,None] )*(.5/dy)
    F[:,:,:,2] = 0
    return F

def makeGridFF( fff,  fname_atom='./Atoms.npy', fname_bond='./Bonds.npy',   dx=0.1, dy=0.1 ):

    atomMap = np.load( fname_atom )
    bondMap = np.load( fname_bond )

    atomMap = blur( atomMap )
    bondMap = blur( bondMap )

    atomMapF = np.empty( (atomMap.shape[0]-2,atomMap.shape[1]-2,1, 3), dtype=np.float64 )
    bondMapF = np.empty( (bondMap.shape[0]-2,bondMap.shape[1]-2,1, 3), dtype=np.float64 )
    atomMapF*=-1
    bondMapF*=-1
    derivGrid_25D( atomMap, atomMapF, dx, dy )
    derivGrid_25D( bondMap, bondMapF, dx, dy )

    lvec = np.array( [[atomMap.shape[0]*dx,0.0,0.0],[0.0,atomMap.shape[1]*dy,0.0],[0.0,0.0,2.0]] )

    print(" shape atomMap, bondMap ", atomMapF.shape, bondMapF.shape)

    return atomMapF, bondMapF, lvec

class EngineFARFF():

    NmaxIter      = 10000
    NstepPerCheck = 10
    Fconv         = 1.-6

    # --- optimizer params
    dt      = 0.05
    damp    = 0.2
    f_limit = 100.0
    l_limit = 0.2

    def __init__(self):
        pass

    def preform_relaxation( self, molecule=None, xyzs=None, Zs=None, qs=None, lvec=None, atomMap=None, bondMap=None, Fconv=-1e-5 ):
        if molecule is not None:
            xyzs = molecule.xyzs
            molecule.Zs
            molecule.qs
        print( " # preform_relaxation - init " )

        natom  = len(xyzs)
        ndof   = reallocFF(natom)
        norb   = ndof - natom
        self.natom = natom; self.ndof = ndof; self.norb = norb;
        self.dofs   = getDofs(self.ndof)   ; print("dofs.shape ", self.dofs.shape)
        self.apos   = self.dofs[:natom]    ; print("apos.shape ", self.apos.shape)
        self.opos   = self.dofs[natom:]    ; print("opos.shape ", self.opos.shape)
        self.apos   = xyzs.copy()

        print( " # preform_relaxation - set DOFs " )

        # --- subtract center of mass
        cog = np.sum( self.apos, axis=0 )
        cog*=(1./natom)
        self.apos -= cog[None,:]
        print( " # preform_relaxation - set DOFs [1] " )
        setupFF     (n=natom)   # use default atom type
        print( " # preform_relaxation - set DOFs [2]" )
        print( " # preform_relaxation - set DOFs [3]" )
        self.atomMap = atomMap # we have to keep it so it is not garbage collected
        self.bondMap = bondMap
        if atomMap is not None:
            print( " atomMap.shape ", atomMap.shape  )
            setGridShape( atomMap.shape[:3]+(1,), lvec )     # ToDo :    this should change if 3D force-field is used
        bindGrids   ( atomMap, bondMap )
        print( " # preform_relaxation - set DOFs [4]" )
        setupOpt(dt=self.dt, damp=self.damp, f_limit=self.f_limit, l_limit=self.l_limit )

        # ! this must be done after setupFF
        self.atomMapStrenghs = getAtomMapStrenghs(natom)
        self.bondMapStrenghs = getBondMapStrenghs(norb)
        self.atomMapStrenghs[:] = 0.0
        self.bondMapStrenghs[:] = 0.0

        print( " # preform_relaxation - to Loop " )

        self.NmaxIter = 10; print( "DEBUG self.NmaxIter = 10; " )
        for i in range( int( self.NmaxIter/self.NstepPerCheck )+1 ):
            F2err = relaxNsteps( self.NstepPerCheck, Fconv=Fconv, ialg=0 )
            print("[%i]|F| %g " %(i*self.NstepPerCheck, np.sqrt(F2err) ) )
            if F2err<(Fconv*Fconv):
                break
        return self.apos[:,:].copy()


if __name__ == "__main__":

    import time

    from .. import io
    if __package__ is None:
        import atomicUtils as au
        import GLView as glv
    else:
        from .. import atomicUtils as au
        from . import GLView as glv

    fff = sys.modules[__name__]
    xyzs, Zs, qs, _ = io.loadXYZ("input.xyz")

    natom  = len(xyzs)
    ndof   = fff.reallocFF(natom)
    norb   = ndof - natom
    dofs   = fff.getDofs(ndof)       ; print("dofs.shape ", dofs.shape)
    apos   = dofs[:natom]            ; print("apos.shape ", apos.shape)
    opos   = dofs[natom:]            ; print("opos.shape ", opos.shape)

    apos[:,:] = xyzs[:,:] #

    cog = np.sum( apos, axis=0 )
    cog*=(1./natom)
    apos -= cog[None,:]

    fff.setupFF(n=natom)   # use default atom type

    try:
        print( "loading atomsMap and bondMap ... " )
        atomMapF, bondMapF, lvecMap = makeGridFF( fff )    # prevent GC from deleting atomMapF, bondMapFF
        if atomMapF is not None:
            setGridShape( atomMapF.shape[:3]+(1,), lvecMap )     # ToDo :    this should change if 3D force-field is used
        bindGrids   ( atomMapF, bondMapF )
    except Exception as e:
        print( e )
        print( "CANNOT LOAD atomsMap and bondMap !!!! " )

    fff.setupOpt(dt=0.05, damp=0.2, f_limit=100.0, l_limit=0.2 )

    glview = glv.GLView()
    for i in range(1000000):
        glview.pre_draw()
        F2err = fff.relaxNsteps(1, Fconv=1e-6, ialg=0)
        print("|F| ", np.sqrt(F2err))
        if F2err<1e-12:
            print( "CONVERGED" )
            exit()
        if glview.post_draw(): break
        time.sleep(.05)
