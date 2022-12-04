import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os

LIB_PATH      = os.path.dirname( os.path.realpath(__file__) )
LIB_PATH_CPP  = os.path.normpath(LIB_PATH+'../../../'+'/cpp/Build/libs/Molecular')
LIB_PATH_CPP  = os.path.normpath(LIB_PATH+'/../bin/')

def recompile(path):
    print(( "recompile path :", path ))
    dir_bak = os.getcwd()
    os.chdir( path)
    os.system("make" )
    os.chdir( dir_bak )
    print(( os.getcwd() ))

lib = ctypes.CDLL( LIB_PATH_CPP+"/libRigidMol.so_" )

array1ui = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')
array1i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array2i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=2, flags='CONTIGUOUS')
array1d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

default_icolor = int("0xFF101010", 0)

# ========= C functions

#void initRigidSubstrate( char* fname, int* ns, double* pos0, double* cell ){
lib.initRigidSubstrate.argtypes = [c_char_p,array1i,array1d,array2d]
lib.initRigidSubstrate.restype  = None
def initRigidSubstrate( fname, ns, pos, cell ):
    lib.initRigidSubstrate( fname, ns, pos, cell  )

#void recalcGridFF( int* ns){
lib.recalcGridFF.argtypes = [array1i]
lib.recalcGridFF.restype  = None
def recalcGridFF( ns ):
    lib.recalcGridFF( ns )

#void saveGridFF(){
lib.saveGridFF.argtypes = []
lib.saveGridFF.restype  = None
def saveGridFF( ):
    lib.saveGridFF( )

#void loadGridFF(){
lib.loadGridFF.argtypes = []
lib.loadGridFF.restype  = None
def loadGridFF( ):
    lib.loadGridFF( )

#void debugSaveGridFF(const char* fname, double* testREQ ){
lib.debugSaveGridFF.argtypes = [c_char_p, array1d]
lib.debugSaveGridFF.restype  = None
def debugSaveGridFF(fname, testREQ ):
    lib.debugSaveGridFF(fname, testREQ )

#void initParams( char* fname_atomTypes, char* fname_bondTypes ){
lib.initParams.argtypes = [c_char_p, c_char_p]
lib.initParams.restype  = None
def initParams( fname_atomTypes, fname_bondTypes):
    lib.initParams(fname_atomTypes, fname_bondTypes )

#int loadMolType   ( const char* fname ){
lib.loadMolType.argtypes = [c_char_p]
lib.loadMolType.restype  = c_int
def loadMolType( fname ):
    return lib.loadMolType( fname )

#int insertMolecule( int itype, double* pos, double* rot, bool rigid ){
lib.insertMolecule.argtypes = [c_int, array1d, array2d, c_bool]
lib.insertMolecule.restype  = c_int
def insertMolecule( itype, pos, rot, rigid ):
    return lib.insertMolecule( itype, pos, rot, rigid )

#void bakeMMFF(){
lib.clear.argtypes = []
lib.clear.restype  = None
def clear( ):
    lib.clear( )

#void bakeMMFF(){
lib.bakeMMFF.argtypes = []
lib.bakeMMFF.restype  = None
def bakeMMFF( ):
    lib.bakeMMFF( )

#void prepareOpt(){
lib.prepareOpt.argtypes = []
lib.prepareOpt.restype  = None
def prepareOpt( ):
    lib.prepareOpt( )

#double relaxNsteps( int nsteps, double F2conf ){
lib.relaxNsteps.argtypes = [c_int, c_double]
lib.relaxNsteps.restype  = c_double
def relaxNsteps( nsteps, F2conf ):
    return lib.relaxNsteps( nsteps, F2conf )

#void save2xyz( char * fname ){
lib.save2xyz.argtypes = [c_char_p]
lib.save2xyz.restype  = None
def save2xyz( fname ):
    lib.save2xyz( fname )

#write2xyz( int i ){
lib.write2xyz.argtypes = [c_int]
lib.write2xyz.restype  = None
def write2xyz( i ):
    lib.write2xyz( i )

#openf(char* fname, int i, char* mode ){
lib.openf.argtypes = [c_char_p, c_int, c_char_p ]
lib.openf.restype  = c_int
def openf( fname, i, mode ):
    return lib.openf( fname, i, mode )

#closef(int i){
lib.closef.argtypes = [c_int]
lib.closef.restype  = None
def closef( i ):
    lib.closef( i )

lib.getPoses.argtypes = [ctypes.POINTER(c_int)]
lib.getPoses.restype  = ctypes.POINTER(c_double)
def getPoses():
    n=c_int(0)
    ptr = lib.getPoses(ctypes.byref(n))
    #print "n",n
    return np.ctypeslib.as_array(ptr, shape=(n.value,8))

lib.getAtomPos.argtypes = [ctypes.POINTER(c_int)]
lib.getAtomPos.restype  = ctypes.POINTER(c_double)
def getAtomPos():
    n=c_int(0)
    ptr = lib.getAtomPos(ctypes.byref(n));
    return np.ctypeslib.as_array( ptr, shape=(n.value,3))

#void setOptFIRE( double dt_max, double dt_min, double damp_max, int    minLastNeg, double finc, double fdec, double falpha, double kickStart ){
lib.setOptFIRE.argtypes = [ c_double, c_double, c_double, c_int    , c_double , c_double , c_double , c_double  ]
lib.setOptFIRE.restype  = None
def setOptFIRE( dt_max=0.05, dt_min=0.005, damp_max=0.1, minLastNeg=5, finc=1.1, fdec=0.5, falpha=0.98, kickStart=1.0 ):
    lib.setOptFIRE( dt_max, dt_min, damp_max, minLastNeg, finc, fdec, falpha, kickStart )
