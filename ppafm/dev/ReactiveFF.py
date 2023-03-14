import ctypes
from ctypes import c_double, c_int

import numpy as np

from . import cpp_utils

cpp_name='ReactiveFF'
cpp_utils.make("RR")
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes

array1ui = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')
array1i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array2i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=2, flags='CONTIGUOUS')
array1d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

# ========= C functions

#void insertAtomType( int nbond, int ihyb, double rbond0, double aMorse, double bMorse, double c6, double R2vdW, double Epz ){
lib.insertAtomType.argtypes = [ c_int, c_int, c_double, c_double, c_double, c_double, c_double, c_double ]
lib.insertAtomType.restype  = c_int
def insertAtomType( nbond, ihyb, rbond0, aMorse, bMorse, c6, R2vdW, Epz ):
    return lib.insertAtomType( nbond, ihyb, rbond0, aMorse, bMorse, c6, R2vdW, Epz )

#void reallocFF(int natom){
lib.reallocFF.argtypes = [c_int]
lib.reallocFF.restype  = None
def reallocFF(natom):
    lib.reallocFF(natom)

#void reallocFF(int natom){
lib.clean.argtypes = []
lib.clean.restype  = None
def clean():
    lib.clean()

#int*    getTypes(){
lib.getTypes.argtypes = []
lib.getTypes.restype  = ctypes.POINTER(c_int)
def getTypes(natom):
    ptr = lib.getTypes( )
    print(ptr)
    return np.ctypeslib.as_array( ptr, shape=(natom,))

#double* getPoss (){
lib.getPoss.argtypes = []
lib.getPoss.restype  = ctypes.POINTER(c_double)
def getPoss(natom):
    return np.ctypeslib.as_array( lib.getPoss( ), shape=(natom,3))

#double* getQrots(){
lib.getQrots.argtypes = []
lib.getQrots.restype  = ctypes.POINTER(c_double)
def getQrots(natom):
    return np.ctypeslib.as_array( lib.getQrots( ), shape=(natom,4))

#double* getHbonds(){
lib.getHbonds.argtypes = []
lib.getHbonds.restype  = ctypes.POINTER(c_double)
def getHbonds(natom):
    return np.ctypeslib.as_array( lib.getHbonds( ), shape=(natom,4,3) )

#double* getEbonds(){
lib.getEbonds.argtypes = []
lib.getEbonds.restype  = ctypes.POINTER(c_double)
def getEbonds(natom):
    return np.ctypeslib.as_array( lib.getEbonds( ), shape=(natom,4) )

#double* getBondCaps(){
lib.getBondCaps.argtypes = []
lib.getBondCaps.restype  = ctypes.POINTER(c_int)
def getBondCaps(natom):
    return np.ctypeslib.as_array( lib.getBondCaps( ), shape=(natom,4) )

#void setTypes( int natoms, int* types ){
lib.setTypes.argtypes = [c_int, array1i]
lib.setTypes.restype  = None
def setTypes(natom, itypes):
    lib.setTypes(natom, itypes)

#void setSurf(double K, double x0, double* h ){
lib.setSurf.argtypes = [c_double, c_double, array1d]
lib.setSurf.restype  = None
def setSurf(K=-1.0, x0=0.0, h=np.array([0.0,0.0,1.0]) ):
    lib.setSurf(K, x0, h)

#void setBox(double K, double fmax, double* p0, double* p1 ){
lib.setBox.argtypes = [c_double, c_double, array1d, array1d]
lib.setBox.restype  = None
def setBox( p0, p1, K=-1.0, fmax=1.0 ):
    lib.setBox(K, fmax, p0, p1)

#int passivateBonds( double Ecut ){
lib.passivateBonds.argtypes = [c_double]
lib.passivateBonds.restype  = c_int
def passivateBonds( Ecut=-0.1 ):
    return lib.passivateBonds( Ecut )

#double relaxNsteps( int nsteps, double F2conf ){
lib.relaxNsteps.argtypes = [ c_int, c_double, c_double, c_double ]
lib.relaxNsteps.restype  = c_double
def relaxNsteps( nsteps=10, F2conf=0.0, dt=0.05, damp=0.9, ):
    return lib.relaxNsteps( nsteps, F2conf, dt, damp )


# === Charge

# double* getChargeJ        ()
lib.getChargeJ.argtypes = []
lib.getChargeJ.restype  = ctypes.POINTER(c_double)
def getChargeJ(natom):
    return np.ctypeslib.as_array( lib.getChargeJ( ), shape=(natom,natom) )

#double* getChargeQs       ()
lib.getChargeQs.argtypes = []
lib.getChargeQs.restype  = ctypes.POINTER(c_double)
def getChargeQs(natom):
    return np.ctypeslib.as_array( lib.getChargeQs( ), shape=(natom,) )

#double* getChargeFs       ()
lib.getChargeFs.argtypes = []
lib.getChargeFs.restype  = ctypes.POINTER(c_double)
def getChargeFs(natom):
    return np.ctypeslib.as_array( lib.getChargeFs( ), shape=(natom,) )

#double* getChargeAffinitis()
lib.getChargeAffinitis.argtypes = []
lib.getChargeAffinitis.restype  = ctypes.POINTER(c_double)
def getChargeAffinitis(natom):
    return np.ctypeslib.as_array( lib.getChargeAffinitis( ), shape=(natom,) )

#double* getChargeHardness ()
lib.getChargeHardness.argtypes = []
lib.getChargeHardness.restype  = ctypes.POINTER(c_double)
def getChargeHardness(natom):
    return np.ctypeslib.as_array( lib.getChargeHardness( ), shape=(natom,) )

#void    setTotalCharge(double q)
lib.setTotalCharge.argtypes = [ c_double ]
lib.setTotalCharge.restype  = None
def setTotalCharge( q ):
    return lib.setTotalCharge( q )

#void setupChargePos( int natom, double* pos, int* itypes, double* taffins, double* thards ){
lib.setupChargePos.argtypes = [ c_int, array2d, array1i, array1d, array1d ]
lib.setupChargePos.restype  = None
def setupChargePos( pos, itypes, taffins, thards  ):
    return lib.setupChargePos( len(itypes), pos, itypes, taffins, thards )

#double relaxCharge( int nsteps, double F2conf, double dt, double damp ){
lib.relaxCharge.argtypes = [ c_int, c_double, c_double, c_double ]
lib.relaxCharge.restype  = c_double
def relaxCharge( nsteps=10, F2conf=0.0, dt=0.05, damp=0.9, ):
    return lib.relaxCharge( nsteps, F2conf, dt, damp )

# ========= Python Functions

def h2bonds( itypes, poss, hbonds, bsc=1.1 ):
    natom = len(poss)
    #ntot  = natom + natom*4
    xyzs    = np.zeros( (natom,5,3) )
    itypes_ = np.zeros( (natom,5), dtype=np.int32 )

    mask = itypes>0

    xyzs[:,0,:] = poss[:,:]
    xyzs[:,1,:] = poss[:,:] + hbonds[:,0,:]*bsc
    xyzs[:,2,:] = poss[:,:] + hbonds[:,1,:]*bsc
    xyzs[:,3,:] = poss[:,:] + hbonds[:,2,:]*bsc
    xyzs[:,4,:] = poss[:,:] + hbonds[:,3,:]*bsc
    itypes_[:,0 ] = 5 + itypes[:]
    itypes_[:   ,1:4] = 1
    itypes_[mask,4  ] = 1
    return xyzs.reshape(-1,3), itypes_.reshape(-1)

def ebond2caps( ebonds, Ecut=-0.1 ):
    caps = np.zeros(ebonds.shape,dtype=np.int32) - 1
    caps[ebonds>Ecut] = 1;
    return caps

def removeSaturatedBonds(caps, itypes, xyzs,  ):
    itypes = itypes.reshape(-1,5)
    xyzs   = xyzs  .reshape(-1,5,3)
    #print ebonds
    #mask  = ebonds > Ecut
    mask   = caps >= 0
    mask[ itypes[:,4]==0,3] = False
    #print mask
    xyzs_   = [ xyzs  [:,0,:], xyzs  [mask[:,0],1,:], xyzs  [mask[:,1],2,:], xyzs  [mask[:,2],3,:], xyzs  [mask[:,3],4,:] ]
    itypes_ = [ itypes[:,0  ], itypes[mask[:,0],1  ], itypes[mask[:,1],2  ], itypes[mask[:,2],3  ], itypes[mask[:,3],4  ] ]
    return np.concatenate(xyzs_), np.concatenate(itypes_)

class RFF():

    def __init__(self, n ):
        self.prepare(n)

    def prepare(self,n):
        self.natom = 20
        reallocFF(natom)
        self.types  = getTypes(natom)
        self.poss   = getPoss(natom)
        self.qrots  = getQrots(natom)
        self.hbonds = getHbonds(natom)
        self.ebonds = getEbonds(natom)
        self.caps   = getBondCaps(natom)

    def genRandom():
        self.itypes  = (np.random.rand( self.natom )*1.3 ).astype(np.int32); print("itypes", itypes)
        setTypes( natom, self.itypes )
        self.poss [:,:]  = ( np.random.rand(self.natom,3)-0.5 ) * 10.0
        self.poss [:,2]  = 0.15
        self.qrots[:,:]  = np.random.rand(self.natom,4)-0.5
        rs               = np.sum(self.qrots**2, axis=1 )
        self.qrots      /= rs[:,None]

        setBox( p0=np.array([-5.0,-5.0,-1.0]), p1=np.array([5.0,5.0,1.0]), K=-1.0, fmax=1.0  )
        setSurf(K=-0.2, x0=0.0, h=np.array([0.0,0.0,1.0]) )

    def relax( nstep ):
        for itr in range(10):
            F2 = relaxNsteps( nsteps=50, F2conf=0.0, dt=0.15, damp=0.9 )

    def relaxAndPassivate():
        t1 = time.clock();
        self.relax( nstep )
        passivateBonds( -0.1 )
        self.relax( nstep )
        t2 = time.clock();
        print("Relaxation time ", t2-t1)
