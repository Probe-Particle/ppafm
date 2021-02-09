import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os
from . import cpp_utils

cpp_name='PolyCycles'
#cpp_utils.compile_lib( cpp_name  )
cpp_utils.make("PolyCycles")
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

array1ui = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')
array1i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array2i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=2, flags='CONTIGUOUS')
array1d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

# ========= C functions


'''
#double* getVpos  (){
lib.getVpos.argtypes = []
lib.getVpos.restype  = ctypes.POINTER(c_double)
def getVpos(nv):
    return np.ctypeslib.as_array( lib.getVpos(), shape=(nv,2) )

#double* getCpos  (){ 
lib.getCpos.argtypes = []
lib.getCpos.restype  = ctypes.POINTER(c_double)
def getCpos(nc):
    return np.ctypeslib.as_array( lib.getCpos(), shape=(nc,2) )
'''


#double* getPos  (){ 
lib.getPos.argtypes = []
lib.getPos.restype  = ctypes.POINTER(c_double)
def getPos(nc,nv):
    arr = np.ctypeslib.as_array( lib.getPos(), shape=(nc+nv,2) )
    return arr[:nc],arr[nc:] 

# double setupOpt( double dt, double damp, double f_limit, double v_limit ){
lib.setupOpt.argtypes = [c_double,c_double,c_double,c_double]
lib.setupOpt.restype  = c_int
def setupOpt(dt=1.0,damping=0.05,f_limit=10.0,v_limit=10.0):
    return lib.setupOpt(dt,damping,f_limit,v_limit );

#void setup( int ncycles, int* nvs ){
lib.setup.argtypes = [c_int,array1i]
lib.setup.restype  = c_int
def setup(nvs):
    return lib.setup(len(nvs),nvs)

#void init( double* angles ){
lib.init.argtypes = [array1d]
lib.init.restype  = None
def init(angles=None):
    lib.init(angles)

#double relaxNsteps( int kind, int nsteps, double F2conf, double dt, double damp ){
lib.relaxNsteps.argtypes = [c_int, c_int,c_double]
lib.relaxNsteps.restype  = c_double
def relaxNsteps(kind=1, nsteps=10, F2conf=-1.0):
    return lib.relaxNsteps(kind, nsteps, F2conf )


'''
class RFF():

    def __init__(self, n ):
        self.prepare(n)

    def prepare(self,n):
        self.natom = 20
        ralloc(natom)
        self.types  = getTypes(natom)
        self.poss   = getPoss(natom)
        self.qrots  = getQrots(natom)
        self.hbonds = getHbonds(natom)
        self.ebonds = getEbonds(natom)
        self.caps   = getBondCaps(natom)

    def genRandom():
        #itypes  = np.random.randint( 2, size=natom, dtype=np.int32 ); print "itypes", itypes
        self.itypes  = (np.random.rand( self.natom )*1.3 ).astype(np.int32); print "itypes", itypes
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
            #print ">> itr ", itr," F2 ", F2 #, caps
            #xyzs, itypes_ = rff.h2bonds( itypes, poss, hbonds, bsc=1.1 )
            #xyzs, itypes_ = rff.removeSaturatedBonds(caps, itypes_, xyzs )
            #au.writeToXYZ( fout, itypes_, xyzs  )

    def relaxAndPassivate():
        t1 = time.clock();
        #fout = open( "rff_movie.xyz",'w')
        self.relax( nstep )
        passivateBonds( -0.1 )
        #print "passivation ", caps
        self.relax( nstep )
        #fout.close()
        t2 = time.clock();
        print "Relaxation time ", t2-t1
'''

